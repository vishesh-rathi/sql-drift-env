"""OpenEnv ``Environment`` implementation for SQLDrift.

Responsibilities:

* Own the private :class:`engine.runtime.RuntimeEpisodeState` and the
  composite :class:`engine.reward.SqlDriftRubric` for the current episode.
* Dispatch each of the eight tool-call payloads to a dedicated
  ``_handle_<tool>`` method that returns a typed
  :class:`models.ToolResult` (or :class:`models.ToolError`).
* Fire drift on a schedule blended with a cooldown: ``max(scheduled,
  first_run_query_step + cooldown)`` before the agent acts on the step
  where drift applies, then recompute the post-drift ground truth hash.
* Publish public observations (:class:`models.SqlDriftObservation`) and a
  strictly sanitised public state snapshot (:class:`models.SqlDriftState`).

Privacy: ``self._runtime`` holds the DuckDB handle, ground-truth hashes,
baseline runtime, and seed. They stay inside this class; the rubric reads
them via a closure, and ``env.state`` exposes only a fixed whitelist of fields.
"""

from __future__ import annotations

import contextlib
import math
import re
import secrets
from random import Random
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import sqlglot
from openenv.core.env_server.interfaces import Environment
from pydantic import BaseModel, ConfigDict, Field

from actors import dba_oracle
from actors.engineering_manager import author_changelog
from engine.drift import apply_drift
from engine.profiler import (
    QueryWatchdogEscalationError,
    execute_hash_timed,
    execute_once_timed,
    execute_once_with_columns,
)
from engine.reward import (
    SPEEDUP_CAP_FOR_INFTY,
    STEP_REBATE_DESCRIBE_TABLE,
    STEP_REBATE_EXPLAIN_QUERY,
    STEP_REBATE_LIST_TABLES,
    STEP_REBATE_READ_CHANGELOG,
    STEP_REBATE_RUN_QUERY,
    STEP_REBATE_SAMPLE_ROWS,
    SqlDriftRubric,
    canonicalize_sql,
    effective_speedup,
)
from engine.runtime import RuntimeEpisodeState
from engine.verifier import canonical_row_hash
from models import (
    REWARD_COMPONENT_KEYS,
    ConsultDBAPayload,
    ConsultDBAResult,
    DescribeTablePayload,
    DescribeTableResult,
    EpisodePhase,
    ExplainQueryPayload,
    ExplainQueryResult,
    ListTablesPayload,
    ListTablesResult,
    ReadChangelogPayload,
    ReadChangelogResult,
    RunQueryPayload,
    RunQueryResult,
    SampleRowsPayload,
    SampleRowsResult,
    SqlDriftAction,
    SqlDriftObservation,
    SqlDriftState,
    SubmitRewritePayload,
    SubmitRewriteResult,
    ToolError,
    ToolErrorCode,
    ToolResult,
)
from scenarios import REGISTRY, get_spec
from skill_library import PlaybookEntry, Store, load_all, retrieve
from utilities.logger import get_module_logger, log_env_reset, log_env_step, log_interaction

from . import settings

if TYPE_CHECKING:
    from scenarios.base import ScenarioSpec

_LOG = get_module_logger(__name__)

DEFAULT_STEP_BUDGET: int = settings.DEFAULT_STEP_BUDGET
MAX_RESULT_ROWS: int = settings.MAX_RESULT_ROWS
QUERY_TIMEOUT_S: float = settings.QUERY_TIMEOUT_S


class _ResetOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str | None = None
    enable_dba_oracle: bool | None = None
    difficulty: Literal["easy", "normal", "hard"] = "normal"
    budget_steps: int = Field(default=DEFAULT_STEP_BUDGET, ge=1)


_READ_ONLY_EXPRESSION_KEYS: frozenset[str] = frozenset({"select", "with"})

# DuckDB exposes a family of table-valued functions and scalar helpers
# that read from the host filesystem or leak introspection state —
# ``read_csv``, ``read_parquet``, ``read_json``, ``read_text``,
# ``parquet_metadata``, ``duckdb_secrets``, ``glob``, etc. They are
# *technically* SELECT-shaped calls so the statement-key check alone
# admits them. We reject any function whose lowercased name starts with
# one of these prefixes or exactly matches one of the known-dangerous
# standalone names. Agent-facing SQL has no legitimate need for any of
# them — the DuckDB connection is pre-populated by the scenario builder.
_DENYLIST_PREFIXES: tuple[str, ...] = (
    "read_",
    "write_",
    "copy_",
    "duckdb_",
    "pragma_",
    "sniff_",
    "parquet_",
    "arrow_",
    "json_table",
    "json_each",
    "sqlite_",
    "load_",
    "install_",
)
_DENYLIST_EXACT: frozenset[str] = frozenset(
    {
        "glob",
        "attach",
        "detach",
        "checkpoint",
        "force_checkpoint",
        "set_secret",
        "create_secret",
        "drop_secret",
        "enable_profiling",
        "disable_profiling",
        "enable_object_cache",
    }
)


def _is_denylisted_function_name(name: str) -> bool:
    """Return True iff ``name`` (case-insensitively) matches a sandbox-escape."""
    lowered = name.lower()
    if lowered in _DENYLIST_EXACT:
        return True
    return any(lowered.startswith(p) for p in _DENYLIST_PREFIXES)


def _function_names(node: sqlglot.exp.Func) -> list[str]:
    """All plausible names to check against the denylist for one AST node.

    sqlglot lowers a few DuckDB calls into dedicated expression classes
    (``ReadCSV``, ``ReadParquet``, …) whose ``.name`` is actually the
    first positional arg — the file path — not the function name. We
    recover the function name from the class name in that case and fall
    back to ``.name`` for the ``Anonymous`` form that covers everything
    else. Including both lets one denylist lookup cover both lowerings.
    """
    cls = type(node).__name__
    out: list[str] = []
    # Derive a snake-case function name from the class name. We insert
    # an underscore at two kinds of CamelCase boundaries:
    #
    # * ``aB``  — normal lower-to-upper (``ReadParquet`` → ``read_parquet``)
    # * ``ABc`` — end of an acronym run (``ReadCSVAuto`` → ``read_csv_auto``)
    #
    # Purely-lowercase class names (``Anonymous``) produce no prefix
    # match; we fall through to ``.name`` below for those.
    if cls and cls[0].isupper():
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", cls).lower()
        out.append(snake)
    name_attr = getattr(node, "name", None)
    if isinstance(name_attr, str) and name_attr:
        out.append(name_attr)
    return out


_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_timeout_s(timeout_s: float | None) -> float:
    """Caller-supplied per-step timeout or the module default.

    ``timeout_s`` is accepted on every OpenEnv ``step()`` (the abstract
    base mandates the keyword). When the caller provides a positive
    value we honour it as the wall-clock budget for any DuckDB query
    this step runs; ``None`` and non-positive values fall back to the
    module-level :data:`QUERY_TIMEOUT_S` so a mis-configured client
    cannot silently disable the watchdog.
    """
    if timeout_s is None or timeout_s <= 0:
        return QUERY_TIMEOUT_S
    return float(timeout_s)


def _initial_schema_synopsis(spec: ScenarioSpec, synopsis: str) -> str:
    """Reset-time synopsis with future drift details removed.

    Drift scenarios should not reveal the exact schema/business-rule
    change before the changelog is published at runtime. We therefore
    trim the authored synopsis at the first ``" Under drift"`` clause on
    reset and only surface the pre-drift schema shape.
    """
    if spec.drift_config is None:
        return synopsis
    predrift, marker, _ = synopsis.partition(" Under drift")
    return predrift if marker else synopsis


def _validate_read_only_sql(sql: str) -> None:
    """Reject anything that isn't a single-statement read-only SELECT/CTE.

    Raises ``ValueError`` so the caller can translate to a typed
    :class:`models.ToolError` with :attr:`ToolErrorCode.INVALID_TOOL_ARGUMENT`.
    This is the only place that mediates what the policy may execute;
    scenario builders and drift DDL call DuckDB directly with privileged
    SQL and deliberately bypass this check.

    Beyond the statement-level gate, this walker also rejects two
    sandbox-escape vectors that would otherwise ride along inside a
    perfectly-shaped SELECT:

    1. Table-valued functions that read from the host filesystem
       (``read_csv``, ``read_parquet``, ``read_json_auto``, ``glob``,
       ``read_text``, …) or leak engine introspection (``duckdb_secrets``
       carries credentials; ``duckdb_settings`` /``duckdb_functions``
       can enumerate available exploits). See :data:`_DENYLIST_PREFIXES`
       / :data:`_DENYLIST_EXACT`.
    2. ``SELECT * FROM 'path/to/x.csv'`` — DuckDB treats a bare string
       literal in a FROM clause as a filesystem path and auto-detects
       the format. There is no function node to inspect in this form,
       so we separately reject any :class:`sqlglot.exp.Table` whose
       backing expression is a string literal.
    """
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL failed to parse: {exc}") from exc

    non_empty = [s for s in statements if s is not None]
    if len(non_empty) != 1:
        raise ValueError("multi-statement SQL is not allowed; submit one SELECT")
    expr = non_empty[0]
    if expr.key not in _READ_ONLY_EXPRESSION_KEYS:
        raise ValueError(
            f"only read-only SELECT/CTE queries are allowed (got {expr.key.upper()} statement)"
        )

    for node in expr.walk():
        # (1) Function-valued sandbox escapes. Inspect both the class
        # name (catches ``ReadCSV`` / ``ReadParquet`` lowerings where
        # ``.name`` holds the file path, not the function name) and
        # ``.name`` (catches the generic ``Anonymous`` form).
        if isinstance(node, sqlglot.exp.Func):
            for fn_name in _function_names(node):
                if _is_denylisted_function_name(fn_name):
                    raise ValueError(
                        f"function {fn_name!r} is not allowed — agent-facing SQL may "
                        "only touch the scenario's in-memory tables"
                    )
        # (2) Bare-path FROM form: ``SELECT * FROM 'x.csv'`` or
        # ``SELECT * FROM '/etc/passwd'``. sqlglot normalises both
        # single- and double-quoted identifiers to
        # ``Identifier(quoted=True)``, so we can't rely on the quote
        # flavour to distinguish a file path from a legitimately-quoted
        # table name. Instead we require every agent-facing table name
        # to be a valid unquoted SQL identifier — the scenarios never
        # emit anything else, and paths always contain ``/``, ``.`` or
        # ``~`` which fail the identifier regex.
        if isinstance(node, sqlglot.exp.Table):
            inner = node.this
            if isinstance(inner, sqlglot.exp.Identifier):
                ident_name = inner.name
                if ident_name and not _VALID_IDENTIFIER_RE.match(ident_name):
                    raise ValueError(
                        f"table identifier {ident_name!r} is not a valid unquoted SQL "
                        "name — reading from file paths or other engine-specific "
                        "resources is not allowed"
                    )


class SqlDriftEnvironment(Environment[SqlDriftAction, SqlDriftObservation, SqlDriftState]):
    """OpenEnv environment for SQL repair + optimization under schema drift."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        skill_store: Store | None = None,
        cleanup_on_close: bool = False,
    ) -> None:
        self._runtime: RuntimeEpisodeState | None = None
        self._skill_store: Store | None = skill_store
        # When True, the skill-store directory is deleted when close() is called.
        # Set this for server-managed per-session stores so disk usage doesn't grow
        # monotonically; see design/codereview.md (session store issue).
        self._cleanup_on_close: bool = cleanup_on_close
        super().__init__(
            rubric=SqlDriftRubric(ctx_provider=lambda: self._require_runtime()),
        )

    # ------------------------------------------------------------------
    # OpenEnv contract
    # ------------------------------------------------------------------

    @log_env_reset
    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SqlDriftObservation:
        options = _ResetOptions.model_validate(kwargs)
        scenario_id = options.scenario_id
        enable_dba_oracle = dba_oracle.is_enabled(options.enable_dba_oracle)
        difficulty = options.difficulty
        budget_steps = options.budget_steps

        if seed is None:
            seed = secrets.randbits(31)
        if episode_id is None:
            episode_id = f"ep-{seed:08x}"
        if scenario_id is None:
            scenario_id = self._pick_scenario_for_seed(seed)

        spec = get_spec(scenario_id)
        instance = spec.materialize(seed, difficulty=difficulty)

        drift_scheduled_step: int | None = None
        if instance.drift_config is not None:
            drift_scheduled_step = Random(seed).randint(
                instance.drift_config.min_step,
                instance.drift_config.max_step,
            )

        self._close_existing_runtime()
        self._runtime = RuntimeEpisodeState(
            episode_id=episode_id,
            seed=seed,
            scenario_id=scenario_id,
            instance=instance,
            conn=instance.conn,
            gt_result_hash_predrift=instance.gt_result_hash_predrift,
            gt_result_hash_postdrift=None,
            baseline_runtime_ms=instance.baseline_runtime_ms,
            baseline_tokens=instance.baseline_tokens,
            baseline_sql_canonical=canonicalize_sql(instance.baseline_sql),
            baseline_postdrift_raises=False,
            drift_scheduled_step=drift_scheduled_step,
            budget_steps=budget_steps,
            dba_oracle_enabled=enable_dba_oracle,
        )

        self._reset_rubric()

        learned_hints = kwargs.get("learned_hints")
        if learned_hints is None:
            learned_hints = self._render_learned_hints(spec, include_drift_cards=False)
        if len(learned_hints) > 800:
            learned_hints = learned_hints[:800]

        rt = self._require_runtime()
        return SqlDriftObservation(
            step=0,
            phase=EpisodePhase.DIAGNOSE,
            last_tool=None,
            tool_result=None,
            drift_fired=False,
            drift_acknowledged=False,
            learned_hints=learned_hints,
            baseline_sql=instance.baseline_sql,
            schema_synopsis=_initial_schema_synopsis(spec, instance.schema_synopsis),
            budget_steps_remaining=rt.budget_steps_remaining,
            reward_components={key: 0.0 for key in REWARD_COMPONENT_KEYS},
            done=False,
            reward=None,
        )

    @log_env_step
    def step(
        self,
        action: SqlDriftAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SqlDriftObservation:
        rt = self._require_runtime()
        if rt.submitted or rt.budget_steps_remaining <= 0:
            raise ValueError("Episode is already finished; call reset() to start a new episode.")
        rt.step_count += 1
        rt.last_step_was_tool_error = False
        rt.last_step_was_repeat_failing_query = False
        rt.last_step_repeat_failing_query_count = 0
        rt.last_step_productive_rebate = 0.0

        self._maybe_fire_drift()

        effective_timeout_s = _resolve_timeout_s(timeout_s)
        try:
            tool_result = self._dispatch(action, timeout_s=effective_timeout_s)
        except QueryWatchdogEscalationError:
            rt.connection_poisoned = True
            rt.phase = EpisodePhase.FINALIZE
            rt.step_count = max(rt.step_count, rt.budget_steps)
            _LOG.error("episode %s aborted after watchdog escalation", rt.episode_id)
            raise
        rt.last_step_was_tool_error = isinstance(tool_result, ToolError)
        if rt.last_step_was_tool_error:
            rt.consecutive_tool_errors += 1
        else:
            rt.consecutive_tool_errors = 0

        done = rt.submitted or rt.budget_steps_remaining <= 0

        obs = SqlDriftObservation(
            step=rt.step_count,
            phase=rt.phase,
            last_tool=action.tool,
            tool_result=tool_result,
            drift_fired=rt.drift_fired,
            drift_acknowledged=rt.drift_acknowledged,
            learned_hints="",
            baseline_sql="",
            schema_synopsis="",
            budget_steps_remaining=rt.budget_steps_remaining,
            reward_components={key: 0.0 for key in REWARD_COMPONENT_KEYS},
            done=done,
            reward=None,
        )
        if rt.drift_acknowledged:
            spec = get_spec(rt.scenario_id)
            obs.learned_hints = self._render_learned_hints(spec, include_drift_cards=True)

        obs.reward = self._apply_rubric(action, obs)
        if self.rubric is not None:
            obs.reward_components = self.rubric.component_scores()

        if done and rt.submitted:
            self._maybe_persist_learned_entry()
        return obs

    def render(self) -> dict[str, Any]:
        """Render the current public state and log the render interaction."""
        rt = self._require_runtime()
        state = self.state
        payload = state.model_dump(mode="json")
        log_interaction(
            event_type="render",
            agent_id=rt.episode_id,
            observation_returned=payload,
            done=rt.submitted or rt.budget_steps_remaining <= 0,
        )
        return payload

    @property
    def state(self) -> SqlDriftState:
        """Sanitised public state snapshot (explicit whitelist)."""
        rt = self._require_runtime()
        return SqlDriftState(
            episode_id=rt.episode_id,
            step_count=rt.step_count,
            scenario_id=rt.scenario_id,
            phase=rt.phase,
            budget_steps_remaining=rt.budget_steps_remaining,
            drift_fired=rt.drift_fired,
            consultations_used=rt.consultations_used,
            submitted=rt.submitted,
        )

    def effective_speedup(self) -> float | None:
        """Return the current episode's effective speedup, if any."""
        rt = self._runtime
        if rt is None:
            return None
        return effective_speedup(rt)

    def close(self) -> None:
        self._close_existing_runtime()
        if self._cleanup_on_close and self._skill_store is not None:
            import shutil

            store_dir = self._skill_store.dir
            shutil.rmtree(store_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Skill-library wiring
    # ------------------------------------------------------------------

    def _render_learned_hints(self, spec: ScenarioSpec, *, include_drift_cards: bool = True) -> str:
        playbook, drift_cards = load_all(self._skill_store)
        drift_kind = None
        if include_drift_cards and spec.drift_config is not None:
            drift_kind = spec.drift_config.kind
        result = retrieve(
            query_tags=spec.tags,
            drift_kind=drift_kind,
            playbook=playbook,
            drift_cards=drift_cards,
        )
        return result.render(max_chars=800)

    def _maybe_persist_learned_entry(self) -> None:
        """Append a PlaybookEntry on terminal success with a meaningful speedup.

        Failures to persist are logged but never re-raised: a training
        rollout should not crash because the on-disk playbook is under
        contention. The skill store itself is crash-safe (atomic writes
        + file-lock) so at-most-once semantics are sufficient here.
        """
        if self._skill_store is None:
            return
        rt = self._require_runtime()
        if not rt.submitted:
            return
        if self.rubric is None:
            return
        scores = self.rubric.component_scores()
        if scores.get("r_correct", 0.0) < 1.0:
            return
        spec = get_spec(rt.scenario_id)
        raw_speedup = effective_speedup(rt)
        # effective_speedup cannot return None here — rt.submitted is True
        # so submitted_runtime_ms is populated — but we guard defensively.
        # ``+∞`` (drift invalidated the baseline) is capped so the on-disk
        # playbook doesn't serialize ``Infinity``, which would round-trip
        # as a JSON parse error on load.
        if raw_speedup is None or math.isinf(raw_speedup):
            speedup_val = float(SPEEDUP_CAP_FOR_INFTY)
        else:
            speedup_val = float(raw_speedup)
        entry = PlaybookEntry(
            tag_set=spec.tags,
            before_snippet=rt.instance.baseline_sql[:200],
            after_snippet=(rt.submitted_sql or "")[:200],
            avg_speedup=speedup_val,
            scenario_family=spec.family,
            source="learned",
        )
        try:
            self._skill_store.append_playbook(entry)
        except Exception as exc:
            _LOG.warning("skill-library append_playbook failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grant_step_rebate_once(self, *, attr: str, rebate: float) -> None:
        rt = self._require_runtime()
        if getattr(rt, attr):
            return
        setattr(rt, attr, True)
        rt.last_step_productive_rebate += rebate

    def _grant_step_rebate_for_table(
        self, *, rewarded_tables_attr: str, table: str, rebate: float
    ) -> None:
        rt = self._require_runtime()
        rewarded = getattr(rt, rewarded_tables_attr)
        if table in rewarded:
            return
        rewarded.add(table)
        rt.last_step_productive_rebate += rebate

    @staticmethod
    def _pick_scenario_for_seed(seed: int) -> str:
        """Deterministic round-robin over the sorted scenario registry."""
        ids = sorted(REGISTRY)
        if not ids:
            raise RuntimeError("no scenarios registered")
        return ids[seed % len(ids)]

    def _require_runtime(self) -> RuntimeEpisodeState:
        if self._runtime is None:
            raise RuntimeError("SqlDriftEnvironment.reset() must be called before step()/state.")
        return self._runtime

    def _close_existing_runtime(self) -> None:
        if self._runtime is not None:
            if self._runtime.connection_poisoned:
                _LOG.error(
                    "skipping close for poisoned DuckDB connection in episode %s",
                    self._runtime.episode_id,
                )
            else:
                with contextlib.suppress(duckdb.Error):
                    self._runtime.conn.close()
            self._runtime = None

    def _maybe_fire_drift(self) -> None:
        """Apply drift when the step index crosses the schedule/cooldown threshold."""
        rt = self._require_runtime()
        if rt.drift_fired:
            return
        if rt.drift_scheduled_step is None:
            return
        if rt.first_run_query_step is None:
            return
        cfg = rt.instance.drift_config
        assert cfg is not None
        minimum = max(rt.drift_scheduled_step, rt.first_run_query_step + cfg.cooldown_steps)
        if rt.step_count < minimum:
            return
        self._fire_drift()

    def _fire_drift(self) -> None:
        """Apply drift, author a changelog, and resolve the post-drift GT hash.

        Failure to recompute the post-drift GT hash is an authoring bug
        (the scenario's ``gt_sql_postdrift`` must execute against the
        just-mutated DB) and we re-raise loudly so it cannot silently
        make every post-drift submission score ``r_correct=0``.
        """
        rt = self._require_runtime()
        cfg = rt.instance.drift_config
        assert cfg is not None
        apply_drift(rt.conn, cfg.kind, cfg.payload)
        rt.drift_fired_step = rt.step_count
        rt.phase = EpisodePhase.DRIFT_RECOVERY
        rt.changelog_entries.append(author_changelog(cfg))

        try:
            rt.conn.execute(rt.instance.baseline_sql).fetchall()
            rt.baseline_postdrift_raises = False
        except duckdb.Error:
            rt.baseline_postdrift_raises = True

        if rt.instance.gt_sql_postdrift is not None:
            try:
                rows = rt.conn.execute(rt.instance.gt_sql_postdrift).fetchall()
            except duckdb.Error as exc:
                raise RuntimeError(
                    f"scenario {rt.scenario_id!r}: authored gt_sql_postdrift failed "
                    f"after drift: {exc}"
                ) from exc
            rt.gt_result_hash_postdrift = canonical_row_hash(rows)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: SqlDriftAction, *, timeout_s: float) -> ToolResult:
        payload = action.payload
        try:
            if isinstance(payload, ListTablesPayload):
                return self._handle_list_tables()
            if isinstance(payload, DescribeTablePayload):
                return self._handle_describe_table(payload)
            if isinstance(payload, SampleRowsPayload):
                return self._handle_sample_rows(payload)
            if isinstance(payload, RunQueryPayload):
                return self._handle_run_query(payload, timeout_s=timeout_s)
            if isinstance(payload, ExplainQueryPayload):
                return self._handle_explain_query(payload, timeout_s=timeout_s)
            if isinstance(payload, ReadChangelogPayload):
                return self._handle_read_changelog()
            if isinstance(payload, SubmitRewritePayload):
                return self._handle_submit_rewrite(payload, timeout_s=timeout_s)
            if isinstance(payload, ConsultDBAPayload):
                return self._handle_consult_dba(payload)
        except duckdb.Error as exc:
            return ToolError(code=ToolErrorCode.DB_ERROR, message=str(exc)[:2000])
        except TimeoutError as exc:
            return ToolError(code=ToolErrorCode.QUERY_TIMEOUT, message=str(exc)[:2000])
        # Unreachable — the discriminated-union validator rejects unknown payloads.
        return ToolError(
            code=ToolErrorCode.INVALID_TOOL_ARGUMENT,
            message=f"unknown payload type: {type(payload).__name__}",
        )

    def _handle_list_tables(self) -> ListTablesResult:
        rt = self._require_runtime()
        rows = rt.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        self._grant_step_rebate_once(attr="listed_tables_rewarded", rebate=STEP_REBATE_LIST_TABLES)
        self._mark_diagnostic()
        return ListTablesResult(tables=[r[0] for r in rows])

    def _handle_describe_table(
        self, payload: DescribeTablePayload
    ) -> DescribeTableResult | ToolError:
        rt = self._require_runtime()
        rows = rt.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [payload.table],
        ).fetchall()
        if not rows:
            return ToolError(
                code=ToolErrorCode.UNKNOWN_TABLE,
                message=f"unknown table: {payload.table}",
            )
        self._grant_step_rebate_for_table(
            rewarded_tables_attr="described_tables_rewarded",
            table=payload.table,
            rebate=STEP_REBATE_DESCRIBE_TABLE,
        )
        self._mark_diagnostic()
        return DescribeTableResult(
            table=payload.table,
            columns=[{"name": r[0], "type": r[1]} for r in rows],
        )

    def _handle_sample_rows(self, payload: SampleRowsPayload) -> SampleRowsResult | ToolError:
        rt = self._require_runtime()
        exists = rt.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [payload.table],
        ).fetchone()
        if not exists or exists[0] == 0:
            return ToolError(
                code=ToolErrorCode.UNKNOWN_TABLE,
                message=f"unknown table: {payload.table}",
            )
        cur = rt.conn.execute(f'SELECT * FROM "{payload.table}" LIMIT {payload.limit}')
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        self._grant_step_rebate_for_table(
            rewarded_tables_attr="sampled_tables_rewarded",
            table=payload.table,
            rebate=STEP_REBATE_SAMPLE_ROWS,
        )
        self._mark_diagnostic()
        return SampleRowsResult(table=payload.table, columns=columns, rows=rows)

    def _handle_run_query(
        self, payload: RunQueryPayload, *, timeout_s: float
    ) -> RunQueryResult | ToolError:
        rt = self._require_runtime()
        sql = payload.sql
        try:
            _validate_read_only_sql(sql)
        except ValueError as exc:
            return ToolError(code=ToolErrorCode.INVALID_TOOL_ARGUMENT, message=str(exc)[:2000])

        # Drift timing: after a valid
        # ``run_query`` attempt, the pre-drift probe invariant is
        # satisfied regardless of whether the execution ultimately
        # returned rows, raised, or was capped for size. Assigning
        # *before* execution means truncation, DB errors, and timeouts
        # can no longer suppress drift firing in later steps.
        if rt.first_run_query_step is None:
            rt.first_run_query_step = rt.step_count

        try:
            result = execute_once_with_columns(
                rt.conn, sql, timeout_s=timeout_s, max_rows=MAX_RESULT_ROWS
            )
        except TimeoutError as exc:
            return ToolError(code=ToolErrorCode.QUERY_TIMEOUT, message=str(exc)[:2000])
        except duckdb.Error as exc:
            # Canonicalize *before* hashing so whitespace-/case-only
            # variants of the same broken query count as the same repeat
            # offence. canonicalize_sql falls back to a whitespace fold
            # for SQL that sqlglot can't parse — still normalises the
            # vast majority of "retried the same typo" cases.
            failure_hash = canonical_row_hash([(canonicalize_sql(sql),)])
            count = rt.failed_query_counts.get(failure_hash, 0) + 1
            rt.failed_query_counts[failure_hash] = count
            rt.failed_query_hashes.add(failure_hash)
            rt.last_step_repeat_failing_query_count = count
            rt.last_step_was_repeat_failing_query = count > 1
            return ToolError(code=ToolErrorCode.DB_ERROR, message=str(exc)[:2000])

        if result.truncated:
            return ToolError(
                code=ToolErrorCode.RESULT_TOO_LARGE,
                message=(
                    f"result exceeded {MAX_RESULT_ROWS}-row cap — narrow the "
                    "projection, add a LIMIT, or aggregate"
                ),
            )

        self._grant_step_rebate_once(attr="run_query_rewarded", rebate=STEP_REBATE_RUN_QUERY)
        self._mark_diagnostic()
        return RunQueryResult(
            columns=result.columns,
            rows=[list(r) for r in result.rows],
            runtime_ms=result.elapsed_ms,
            row_count=len(result.rows),
        )

    def _handle_explain_query(
        self, payload: ExplainQueryPayload, *, timeout_s: float
    ) -> ExplainQueryResult | ToolError:
        rt = self._require_runtime()
        try:
            _validate_read_only_sql(payload.sql)
        except ValueError as exc:
            return ToolError(code=ToolErrorCode.INVALID_TOOL_ARGUMENT, message=str(exc)[:2000])
        # EXPLAIN is plan-only (no data materialisation) but we still
        # route it through the watchdog so a pathological query cannot
        # burn the step budget past the caller's wall-clock deadline.
        explain_rows, _ = execute_once_timed(rt.conn, f"EXPLAIN {payload.sql}", timeout_s=timeout_s)
        plan = "\n".join(str(r[-1]) if r else "" for r in explain_rows)
        self._grant_step_rebate_once(
            attr="explain_query_rewarded", rebate=STEP_REBATE_EXPLAIN_QUERY
        )
        self._mark_diagnostic()
        return ExplainQueryResult(plan=plan[:10_000])

    def _handle_read_changelog(self) -> ReadChangelogResult:
        rt = self._require_runtime()
        if rt.changelog_entries:
            rt.drift_acknowledged = True
            self._grant_step_rebate_once(
                attr="changelog_rewarded_after_drift",
                rebate=STEP_REBATE_READ_CHANGELOG,
            )
        self._mark_diagnostic()
        return ReadChangelogResult(entries=list(rt.changelog_entries))

    def _handle_submit_rewrite(
        self, payload: SubmitRewritePayload, *, timeout_s: float
    ) -> SubmitRewriteResult | ToolError:
        rt = self._require_runtime()
        if not rt.diagnostic_actions_taken:
            return ToolError(
                code=ToolErrorCode.SUBMIT_BEFORE_DIAGNOSE,
                message=(
                    "submit_rewrite rejected: the agent must take at least one "
                    "diagnostic action (list_tables, describe_table, sample_rows, "
                    "run_query, explain_query, or read_changelog) before submitting."
                ),
            )
        sql = payload.sql
        try:
            _validate_read_only_sql(sql)
        except ValueError as exc:
            return ToolError(code=ToolErrorCode.INVALID_TOOL_ARGUMENT, message=str(exc)[:2000])
        try:
            agent_hash, elapsed_ms = execute_hash_timed(rt.conn, sql, timeout_s=timeout_s)
        except TimeoutError as exc:
            return ToolError(code=ToolErrorCode.QUERY_TIMEOUT, message=str(exc)[:2000])
        except duckdb.Error as exc:
            return ToolError(code=ToolErrorCode.DB_ERROR, message=str(exc)[:2000])
        gt_hash = (
            rt.gt_result_hash_postdrift
            if rt.drift_fired and rt.gt_result_hash_postdrift is not None
            else rt.gt_result_hash_predrift
        )
        matches = agent_hash == gt_hash

        rt.submitted = True
        rt.submitted_sql = sql
        rt.submitted_sql_canonical = canonicalize_sql(sql)
        rt.submitted_result_hash = agent_hash
        rt.submitted_runtime_ms = elapsed_ms
        rt.phase = EpisodePhase.FINALIZE
        return SubmitRewriteResult(
            accepted=True,
            runtime_ms=elapsed_ms,
            matches_ground_truth=matches,
        )

    def _handle_consult_dba(self, payload: ConsultDBAPayload) -> ConsultDBAResult | ToolError:
        rt = self._require_runtime()
        if not rt.dba_oracle_enabled:
            return ToolError(
                code=ToolErrorCode.INVALID_TOOL_ARGUMENT,
                message="consult_dba disabled — set enable_dba_oracle=True at reset()",
            )
        if not dba_oracle.has_hints(rt.scenario_id):
            return ToolError(
                code=ToolErrorCode.INVALID_TOOL_ARGUMENT,
                message=f"no DBA hints registered for scenario={rt.scenario_id!r}",
            )
        rt.consultations_used += 1
        tier = min(rt.consultations_used, 3)
        hint = dba_oracle.get_hint(rt.scenario_id, tier)
        del payload  # question is free-text context only; hints are scenario-keyed.
        return ConsultDBAResult(tier=tier, hint=hint)

    def _mark_diagnostic(self) -> None:
        """Record a successful diagnostic tool call and advance the phase machine."""
        rt = self._require_runtime()
        rt.diagnostic_actions_taken += 1
        if rt.phase == EpisodePhase.DIAGNOSE:
            rt.phase = EpisodePhase.REWRITE


__all__ = [
    "DEFAULT_STEP_BUDGET",
    "MAX_RESULT_ROWS",
    "QUERY_TIMEOUT_S",
    "SqlDriftEnvironment",
]
