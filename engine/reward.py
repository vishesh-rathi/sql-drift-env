"""SQLDrift composed rubric (correctness, drift, speedup, gates, DBA tax).

Six child rubrics, one per reward component (:data:`REWARD_COMPONENT_KEYS`):

    r_correct      correctness vs ground-truth hash, gated on ≥ 1.2× speedup
    r_drift        bonus/penalty for (not) adapting to post-drift identifiers
    r_speedup      tanh-shaped speedup bonus, gated on r_correct > 0
    r_step_tax     base step tax plus bounded productive-action rebates
    r_gatekeepers  escalating tool-error / repeat-failing / no-op penalties
    r_consult_dba  DBA-oracle consult penalties (feature-flagged; 0 when off)

All child rubrics share a single ``ctx_provider`` that returns the private
:class:`engine.runtime.RuntimeEpisodeState`; this keeps the rubric
stateless relative to the environment and makes each component
individually unit-testable with a synthesized triple
``(RuntimeEpisodeState, SqlDriftAction, SqlDriftObservation)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from openenv.core.rubrics import Rubric

from models import (
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewriteResult,
    ToolError,
    ToolName,
)

if TYPE_CHECKING:
    from engine.runtime import RuntimeEpisodeState


# Tunable thresholds pulled out to module level so tests and future
# curriculum code share a single source of truth.
SPEEDUP_MIN: float = 1.2
SPEEDUP_CAP_FOR_INFTY: float = 64.0
STEP_TAX: float = -0.03
STEP_REBATE_LIST_TABLES: float = 0.04
STEP_REBATE_DESCRIBE_TABLE: float = 0.06
STEP_REBATE_SAMPLE_ROWS: float = 0.05
STEP_REBATE_RUN_QUERY: float = 0.04
STEP_REBATE_EXPLAIN_QUERY: float = 0.04
STEP_REBATE_READ_CHANGELOG: float = 0.08

GATE_MALFORMED_TOOL_CALL: float = -0.3
GATE_CONSECUTIVE_TOOL_ERROR: float = -0.1
GATE_REPEAT_FAILING_QUERY: float = -0.1
GATE_BASELINE_VERBATIM: float = -0.2
_MAX_ESCALATION_STEPS: int = 3

CONSULT_ESCALATION: tuple[float, float, float] = (-0.1, -0.3, -0.8)


# =============================================================================
# Helpers
# =============================================================================


def canonicalize_sql(sql: str) -> str:
    """Whitespace/case/alias-insensitive canonical form.

    Uses sqlglot's duckdb dialect round-trip so reorders/reformats agree;
    falls back to a simple whitespace fold if sqlglot rejects the SQL
    (e.g. during the baseline-verbatim check on an agent-submitted blob).
    """
    try:
        import sqlglot

        expr = sqlglot.parse_one(sql, dialect="duckdb")
        return expr.sql(dialect="duckdb", comments=False, normalize=True).strip().lower()
    except Exception:
        return " ".join(sql.lower().split())


_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _extract_identifiers(sql: str) -> frozenset[str]:
    """Loose case-preserving identifier harvest.

    Strips string literals before tokenizing so e.g. `WHERE x = 'ACTIVE'`
    does not leak 'ACTIVE' into the identifier set.
    """
    stripped = re.sub(r"'[^']*'", "", sql)
    return frozenset(m.group(0) for m in _IDENT_RE.finditer(stripped))


def _extract_column_references(sql: str) -> frozenset[str]:
    """Column-reference identifiers via sqlglot AST (excludes alias labels).

    For drift-adapt scoring, ``SELECT account_id AS user_id`` references
    the new column and merely labels the output — the alias must not
    count as a surviving pre-drift marker. Falls back to the regex
    extractor on parse failure so malformed SQL still scores something.
    """
    try:
        import sqlglot
        expr = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:
        return _extract_identifiers(sql)
    if expr is None:
        return _extract_identifiers(sql)
    return frozenset(
        n.name for n in expr.walk() if isinstance(n, sqlglot.exp.Column) and n.name
    )


def _literals(sql: str) -> frozenset[str]:
    """All `'..'`-quoted string literals in `sql`."""
    return frozenset(re.findall(r"'([^']*)'", sql))


_AGENT_MS_EPSILON: float = 1e-6


def effective_speedup(rt: RuntimeEpisodeState) -> float | None:
    """Compute ``effective_speedup`` from the runtime snapshot (speedup rubric).

    Single source of truth for the speedup number used across the code
    base — rubric scoring, the skill library's ``avg_speedup`` field,
    and the training evaluator all route through here so divergent
    definitions cannot drift apart.

    Returns:

    * ``None`` — no submission has happened yet. Callers that need a
      numeric default (e.g. the rubric, which is only invoked
      post-submission) should verify ``rt.submitted`` first.
    * ``+∞`` — drift has fired and the pre-drift baseline SQL no longer
      executes against the post-drift schema; any correct submission is
      definitionally "infinitely faster" than an unrunnable baseline.
    * ``baseline_ms / max(agent_ms, ε)`` otherwise. A tiny ``ε`` clamp
      guards against zero/negative timings from sub-microsecond queries
      and treats them as "as fast as possible" (very large, finite
      speedup) rather than silently collapsing the reward.
    """
    if rt.submitted_runtime_ms is None:
        return None
    if rt.drift_fired and rt.baseline_postdrift_raises:
        return math.inf
    agent_ms = max(rt.submitted_runtime_ms, _AGENT_MS_EPSILON)
    return rt.baseline_runtime_ms / agent_ms


def _speedup_for_reward(rt: RuntimeEpisodeState) -> float:
    """Rubric-facing speedup that never returns ``None``.

    The rubric is only invoked once ``rt.submitted`` is True, so
    :func:`effective_speedup` cannot return ``None`` from these call
    sites; we assert that and coerce to ``0.0`` defensively if it ever
    does (prevents a silent ``TypeError`` inside the reward math).
    """
    val = effective_speedup(rt)
    return 0.0 if val is None else val


def _is_terminal_submission(
    action: SqlDriftAction,
    observation: SqlDriftObservation,
    rt: RuntimeEpisodeState,
) -> bool:
    """True iff this step is the submission step.

    The env sets ``done=True`` on a successful submission and attaches a
    :class:`SubmitRewriteResult`; we gate terminal rewards on both
    signals so repeated rubric calls on an unchanged state don't
    double-score.
    """
    if not rt.submitted:
        return False
    if action.tool != ToolName.SUBMIT_REWRITE:
        return False
    tr = observation.tool_result
    return isinstance(tr, SubmitRewriteResult)


def _gt_hash(rt: RuntimeEpisodeState) -> str | None:
    if rt.drift_fired and rt.gt_result_hash_postdrift is not None:
        return rt.gt_result_hash_postdrift
    return rt.gt_result_hash_predrift


# =============================================================================
# Child rubrics
# =============================================================================


class _CtxChild(Rubric):
    """Base child rubric sharing the ctx provider."""

    def __init__(self, ctx_provider: Callable[[], RuntimeEpisodeState]) -> None:
        super().__init__()
        object.__setattr__(self, "_ctx", ctx_provider)

    def forward(
        self,
        action: SqlDriftAction,
        observation: SqlDriftObservation,
    ) -> float:
        raise NotImplementedError


class Correctness(_CtxChild):
    """Terminal-only correctness: +1.0 / +0.5 / -1.0 by hash and speedup."""

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        if not _is_terminal_submission(action, observation, rt):
            return 0.0
        gt = _gt_hash(rt)
        agent_hash = rt.submitted_result_hash
        if gt is None or agent_hash is None:
            return 0.0
        if agent_hash != gt:
            return -1.0
        speedup = _speedup_for_reward(rt)
        if speedup >= SPEEDUP_MIN:
            return 1.0
        return 0.5


class DriftAdapt(_CtxChild):
    """+0.5 for a correctly-adapted submission, -0.5 for a pre-drift-only
    submission after drift fired.

    Adaptation is detected against two scenario-declared identifier sets:

    * ``postdrift_identifiers`` — identifiers/literals that only a
      correct post-drift rewrite will introduce (e.g. ``account_id``
      after a column rename, ``'ACTIVE'`` after an enum split).
    * ``predrift_identifiers`` — identifiers/literals a submission that
      ignored the drift would retain (e.g. ``user_id``, ``'active'``,
      the ISO anchor strings under date-format drift).

    A submission is considered "adapted" when it either surfaces a
    post-drift marker *or* the scenario declares no distinctive
    post-drift identifiers (e.g. date-format drift keeps the same
    column name and only the literal shape changes) AND it does not
    retain any pre-drift marker. The penalty fires only when the
    submission still carries pre-drift markers AND produced the wrong
    post-drift result — so a merely partial rewrite (neither pre-
    nor post-flavoured) never earns a penalty it can't diagnose.
    """

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        # Only drift scenarios participate.
        if rt.gt_result_hash_postdrift is None and not rt.drift_fired:
            return 0.0
        if not _is_terminal_submission(action, observation, rt):
            return 0.0
        inst = getattr(rt, "instance", None)
        post_ids: frozenset[str] = (
            getattr(inst, "postdrift_identifiers", frozenset()) or frozenset()
        )
        pre_ids: frozenset[str] = getattr(inst, "predrift_identifiers", frozenset()) or frozenset()
        agent_sql = rt.submitted_sql or ""

        idents = _extract_column_references(agent_sql)
        literals = _literals(agent_sql)
        markers = idents | literals
        uses_post = bool(post_ids & markers)
        uses_pre = bool(pre_ids & markers)

        # Treat "no distinctive post identifier" scenarios as
        # satisfied by absence-of-pre (see class docstring).
        adapted = (uses_post or not post_ids) and not uses_pre

        agent_hash = rt.submitted_result_hash
        gt_post = rt.gt_result_hash_postdrift

        if rt.drift_fired and agent_hash == gt_post and adapted:
            return 0.5
        if rt.drift_fired and uses_pre and agent_hash != gt_post:
            return -0.5
        return 0.0


class Speedup(_CtxChild):
    """Terminal-only, gated on r_correct > 0: 0.3·tanh(log2(speedup)/3)."""

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        if not _is_terminal_submission(action, observation, rt):
            return 0.0
        gt = _gt_hash(rt)
        if gt is None or rt.submitted_result_hash != gt:
            return 0.0
        raw = _speedup_for_reward(rt)
        if math.isinf(raw):
            raw = SPEEDUP_CAP_FOR_INFTY
        if raw <= 1.0:
            return 0.0
        return 0.3 * math.tanh(math.log2(raw) / 3.0)


class StepTax(_CtxChild):
    """Base step tax plus bounded rebates for productive exploration."""

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        if _is_terminal_submission(action, observation, rt):
            return 0.0
        rebate = max(0.0, float(getattr(rt, "last_step_productive_rebate", 0.0)))
        return STEP_TAX + rebate


class Gatekeepers(_CtxChild):
    """Sum of three independent penalties; repeats escalate up to a cap."""

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        penalty = 0.0
        # 1. Malformed / failed tool call — ToolError emitted this step.
        if isinstance(observation.tool_result, ToolError):
            penalty += GATE_MALFORMED_TOOL_CALL
            streak = max(0, int(getattr(rt, "consecutive_tool_errors", 0)) - 1)
            penalty += GATE_CONSECUTIVE_TOOL_ERROR * min(streak, _MAX_ESCALATION_STEPS)
        # 2. Repeat failing query — env marks the flag on the runtime
        #    state immediately before invoking the rubric.
        repeats = max(0, int(getattr(rt, "last_step_repeat_failing_query_count", 0)) - 1)
        if repeats > 0:
            penalty += GATE_REPEAT_FAILING_QUERY * min(repeats, _MAX_ESCALATION_STEPS)
        # 3. Baseline-verbatim submission (Rev-3 gate — stacks with
        #    correctness's +0.5 partial to cap the no-op rewrite at +0.3).
        if (
            action.tool == ToolName.SUBMIT_REWRITE
            and _is_terminal_submission(action, observation, rt)
            and rt.submitted_sql_canonical == rt.baseline_sql_canonical
        ):
            penalty += GATE_BASELINE_VERBATIM
        return penalty


class ConsultDBA(_CtxChild):
    """Escalating penalties -0.1 / -0.3 / -0.8 per consult when the flag is on."""

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        rt = self._ctx()
        oracle_enabled = getattr(rt, "dba_oracle_enabled", False)
        if not oracle_enabled:
            return 0.0
        if action.tool != ToolName.CONSULT_DBA:
            return 0.0
        # Count the consult THIS step by indexing into the escalation
        # table using the pre-increment value (env increments on the same step).
        tier = min(rt.consultations_used, len(CONSULT_ESCALATION))
        if tier <= 0:
            return CONSULT_ESCALATION[0]
        return CONSULT_ESCALATION[tier - 1]


# =============================================================================
# Composite
# =============================================================================


class SqlDriftRubric(Rubric):
    """Composite rubric: sum of six children.

    Registration as attributes auto-enrolls them in
    :meth:`Rubric.named_rubrics` so training loops can introspect
    per-component scores.
    """

    def __init__(self, ctx_provider: Callable[[], RuntimeEpisodeState]) -> None:
        super().__init__()
        # NOTE: order matters — correctness must populate last_score before
        # speedup reads it via the shared ctx_provider (both are pure
        # functions of the runtime state, so identical output — but the
        # explicit ordering documents the intent).
        self.correctness = Correctness(ctx_provider)
        self.drift_adapt = DriftAdapt(ctx_provider)
        self.speedup = Speedup(ctx_provider)
        self.step_tax = StepTax(ctx_provider)
        self.gatekeepers = Gatekeepers(ctx_provider)
        self.consult_dba = ConsultDBA(ctx_provider)

    def forward(self, action: SqlDriftAction, observation: SqlDriftObservation) -> float:
        total = (
            self.correctness(action, observation)
            + self.drift_adapt(action, observation)
            + self.speedup(action, observation)
            + self.step_tax(action, observation)
            + self.gatekeepers(action, observation)
            + self.consult_dba(action, observation)
        )
        return total

    def component_scores(self) -> dict[str, float]:
        """Return the most-recent per-component scores, keyed for W&B.

        Keys match :data:`models.REWARD_COMPONENT_KEYS` so the observation
        envelope and the demo plots agree on a stable schema.
        """
        return {
            "r_correct": float(self.correctness.last_score or 0.0),
            "r_drift": float(self.drift_adapt.last_score or 0.0),
            "r_speedup": float(self.speedup.last_score or 0.0),
            "r_step_tax": float(self.step_tax.last_score or 0.0),
            "r_gatekeepers": float(self.gatekeepers.last_score or 0.0),
            "r_consult_dba": float(self.consult_dba.last_score or 0.0),
        }


__all__ = [
    "CONSULT_ESCALATION",
    "ConsultDBA",
    "Correctness",
    "DriftAdapt",
    "GATE_BASELINE_VERBATIM",
    "GATE_CONSECUTIVE_TOOL_ERROR",
    "GATE_MALFORMED_TOOL_CALL",
    "GATE_REPEAT_FAILING_QUERY",
    "Gatekeepers",
    "SPEEDUP_CAP_FOR_INFTY",
    "SPEEDUP_MIN",
    "STEP_REBATE_DESCRIBE_TABLE",
    "STEP_REBATE_EXPLAIN_QUERY",
    "STEP_REBATE_LIST_TABLES",
    "STEP_REBATE_READ_CHANGELOG",
    "STEP_REBATE_RUN_QUERY",
    "STEP_REBATE_SAMPLE_ROWS",
    "STEP_TAX",
    "Speedup",
    "SqlDriftRubric",
    "StepTax",
    "canonicalize_sql",
    "effective_speedup",
]
