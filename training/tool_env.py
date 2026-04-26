"""TRL-compatible tool environment for SQLDrift.

TRL's :class:`trl.GRPOTrainer` drives multi-turn OpenEnv rollouts via
the ``environment_factory=`` kwarg: it instantiates one instance of the
class per generation, calls :meth:`reset` with the sampled dataset row
columns as keyword arguments, and exposes every public method with a
typed ``Args:`` docstring as a function-calling tool. We therefore
unroll the eight SQLDrift tools (``list_tables``, ``describe_table``,
``sample_rows``, ``run_query``, ``explain_query``, ``read_changelog``,
``submit_rewrite``, ``consult_dba``) as individual methods with
minimal, model-facing surface area — exactly the shape TRL's tool
schema extractor expects.

Side-channel state the trainer's reward function reads:

* ``self.reward`` — the *latest* scalar reward from the env (overwritten
  on every tool call). TRL reward functions consume this after the
  episode finishes, so the final value is what matters for GRPO group
  comparisons.
* ``self.episode_return`` — the running sum of per-step rewards.
* ``self.terminal_reward`` — sticky copy of the final reward emitted
  when the env set ``done=True``.
* ``self.done`` — True once the episode terminated.
* ``self.components`` — most recent per-rubric component scores.

The default client factory opens a synchronous OpenEnv session against
the caller-supplied ``env_url`` (falling back to the
``SQL_DRIFT_ENV_URL`` environment variable). Tests substitute an
in-process factory via
:func:`set_client_factory` so CPU-only CI can exercise the class
without a running server.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from client import SqlDriftEnv
from utilities.env_loader import env_str

_DEFAULT_ENV_URL = env_str("SQL_DRIFT_ENV_URL", "http://localhost:8000")


class _SyncClientProtocol:
    """Minimal subset of :class:`openenv.core.env_client.EnvClient` we rely on."""

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> Any: ...
    def step(self, action: Any) -> Any: ...
    def close(self) -> None: ...


ClientFactory = Callable[[], _SyncClientProtocol]


def _default_client_factory(base_url: str) -> _SyncClientProtocol:
    """Build the standard sync WebSocket-backed SQLDrift client."""
    client: _SyncClientProtocol = SqlDriftEnv(base_url=base_url).sync()
    return client


_client_factory_override: ClientFactory | None = None


def set_client_factory(factory: ClientFactory | None) -> None:
    """Override the client factory (tests and in-process rollouts).

    Passing ``None`` restores the default HTTP/WS-backed factory.
    """
    global _client_factory_override
    _client_factory_override = factory


# ---------------------------------------------------------------------------
# Internal helpers — kept private (underscore-prefixed) so TRL does NOT
# expose them as tools.
# ---------------------------------------------------------------------------


def _format_columns(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "(none)"


def _format_rows(columns: list[str], rows: list[list[Any]], *, limit: int = 20) -> str:
    """Render a small result set as a fixed-width table string.

    Large result sets are truncated to ``limit`` rows with a trailing
    ``... (N more)`` marker so the model's context does not balloon
    even when a query legitimately returns many rows.
    """
    if not rows:
        return f"columns: {_format_columns(columns)}\n(0 rows)"
    head = rows[:limit]
    body = "\n".join(" | ".join(str(cell) for cell in row) for row in head)
    more = f"\n... ({len(rows) - len(head)} more)" if len(rows) > len(head) else ""
    return f"columns: {_format_columns(columns)}\n{body}{more}"


# ---------------------------------------------------------------------------
# Tool-env class — TRL consumes the public methods as tool schemas.
# Method order here matches :class:`models.ToolName` so the model's tool
# menu stays aligned with the server's dispatch table.
# ---------------------------------------------------------------------------


class SqlDriftToolEnv:
    """TRL-facing wrapper around the SQLDrift OpenEnv client.

    One instance per GRPO generation. TRL discovers each public method
    (list_tables, describe_table, …) as a callable tool; the class
    itself stores the per-episode reward so the reward function can
    read it back after the rollout finishes.
    """

    def __init__(self, *, env_url: str | None = None) -> None:
        base_url = env_url or os.environ.get("SQL_DRIFT_ENV_URL", _DEFAULT_ENV_URL)
        if _client_factory_override is None:
            self._client = _default_client_factory(base_url)
        else:
            self._client = _client_factory_override()
        self.reward: float = 0.0
        self.terminal_reward: float = 0.0
        self.episode_return: float = 0.0
        self.done: bool = False
        self.submitted: bool = False
        self.drift_fired: bool = False
        self.components: dict[str, float] = {}
        self._last_observation_text: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> str | None:
        """Begin a new SQLDrift episode.

        TRL invokes ``reset`` with every dataset column as a keyword
        argument. We accept ``seed``, ``scenario_id``,
        ``enable_dba_oracle``, ``budget_steps``, and ``difficulty`` (all
        optional — the env supplies deterministic defaults when omitted)
        and ignore the rest.
        """
        seed = kwargs.get("seed")
        scenario_id = kwargs.get("scenario_id")
        enable_dba_oracle = kwargs.get("enable_dba_oracle")
        budget_steps = kwargs.get("budget_steps")
        difficulty = kwargs.get("difficulty")

        reset_kwargs: dict[str, Any] = {}
        if scenario_id is not None:
            reset_kwargs["scenario_id"] = scenario_id
        if enable_dba_oracle is not None:
            reset_kwargs["enable_dba_oracle"] = enable_dba_oracle
        if budget_steps is not None:
            reset_kwargs["budget_steps"] = budget_steps
        if difficulty is not None:
            reset_kwargs["difficulty"] = difficulty

        result = self._client.reset(
            seed=int(seed) if seed is not None else None,
            **reset_kwargs,
        )
        observation = _extract_observation(result)
        self.reward = 0.0
        self.terminal_reward = 0.0
        self.episode_return = 0.0
        self.done = False
        self.submitted = False
        self.drift_fired = False
        self.components = {}
        return _format_observation_prelude(observation)

    def close(self) -> None:
        """Release the underlying client session (idempotent)."""
        with _suppress_client_errors():
            self._client.close()

    # ------------------------------------------------------------------
    # Tools — exposed to the model via TRL's tool-schema extractor.
    # Each method has a typed signature + ``Args:`` docstring so the
    # schema generator can produce a clean JSON tool spec.
    # ------------------------------------------------------------------

    def list_tables(self) -> str:
        """Enumerate every table visible in the current SQLDrift database.

        Returns:
            A comma-separated list of table names.
        """
        return self._dispatch(SqlDriftEnv.action_list_tables())

    def describe_table(self, table: str) -> str:
        """Return column names and types for a single table.

        Args:
            table: Name of the table to describe.

        Returns:
            A formatted list of ``column: type`` pairs, one per line.
        """
        return self._dispatch(SqlDriftEnv.action_describe_table(table=table))

    def sample_rows(self, table: str, limit: int = 5) -> str:
        """Peek at up to five rows from a table for fast schema intuition.

        Args:
            table: Name of the table to sample from.
            limit: Number of rows to return (1-5, inclusive).

        Returns:
            A rendered table of the sampled rows.
        """
        return self._dispatch(SqlDriftEnv.action_sample_rows(table=table, limit=int(limit)))

    def run_query(self, sql: str) -> str:
        """Execute a read-only SELECT and return its result set.

        Args:
            sql: The SELECT statement to execute. DDL/DML is rejected.

        Returns:
            The rendered result table (truncated to a handful of rows).
        """
        return self._dispatch(SqlDriftEnv.action_run_query(sql=sql))

    def explain_query(self, sql: str) -> str:
        """Return the DuckDB query plan for a SELECT without executing it.

        Args:
            sql: The SELECT statement to plan.

        Returns:
            The textual query plan.
        """
        return self._dispatch(SqlDriftEnv.action_explain_query(sql=sql))

    def read_changelog(self) -> str:
        """Read every drift deploy note that has been published so far.

        Returns:
            The concatenated changelog entries, most recent last.
        """
        return self._dispatch(SqlDriftEnv.action_read_changelog())

    def submit_rewrite(self, sql: str) -> str:
        """Commit a final SELECT and end the episode.

        Args:
            sql: The rewritten SELECT to submit as the final answer.

        Returns:
            A short verdict summarising whether the rewrite matched the
            ground truth and its measured runtime.
        """
        return self._dispatch(SqlDriftEnv.action_submit_rewrite(sql=sql))

    def consult_dba(self, question: str) -> str:
        """Ask the on-call DBA for a hint (enabled only when the DBA
        oracle flag is on; each call incurs an escalating penalty).

        Args:
            question: Free-text question for context — the actual hint
                is scenario-keyed, not question-keyed.

        Returns:
            The DBA's hint plus its tier.
        """
        return self._dispatch(SqlDriftEnv.action_consult_dba(question=question))

    # ------------------------------------------------------------------
    # Internal dispatch — wraps every env.step() call, records reward
    # state, and turns tool errors into plain-text returns so the
    # model sees a recoverable message instead of a crash.
    # ------------------------------------------------------------------

    def _dispatch(self, action: Any) -> str:
        if self.done:
            # Episode already terminated (budget exhausted or a prior
            # submit). Raising a ValueError is the TRL-sanctioned way
            # to tell the model the episode is over (see the TRL
            # OpenEnv integration guide on "Error handling").
            raise ValueError("Episode is already finished; the environment is closed.")
        result = self._client.step(action)
        obs = _extract_observation(result)
        self._ingest_observation(obs)
        return _format_tool_result(obs)

    def _ingest_observation(self, obs: Any) -> None:
        reward_val = _safe_float(getattr(obs, "reward", 0.0))
        self.reward = reward_val
        self.episode_return += reward_val
        components = getattr(obs, "reward_components", None)
        if isinstance(components, dict):
            self.components = {k: float(v) for k, v in components.items()}
        self.drift_fired = bool(getattr(obs, "drift_fired", self.drift_fired))
        done = bool(getattr(obs, "done", False))
        if done:
            self.done = True
            self.terminal_reward = reward_val
            # ``submit_rewrite`` is the only tool that ends an episode
            # by setting ``submitted=True``; budget exhaustion ends it
            # without submission. Surface both so reward functions can
            # decide whether the rollout produced a real submission.
            tool_result = getattr(obs, "tool_result", None)
            submitted = getattr(tool_result, "accepted", None)
            if submitted is not None:
                self.submitted = bool(submitted)


# ---------------------------------------------------------------------------
# Observation rendering — kept module-private so TRL sees them as
# helpers rather than tools.
# ---------------------------------------------------------------------------


def _extract_observation(result: Any) -> Any:
    """Return the ``SqlDriftObservation`` out of an ``EnvClient`` return value.

    The sync ``EnvClient`` API emits a :class:`StepResult` wrapper; the
    actual observation lives on the ``.observation`` attribute. Reset
    uses the same shape. We defensively fall back to the value itself
    when it is already an observation (which is how our in-process
    test shim reports results).
    """
    observation = getattr(result, "observation", None)
    return result if observation is None else observation


def _safe_float(value: Any) -> float:
    """Coerce any reward-shaped value to a finite float (0.0 on failure)."""
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0


def _format_observation_prelude(obs: Any) -> str:
    """Render the initial reset observation for the model."""
    parts: list[str] = []
    baseline = getattr(obs, "baseline_sql", "")
    synopsis = getattr(obs, "schema_synopsis", "")
    hints = getattr(obs, "learned_hints", "")
    budget = getattr(obs, "budget_steps_remaining", None)
    if synopsis:
        parts.append(f"Schema synopsis:\n{synopsis}")
    if baseline:
        parts.append(f"Baseline query:\n{baseline}")
    if budget is not None:
        parts.append(f"Remaining step budget: {budget}")
    if hints:
        parts.append(f"Learned hints:\n{hints}")
    return "\n\n".join(parts) if parts else ""


def _format_tool_result(obs: Any) -> str:
    """Render ``obs.tool_result`` as a plain-text response for the model."""
    tool_result = getattr(obs, "tool_result", None)
    if tool_result is None:
        text = "(no tool result)"
        hints = getattr(obs, "learned_hints", "")
        return f"{text}\n\nLearned hints:\n{hints}" if hints else text

    # Type-based dispatch keeps the code straightforward and avoids
    # a heavy dependency on the Pydantic discriminator tags at
    # runtime. Each branch returns a string that the model can parse.
    from models import (  # local import avoids a circular hazard at module import
        ConsultDBAResult,
        DescribeTableResult,
        ExplainQueryResult,
        ListTablesResult,
        ReadChangelogResult,
        RunQueryResult,
        SampleRowsResult,
        SubmitRewriteResult,
        ToolError,
    )

    if isinstance(tool_result, ToolError):
        text = f"error[{tool_result.code.value}]: {tool_result.message}"
    elif isinstance(tool_result, ListTablesResult):
        text = _format_columns(tool_result.tables)
    elif isinstance(tool_result, DescribeTableResult):
        cols = "\n".join(f"{c['name']}: {c['type']}" for c in tool_result.columns)
        text = f"table: {tool_result.table}\n{cols}"
    elif isinstance(tool_result, SampleRowsResult | RunQueryResult):
        text = _format_rows(tool_result.columns, tool_result.rows)
    elif isinstance(tool_result, ExplainQueryResult):
        text = tool_result.plan
    elif isinstance(tool_result, ReadChangelogResult):
        text = "\n---\n".join(tool_result.entries) if tool_result.entries else "(no changelog yet)"
    elif isinstance(tool_result, SubmitRewriteResult):
        verdict = "matched ground truth" if tool_result.matches_ground_truth else "did NOT match"
        text = f"submitted (runtime={tool_result.runtime_ms:.2f}ms) — {verdict}"
    elif isinstance(tool_result, ConsultDBAResult):
        text = f"[tier {tool_result.tier}] {tool_result.hint}"
    else:
        text = str(tool_result)
    hints = getattr(obs, "learned_hints", "")
    return f"{text}\n\nLearned hints:\n{hints}" if hints else text


class _suppress_client_errors:
    """Tiny context manager — swallows teardown-time errors from the client."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc is not None  # swallow any exception from close()


__all__ = [
    "ClientFactory",
    "SqlDriftToolEnv",
    "set_client_factory",
]
