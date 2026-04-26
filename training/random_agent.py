"""Deterministic random agent used in the CPU smoke-test loop.

The agent mirrors the tool surface exposed by
:class:`models.SqlDriftAction` but makes no LLM call; all choices
come from a seeded :class:`random.Random`.  It is intentionally noisy
so ``tests/integration/test_env_random_smoke.py`` sees reward variance
(> 0.1 std across 5 rollouts × 10 scenarios) and exercises every tool.

Design notes
------------
The agent is stateful per episode: it remembers every table name it
has observed via :class:`ListTablesResult` / :class:`DescribeTableResult`
so later ``run_query`` / ``submit_rewrite`` / ``sample_rows`` actions
can target real identifiers rather than fabricated ones (which would
only ever yield ``ToolError`` with code ``unknown_table``).

The agent is not meant to score well: it is a ground-truth
"does the env crash under arbitrary-but-syntactically-valid input?"
harness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from models import (
    ConsultDBAPayload,
    DescribeTablePayload,
    DescribeTableResult,
    ExplainQueryPayload,
    ListTablesPayload,
    ListTablesResult,
    ReadChangelogPayload,
    RunQueryPayload,
    SampleRowsPayload,
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewritePayload,
    ToolName,
)

# Tools the agent may sample. submit_rewrite is gated to fire at most
# once per episode (otherwise the first draw ends the episode before
# any real exploration happens) — see :meth:`RandomAgent.act`.
_EXPLORATORY_TOOLS: tuple[ToolName, ...] = (
    ToolName.LIST_TABLES,
    ToolName.DESCRIBE_TABLE,
    ToolName.SAMPLE_ROWS,
    ToolName.RUN_QUERY,
    ToolName.EXPLAIN_QUERY,
    ToolName.READ_CHANGELOG,
    ToolName.CONSULT_DBA,
)


@dataclass
class RandomAgent:
    """Seeded agent that emits valid ``SqlDriftAction`` envelopes."""

    seed: int = 0
    submit_probability: float = 0.08
    """Probability of drawing ``SUBMIT_REWRITE`` once at least one
    table name is known. Small by design — we want diverse rollouts."""

    _rng: random.Random = field(init=False)
    _known_tables: list[str] = field(init=False)
    _submitted: bool = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._known_tables = []
        self._submitted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, scenario_id: str | None = None) -> None:
        """Reset per-episode state. Optionally reseed the RNG."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._known_tables = []
        self._submitted = False

    # ------------------------------------------------------------------
    # Observation ingestion — harvest table names so later calls target
    # real identifiers.
    # ------------------------------------------------------------------
    def observe(self, obs: SqlDriftObservation) -> None:
        result = obs.tool_result
        if isinstance(result, ListTablesResult):
            for t in result.tables:
                if t not in self._known_tables:
                    self._known_tables.append(t)
        elif isinstance(result, DescribeTableResult) and (
            result.table and result.table not in self._known_tables
        ):
            self._known_tables.append(result.table)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def act(self, obs: SqlDriftObservation) -> SqlDriftAction:
        """Return the next ``SqlDriftAction`` given ``obs``.

        Exploration heuristic:

        1. If we have never seen any table, draw ``LIST_TABLES`` first
           so downstream tools have something to aim at.
        2. With :attr:`submit_probability` (once we have table names),
           emit ``SUBMIT_REWRITE`` and terminate.
        3. Otherwise uniformly sample a tool from
           :data:`_EXPLORATORY_TOOLS`.
        """
        self.observe(obs)

        if not self._known_tables:
            return SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload())

        if not self._submitted and self._rng.random() < self.submit_probability:
            self._submitted = True
            return self._build_submit()

        tool = self._rng.choice(_EXPLORATORY_TOOLS)
        return self._build(tool)

    # ------------------------------------------------------------------
    # Per-tool payload builders
    # ------------------------------------------------------------------
    def _pick_table(self) -> str:
        return self._rng.choice(self._known_tables)

    def _build(self, tool: ToolName) -> SqlDriftAction:
        if tool is ToolName.LIST_TABLES:
            return SqlDriftAction(tool=tool, payload=ListTablesPayload())
        if tool is ToolName.DESCRIBE_TABLE:
            return SqlDriftAction(tool=tool, payload=DescribeTablePayload(table=self._pick_table()))
        if tool is ToolName.SAMPLE_ROWS:
            limit = self._rng.randint(1, 5)
            return SqlDriftAction(
                tool=tool,
                payload=SampleRowsPayload(table=self._pick_table(), limit=limit),
            )
        if tool is ToolName.RUN_QUERY:
            return SqlDriftAction(
                tool=tool,
                payload=RunQueryPayload(sql=f"SELECT * FROM {self._pick_table()} LIMIT 1"),
            )
        if tool is ToolName.EXPLAIN_QUERY:
            return SqlDriftAction(
                tool=tool,
                payload=ExplainQueryPayload(sql=f"SELECT * FROM {self._pick_table()}"),
            )
        if tool is ToolName.READ_CHANGELOG:
            return SqlDriftAction(tool=tool, payload=ReadChangelogPayload())
        if tool is ToolName.CONSULT_DBA:
            return SqlDriftAction(
                tool=tool,
                payload=ConsultDBAPayload(
                    question=self._rng.choice(
                        (
                            "What is the biggest anti-pattern here?",
                            "How do I make this faster?",
                            "Did anything change?",
                        )
                    )
                ),
            )
        raise RuntimeError(f"unhandled tool {tool!r}")

    def _build_submit(self) -> SqlDriftAction:
        # The submit is intentionally trivial: a random-table SELECT *.
        # It will rarely match ground truth, but that's the point — we
        # want the agent to terminate episodes so we observe the full
        # reward pipeline (including the baseline-verbatim gate).
        table = self._pick_table()
        return SqlDriftAction(
            tool=ToolName.SUBMIT_REWRITE,
            payload=SubmitRewritePayload(sql=f"SELECT * FROM {table}"),
        )


__all__ = ["RandomAgent"]
