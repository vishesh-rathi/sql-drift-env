"""P12 — :class:`training.random_agent.RandomAgent` unit tests."""

from __future__ import annotations

from models import (
    DescribeTableResult,
    EpisodePhase,
    ListTablesResult,
    SqlDriftObservation,
    SubmitRewritePayload,
    ToolName,
)
from training.random_agent import RandomAgent


def _empty_obs() -> SqlDriftObservation:
    return SqlDriftObservation(step=0, phase=EpisodePhase.DIAGNOSE, budget_steps_remaining=25)


def _obs_with(result) -> SqlDriftObservation:
    return SqlDriftObservation(
        step=1,
        phase=EpisodePhase.DIAGNOSE,
        budget_steps_remaining=24,
        tool_result=result,
    )


class TestRandomAgent:
    def test_first_action_is_list_tables(self) -> None:
        agent = RandomAgent(seed=0)
        action = agent.act(_empty_obs())
        assert action.tool is ToolName.LIST_TABLES

    def test_seeded_trajectories_are_deterministic(self) -> None:
        def run() -> list[str]:
            a = RandomAgent(seed=42)
            # Feed one ListTables result so the agent can target tables.
            a.observe(_obs_with(ListTablesResult(tables=["t1", "t2"])))
            return [a.act(_empty_obs()).tool.value for _ in range(15)]

        assert run() == run()

    def test_different_seeds_diverge(self) -> None:
        def run(seed: int) -> list[str]:
            a = RandomAgent(seed=seed)
            a.observe(_obs_with(ListTablesResult(tables=["t1", "t2"])))
            return [a.act(_empty_obs()).tool.value for _ in range(30)]

        assert run(1) != run(2)

    def test_observe_harvests_known_tables(self) -> None:
        a = RandomAgent(seed=0)
        a.observe(_obs_with(ListTablesResult(tables=["orders", "users"])))
        a.observe(_obs_with(DescribeTableResult(table="products", columns=[])))
        # Internal state — but we probe it via many actions; every
        # table-pinned payload should hit one of those three names.
        for _ in range(100):
            action = a.act(_empty_obs())
            tbl = getattr(action.payload, "table", None)
            if tbl is not None:
                assert tbl in {"orders", "users", "products"}

    def test_submit_only_fires_once(self) -> None:
        agent = RandomAgent(seed=0, submit_probability=1.0)
        agent.observe(_obs_with(ListTablesResult(tables=["t"])))

        first = agent.act(_empty_obs())
        assert first.tool is ToolName.SUBMIT_REWRITE
        assert isinstance(first.payload, SubmitRewritePayload)

        # Subsequent calls must pick a non-submit tool.
        for _ in range(50):
            nxt = agent.act(_empty_obs())
            assert nxt.tool is not ToolName.SUBMIT_REWRITE

    def test_reset_clears_state(self) -> None:
        a = RandomAgent(seed=5)
        a.observe(_obs_with(ListTablesResult(tables=["x"])))
        a.reset()
        # After reset the very first action is list_tables again.
        assert a.act(_empty_obs()).tool is ToolName.LIST_TABLES
