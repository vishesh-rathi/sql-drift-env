"""SQLDrift ``EnvClient`` — tool-aware payload constructors + response parser.

Inherits :class:`openenv.core.env_client.EnvClient` so TRL rollouts,
notebook exploration, and integration tests all use the same WS-backed
session semantics. Stateful episodes MUST go through the ``/ws`` channel
(HTTP ``/step`` is stateless: one fresh env per request).

Convenience constructors (:meth:`SqlDriftEnv.action_list_tables`, etc.)
hide the discriminated-union boilerplate so agent code reads naturally::

    env = SqlDriftEnv(base_url="http://localhost:8000").sync()
    with env:
        r = env.reset(seed=42, scenario_id="03_cartesian_join")
        r = env.step(SqlDriftEnv.action_run_query("SELECT COUNT(*) FROM events"))
        ...
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import (
    ConsultDBAPayload,
    DescribeTablePayload,
    ExplainQueryPayload,
    ListTablesPayload,
    ReadChangelogPayload,
    RunQueryPayload,
    SampleRowsPayload,
    SqlDriftAction,
    SqlDriftObservation,
    SqlDriftState,
    SubmitRewritePayload,
    ToolName,
)


class SqlDriftEnv(EnvClient[SqlDriftAction, SqlDriftObservation, SqlDriftState]):
    """Tool-aware client for the SQLDrift OpenEnv environment."""

    # ------------------------------------------------------------------
    # EnvClient ABC implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: SqlDriftAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SqlDriftObservation]:
        obs_data = payload.get("observation", {})
        observation = SqlDriftObservation.model_validate(obs_data)
        # Base transport strips reward + done off the observation dict — we
        # re-populate them so the agent can read straight off `.observation`.
        reward = payload.get("reward")
        done = bool(payload.get("done", False))
        observation.reward = reward
        observation.done = done
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> SqlDriftState:
        return SqlDriftState.model_validate(payload)

    # ------------------------------------------------------------------
    # Action factories — one per tool, accepting only the args that tool
    # cares about; payload.kind is filled in automatically.
    # ------------------------------------------------------------------

    @staticmethod
    def action_list_tables() -> SqlDriftAction:
        return SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload())

    @staticmethod
    def action_describe_table(table: str) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.DESCRIBE_TABLE,
            payload=DescribeTablePayload(table=table),
        )

    @staticmethod
    def action_sample_rows(table: str, limit: int = 5) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.SAMPLE_ROWS,
            payload=SampleRowsPayload(table=table, limit=limit),
        )

    @staticmethod
    def action_run_query(sql: str) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.RUN_QUERY,
            payload=RunQueryPayload(sql=sql),
        )

    @staticmethod
    def action_explain_query(sql: str) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.EXPLAIN_QUERY,
            payload=ExplainQueryPayload(sql=sql),
        )

    @staticmethod
    def action_read_changelog() -> SqlDriftAction:
        return SqlDriftAction(tool=ToolName.READ_CHANGELOG, payload=ReadChangelogPayload())

    @staticmethod
    def action_submit_rewrite(sql: str) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.SUBMIT_REWRITE,
            payload=SubmitRewritePayload(sql=sql),
        )

    @staticmethod
    def action_consult_dba(question: str) -> SqlDriftAction:
        return SqlDriftAction(
            tool=ToolName.CONSULT_DBA,
            payload=ConsultDBAPayload(question=question),
        )


__all__ = ["SqlDriftEnv"]
