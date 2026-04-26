"""P1 — unit tests for public data models."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from models import (
    REWARD_COMPONENT_KEYS,
    TOOL_TO_PAYLOAD_KIND,
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
    ToolName,
    ToolResult,
)


class TestActionDiscriminator:
    def test_roundtrip_every_tool_payload(self) -> None:
        cases: list[tuple[ToolName, dict]] = [
            (ToolName.LIST_TABLES, {"kind": "list_tables"}),
            (ToolName.DESCRIBE_TABLE, {"kind": "describe_table", "table": "orders"}),
            (ToolName.SAMPLE_ROWS, {"kind": "sample_rows", "table": "orders", "limit": 3}),
            (ToolName.RUN_QUERY, {"kind": "run_query", "sql": "SELECT 1"}),
            (ToolName.EXPLAIN_QUERY, {"kind": "explain_query", "sql": "SELECT 1"}),
            (ToolName.READ_CHANGELOG, {"kind": "read_changelog"}),
            (ToolName.SUBMIT_REWRITE, {"kind": "submit_rewrite", "sql": "SELECT 1"}),
            (ToolName.CONSULT_DBA, {"kind": "consult_dba", "question": "help"}),
        ]
        for tool, payload in cases:
            action = SqlDriftAction.model_validate({"tool": tool, "payload": payload})
            dumped = action.model_dump_json()
            reparsed = SqlDriftAction.model_validate_json(dumped)
            assert reparsed.tool == tool
            assert reparsed.payload.kind == TOOL_TO_PAYLOAD_KIND[tool]

    def test_mismatched_tool_and_payload_kind_raises(self) -> None:
        with pytest.raises(ValidationError, match="tool/payload mismatch"):
            SqlDriftAction.model_validate(
                {"tool": ToolName.RUN_QUERY, "payload": ListTablesPayload()}
            )

    def test_unknown_tool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SqlDriftAction.model_validate_json(
                '{"tool": "bogus", "payload": {"kind": "list_tables"}}'
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RunQueryPayload.model_validate({"kind": "run_query", "sql": "SELECT 1", "x": 1})


class TestPayloadBounds:
    def test_sql_length_cap(self) -> None:
        with pytest.raises(ValidationError):
            RunQueryPayload(sql="x" * 10_001)

    def test_table_name_cap(self) -> None:
        with pytest.raises(ValidationError):
            DescribeTablePayload(table="x" * 64)

    def test_sample_limit_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            SampleRowsPayload(table="t", limit=6)

    def test_consult_dba_question_cap(self) -> None:
        with pytest.raises(ValidationError):
            ConsultDBAPayload(question="q" * 401)


class TestToolResultDiscriminator:
    def test_every_result_kind_parses_through_union(self) -> None:
        adapter = TypeAdapter(ToolResult)
        cases: list[dict] = [
            {"kind": "list_tables_result", "tables": ["a", "b"]},
            {
                "kind": "describe_table_result",
                "table": "orders",
                "columns": [{"name": "id", "type": "BIGINT"}],
            },
            {
                "kind": "sample_rows_result",
                "table": "orders",
                "columns": ["id"],
                "rows": [[1], [2]],
            },
            {
                "kind": "run_query_result",
                "columns": ["c"],
                "rows": [[1]],
                "runtime_ms": 1.2,
                "row_count": 1,
            },
            {"kind": "explain_query_result", "plan": "PROJECT -> SEQ_SCAN"},
            {"kind": "read_changelog_result", "entries": ["e1", "e2"]},
            {
                "kind": "submit_rewrite_result",
                "accepted": True,
                "runtime_ms": 4.2,
                "matches_ground_truth": True,
            },
            {"kind": "consult_dba_result", "tier": 2, "hint": "try EXPLAIN"},
            {"kind": "tool_error", "code": "db_error", "message": "boom"},
        ]
        for payload in cases:
            parsed = adapter.validate_python(payload)
            assert parsed.kind == payload["kind"]

    def test_tool_error_code_enum(self) -> None:
        err = ToolError(code=ToolErrorCode.QUERY_TIMEOUT, message="timed out")
        assert err.code == ToolErrorCode.QUERY_TIMEOUT


class TestObservation:
    def test_observation_carries_reward_components(self) -> None:
        obs = SqlDriftObservation(
            step=3,
            phase=EpisodePhase.DIAGNOSE,
            last_tool=ToolName.RUN_QUERY,
            tool_result=RunQueryResult(columns=["c"], rows=[[1]], runtime_ms=0.5, row_count=1),
            drift_fired=False,
            drift_acknowledged=False,
            learned_hints="",
            budget_steps_remaining=22,
            reward_components={k: 0.0 for k in REWARD_COMPONENT_KEYS},
            done=False,
            reward=0.0,
        )
        assert set(obs.reward_components.keys()) == set(REWARD_COMPONENT_KEYS)

    def test_observation_roundtrip_preserves_discriminator(self) -> None:
        obs = SqlDriftObservation(
            step=1,
            phase=EpisodePhase.DIAGNOSE,
            last_tool=ToolName.LIST_TABLES,
            tool_result=ListTablesResult(tables=["orders"]),
            budget_steps_remaining=24,
            reward=0.0,
        )
        clone = SqlDriftObservation.model_validate_json(obs.model_dump_json())
        assert clone.tool_result is not None
        assert clone.tool_result.kind == "list_tables_result"


class TestStatePublicFace:
    def test_state_forbids_extra_fields(self) -> None:
        """Rev-3 — public state must reject any field we didn't explicitly model."""
        with pytest.raises(ValidationError):
            SqlDriftState.model_validate(
                {
                    "scenario_id": "x",
                    "phase": "diagnose",
                    "budget_steps_remaining": 25,
                    "conn": "SECRET_HANDLE",  # MUST be rejected
                }
            )

    def test_state_json_contains_no_private_names(self) -> None:
        state = SqlDriftState(
            scenario_id="01_correlated_subquery",
            phase=EpisodePhase.DIAGNOSE,
            budget_steps_remaining=25,
            drift_fired=False,
            consultations_used=0,
            submitted=False,
        )
        payload = state.model_dump_json()
        for denylisted in (
            "seed",
            "conn",
            "gt_result_hash_predrift",
            "gt_result_hash_postdrift",
            "baseline_runtime_ms",
            "baseline_sql_canonical",
            "submitted_sql",
        ):
            assert denylisted not in payload

    def test_state_negative_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SqlDriftState(
                scenario_id="x",
                phase=EpisodePhase.DIAGNOSE,
                budget_steps_remaining=-1,
            )


class TestResultHelpers:
    def test_describe_result_schema_shape(self) -> None:
        res = DescribeTableResult(table="orders", columns=[{"name": "id", "type": "BIGINT"}])
        assert res.columns[0]["name"] == "id"

    def test_submit_result_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            SubmitRewriteResult.model_validate({"kind": "submit_rewrite_result", "accepted": True})

    def test_consult_tier_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ConsultDBAResult(tier=4, hint="nope")
        with pytest.raises(ValidationError):
            ConsultDBAResult(tier=0, hint="nope")
        ConsultDBAResult(tier=1, hint="ok")
        ConsultDBAResult(tier=3, hint="ok")


# Quiet unused-import linter: every result type above is exercised either
# through the TypeAdapter union or direct construction.
_USED = (
    ExplainQueryPayload,
    ExplainQueryResult,
    ReadChangelogPayload,
    ReadChangelogResult,
    SampleRowsResult,
    SubmitRewritePayload,
)
