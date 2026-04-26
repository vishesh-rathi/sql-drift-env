"""Public data models for SQLDrift.

Rev 3 design notes enforced here:

- Action is a discriminated union over a public `kind: Literal[...]` tag on
  each payload sub-model. Pydantic v2 forbids leading-underscore names as
  discriminator keys (reserved for private attrs), so we keep the tag public.
- `SqlDriftAction` cross-validates that the envelope-level `tool` matches
  `payload.kind` (prevents inconsistent envelopes from being constructed).
- `SqlDriftObservation.tool_result` is itself a discriminated union over the
  eight concrete result types plus `ToolError` (for in-env semantic failures;
  envelope-level `ValidationError` is a transport-layer concern, not an in-env code).
- `SqlDriftState` is the public state snapshot shipped over `/state`. It
  never carries ground truth, DB handles, baseline runtime, or seeds;
  `extra="forbid"` guarantees no accidental leak as new fields are added.
  The private `RuntimeEpisodeState` lives in :mod:`engine.runtime`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

# =============================================================================
# Enums
# =============================================================================


class ToolName(StrEnum):
    LIST_TABLES = "list_tables"
    DESCRIBE_TABLE = "describe_table"
    SAMPLE_ROWS = "sample_rows"
    RUN_QUERY = "run_query"
    EXPLAIN_QUERY = "explain_query"
    READ_CHANGELOG = "read_changelog"
    SUBMIT_REWRITE = "submit_rewrite"
    CONSULT_DBA = "consult_dba"


class EpisodePhase(StrEnum):
    DIAGNOSE = "diagnose"
    REWRITE = "rewrite"
    DRIFT_RECOVERY = "drift_recovery"
    FINALIZE = "finalize"


class ToolErrorCode(StrEnum):
    """In-environment semantic failure codes (API contract).

    Envelope-level `pydantic.ValidationError` is handled by the OpenEnv
    transport layer (HTTP 422 / `/ws` error frame) and never reaches
    `env.step`, so it has no code here.
    """

    DB_ERROR = "db_error"
    UNKNOWN_TABLE = "unknown_table"
    QUERY_TIMEOUT = "query_timeout"
    RESULT_TOO_LARGE = "result_too_large"
    SUBMIT_BEFORE_DIAGNOSE = "submit_before_diagnose"
    INVALID_TOOL_ARGUMENT = "invalid_tool_argument"


# =============================================================================
# Tool payloads (request side of `SqlDriftAction`)
# =============================================================================


class _BasePayload(BaseModel):
    """Shared config for every tool-call payload."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ListTablesPayload(_BasePayload):
    kind: Literal["list_tables"] = "list_tables"


class DescribeTablePayload(_BasePayload):
    kind: Literal["describe_table"] = "describe_table"
    table: str = Field(min_length=1, max_length=63)


class SampleRowsPayload(_BasePayload):
    kind: Literal["sample_rows"] = "sample_rows"
    table: str = Field(min_length=1, max_length=63)
    limit: int = Field(default=5, ge=1, le=5)


class RunQueryPayload(_BasePayload):
    kind: Literal["run_query"] = "run_query"
    sql: str = Field(min_length=1, max_length=10_000)


class ExplainQueryPayload(_BasePayload):
    kind: Literal["explain_query"] = "explain_query"
    sql: str = Field(min_length=1, max_length=10_000)


class ReadChangelogPayload(_BasePayload):
    kind: Literal["read_changelog"] = "read_changelog"


class SubmitRewritePayload(_BasePayload):
    kind: Literal["submit_rewrite"] = "submit_rewrite"
    sql: str = Field(min_length=1, max_length=10_000)


class ConsultDBAPayload(_BasePayload):
    kind: Literal["consult_dba"] = "consult_dba"
    question: str = Field(min_length=1, max_length=400)


ToolPayload = Annotated[
    ListTablesPayload
    | DescribeTablePayload
    | SampleRowsPayload
    | RunQueryPayload
    | ExplainQueryPayload
    | ReadChangelogPayload
    | SubmitRewritePayload
    | ConsultDBAPayload,
    Field(discriminator="kind"),
]


# Tool -> payload-kind mapping; single source of truth for cross-validation
# and for the server-side dispatcher in P7.
TOOL_TO_PAYLOAD_KIND: dict[ToolName, str] = {
    ToolName.LIST_TABLES: "list_tables",
    ToolName.DESCRIBE_TABLE: "describe_table",
    ToolName.SAMPLE_ROWS: "sample_rows",
    ToolName.RUN_QUERY: "run_query",
    ToolName.EXPLAIN_QUERY: "explain_query",
    ToolName.READ_CHANGELOG: "read_changelog",
    ToolName.SUBMIT_REWRITE: "submit_rewrite",
    ToolName.CONSULT_DBA: "consult_dba",
}


# =============================================================================
# SqlDriftAction envelope
# =============================================================================


class SqlDriftAction(Action):
    """Tool-call envelope.

    JSON wire format::

        {"tool": "run_query", "payload": {"kind": "run_query", "sql": "..."}}

    The `tool` field and `payload.kind` must agree; mismatch raises at
    validation time.
    """

    tool: ToolName
    payload: ToolPayload

    @model_validator(mode="after")
    def _tool_matches_payload(self) -> SqlDriftAction:
        expected = TOOL_TO_PAYLOAD_KIND[self.tool]
        if self.payload.kind != expected:
            # PydanticCustomError keeps ``ctx`` JSON-serializable (plain
            # strings only), unlike a bare ``ValueError`` which Pydantic
            # wraps with ``ctx={"error": ValueError(...)}`` and breaks
            # FastAPI HTTPException JSON encoder (422 responses).
            raise PydanticCustomError(
                "tool_payload_mismatch",
                "tool/payload mismatch: tool={tool} expects payload.kind={expected}, got {got}",
                {
                    "tool": self.tool.value,
                    "expected": expected,
                    "got": self.payload.kind,
                },
            )
        return self


# =============================================================================
# Tool results (response side of `SqlDriftObservation.tool_result`)
# =============================================================================


class _BaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ListTablesResult(_BaseResult):
    kind: Literal["list_tables_result"] = "list_tables_result"
    tables: list[str]


class DescribeTableResult(_BaseResult):
    kind: Literal["describe_table_result"] = "describe_table_result"
    table: str
    columns: list[dict[str, str]]  # [{"name": "...", "type": "..."}]


class SampleRowsResult(_BaseResult):
    kind: Literal["sample_rows_result"] = "sample_rows_result"
    table: str
    columns: list[str]
    rows: list[list[Any]]


class RunQueryResult(_BaseResult):
    kind: Literal["run_query_result"] = "run_query_result"
    columns: list[str]
    rows: list[list[Any]]
    runtime_ms: float
    row_count: int


class ExplainQueryResult(_BaseResult):
    kind: Literal["explain_query_result"] = "explain_query_result"
    plan: str


class ReadChangelogResult(_BaseResult):
    kind: Literal["read_changelog_result"] = "read_changelog_result"
    entries: list[str]


class SubmitRewriteResult(_BaseResult):
    kind: Literal["submit_rewrite_result"] = "submit_rewrite_result"
    accepted: bool
    runtime_ms: float
    matches_ground_truth: bool


class ConsultDBAResult(_BaseResult):
    kind: Literal["consult_dba_result"] = "consult_dba_result"
    tier: int = Field(ge=1, le=3)
    hint: str


class ToolError(_BaseResult):
    kind: Literal["tool_error"] = "tool_error"
    code: ToolErrorCode
    message: str = Field(max_length=2_000)


ToolResult = Annotated[
    ListTablesResult
    | DescribeTableResult
    | SampleRowsResult
    | RunQueryResult
    | ExplainQueryResult
    | ReadChangelogResult
    | SubmitRewriteResult
    | ConsultDBAResult
    | ToolError,
    Field(discriminator="kind"),
]


# The six reward-component keys match the composed rubric; tests and telemetry
# rely on this exact schema.
REWARD_COMPONENT_KEYS: tuple[str, ...] = (
    "r_correct",
    "r_drift",
    "r_speedup",
    "r_step_tax",
    "r_gatekeepers",
    "r_consult_dba",
)


# =============================================================================
# SqlDriftObservation
# =============================================================================


def _zero_reward_components() -> dict[str, float]:
    """Six-key reward envelope initialised to zero.

    Every observation, including the reset observation, carries the full
    six-key schema so telemetry and tests can index it unconditionally.
    """
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


class SqlDriftObservation(Observation):
    """Observation returned by :meth:`SqlDriftEnvironment.step`.

    Inherits `done: bool` and `reward: float | None` from base Observation.

    The task payload (`baseline_sql`, `schema_synopsis`) is delivered on
    the reset observation and kept empty on subsequent steps: the agent
    is expected to capture it once and hold it in its own context.
    """

    step: int = Field(ge=0)
    phase: EpisodePhase
    last_tool: ToolName | None = None
    tool_result: ToolResult | None = None
    drift_fired: bool = False
    drift_acknowledged: bool = False
    learned_hints: str = Field(default="", max_length=800)
    baseline_sql: str = Field(default="", max_length=10_000)
    schema_synopsis: str = Field(default="", max_length=2_000)
    budget_steps_remaining: int = Field(ge=0)
    reward_components: dict[str, float] = Field(default_factory=_zero_reward_components)


# =============================================================================
# SqlDriftState — PUBLIC state (sanitized)
# =============================================================================


class SqlDriftState(State):
    """Public state snapshot — serialized over `/state`.

    Ground truth, DB handles, seeds, and baseline SQL live in
    :class:`engine.runtime.RuntimeEpisodeState` and are never exposed here.
    `extra="forbid"` guarantees no accidental leak via future field additions.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    scenario_id: str
    phase: EpisodePhase
    budget_steps_remaining: int = Field(ge=0)
    drift_fired: bool = False
    consultations_used: int = Field(default=0, ge=0)
    submitted: bool = False


__all__ = [
    "ConsultDBAPayload",
    "ConsultDBAResult",
    "DescribeTablePayload",
    "DescribeTableResult",
    "EpisodePhase",
    "ExplainQueryPayload",
    "ExplainQueryResult",
    "ListTablesPayload",
    "ListTablesResult",
    "REWARD_COMPONENT_KEYS",
    "ReadChangelogPayload",
    "ReadChangelogResult",
    "RunQueryPayload",
    "RunQueryResult",
    "SampleRowsPayload",
    "SampleRowsResult",
    "SqlDriftAction",
    "SqlDriftObservation",
    "SqlDriftState",
    "SubmitRewritePayload",
    "SubmitRewriteResult",
    "TOOL_TO_PAYLOAD_KIND",
    "ToolError",
    "ToolErrorCode",
    "ToolName",
    "ToolPayload",
    "ToolResult",
]
