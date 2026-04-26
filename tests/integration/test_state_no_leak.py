"""P10 — ``env.state.model_dump_json()`` never leaks private runtime fields.

This is the public-state privacy invariant. New fields added to
:class:`RuntimeEpisodeState` must NOT accidentally appear in the public
state dump. ``SqlDriftState`` uses ``extra="forbid"`` so any such leak
would require an explicit code change to the public model — and we
enforce it with a deny-list here so reviewers notice.
"""

from __future__ import annotations

import json

import pytest

from server import SqlDriftEnvironment

# Field-name substrings that must never appear in a public state dump.
PRIVATE_TOKENS = (
    "seed",
    "conn",
    "gt_result_hash",
    "baseline_runtime_ms",
    "baseline_tokens",
    "baseline_sql_canonical",
    "baseline_postdrift_raises",
    "drift_scheduled_step",
    "first_run_query_step",
    "failed_query_hashes",
    "changelog_entries",
    "submitted_sql",
    "submitted_result_hash",
    "submitted_runtime_ms",
    "last_step_was_tool_error",
    "last_step_was_repeat_failing_query",
)


@pytest.fixture()
def env():
    e = SqlDriftEnvironment()
    yield e
    e.close()


@pytest.mark.parametrize(
    "scenario_id",
    [
        "01_correlated_subquery",
        "03_cartesian_join",
        "07_drift_column_rename",
        "09_drift_enum_rule",
    ],
)
def test_public_state_has_no_private_tokens(env, scenario_id: str) -> None:
    env.reset(seed=42, scenario_id=scenario_id)
    j = env.state.model_dump_json()
    parsed = json.loads(j)
    assert "scenario_id" in parsed
    for tok in PRIVATE_TOKENS:
        assert tok not in j, f"leaked {tok!r} in state JSON: {j}"


def test_public_state_schema_is_explicit(env) -> None:
    env.reset(seed=1, scenario_id="01_correlated_subquery")
    schema = env.state.model_json_schema()
    expected_fields = {
        "episode_id",
        "step_count",
        "scenario_id",
        "phase",
        "budget_steps_remaining",
        "drift_fired",
        "consultations_used",
        "submitted",
    }
    actual_fields = set(schema.get("properties", {}).keys())
    assert actual_fields == expected_fields
    # Extra forbidden — adding a new field requires an explicit change.
    assert schema.get("additionalProperties") is False
