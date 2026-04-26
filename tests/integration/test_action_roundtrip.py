"""P10 — every ``SqlDriftAction`` payload round-trips through JSON.

The discriminated-union `kind` tag means serialization + parsing must
agree on which sub-payload to instantiate. Any divergence here would
silently corrupt agent actions over the wire.
"""

from __future__ import annotations

import json

import pytest

from client import SqlDriftEnv
from models import SqlDriftAction

ACTION_FACTORIES = [
    SqlDriftEnv.action_list_tables(),
    SqlDriftEnv.action_describe_table("users"),
    SqlDriftEnv.action_sample_rows("users", limit=3),
    SqlDriftEnv.action_run_query("SELECT 1"),
    SqlDriftEnv.action_explain_query("SELECT 1"),
    SqlDriftEnv.action_read_changelog(),
    SqlDriftEnv.action_submit_rewrite("SELECT 1 AS x"),
    SqlDriftEnv.action_consult_dba("What's up?"),
]


@pytest.mark.parametrize("action", ACTION_FACTORIES, ids=[a.tool.value for a in ACTION_FACTORIES])
def test_action_roundtrip(action: SqlDriftAction) -> None:
    blob = action.model_dump_json()
    parsed = SqlDriftAction.model_validate_json(blob)
    assert parsed == action
    # And round-trip via dict too (the JSON transport hits both paths).
    as_dict = json.loads(blob)
    reparsed = SqlDriftAction.model_validate(as_dict)
    assert reparsed == action
    # The kind tag matches the tool name (enforced by model_validator).
    assert reparsed.payload.kind == reparsed.tool.value


def test_all_eight_tools_covered() -> None:
    kinds = {a.payload.kind for a in ACTION_FACTORIES}
    assert len(kinds) == 8
