"""Unit tests for the LLM-agent completion parser.

We don't instantiate :class:`training.llm_agent.LLMAgent` here — that
requires ``transformers`` + ``peft`` + a GPU model on disk. Instead we
cover :func:`training.llm_agent._parse_completion_as_action` (and its
helpers) because that is the only part that runs hot in eval: the
robustness of SQLDrift eval on trained checkpoints depends on it
tolerating noisy model outputs without crashing the rollout.
"""

from __future__ import annotations

from models import DescribeTablePayload, EpisodePhase, RunQueryPayload, SqlDriftObservation
from training.llm_agent import (
    LLMAgent,
    _fallback_action,
    _parse_completion_as_action,
)


class TestParseCompletion:
    def test_clean_json_roundtrips(self) -> None:
        text = '{"tool": "list_tables", "payload": {"kind": "list_tables"}}'
        action, ok = _parse_completion_as_action(text)
        assert ok is True
        assert action.tool.value == "list_tables"

    def test_json_with_surrounding_prose(self) -> None:
        text = (
            "Here is my tool call:\n"
            '{"tool": "run_query", "payload": {"kind": "run_query", "sql": "SELECT 1"}}\n'
            "hope this works."
        )
        action, ok = _parse_completion_as_action(text)
        assert ok is True
        assert action.tool.value == "run_query"
        assert isinstance(action.payload, RunQueryPayload)
        assert action.payload.sql == "SELECT 1"

    def test_fenced_json_parses(self) -> None:
        text = (
            "```json\n"
            '{"tool": "describe_table", "payload": {"kind": "describe_table", "table": "users"}}\n'
            "```"
        )
        action, ok = _parse_completion_as_action(text)
        assert ok is True
        assert action.tool.value == "describe_table"
        assert isinstance(action.payload, DescribeTablePayload)
        assert action.payload.table == "users"

    def test_mismatched_tool_and_payload_kind_falls_back(self) -> None:
        text = '{"tool": "run_query", "payload": {"kind": "list_tables"}}'
        action, ok = _parse_completion_as_action(text)
        assert ok is False
        assert action == _fallback_action()

    def test_garbage_output_falls_back_without_raising(self) -> None:
        for text in ["", "nope", "{not json}", "{}"]:
            action, ok = _parse_completion_as_action(text)
            assert ok is False
            assert action == _fallback_action()

    def test_nested_json_parses_balanced_envelope(self) -> None:
        text = (
            "Thinking: the baseline uses ts so I'll submit.\n"
            '{"tool": "submit_rewrite", "payload": {"kind": "submit_rewrite", '
            '"sql": "SELECT id FROM orders WHERE id IN (1,2,3)"}}'
        )
        action, ok = _parse_completion_as_action(text)
        assert ok is True
        assert action.tool.value == "submit_rewrite"


class TestLLMAgentReset:
    def test_reset_records_scenario_id_without_loading_model(self) -> None:
        agent = LLMAgent.__new__(LLMAgent)
        agent.seed = 0
        agent._scenario_id = "unknown"
        agent._system_prompt = "stale"
        agent._history = [{"role": "user", "content": "stale"}]

        LLMAgent.reset(agent, seed=7, scenario_id="07_drift_column_rename")

        assert agent.seed == 7
        assert agent._scenario_id == "07_drift_column_rename"
        assert agent._system_prompt == ""
        assert agent._history == []


class TestLLMAgentUserMessage:
    def test_render_user_message_includes_post_drift_hints(self) -> None:
        agent = LLMAgent.__new__(LLMAgent)
        agent._history = []
        obs = SqlDriftObservation(
            step=3,
            phase=EpisodePhase.DRIFT_RECOVERY,
            budget_steps_remaining=12,
            drift_fired=True,
            learned_hints="- [drift:column_rename] update old identifiers",
        )

        text = LLMAgent._render_user_message(agent, obs)

        assert "Drift has fired." in text
        assert "Learned hints:" in text
        assert "drift:column_rename" in text
