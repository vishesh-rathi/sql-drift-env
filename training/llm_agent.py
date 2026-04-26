"""LLM-backed agent used by :mod:`training.eval` for trained checkpoints.

The eval harness only needs the :class:`.eval.Agent` protocol
(``reset(seed)`` + ``act(obs) -> SqlDriftAction``). This module
supplies a minimal, chat-template-driven policy that:

1. Loads a saved model directory (either a full HF checkpoint or a PEFT
   adapter pointing at a base model).
2. Maintains a bounded chat history across the episode so the model
   sees its own prior tool calls and their observations.
3. Prompts the model to emit *exactly one* JSON tool-call envelope per
   turn (``{"tool": "...", "payload": {...}}``) and parses it into a
   :class:`models.SqlDriftAction`.
4. Falls back to a safe default action on parse failure rather than
   crashing the rollout — this matches the random-agent contract and
   keeps eval sweeps resilient to occasional generation noise.

All heavy ML imports (``torch``, ``transformers``, ``peft``) are
deferred into :meth:`LLMAgent.__init__` so the module is importable on
CPU-only CI boxes for type checking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from models import (
    ConsultDBAResult,
    DescribeTableResult,
    ExplainQueryResult,
    ListTablesPayload,
    ListTablesResult,
    ReadChangelogResult,
    RunQueryResult,
    SampleRowsResult,
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewriteResult,
    ToolError,
    ToolName,
    ToolPayload,
)
from training.prompt import render_system_prompt
from utilities.logger import get_module_logger, log_interaction

_LOG = get_module_logger(__name__)

# Compact, model-facing JSON contract. Kept short because it ships with
# every turn and its tokens count against ``max_seq_length``.
_TOOL_CONTRACT = (
    "Respond with EXACTLY ONE JSON object per turn and nothing else:\n"
    '{"tool": "<tool_name>", "payload": {...}}\n'
    "Valid tool names: list_tables, describe_table, sample_rows, run_query, "
    "explain_query, read_changelog, submit_rewrite, consult_dba.\n"
    "Payload schemas (match one):\n"
    '- list_tables: {"kind": "list_tables"}\n'
    '- describe_table: {"kind": "describe_table", "table": "<str>"}\n'
    '- sample_rows: {"kind": "sample_rows", "table": "<str>", "limit": 1..5}\n'
    '- run_query: {"kind": "run_query", "sql": "<SELECT ...>"}\n'
    '- explain_query: {"kind": "explain_query", "sql": "<SELECT ...>"}\n'
    '- read_changelog: {"kind": "read_changelog"}\n'
    '- submit_rewrite: {"kind": "submit_rewrite", "sql": "<SELECT ...>"}\n'
    '- consult_dba: {"kind": "consult_dba", "question": "<str>"}\n'
    "Never emit prose; never wrap JSON in Markdown fences."
)

# The first capture group picks up the first balanced JSON object in the
# completion. We keep this forgiving (``re.DOTALL``) so models that wrap
# the JSON in markdown code fences still parse.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


class LLMAgent:
    """Chat-template agent compatible with :class:`training.eval.Agent`.

    Parameters
    ----------
    model_path:
        Directory containing either a HF ``AutoModelForCausalLM``
        checkpoint or a PEFT adapter (detected by presence of
        ``adapter_config.json``).
    base_model:
        Optional explicit base model id. Required only if the adapter's
        ``adapter_config.json`` does not record ``base_model_name_or_path``.
    max_new_tokens:
        Cap on tokens generated per turn. 128 tokens is enough for any
        tool envelope while keeping rollouts brisk.
    temperature:
        Sampling temperature. ``0.0`` switches to greedy decoding,
        which is what we want for deterministic eval sweeps.
    history_turns:
        How many past ``(user, assistant)`` turns to keep in the rolling
        context. Older turns are dropped to keep the prompt bounded.
    """

    def __init__(
        self,
        model_path: str,
        *,
        base_model: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        history_turns: int = 6,
        seed: int = 0,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path)
        is_adapter = (path / "adapter_config.json").exists()
        if is_adapter:
            adapter_cfg = json.loads((path / "adapter_config.json").read_text())
            resolved_base = base_model or adapter_cfg.get("base_model_name_or_path") or ""
            if not resolved_base:
                raise ValueError(
                    f"adapter at {path!s} lacks base_model_name_or_path; "
                    "pass base_model= explicitly"
                )
            tokenizer = AutoTokenizer.from_pretrained(resolved_base)
            model = AutoModelForCausalLM.from_pretrained(
                resolved_base,
                torch_dtype="auto",
                device_map="auto",
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                torch_dtype="auto",
                device_map="auto",
            )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._history_turns = max(history_turns, 1)
        self.seed = seed

        self._scenario_id = "unknown"
        self._system_prompt = ""
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, scenario_id: str | None = None) -> None:
        """Clear per-episode history (called at the start of every episode)."""
        if seed is not None:
            self.seed = seed
        self._scenario_id = scenario_id or "unknown"
        self._system_prompt = ""
        self._history = []

    def act(self, obs: SqlDriftObservation) -> SqlDriftAction:
        """Return the next :class:`SqlDriftAction` given the latest observation."""
        if not self._system_prompt:
            self._system_prompt = self._initial_system_prompt(obs)
        user_message = self._render_user_message(obs)
        messages = self._build_messages(user_message)

        completion = self._generate(messages)
        action, parsed_ok = _parse_completion_as_action(completion)

        # Record this turn *after* generation so parse failures do not
        # poison the history with obviously-broken assistant text.
        self._history.append({"role": "user", "content": user_message})
        assistant_text = completion if parsed_ok else _canonicalise_action(action)
        self._history.append({"role": "assistant", "content": assistant_text})
        self._trim_history()

        return action

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _initial_system_prompt(self, obs: SqlDriftObservation) -> str:
        """Build the per-episode system prompt.

        We concatenate the shared :func:`render_system_prompt` (so the
        tool catalog stays in lockstep with :class:`models.ToolName`)
        with :data:`_TOOL_CONTRACT` (the compact JSON shape the model
        must emit). The first turn also carries the baseline SQL and
        schema synopsis so the model does not have to discover them
        before it can do anything useful.
        """
        base = render_system_prompt(
            scenario_id=self._scenario_id,
            learned_hints=obs.learned_hints,
            phase=obs.phase,
            budget_steps_remaining=obs.budget_steps_remaining,
            drift_fired=obs.drift_fired,
        )
        task_block = ""
        if obs.schema_synopsis:
            task_block += f"\n\nSchema synopsis:\n{obs.schema_synopsis}"
        if obs.baseline_sql:
            task_block += f"\n\nBaseline query:\n{obs.baseline_sql}"
        return f"{base}{task_block}\n\n{_TOOL_CONTRACT}"

    def _render_user_message(self, obs: SqlDriftObservation) -> str:
        """Summarise the env's response to the previous tool call."""
        parts: list[str] = []
        if obs.drift_fired and (not self._history or self._most_recent_user_mentions_drift()):
            parts.append("Drift has fired.")
        if obs.budget_steps_remaining is not None:
            parts.append(f"Remaining steps: {obs.budget_steps_remaining}")
        tool_summary = _summarise_tool_result(obs)
        if tool_summary:
            parts.append(tool_summary)
        if obs.learned_hints:
            parts.append("Learned hints:\n" + obs.learned_hints)
        if not parts:
            parts.append("Pick the next tool call.")
        return "\n".join(parts)

    def _most_recent_user_mentions_drift(self) -> bool:
        for msg in reversed(self._history):
            if msg["role"] == "user":
                return "Drift has fired" not in msg["content"]
        return True

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        return (
            [{"role": "system", "content": self._system_prompt}]
            + self._history
            + [{"role": "user", "content": user_message}]
        )

    def _trim_history(self) -> None:
        # Each logical turn is a (user, assistant) pair.
        max_messages = self._history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        tok = self._tokenizer
        inputs = tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self._model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "pad_token_id": tok.pad_token_id,
        }
        if self._temperature > 0.0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self._temperature,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.inference_mode():
                output = self._model.generate(inputs, **gen_kwargs)
        except Exception as exc:
            log_interaction(
                event_type="llm_call",
                agent_id=self._agent_id(),
                llm_prompt=messages,
                error=repr(exc),
            )
            raise
        completion_ids = output[0, inputs.shape[-1] :]
        text: str = tok.decode(completion_ids, skip_special_tokens=True)
        response = text.strip()
        log_interaction(
            event_type="llm_call",
            agent_id=self._agent_id(),
            llm_prompt=messages,
            llm_response=response,
        )
        return response

    def _agent_id(self) -> str:
        return f"llm_agent:{self._scenario_id}:{self.seed}"


# ---------------------------------------------------------------------------
# Module-level helpers — shared with unit tests.
# ---------------------------------------------------------------------------


def _canonicalise_action(action: SqlDriftAction) -> str:
    """Render a ``SqlDriftAction`` the same way the model is asked to."""
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"))


def _parse_completion_as_action(text: str) -> tuple[SqlDriftAction, bool]:
    """Turn a model completion into a valid action.

    Returns a ``(action, parsed_ok)`` tuple. Callers use the boolean to
    decide whether to record the raw completion or a sanitised form in
    the chat history.

    On any parse failure we fall back to ``list_tables`` (the cheapest,
    always-safe action) and log at INFO so eval runs surface the miss
    without crashing.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            try:
                return SqlDriftAction.model_validate(payload), True
            except Exception as exc:  # pydantic.ValidationError & friends
                _LOG.info("LLM agent produced invalid action JSON: %s", exc)
    return _fallback_action(), False


def _fallback_action() -> SqlDriftAction:
    payload: ToolPayload = ListTablesPayload()
    return SqlDriftAction(tool=ToolName.LIST_TABLES, payload=payload)


def _summarise_tool_result(obs: SqlDriftObservation) -> str:
    """Compact textual view of the last tool_result, bounded in length."""
    tr = obs.tool_result
    if tr is None:
        return ""
    if isinstance(tr, ToolError):
        return f"Tool error [{tr.code.value}]: {tr.message}"
    if isinstance(tr, ListTablesResult):
        return "Tables: " + ", ".join(tr.tables[:30])
    if isinstance(tr, DescribeTableResult):
        cols = ", ".join(f"{c.get('name', '')}:{c.get('type', '')}" for c in tr.columns[:20])
        return f"{tr.table} columns: {cols}"
    if isinstance(tr, SampleRowsResult):
        return _render_small_table(tr.columns, tr.rows)
    if isinstance(tr, RunQueryResult):
        return f"{tr.row_count} rows in {tr.runtime_ms:.1f}ms\n" + _render_small_table(
            tr.columns, tr.rows
        )
    if isinstance(tr, ExplainQueryResult):
        return "Plan:\n" + tr.plan[:1500]
    if isinstance(tr, ReadChangelogResult):
        return "Changelog:\n" + ("\n---\n".join(tr.entries[-3:]) or "(no entries)")
    if isinstance(tr, SubmitRewriteResult):
        verdict = "matched" if tr.matches_ground_truth else "mismatch"
        return f"Submitted ({verdict}, {tr.runtime_ms:.1f}ms)"
    if isinstance(tr, ConsultDBAResult):
        return f"[DBA tier {tr.tier}] {tr.hint}"
    return ""


def _render_small_table(columns: list[str], rows: list[list[Any]], limit: int = 5) -> str:
    header = ", ".join(columns) if columns else "(no columns)"
    head = rows[:limit]
    body = "\n".join(" | ".join(str(cell) for cell in row) for row in head)
    more = f"\n... ({len(rows) - len(head)} more)" if len(rows) > len(head) else ""
    return f"[{header}]\n{body}{more}" if body else f"[{header}] (0 rows)"


__all__ = ["LLMAgent"]
