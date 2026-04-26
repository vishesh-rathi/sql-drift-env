"""Run one SQLDrift episode with an OpenAI-compatible chat endpoint.

Usage:

    uv run python utilities/verbose_api_rollout.py --scenario 07_drift_column_rename

Configuration is read from the repo-local ``.env`` file:

* ``SQL_DRIFT_API_KEY``
* ``SQL_DRIFT_API_BASE_URL``
* ``SQL_DRIFT_API_MODEL``

The environment itself runs in-process, so you do NOT need to start the
OpenEnv server just to watch one episode.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import cast

# Make the repo root importable when the script is run as
# ``python utilities/verbose_api_rollout.py`` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from models import SqlDriftAction, SqlDriftObservation
from server import SqlDriftEnvironment
from training.llm_agent import (
    _TOOL_CONTRACT,
    _canonicalise_action,
    _parse_completion_as_action,
    _summarise_tool_result,
)
from training.prompt import render_system_prompt
from utilities.env_loader import env_str
from utilities.logger import log_interaction


class ApiEpisodeAgent:
    """Minimal chat agent backed by an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        history_turns: int,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._history_turns = max(history_turns, 1)
        self._history: list[dict[str, str]] = []
        self._scenario_id = "unknown"
        self._system_prompt = ""
        self.last_completion = ""
        self.last_parsed_ok = False

    def reset(self, *, scenario_id: str) -> None:
        self._history = []
        self._scenario_id = scenario_id
        self._system_prompt = ""
        self.last_completion = ""
        self.last_parsed_ok = False

    def act(self, obs: SqlDriftObservation) -> SqlDriftAction:
        if not self._system_prompt:
            self._system_prompt = self._initial_system_prompt(obs)
        user_message = self._render_user_message(obs)
        completion = self._generate(user_message)
        action, parsed_ok = _parse_completion_as_action(completion)

        self.last_completion = completion
        self.last_parsed_ok = parsed_ok
        self._history.append({"role": "user", "content": user_message})
        self._history.append(
            {
                "role": "assistant",
                "content": completion if parsed_ok else _canonicalise_action(action),
            }
        )
        self._trim_history()
        return action

    def _initial_system_prompt(self, obs: SqlDriftObservation) -> str:
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
        parts: list[str] = []
        if obs.drift_fired and not self._history_mentions_drift():
            parts.append("Drift has fired.")
        parts.append(f"Remaining steps: {obs.budget_steps_remaining}")
        tool_summary = _summarise_tool_result(obs)
        if tool_summary:
            parts.append(tool_summary)
        if obs.learned_hints:
            parts.append("Learned hints:\n" + obs.learned_hints)
        else:
            parts.append("Pick the next tool call.")
        return "\n".join(parts)

    def _history_mentions_drift(self) -> bool:
        for msg in reversed(self._history):
            if msg["role"] == "user":
                return "Drift has fired." in msg["content"]
        return False

    def _generate(self, user_message: str) -> str:
        messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": "system", "content": self._system_prompt}]
            + self._history
            + [{"role": "user", "content": user_message}],
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            log_interaction(
                event_type="llm_call",
                agent_id=self._agent_id(),
                llm_prompt=messages,
                error=repr(exc),
            )
            raise
        content = response.choices[0].message.content
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = str(content).strip()
        log_interaction(
            event_type="llm_call",
            agent_id=self._agent_id(),
            llm_prompt=messages,
            llm_response=text,
        )
        return text

    def _trim_history(self) -> None:
        max_messages = self._history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _agent_id(self) -> str:
        return f"api_agent:{self._scenario_id}:{self._model}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verbose one-episode API rollout for SQLDrift.")
    parser.add_argument("--scenario", default="07_drift_column_rename")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="normal")
    parser.add_argument("--budget-steps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--enable-dba-oracle", action="store_true")
    return parser.parse_args()


def _required_env(name: str) -> str:
    value = env_str(name, "")
    if not value:
        raise RuntimeError(
            f"Missing {name}. Set it in `.env` (see `.env.example`) or export it before running."
        )
    return value


def _print_header(args: argparse.Namespace, model: str, base_url: str) -> None:
    print("=" * 88)
    print("SQLDrift verbose API rollout")
    print("=" * 88)
    print(
        f"scenario={args.scenario} seed={args.seed} difficulty={args.difficulty} "
        f"budget_steps={args.budget_steps}"
    )
    print(f"model={model}")
    print(f"base_url={base_url}")


def _print_reset(obs: SqlDriftObservation) -> None:
    print("\n[reset]")
    print(f"phase={obs.phase} budget={obs.budget_steps_remaining} drift_fired={obs.drift_fired}")
    if obs.schema_synopsis:
        print("\nSchema synopsis:")
        print(obs.schema_synopsis)
    if obs.baseline_sql:
        print("\nBaseline SQL:")
        print(obs.baseline_sql)
    if obs.learned_hints:
        print("\nLearned hints:")
        print(obs.learned_hints)


def _print_step(
    step: int, agent: ApiEpisodeAgent, action: SqlDriftAction, obs: SqlDriftObservation
) -> None:
    print("\n" + "-" * 88)
    print(f"[step {step:02d}] model completion")
    print(textwrap.indent(agent.last_completion or "(empty)", "  "))
    print(f"[step {step:02d}] parsed_ok={agent.last_parsed_ok}")
    print(f"[step {step:02d}] action={action.model_dump(mode='json')}")
    print(
        f"[step {step:02d}] phase={obs.phase} drift_fired={obs.drift_fired} "
        f"budget={obs.budget_steps_remaining} reward={obs.reward}"
    )
    summary = _summarise_tool_result(obs) or "(no tool result)"
    print(f"[step {step:02d}] result:")
    print(textwrap.indent(summary, "  "))
    if obs.learned_hints:
        print(f"[step {step:02d}] learned_hints:")
        print(textwrap.indent(obs.learned_hints, "  "))
    if obs.reward_components:
        print(f"[step {step:02d}] reward_components={obs.reward_components}")


def _print_final(env: SqlDriftEnvironment, obs: SqlDriftObservation) -> None:
    print("\n" + "=" * 88)
    print("Episode finished")
    print("=" * 88)
    print(f"done={obs.done} reward={obs.reward}")
    print(f"reward_components={obs.reward_components}")
    print(f"effective_speedup={env.effective_speedup()}")
    print(f"final_state={env.state.model_dump()}")


def main() -> None:
    args = _parse_args()
    model = _required_env("SQL_DRIFT_API_MODEL")
    base_url = _required_env("SQL_DRIFT_API_BASE_URL")
    api_key = _required_env("SQL_DRIFT_API_KEY")

    agent = ApiEpisodeAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        history_turns=args.history_turns,
    )
    env = SqlDriftEnvironment()
    try:
        obs = env.reset(
            seed=args.seed,
            scenario_id=args.scenario,
            difficulty=args.difficulty,
            budget_steps=args.budget_steps,
            enable_dba_oracle=args.enable_dba_oracle,
        )
        agent.reset(scenario_id=args.scenario)
        _print_header(args, model, base_url)
        _print_reset(obs)

        for step in range(1, args.max_steps + 1):
            if obs.done:
                break
            action = agent.act(obs)
            obs = env.step(action)
            _print_step(step, agent, action, obs)

        _print_final(env, obs)
    finally:
        env.close()


if __name__ == "__main__":
    main()
