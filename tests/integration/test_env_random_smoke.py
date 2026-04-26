"""CPU random-rollout smoke covering every scenario.

* CI default (``pytest -m "not slow"``): 5 random rollouts × 10
  scenarios = 50 episodes. Must terminate within budget and show
  non-trivial reward variance.
* Slow suite (``pytest -m slow``): 50 × 10 = 500 episodes.

Seeds are derived via ``hashlib.blake2b`` so they are stable across
processes. Python's built-in ``hash()`` is process-salted, which made
the non-slow suite flaky before the switch.
"""

from __future__ import annotations

import hashlib
import statistics

import pytest

from server import SqlDriftEnvironment
from training.config import ALL_SCENARIOS
from training.random_agent import RandomAgent


def _stable_seed(scenario_id: str, k: int) -> int:
    """Deterministic 31-bit seed, stable across Python invocations."""
    digest = hashlib.blake2b(f"{scenario_id}:{k}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _run_episode(
    env: SqlDriftEnvironment,
    *,
    scenario_id: str,
    seed: int,
    max_steps: int,
) -> tuple[float, int]:
    obs = env.reset(seed=seed, scenario_id=scenario_id)
    agent = RandomAgent(seed=seed)
    total_reward = 0.0
    steps = 0
    while not obs.done and steps < max_steps:
        obs = env.step(agent.act(obs))
        total_reward += obs.reward or 0.0
        steps += 1
    return total_reward, steps


def _rollout_batch(n_per_scenario: int) -> tuple[list[float], list[int]]:
    rewards: list[float] = []
    steps_list: list[int] = []
    env = SqlDriftEnvironment()
    try:
        for scenario_id in ALL_SCENARIOS:
            for k in range(n_per_scenario):
                reward, steps = _run_episode(
                    env,
                    scenario_id=scenario_id,
                    seed=_stable_seed(scenario_id, k),
                    max_steps=30,
                )
                rewards.append(reward)
                steps_list.append(steps)
    finally:
        env.close()
    return rewards, steps_list


def test_random_smoke_5x10() -> None:
    rewards, steps = _rollout_batch(n_per_scenario=5)
    assert len(rewards) == 5 * len(ALL_SCENARIOS)
    assert max(steps) <= 27, f"some episode exceeded budget: {max(steps)}"
    std = statistics.pstdev(rewards)
    assert std > 0.1, f"reward std too low — policy/env not exercised: std={std:.3f}"


@pytest.mark.slow
def test_random_smoke_50x10() -> None:
    """500-episode nightly-style sweep (multi-seed)."""
    rewards, steps = _rollout_batch(n_per_scenario=50)
    assert len(rewards) == 50 * len(ALL_SCENARIOS)
    assert max(steps) <= 27
    assert statistics.pstdev(rewards) > 0.1
