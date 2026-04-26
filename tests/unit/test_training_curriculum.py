"""P12 — :func:`training.grpo_train.iter_curriculum` unit tests."""

from __future__ import annotations

from itertools import islice

from training.config import CurriculumConfig, GRPOConfig
from training.grpo_train import iter_curriculum


def test_uniform_samples_all_scenarios_given_enough_draws() -> None:
    cfg = GRPOConfig(curriculum=CurriculumConfig(mode="uniform"))
    draws = list(islice(iter_curriculum(cfg, seed=0), 400))
    assert {scenario for scenario, _ in draws} == set(cfg.curriculum.scenarios)


def test_static_order_is_round_robin() -> None:
    cfg = GRPOConfig(
        curriculum=CurriculumConfig(
            mode="static_order",
            scenarios=("a", "b", "c"),
        )
    )
    draws = [s for s, _ in islice(iter_curriculum(cfg, seed=0), 7)]
    assert draws == ["a", "b", "c", "a", "b", "c", "a"]


def test_weighted_respects_distribution() -> None:
    cfg = GRPOConfig(
        curriculum=CurriculumConfig(
            mode="weighted",
            scenarios=("hot", "cold"),
            weights=(9.0, 1.0),
        )
    )
    draws = [s for s, _ in islice(iter_curriculum(cfg, seed=0), 1000)]
    hot = draws.count("hot")
    assert 0.80 < hot / 1000 < 0.98  # ~0.9 in expectation


def test_seeds_are_in_configured_range() -> None:
    cfg = GRPOConfig(curriculum=CurriculumConfig(seed_range=(10, 20)))
    draws = list(islice(iter_curriculum(cfg, seed=1), 200))
    assert all(10 <= s < 20 for _, s in draws)


def test_seeded_iter_is_deterministic() -> None:
    cfg = GRPOConfig(curriculum=CurriculumConfig())
    a = list(islice(iter_curriculum(cfg, seed=123), 20))
    b = list(islice(iter_curriculum(cfg, seed=123), 20))
    assert a == b
