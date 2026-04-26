"""P12 — :mod:`training.config` unit tests."""

from __future__ import annotations

import importlib

import pytest

import training.config as config_mod
from training.config import ALL_SCENARIOS, CurriculumConfig, GRPOConfig


class TestCurriculumConfig:
    def test_defaults_cover_all_scenarios(self) -> None:
        c = CurriculumConfig()
        assert c.scenarios == ALL_SCENARIOS
        assert len(c.scenarios) == len(ALL_SCENARIOS)
        assert c.mode == "uniform"

    def test_weighted_requires_matching_weights(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            CurriculumConfig(mode="weighted", weights=(1.0,))

    def test_weighted_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            CurriculumConfig(
                mode="weighted",
                scenarios=("a", "b"),
                weights=(1.0, -0.5),
            )

    def test_weighted_rejects_all_zero(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            CurriculumConfig(
                mode="weighted",
                scenarios=("a", "b"),
                weights=(0.0, 0.0),
            )

    def test_empty_scenarios_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CurriculumConfig(scenarios=())

    def test_bad_seed_range(self) -> None:
        with pytest.raises(ValueError, match="seed_range"):
            CurriculumConfig(seed_range=(5, 5))
        with pytest.raises(ValueError, match="seed_range"):
            CurriculumConfig(seed_range=(-1, 10))


class TestGRPOConfig:
    def test_defaults_sane(self) -> None:
        c = GRPOConfig()
        assert c.group_size >= 2
        assert c.max_steps > 0
        assert c.seed == 0
        assert c.lora_r == 16
        assert c.fp16 is True
        assert c.bf16 is False
        assert c.curriculum.scenarios == ALL_SCENARIOS

    def test_env_base_url_defaults_from_environment(self, monkeypatch) -> None:
        with monkeypatch.context() as m:
            m.setenv("SQL_DRIFT_ENV_URL", "http://localhost:8123")
            reloaded = importlib.reload(config_mod)
            c = reloaded.GRPOConfig()
            assert c.env_base_url == "http://localhost:8123"
        importlib.reload(config_mod)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"group_size": 1}, "group_size"),
            ({"max_steps": 0}, "max_steps"),
            ({"seed": -1}, "seed"),
            ({"lora_r": 0}, "lora_r"),
            ({"fp16": True, "bf16": True}, "fp16"),
            ({"temperature": 0.0}, "temperature"),
            ({"temperature": 3.0}, "temperature"),
        ],
    )
    def test_invalid_knobs_rejected(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            GRPOConfig(**kwargs)
