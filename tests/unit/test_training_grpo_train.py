"""Focused unit tests for :mod:`training.grpo_train` entrypoint wiring."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import training.grpo_train as grpo_train
from training.config import GRPOConfig


def test_build_dataset_forwards_dba_flag() -> None:
    rows: dict[str, object] = {}

    class _FakeDataset:
        @staticmethod
        def from_dict(payload):
            rows.update(payload)
            return payload

    previous = sys.modules.get("datasets")
    sys.modules["datasets"] = SimpleNamespace(Dataset=_FakeDataset)
    try:
        cfg = GRPOConfig(dba_oracle_enabled=True)
        dataset = grpo_train.build_dataset(cfg, num_rows=3, seed=7)
    finally:
        if previous is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = previous

    assert dataset["enable_dba_oracle"] == [True, True, True]


def test_train_uses_config_seed_for_global_and_curriculum_rngs(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, int]] = []
    trainer_kwargs: dict[str, object] = {}
    trainer_instances: list[object] = []

    monkeypatch.setattr(grpo_train, "set_seed", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(
        grpo_train,
        "build_dataset",
        lambda config, *, num_rows, seed=0: events.append(("dataset", seed)) or ["row"],
    )
    monkeypatch.setattr(grpo_train, "load_model_and_tokenizer", lambda config: ("model", "tok"))
    monkeypatch.setattr(grpo_train, "build_peft_config", lambda config: object())

    class _FakeTRLGRPOConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeTrainer:
        def __init__(self, **kwargs) -> None:
            trainer_kwargs.update(kwargs)
            self.callbacks: list[object] = []
            trainer_instances.append(self)

        def add_callback(self, cb: object) -> None:
            self.callbacks.append(cb)

        def train(self) -> None:
            return None

        def save_model(self, path: str) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(GRPOConfig=_FakeTRLGRPOConfig, GRPOTrainer=_FakeTrainer),
    )

    # `train()` lazy-imports `transformers.TrainerCallback` inside
    # `_build_flush_log_history_callback`; mocking it here keeps the unit
    # test independent of whether the [train] extra is locally installed
    # (mirrors the `trl` mock above; see Task 1 hardening rationale).
    class _FakeTrainerCallback:
        pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrainerCallback=_FakeTrainerCallback),
    )

    cfg = GRPOConfig(output_dir=str(tmp_path), seed=123, max_steps=1, group_size=2)
    grpo_train.train(cfg)

    assert events == [("seed", 123), ("dataset", 123)]
    assert trainer_kwargs["args"].kwargs["fp16"] is True
    assert trainer_kwargs["args"].kwargs["bf16"] is False
    assert trainer_kwargs["args"].kwargs["max_tool_calling_iterations"] == 32
    assert trainer_kwargs["args"].kwargs["dataloader_num_workers"] == 0
    env_factory = trainer_kwargs["environment_factory"]
    assert env_factory.func is grpo_train.SqlDriftToolEnv
    assert env_factory.keywords == {"env_url": cfg.env_base_url}

    # Production code MUST register the JSONL flush callback so a crash
    # mid-run still leaves curves on disk for utilities/plot_curves.py.
    # Use type().__name__ to avoid importing the private class.
    assert len(trainer_instances) == 1
    registered_callback_names = [type(cb).__name__ for cb in trainer_instances[0].callbacks]
    assert registered_callback_names == ["_FlushLogHistory"], (
        f"expected exactly one _FlushLogHistory callback registered on the trainer, "
        f"got {registered_callback_names!r}"
    )


def test_reward_from_environments_uses_episode_return_over_terminal_reward() -> None:
    envs = [
        SimpleNamespace(
            done=True,
            episode_return=1.25,
            terminal_reward=0.8,
            reward=0.8,
            components={},
        ),
        SimpleNamespace(
            done=False, episode_return=-0.12, terminal_reward=0.0, reward=-0.03, components={}
        ),
    ]

    assert grpo_train.reward_from_environments(envs) == [1.25, -0.12]


def test_reward_from_environments_diversity_does_not_dominate() -> None:
    """A modest spread in rubric components nudges the scalar without swamping the return."""
    env = SimpleNamespace(
        episode_return=0.0,
        components={"a": 1.0, "b": -0.5},
    )
    r = grpo_train.reward_from_environments([env])[0]
    # spread=1.5, diversity=0.04*min(2,1.5)=0.06
    assert 0.05 < r < 0.1
