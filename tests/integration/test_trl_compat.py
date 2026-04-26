"""TRL API compatibility smoke-tests.

These tests assert that the installed TRL version exposes the exact symbols,
constructor signatures, and config fields that ``training/grpo_train.py``
depends on.  They are intentionally lightweight (no GPU, no dataset, no
forward pass) — the goal is to catch a version upgrade that silently breaks
the training entrypoint before submission day.

Run with::

    pytest -m slow tests/integration/test_trl_compat.py

The tests are excluded from the default ``pytest`` run (``addopts = "-m 'not slow'"``
in pyproject.toml) to keep the fast CI gate clean.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.slow

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _import_trl() -> object:
    """Return the trl module, or skip the test if the [train] extra is absent."""
    return pytest.importorskip(
        "trl",
        reason="trl is not installed; add the [train] extra: uv sync --extra train",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GRPOTrainer.__init__ signature
# ──────────────────────────────────────────────────────────────────────────────


def test_grpo_trainer_is_importable() -> None:
    """GRPOTrainer must be a top-level export of trl."""
    trl = _import_trl()
    assert hasattr(trl, "GRPOTrainer"), (
        "trl.GRPOTrainer is missing; the installed trl version may be too old "
        "(>=0.25.0 required) or the API was renamed."
    )


def test_grpo_trainer_accepts_environment_factory() -> None:
    """GRPOTrainer.__init__ must accept the ``environment_factory`` kwarg (TRL ≥0.25)."""
    trl = _import_trl()
    GRPOTrainer = getattr(trl, "GRPOTrainer", None)
    if GRPOTrainer is None:
        pytest.skip("GRPOTrainer not available")
    sig = inspect.signature(GRPOTrainer.__init__)
    assert "environment_factory" in sig.parameters, (
        "GRPOTrainer.__init__ is missing the ``environment_factory=`` parameter. "
        "grpo_train.train() passes ``environment_factory=partial(SqlDriftToolEnv, …)`` "
        "which drives the multi-turn OpenEnv rollout."
    )


def test_grpo_trainer_accepts_reward_funcs() -> None:
    """GRPOTrainer.__init__ must accept the ``reward_funcs`` kwarg."""
    trl = _import_trl()
    GRPOTrainer = getattr(trl, "GRPOTrainer", None)
    if GRPOTrainer is None:
        pytest.skip("GRPOTrainer not available")
    sig = inspect.signature(GRPOTrainer.__init__)
    assert "reward_funcs" in sig.parameters, (
        "GRPOTrainer.__init__ is missing ``reward_funcs=``. "
        "grpo_train.train() passes ``reward_funcs=reward_from_environments``."
    )


# ──────────────────────────────────────────────────────────────────────────────
# GRPOConfig fields used by grpo_train.train()
# ──────────────────────────────────────────────────────────────────────────────

# All kwargs passed to TRLGRPOConfig in training/grpo_train.py:train().
# NOTE: `max_prompt_length` was removed in TRL 0.25 (prompt length is now
# dataset-driven). Do NOT re-add it here; doing so will green-light a
# call site that crashes with TypeError at construction time.
_REQUIRED_GRPO_CONFIG_KWARGS: frozenset[str] = frozenset(
    {
        "output_dir",
        "learning_rate",
        "max_steps",
        "per_device_train_batch_size",
        "num_generations",
        "gradient_accumulation_steps",
        "warmup_steps",
        "max_completion_length",
        "temperature",
        "top_p",
        "logging_steps",
        "save_steps",
        "chat_template_kwargs",
    }
)


# Kwargs that MUST NOT be present — i.e. removed-from-TRL fields the
# trainer used to pass. If TRL ever brings these back, that's a flag to
# re-evaluate the call site, but until then their absence is the
# invariant we want to lock down.
_REMOVED_GRPO_CONFIG_KWARGS: frozenset[str] = frozenset({"max_prompt_length"})


def test_trl_grpoconfig_accepts_expected_fields() -> None:
    """TRL GRPOConfig.__init__ must accept every field used by grpo_train.train()."""
    trl = _import_trl()
    GRPOConfig = getattr(trl, "GRPOConfig", None)
    if GRPOConfig is None:
        pytest.skip("trl.GRPOConfig not available")
    sig = inspect.signature(GRPOConfig.__init__)
    missing = _REQUIRED_GRPO_CONFIG_KWARGS - sig.parameters.keys()
    assert not missing, (
        f"TRL GRPOConfig.__init__ is missing kwargs used by grpo_train.train(): {sorted(missing)}. "
        "Either the installed TRL version renamed these fields or the training/grpo_train.py "
        "call-site must be updated to match the current TRL API."
    )


def test_trl_grpoconfig_does_not_accept_removed_fields() -> None:
    """Lock in the removal of fields TRL has dropped (e.g. max_prompt_length).

    grpo_train.train() used to pass `max_prompt_length`; TRL 0.25 removed
    it. If a future TRL re-introduces the kwarg under the same name with
    different semantics, we want a CI signal to re-evaluate, not a silent
    behavior change.
    """
    trl = _import_trl()
    GRPOConfig = getattr(trl, "GRPOConfig", None)
    if GRPOConfig is None:
        pytest.skip("trl.GRPOConfig not available")
    sig = inspect.signature(GRPOConfig.__init__)
    resurrected = _REMOVED_GRPO_CONFIG_KWARGS & sig.parameters.keys()
    assert not resurrected, (
        f"TRL GRPOConfig has re-introduced kwargs we deliberately dropped: "
        f"{sorted(resurrected)}. Re-evaluate the call site in "
        "training/grpo_train.py before adding them back."
    )


# ──────────────────────────────────────────────────────────────────────────────
# processing_class kwarg (replaced tokenizer= in TRL ≥0.12)
# ──────────────────────────────────────────────────────────────────────────────


def test_grpo_trainer_accepts_processing_class() -> None:
    """GRPOTrainer.__init__ must accept ``processing_class=`` (not the old ``tokenizer=``)."""
    trl = _import_trl()
    GRPOTrainer = getattr(trl, "GRPOTrainer", None)
    if GRPOTrainer is None:
        pytest.skip("GRPOTrainer not available")
    sig = inspect.signature(GRPOTrainer.__init__)
    assert "processing_class" in sig.parameters, (
        "GRPOTrainer.__init__ is missing ``processing_class=``. "
        "grpo_train.train() passes ``processing_class=tokenizer``; the old ``tokenizer=`` "
        "kwarg was removed in TRL ≥0.12."
    )
