"""GRPO training entrypoint wiring for SQLDrift.

This module hosts the building blocks for training a Qwen3-class
model (default: ``Qwen/Qwen3-4B-Instruct-2507``) against the SQLDrift
OpenEnv environment with :class:`trl.GRPOTrainer`:

* :func:`iter_curriculum` — pure, dependency-free scenario sampler
  used both by :func:`build_dataset` and by unit tests.
* :func:`build_dataset` — turns the curriculum iterator into a
  Hugging Face :class:`datasets.Dataset` whose rows carry the per-
  episode seed / scenario id that :class:`.tool_env.SqlDriftToolEnv`
  consumes in its ``reset(**kwargs)``.
* :func:`load_model_and_tokenizer` — lazy Unsloth loader that keeps
  the CUDA dependency tree out of the module top-level so CPU-only
  CI can still import this file.
* :func:`build_env_client` — sanctioned way to obtain the OpenEnv
  client bound to a running SQLDrift server (used by the eval
  harness and notebooks — the trainer builds its own clients via
  ``environment_factory``).
* :func:`reward_from_environments` — a TRL-compatible reward
  function that reads the cumulative trajectory return off each
  rolled-out :class:`SqlDriftToolEnv` instance.
* :func:`train` — the real entrypoint: builds a curriculum dataset,
  loads the model, instantiates :class:`trl.GRPOTrainer` with
  an ``environment_factory`` bound to :class:`SqlDriftToolEnv` and the
  caller-supplied env URL (the TRL-sanctioned multi-turn OpenEnv
  rollout path, see the TRL OpenEnv integration guide) and runs
  ``trainer.train()``. All heavy imports are lazy so CPU-only CI can
  still import the module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from random import Random
from typing import Any

from training.config import GRPOConfig
from training.prompt import render_system_prompt
from training.seeding import set_seed
from training.tool_env import SqlDriftToolEnv
from utilities.logger import get_module_logger

_LOG = get_module_logger(__name__)


def coerce_trainable_from_bfloat16_to_float32(model: Any) -> int:
    """Re-type trainable parameters off ``bfloat16`` when using legacy fp16 AMP.

    With ``GRPOConfig(fp16=True, bf16=False)``, Hugging Face :class:`Trainer`
    uses a :class:`torch.amp.GradScaler`.  PEFT/TRL may still build new LoRA
    matrices in **bfloat16** (checkpoint / config default).  The scaler’s
    CUDA foreach helpers then fail on T4 and similar with::

        NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda
        not implemented for 'BFloat16'

    Casting any *trainable* ``bfloat16`` parameter to ``float32`` matches the
    usual “fp32 master weight” path and avoids the bug.

    Returns the number of parameters updated.
    """
    import torch

    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float32)
            n += 1
    if n:
        _LOG.info(
            "Coerced %d trainable bfloat16 parameter tensors to float32 (fp16 AMP / GradScaler).", n
        )
    return n


def iter_curriculum(config: GRPOConfig, *, seed: int = 0) -> Iterator[tuple[str, int]]:
    """Yield an infinite stream of ``(scenario_id, episode_seed)`` tuples."""
    rng = Random(seed)
    curr = config.curriculum
    lo, hi = curr.seed_range
    i = 0
    while True:
        if curr.mode == "uniform":
            scenario = rng.choice(curr.scenarios)
        elif curr.mode == "weighted":
            scenario = rng.choices(curr.scenarios, weights=list(curr.weights or ()), k=1)[0]
        else:
            scenario = curr.scenarios[i % len(curr.scenarios)]
        yield scenario, rng.randint(lo, hi - 1)
        i += 1


def build_dataset(config: GRPOConfig, *, num_rows: int, seed: int = 0) -> Any:
    """Build a :class:`datasets.Dataset` of prompt rows for GRPO.

    Every row pre-computes the system prompt for a single ``(scenario,
    seed)`` pair so the trainer sees a normal chat-format ``prompt``
    column. The extra columns (``scenario_id``, ``seed``,
    ``budget_steps``, ``enable_dba_oracle``) ride along verbatim and are forwarded by TRL as
    ``**kwargs`` to :meth:`SqlDriftToolEnv.reset`, which is how we
    reproducibly pin each episode to its curriculum slot.

    This function imports :mod:`datasets` lazily to keep the stdlib-
    only import surface for CPU-only CI.
    """
    if num_rows < 1:
        raise ValueError("build_dataset requires num_rows >= 1")

    from datasets import Dataset

    it = iter_curriculum(config, seed=seed)
    prompts: list[list[dict[str, str]]] = []
    scenario_ids: list[str] = []
    seeds: list[int] = []
    budgets: list[int] = []
    oracle_flags: list[bool] = []

    for _ in range(num_rows):
        scenario_id, episode_seed = next(it)
        system = render_system_prompt(
            scenario_id=scenario_id, dba_enabled=config.dba_oracle_enabled
        )
        prompts.append(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Diagnose, adapt to any drift, and submit a correct "
                        "rewrite of the baseline query using the tools "
                        "provided. Call submit_rewrite when confident."
                    ),
                },
            ]
        )
        scenario_ids.append(scenario_id)
        seeds.append(episode_seed)
        budgets.append(config.episode_step_budget)
        oracle_flags.append(config.dba_oracle_enabled)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "scenario_id": scenario_ids,
            "seed": seeds,
            "budget_steps": budgets,
            "enable_dba_oracle": oracle_flags,
        }
    )


def load_model_and_tokenizer(config: GRPOConfig) -> tuple[Any, Any]:
    """Load the base model with QLoRA-friendly quantization + tokenizer.

    Mirrors the stack used by Hugging Face TRL's own reference notebooks
    (``grpo_trl_lora_qlora.ipynb`` and the OpenEnv examples): plain
    :class:`transformers.AutoModelForCausalLM` + :class:`BitsAndBytesConfig`
    nf4 4-bit quantization. PEFT/LoRA is layered ON the model by the
    GRPOTrainer itself when ``peft_config`` is passed to its constructor —
    we do not call ``get_peft_model`` here.

    Returning ``(model, tokenizer)`` keeps the function signature stable
    for callers (notebooks, eval harness, etc.) that prepared LoRA
    separately.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if config.fp16 else torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        attn_implementation="sdpa",
        dtype="float32" if quantization_config is not None else "auto",
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, tokenizer


def build_peft_config(config: GRPOConfig) -> Any:
    """Build the :class:`peft.LoraConfig` matching ``GRPOConfig`` knobs.

    Returned separately so callers can pass it to
    ``GRPOTrainer(peft_config=...)`` — the TRL-recommended path for
    LoRA/QLoRA training.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_env_client(config: GRPOConfig) -> Any:
    """Instantiate the OpenEnv client bound to the running SQLDrift server.

    The GRPO trainer itself does not use this helper — it owns its own
    client lifecycle via ``environment_factory=SqlDriftToolEnv``. The
    helper exists for the eval harness and notebooks that still want a
    one-shot client.
    """
    from client import SqlDriftEnv

    return SqlDriftEnv(base_url=config.env_base_url).sync()


def reward_from_environments(
    environments: Sequence[SqlDriftToolEnv],
    **_: Any,
) -> list[float]:
    """TRL-compatible reward function.

    SQLDrift's reward shaping is trajectory-based: step tax, tool-error
    penalties, repeat-failing-query penalties, and DBA consult penalties
    accrue *before* the final submit step. GRPO therefore needs the
    running :attr:`SqlDriftToolEnv.episode_return`, not just the last
    step's reward, or it would silently discard most of the shaping
    signal during training.

    A small, bounded *diversity* term derived from
    :attr:`SqlDriftToolEnv.components` (last per-rubric partial scores) is
    added on top. GRPO advantages are group-relative: when every rollout
    in a group plausibly receives similar cumulative returns, within-group
    variance collapses. Spreading the rubric components (without changing
    the true terminal outcome in ``episode_return``) improves signal
    diversity for the policy gradient.
    """
    out: list[float] = []
    for env in environments:
        base = float(env.episode_return)
        comps = getattr(env, "components", None) or {}
        if isinstance(comps, dict) and len(comps) > 0:
            vals = [float(v) for v in comps.values()]
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        else:
            spread = 0.0
        diversity = 0.04 * min(2.0, max(0.0, spread))
        out.append(base + diversity)
    return out


def _build_flush_log_history_callback(out_path: Path) -> Any:
    """Construct a TrainerCallback that appends each log dict to JSONL.

    Why: `trainer.state.log_history` is in-memory only; a crash mid-run
    wipes the curves needed by `utilities/plot_curves.py`. This callback
    flushes per `on_log` so a step-N crash still leaves N records on
    disk for the post-mortem plot.

    The callback class is constructed lazily inside `train()` because
    `transformers.TrainerCallback` is a [train]-extra import.
    """
    from transformers import TrainerCallback

    class _FlushLogHistory(TrainerCallback):
        def __init__(self, target: Path) -> None:
            self._target = target
            self._target.parent.mkdir(parents=True, exist_ok=True)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs:
                return
            record = {"step": state.global_step, **logs}
            with self._target.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    return _FlushLogHistory(out_path)


def train(config: GRPOConfig) -> Any:
    """Run the full GRPO training loop against SQLDrift.

    All heavy imports (``trl``, ``datasets``, ``transformers``,
    ``peft``, ``bitsandbytes``) are deferred into this function body so
    a CPU-only checkout that only imports :mod:`training.grpo_train`
    for, say, :func:`iter_curriculum` never triggers them.

    Stack: :class:`transformers.AutoModelForCausalLM` (+ optional
    ``BitsAndBytesConfig`` 4-bit nf4) → :class:`trl.GRPOTrainer` with
    ``environment_factory=SqlDriftToolEnv`` and ``peft_config`` for
    LoRA. Mirrors Hugging Face TRL's official reference notebooks
    (``grpo_trl_lora_qlora.ipynb``, ``openenv_wordle_grpo.ipynb``).
    """
    set_seed(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grpo_config.json").write_text(
        json.dumps(asdict(config), default=str, indent=2),
    )

    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    dataset = build_dataset(
        config,
        num_rows=max(config.max_steps * config.group_size, config.group_size),
        seed=config.seed,
    )

    # TRL >=0.25 removed `max_prompt_length` (prompt length is now
    # dataset-driven). `max_completion_length` is the TOTAL token budget
    # across the entire multi-turn conversation, not a per-turn cap —
    # see the OpenEnv integration doc:
    # https://huggingface.co/docs/trl/openenv#max_completion_length-in-multi-turn-episodes
    # Do NOT re-add `max_prompt_length` here; the kwarg no longer exists
    # and its presence raises TypeError on construction.
    # Gemma-4 has no <think> wrapper to suppress, so we don't pass
    # chat_template_kwargs here. per_device_train_batch_size=1 +
    # num_generations=group_size mirrors the Gemma-4 sudoku notebook
    # (TRL requires the per-device batch * grad_accum to be divisible
    # by num_generations; 1*2 % 2 == 0 satisfies that for group_size=2).
    trl_args = TRLGRPOConfig(
        output_dir=str(out),
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        per_device_train_batch_size=1,
        num_generations=config.group_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_completion_length=config.max_completion_length,
        max_tool_calling_iterations=config.max_tool_calling_iterations,
        temperature=config.temperature,
        top_p=config.top_p,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        log_completions=True,
        report_to=["tensorboard"],
        logging_dir=str(out / "tb"),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_peft_config(config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_from_environments,
        args=trl_args,
        environment_factory=partial(SqlDriftToolEnv, env_url=config.env_base_url),
        peft_config=peft_config,
    )
    if config.fp16 and not config.bf16:
        coerce_trainable_from_bfloat16_to_float32(trainer.model)
    trainer.add_callback(_build_flush_log_history_callback(out / "log_history.jsonl"))

    _LOG.info(
        "Starting GRPO: scenarios=%d, max_steps=%d, group_size=%d, env_url=%s",
        len(config.curriculum.scenarios),
        config.max_steps,
        config.group_size,
        config.env_base_url,
    )
    trainer.train()
    trainer.save_model(str(out))
    return trainer


def _parse_args(argv: list[str] | None = None) -> GRPOConfig:
    import argparse

    ap = argparse.ArgumentParser(description="GRPO training for SQLDrift.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON manifest overriding GRPOConfig defaults.",
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--env-base-url", type=str, default=None)
    ns = ap.parse_args(argv)

    overrides: dict[str, Any] = {}
    if ns.config is not None:
        overrides.update(json.loads(ns.config.read_text()))
    if ns.max_steps is not None:
        overrides["max_steps"] = ns.max_steps
    if ns.output_dir is not None:
        overrides["output_dir"] = ns.output_dir
    if ns.env_base_url is not None:
        overrides["env_base_url"] = ns.env_base_url
    from training.config import CurriculumConfig

    if "curriculum" in overrides and isinstance(overrides["curriculum"], dict):
        overrides["curriculum"] = CurriculumConfig(**overrides["curriculum"])
    return GRPOConfig(**overrides)


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()


__all__ = [
    "build_dataset",
    "build_env_client",
    "build_peft_config",
    "coerce_trainable_from_bfloat16_to_float32",
    "iter_curriculum",
    "load_model_and_tokenizer",
    "reward_from_environments",
    "train",
]
