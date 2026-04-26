"""Training configuration dataclasses.

Holds every knob the :mod:`training.grpo_train` script or the eval CLI
needs, as plain, frozen dataclasses so they serialize cleanly to JSON
for experiment manifests.

Deliberately lightweight: do not import ``trl`` / ``unsloth`` /
``transformers`` at module import time. Those libraries are CUDA-heavy
and optional. ``grpo_train.py`` resolves them lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from utilities.env_loader import env_str


def _load_all_scenarios() -> tuple[str, ...]:
    from scenarios import iter_specs

    return tuple(spec.scenario_id for spec in iter_specs())


# Derived from the live registry so training defaults stay in sync with
# the scenarios actually shipped under ``scenarios/``.
ALL_SCENARIOS: tuple[str, ...] = _load_all_scenarios()


@dataclass(frozen=True)
class CurriculumConfig:
    """Scenario sampling policy for GRPO rollouts.

    ``mode="uniform"`` samples each id in :attr:`scenarios` with equal
    probability.  ``mode="weighted"`` uses :attr:`weights` (must be the
    same length as :attr:`scenarios`) — useful for over-sampling drift
    scenarios early in training.  ``mode="static_order"`` iterates the
    list round-robin (handy for reproducing eval-style runs).
    """

    scenarios: tuple[str, ...] = ALL_SCENARIOS
    mode: Literal["uniform", "weighted", "static_order"] = "uniform"
    weights: tuple[float, ...] | None = None
    seed_range: tuple[int, int] = (0, 2**31 - 1)

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("CurriculumConfig.scenarios must be non-empty")
        if self.mode == "weighted":
            if self.weights is None or len(self.weights) != len(self.scenarios):
                raise ValueError("mode='weighted' requires weights of the same length as scenarios")
            if any(w < 0 for w in self.weights):
                raise ValueError("weights must all be >= 0")
            if sum(self.weights) <= 0:
                raise ValueError("at least one weight must be > 0")
        lo, hi = self.seed_range
        if lo < 0 or hi <= lo:
            raise ValueError("seed_range must be (lo >= 0, hi > lo)")


@dataclass(frozen=True)
class GRPOConfig:
    """Top-level training config for the GRPO skeleton.

    Defaults pinned to ``Qwen/Qwen3-4B-Instruct-2507`` (Apache-2.0, July 2025
    release, BFCL-v3 = 61.9 native tool-calling) loaded via
    ``transformers.AutoModelForCausalLM`` + ``BitsAndBytesConfig`` 4-bit nf4
    QLoRA + ``peft.LoraConfig`` — the stack used by Hugging Face TRL's own
    reference notebooks (``grpo_trl_lora_qlora.ipynb``, the OpenEnv
    Wordle/Echo examples). Every knob is override-able from the CLI or a
    JSON manifest.
    """

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 4096
    # 4-bit nf4 quantization (QLoRA) — fits a 4B model on a free Colab T4
    # in ~6 GB peak. Flip to False on a >=24 GB GPU for plain LoRA.
    load_in_4bit: bool = True

    # LoRA r=16 / alpha=32 mirrors the TRL grpo_trl_lora_qlora reference.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # GRPO knobs — sized for a free Colab T4 (16 GB) running multi-turn
    # tool-using rollouts: per_device_train_batch_size=1 in TRL,
    # num_generations=2, gradient_accumulation_steps=2.
    group_size: int = 2
    learning_rate: float = 5e-6
    max_steps: int = 500
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 10
    # NOTE: TRL >=0.25 removed `max_prompt_length` from GRPOConfig — prompt
    # length is dataset-driven, not trainer-configurable. And
    # `max_completion_length` is now the TOTAL token budget across the
    # entire multi-turn conversation (all assistant generations + tool
    # results combined), not a per-turn cap. Size it for the worst-case
    # episode (25-step budget * ~80 tokens per turn ≈ 2k).
    max_completion_length: int = 2048
    # TRL's tool loop default is `sys.maxsize` when unset — a chatty model can
    # spend *hours* in one "step". Cap turns to a bit above the env step budget
    # so each rollout always terminates in bounded time.
    max_tool_calling_iterations: int = 32
    # Qwen3-Instruct-2507 recommends temperature=0.7 / top_p=0.8 for
    # non-thinking instruct mode. The 2507 line is non-thinking by default,
    # so no `enable_thinking` toggle is needed.
    temperature: float = 0.7
    top_p: float = 0.8
    seed: int = 0
    # TRL defaults bf16 to True when fp16 is not explicitly set, but Colab
    # T4 supports fp16 only (no bf16 hardware). Keep fp16=True for T4.
    fp16: bool = True
    bf16: bool = False

    # Env wiring
    env_base_url: str = env_str("SQL_DRIFT_ENV_URL", "http://localhost:8000")
    episode_step_budget: int = 25
    dba_oracle_enabled: bool = False

    # IO
    output_dir: str = "outputs/grpo_run"
    logging_steps: int = 1
    save_steps: int = 100

    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("GRPO group_size must be >= 2 (GRPO requires groups).")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.lora_r < 1:
            raise ValueError("lora_r must be >= 1")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 are mutually exclusive")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        if self.max_tool_calling_iterations < 1:
            raise ValueError("max_tool_calling_iterations must be >= 1")


__all__ = ["ALL_SCENARIOS", "CurriculumConfig", "GRPOConfig"]
