"""TRL/Unsloth GRPO training harness — P12/P13.

Public surface:

* :class:`training.config.GRPOConfig` / :class:`training.config.CurriculumConfig`
* :func:`training.prompt.render_system_prompt`
* :class:`training.random_agent.RandomAgent`
* :func:`training.grpo_train.train` (requires GPU + ``[train]`` extra
  plus the CUDA-specific Unsloth stack installed by ``utilities/run_training_job.py``)
"""

from __future__ import annotations

from training.config import ALL_SCENARIOS, CurriculumConfig, GRPOConfig
from training.prompt import (
    render_prompt_from_observation,
    render_system_prompt,
)
from training.random_agent import RandomAgent

__all__ = [
    "ALL_SCENARIOS",
    "CurriculumConfig",
    "GRPOConfig",
    "RandomAgent",
    "render_prompt_from_observation",
    "render_system_prompt",
]
