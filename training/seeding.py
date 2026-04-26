"""Shared seeding helper for training and eval entrypoints."""

from __future__ import annotations

import random
from typing import Any


def set_seed(seed: int) -> None:
    """Seed every RNG surface the training/eval stack relies on."""
    random.seed(seed)

    np_mod: Any | None = None
    try:
        import numpy as _np_mod
    except ImportError:
        pass
    else:
        np_mod = _np_mod
    if np_mod is not None:
        np_mod.random.seed(seed)

    torch_mod: Any | None = None
    try:
        import torch as _torch_mod
    except ImportError:
        pass
    else:
        torch_mod = _torch_mod
    if torch_mod is not None:
        manual_seed = getattr(torch_mod, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        cuda = getattr(torch_mod, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if cuda is not None and callable(is_available) and is_available():
            cuda_manual_seed = getattr(cuda, "manual_seed", None)
            if callable(cuda_manual_seed):
                cuda_manual_seed(seed)
            cuda_manual_seed_all = getattr(cuda, "manual_seed_all", None)
            if callable(cuda_manual_seed_all):
                cuda_manual_seed_all(seed)

    try:
        from transformers import set_seed as transformers_set_seed
    except ImportError:
        transformers_set_seed = None
    if transformers_set_seed is not None:
        transformers_set_seed(seed)


__all__ = ["set_seed"]
