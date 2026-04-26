"""Run SQLDrift GRPO on Hugging Face Jobs (T4 medium) with Gemma-4 E2B.

The install + runtime stack is taken **verbatim** from Unsloth's
official Gemma-4 GRPO notebook
(``gemma4_(e2b)_reinforcement_learning_sudoku_game.py`` at the repo root,
which is the source of truth for "what actually resolves cleanly in
2026-04 against the latest Unsloth/TRL/Transformers/Triton lines").
The notebook works on a free Colab T4; this script mirrors it inside a
CUDA 12.6 docker container on HF Jobs so we can train without a Colab
session.

What this gives you over the previous (Qwen3 / xformers-pinned) version:

* No more torch=2.7.0 ↔ xformers=0.0.30 ↔ unsloth_zoo pin tug-of-war.
  We follow Unsloth's recipe (``torch>=2.8.0``, ``triton>=3.4.0``,
  ``unsloth[base] @ git``, ``unsloth_zoo[base] @ git``, then a final
  ``--no-deps`` upgrade of ``transformers>=5.5.0`` / ``tokenizers`` /
  ``trl>=0.28.0``). Unsloth's git-extras pull a self-consistent set of
  wheels; the ``--no-deps`` upgrade prevents transformers/TRL from
  dragging anything else along.
* Gemma-4 E2B is multimodal, so we install ``timm`` with ``--no-deps``
  for vision/audio support — exactly as the source-of-truth notebook
  does — even though SQLDrift only feeds text.
* All CUDA-sensitive wheels (torch, torchvision) come from the
  ``cu126`` PyTorch index so they land matching the docker image's
  CUDA minor; a downstream ``--no-deps`` upgrade can't silently swap
  them out.

Two operating modes:

* **HTTP env** (default): set ``SQL_DRIFT_ENV_URL`` to a deployed Space
  URL. The trainer's ``SqlDriftToolEnv`` reaches it over HTTP/WS for
  every tool call.
* **In-process env**: leave ``SQL_DRIFT_ENV_URL`` unset. The inline
  training script overrides ``training.tool_env.set_client_factory`` so
  each rollout instantiates ``SqlDriftEnvironment()`` inside the same
  container — no Space deploy required.

Prerequisites:

1. ``HF_TOKEN`` exported with write scope (job creation + log streaming).
2. ``SQL_DRIFT_REPO_URL`` pointing at a git URL the container can
   ``pip install git+<url>`` against. Either:
     - a public GitHub clone URL, or
     - an HF Space repo URL with the token embedded:
       ``https://USER:HF_TOKEN@huggingface.co/spaces/USER/sql-drift-env``.
3. (Optional) ``SQL_DRIFT_ENV_URL`` for HTTP-env mode.

Run it with::

    python utilities/run_training_job.py
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from textwrap import dedent

from dotenv import load_dotenv
from huggingface_hub import fetch_job_logs, inspect_job, run_job

# ---------------------------------------------------------------------------
# Locked hyperparameters
# ---------------------------------------------------------------------------
# Gemma-4 E2B-it. This is the model name baked into the source-of-truth
# notebook; do not switch back to a Qwen line without revisiting the
# install block (Qwen3 needs a TRL response-schema chat-template hack
# that the Gemma family does not).
MODEL_ID = "unsloth/gemma-4-E2B-it"
MAX_STEPS = 80
GROUP_SIZE = 2  # num_generations per GRPO step (T4 16 GB sweet spot)
SEED = 7
MAX_SEQ_LENGTH = 4096
# Multi-turn TOTAL token budget across the entire conversation
# (assistant generations + tool results combined) — TRL >=0.25 redefined
# `max_completion_length` away from a per-turn cap.
# https://huggingface.co/docs/trl/openenv#max_completion_length-in-multi-turn-episodes
MAX_COMPLETION_LENGTH = 2048
SAVE_STEPS = 20
LEARNING_RATE = 5e-6
WARMUP_STEPS = 8
HF_JOB_FLAVOR = "t4-medium"  # 1× T4 16 GB; matches the Colab tier the recipe targets
TIMEOUT = "20h"  # hard cap; billing stops if a run hangs past wall time

# CUDA 12.6 + cuDNN 9 on Ubuntu 24.04 (Python 3.12). Torch 2.8 ships a
# cu126 wheel; keep the docker minor and the wheel tag in lockstep.
DOCKER_IMAGE = "nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04"
TORCH_CUDA_TAG = "cu126"

# Triton kernels commit pinned by the Unsloth Gemma-4 notebook. Keep
# this in sync with that notebook — it's part of the recipe, not a knob.
TRITON_KERNELS_REF = "0add68262ab0a2e33b84524346cb27cbb2787356"

load_dotenv()


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set in the environment. See the docstring of {__file__}.")
    return value


def _build_train_script(env_url: str | None) -> str:
    """Render the in-container training script (stdin to ``python3``).

    Pure stdlib + 3.12 syntax. Reads no CLI args; everything is baked in
    at build time so the container has nothing to negotiate at runtime.
    """
    if env_url:
        in_process_setup = ""
    else:
        # In-process mode: override SqlDriftToolEnv's HTTP client with a
        # zero-overhead in-process SqlDriftEnvironment instance per
        # rollout. Avoids the Space deploy + thousands of HTTP round-trips
        # per training run.
        in_process_setup = (
            "from server import SqlDriftEnvironment\n"
            "from training.tool_env import set_client_factory\n"
            "set_client_factory(lambda: SqlDriftEnvironment())\n"
            "print('[setup] in-process env mode enabled (no SQL_DRIFT_ENV_URL)', flush=True)\n"
        )

    # The wheel layout is site-packages/sql_drift_env/<sibling>; the
    # codebase uses flat imports (``from training...``, ``from server...``).
    # Putting the package directory on sys.path mirrors _cli.py and the
    # PYTHONPATH=/app/env pattern from the Docker entrypoint.
    parts: list[str] = [
        "import sys",
        "import sql_drift_env",
        "",
        "flat = sql_drift_env.__path__[0]",
        "if flat not in sys.path:",
        "    sys.path.insert(0, flat)",
        "",
        "# Sanity-check CUDA before doing anything else.",
        "import torch",
        "if not torch.cuda.is_available():",
        "    raise SystemExit(",
        '        "torch.cuda.is_available() is False. The job is running on CPU; "',
        '        "aborting before wasting GPU minutes."',
        "    )",
        "print(",
        '    f"[setup] torch={torch.__version__} cuda={torch.version.cuda} "',
        '    f"device={torch.cuda.get_device_name(0)}",',
        "    flush=True,",
        ")",
        "",
        in_process_setup,
        "from training.config import ALL_SCENARIOS, CurriculumConfig, GRPOConfig",
        "from training.grpo_train import train",
        "",
        "cfg = GRPOConfig(",
        f"    model_name={MODEL_ID!r},",
        f"    env_base_url={(env_url or 'http://localhost:8000')!r},",
        '    output_dir="/outputs/grpo_run",',
        f"    max_steps={MAX_STEPS},",
        f"    group_size={GROUP_SIZE},",
        f"    learning_rate={LEARNING_RATE},",
        f"    warmup_steps={WARMUP_STEPS},",
        f"    save_steps={SAVE_STEPS},",
        "    logging_steps=1,",
        f"    max_seq_length={MAX_SEQ_LENGTH},",
        f"    max_completion_length={MAX_COMPLETION_LENGTH},",
        f"    seed={SEED},",
        "    fp16=True,",
        "    bf16=False,",
        "    curriculum=CurriculumConfig(",
        "        scenarios=ALL_SCENARIOS,",
        '        mode="weighted",',
        "        weights=(1, 1, 1, 1, 1, 1, 2, 2, 2, 2),",
        "    ),",
        ")",
        'print("[setup] starting train(cfg)", flush=True)',
        "trainer = train(cfg)",
        'print("DONE", flush=True)',
    ]
    return "\n".join(parts)


def _build_preflight_script() -> str:
    """Post-install preflight: catch dep mismatches in the first ~30s.

    Asserts that:

    * torch is the cu126 wheel we asked for (no downstream package
      sneaked a different CUDA tag in)
    * ``torch.cuda.is_available()`` (the image actually has the CUDA stack)
    * ``bitsandbytes`` imports cleanly (its native lib's ``dlopen`` of
      ``libnvJitLink`` succeeds)
    * ``unsloth`` and ``unsloth.FastVisionModel`` import cleanly (the
      Gemma-4 multimodal loader is what we use in
      ``training/grpo_train.py``)
    """
    return dedent(
        f"""
        import torch

        assert torch.cuda.is_available(), (
            f"torch.cuda.is_available() is False (torch={{torch.__version__}}); "
            "image lacks CUDA libs or the wheel is CPU-only."
        )
        assert "+{TORCH_CUDA_TAG}" in torch.__version__, (
            f"torch wheel is {{torch.__version__}} but expected +{TORCH_CUDA_TAG}; "
            "a downstream install bumped torch despite the pip constraint."
        )

        import bitsandbytes  # noqa: F401  triggers native dlopen
        import unsloth  # noqa: F401  triggers triton + unsloth_zoo imports
        from unsloth import FastVisionModel  # noqa: F401  Gemma-4 loader path

        print(
            f"[setup] preflight OK torch={{torch.__version__}} "
            f"cuda={{torch.version.cuda}} device={{torch.cuda.get_device_name(0)}}",
            flush=True,
        )
        """
    ).strip()


def _build_command(repo_url: str, env_url: str | None) -> list[str]:
    """Build the ``["bash", "-c", <script>]`` command for ``run_job``.

    The bash script mirrors Unsloth's official Gemma-4 GRPO install:

    1. apt install python3-pip, python3-venv, python3-dev,
       build-essential, git (build-essential + python3-dev are needed
       at runtime for triton's first-use JIT compile of cuda_utils.so).
    2. Create a venv (avoids PEP 668 externally-managed-environment
       lock on Ubuntu 24.04) and bootstrap pip / wheel / setuptools.
    3. Install torch + torchvision from the ``cu126`` index, version
       capture them into a constraints file, and force every subsequent
       install to honor that pin. This stops a downstream ``--no-deps``
       upgrade from silently swapping CUDA minors under us.
    4. Install ``triton``, ``numpy``, ``pillow``, ``bitsandbytes``, then
       ``unsloth_zoo[base] @ git`` and ``unsloth[base] @ git`` (the
       ``[base]`` extras pull the rest of the Unsloth runtime —
       xformers, etc. — pinned to versions that resolve against torch
       2.8). Then the triton-kernels git pin from the recipe.
    5. ``pip install --upgrade --no-deps`` ``transformers>=5.5.0``,
       ``tokenizers``, ``trl>=0.28.0``, ``unsloth``, ``unsloth_zoo``.
       The ``--no-deps`` is load-bearing: without it the upgrade would
       drag transformers/trl deps that can downgrade torch.
    6. ``pip install --no-deps timm`` (Gemma-4 vision/audio).
    7. Install the SQLDrift repo and its trainer-side companions
       (``datasets``, ``accelerate``, ``peft``, ``jmespath``,
       ``tensorboard``) under the torch constraint.
    8. Run a preflight Python script asserting torch/cuda/bitsandbytes/
       unsloth all import cleanly, then run the training script.
    """
    train_py = _build_train_script(env_url)
    preflight_py = _build_preflight_script()

    bash = dedent(
        f"""
        set -euo pipefail
        echo "[setup] ACCELERATOR=${{ACCELERATOR:-unknown}} CPU=${{CPU_CORES:-?}} MEMORY=${{MEMORY:-?}}"

        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends \\
            python3-pip python3-venv python3-dev \\
            build-essential \\
            git ca-certificates

        python3 -m venv /opt/sqldrift
        # shellcheck disable=SC1091
        . /opt/sqldrift/bin/activate
        python -m pip install --no-cache-dir --upgrade pip wheel setuptools

        # cu126 wheels publish as `2.8.0+cu126` (PEP 440 local segment),
        # which sorts higher than vanilla PyPI 2.8.0 — pip prefers the
        # cu126 wheel automatically when this index is on the path.
        export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/{TORCH_CUDA_TAG}

        # ---------------------------------------------------------------
        # Step 1 (Gemma-4 recipe line 1): torch + triton + numpy/pillow
        # + torchvision + bitsandbytes, pulled with the cu126 index.
        # We don't pin torch to a specific 2.8 patch — the recipe just
        # asks for ">=2.8.0" and we trust unsloth-zoo[base]'s metadata
        # to keep things self-consistent.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir \\
            --index-url https://download.pytorch.org/whl/{TORCH_CUDA_TAG} \\
            --extra-index-url https://pypi.org/simple \\
            "torch>=2.8.0" \\
            torchvision \\
            "triton>=3.4.0" \\
            numpy pillow bitsandbytes

        # Capture the EXACT installed torch/torchvision versions (incl.
        # the +cu126 local segment) into a pip constraints file so every
        # subsequent install is forced to honor the pin. unsloth's
        # resolver, --no-deps upgrades, and the [train]-extra metadata
        # cannot resolve a different wheel without violating the
        # constraint — pip will error rather than silently swap.
        TORCH_PINNED="$(python -c 'import torch; print(torch.__version__)')"
        TORCHVISION_PINNED="$(python -c 'import torchvision; print(torchvision.__version__)')"
        echo "[setup] pinned torch=${{TORCH_PINNED}} torchvision=${{TORCHVISION_PINNED}}"
        case "${{TORCH_PINNED}}" in
          *+{TORCH_CUDA_TAG}) ;;
          *)
            echo "[setup] FATAL: torch wheel is ${{TORCH_PINNED}} but expected +{TORCH_CUDA_TAG}." >&2
            exit 1
            ;;
        esac
        {{
            printf 'torch==%s\\n' "${{TORCH_PINNED}}"
            printf 'torchvision==%s\\n' "${{TORCHVISION_PINNED}}"
        }} > /tmp/constraints.txt

        # ---------------------------------------------------------------
        # Step 2 (Gemma-4 recipe line 1, cont.): unsloth + unsloth-zoo
        # from git with [base] extras. The [base] extras pull the rest
        # of the Unsloth runtime (xformers etc.) at versions that match
        # torch 2.8 — letting Unsloth own this resolution avoids the
        # torch/xformers tug-of-war we hit on the prior pin.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir \\
            --constraint /tmp/constraints.txt \\
            "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
            "unsloth[base] @ git+https://github.com/unslothai/unsloth"

        # ---------------------------------------------------------------
        # Step 3 (Gemma-4 recipe line 1, end): triton-kernels at the
        # commit the recipe pins.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir \\
            --constraint /tmp/constraints.txt \\
            "git+https://github.com/triton-lang/triton.git@{TRITON_KERNELS_REF}#subdirectory=python/triton_kernels"

        # ---------------------------------------------------------------
        # Step 4 (Gemma-4 recipe line 2): upgrade transformers, tokenizers,
        # trl, unsloth, unsloth_zoo with --no-deps. The --no-deps is
        # load-bearing: without it the resolver would drag in deps that
        # can knock torch off the cu126 pin.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir --upgrade --no-deps \\
            "transformers>=5.5.0" \\
            tokenizers \\
            "trl>=0.28.0" \\
            unsloth \\
            unsloth_zoo

        # ---------------------------------------------------------------
        # Step 5 (Gemma-4 recipe line 3): timm with --no-deps. Gemma-4 is
        # multimodal; FastVisionModel imports timm even for text-only
        # workloads.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir --no-deps --upgrade timm

        # ---------------------------------------------------------------
        # Step 6: trainer-side helpers. Installed under the constraint so
        # transformers/torch versions stay where steps 1-5 left them.
        # jmespath is required by TRL's GRPOTrainer when `tools` /
        # `environment_factory` is set; tensorboard is the report_to
        # backend our run uses.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir \\
            --constraint /tmp/constraints.txt \\
            "datasets>=2.20.0,<4.0" \\
            "accelerate>=1.13.0,<2.0" \\
            "peft>=0.19,<1.0" \\
            "jmespath>=1.0,<2.0" \\
            "tensorboard>=2.20,<3.0"

        # ---------------------------------------------------------------
        # Step 7: install the SQLDrift repo. The [train] extra is
        # intentionally omitted: it's a CPU-side floor for static checks,
        # while the real trainer stack is everything we just installed.
        # ---------------------------------------------------------------
        python -m pip install --no-cache-dir --no-deps \\
            "sql-drift-env @ git+{repo_url}@main"
        # Pull SQLDrift's runtime (non-train) deps explicitly so --no-deps
        # above doesn't strand us without duckdb/openenv-core/etc.
        python -m pip install --no-cache-dir \\
            --constraint /tmp/constraints.txt \\
            "duckdb>=1.5.2,<2.0" \\
            "huggingface-hub>=1.5.0,<2.0" \\
            "openenv-core[core]>=0.2.2,<0.4" \\
            "sqlglot>=30.6.0,<40.0" \\
            "pydantic>=2.8.0,<3.0" \\
            "python-dotenv>=1.2.2,<2.0" \\
            "openai>=2.32.0,<3.0"

        # ---------------------------------------------------------------
        # Step 8: preflight — fail fast on a CUDA mismatch BEFORE we
        # touch the GPU. Heredoc terminator MUST be at column 0.
        # ---------------------------------------------------------------
        cat > /tmp/preflight.py <<'PYCHECK'
{preflight_py}
PYCHECK
        python /tmp/preflight.py

        # ---------------------------------------------------------------
        # Step 9: run training under the venv's Python.
        # ---------------------------------------------------------------
        cat > /tmp/run.py <<'PYEOF'
{train_py}
PYEOF
        python /tmp/run.py
        """
    ).strip()
    return ["bash", "-c", bash]


def main() -> None:
    repo_url = _required_env("SQL_DRIFT_REPO_URL")
    hf_token = _required_env("HF_TOKEN")
    env_url = os.environ.get("SQL_DRIFT_ENV_URL", "").strip() or None

    secrets = {"HF_TOKEN": hf_token}
    if env_url:
        secrets["SQL_DRIFT_ENV_URL"] = env_url

    command = _build_command(repo_url, env_url)
    job = run_job(
        image=DOCKER_IMAGE,
        command=command,
        flavor=HF_JOB_FLAVOR,
        timeout=TIMEOUT,
        secrets=secrets,
        env={
            "WANDB_DISABLED": "true",
            "TRL_EXPERIMENTAL_SILENCE": "1",
            "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
        },
    )

    print(f"Job URL : {job.url}", flush=True)
    print(f"Job ID  : {job.id}", flush=True)
    print(
        "Mode    : " + (f"HTTP env ({env_url})" if env_url else "in-process env (no Space)"),
        flush=True,
    )
    print(f"Image   : {DOCKER_IMAGE}", flush=True)
    print(f"Flavor  : {HF_JOB_FLAVOR}  (timeout={TIMEOUT})", flush=True)
    print(f"Model   : {MODEL_ID}", flush=True)

    # Exponential backoff while the job is queued/scheduling so we don't
    # rate-limit the HF API. Sleep grows 5s -> 10s -> 20s -> 40s -> 60s.
    _PRE_RUN_STAGES = {"SCHEDULING", "QUEUED", "PENDING", "INITIALIZING"}
    print("--- waiting for job to start ---", flush=True)
    sleep_times = (5, 10, 20, 40, 60)
    sleeper = itertools.chain(sleep_times, itertools.repeat(sleep_times[-1]))
    while True:
        info = inspect_job(job_id=job.id)
        stage = getattr(info.status, "stage", "?")
        if stage not in _PRE_RUN_STAGES:
            print(f"--- job stage={stage}, opening log stream ---", flush=True)
            break
        print(f"  stage={stage} (polling)", flush=True)
        time.sleep(next(sleeper))

    try:
        for line in fetch_job_logs(job_id=job.id):
            print(line, flush=True)
    except KeyboardInterrupt:
        print(
            "\nlocal stream interrupted; the job continues. "
            f"Cancel billing with: hf jobs cancel {job.id}",
            flush=True,
        )
        return

    info = inspect_job(job_id=job.id)
    stage = getattr(info.status, "stage", "?")
    msg = getattr(info.status, "message", None)
    print(f"--- job stage={stage} message={msg} ---", flush=True)
    if stage == "ERROR":
        sys.exit(1)


if __name__ == "__main__":
    main()
