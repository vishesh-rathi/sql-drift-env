# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

# ─── IMAGE PINNING ────────────────────────────────────────────────────────────
# Using :latest is a moving target: a base-image update between local build
# and judging rebuild can pull a different Python / CUDA / OS version.
#
# TO PIN (run once, commit the digest):
#   docker pull ghcr.io/meta-pytorch/openenv-base:latest
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/meta-pytorch/openenv-base:latest
#   # → ghcr.io/meta-pytorch/openenv-base@sha256:<digest>
#
# Then replace the ARG default below with the digest, e.g.:
#   ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base@sha256:<digest>
#
# The sha256 digest is immutable; the image will not change between builds.
# ──────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=sql_drift_env

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies from the frozen lock file.  Failing fast when uv.lock
# is absent prevents a silent re-resolution that could pull different versions
# than the ones tested (see design/codereview.md, dependency pins).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# Final runtime stage — inherits the same pinned/latest image chosen above.
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the uv-managed Python interpreter from the builder so that the
# shebangs baked into ``/app/env/.venv/bin/*`` (e.g. uvicorn) resolve.
# Without this, execve() on those scripts silently falls back to the
# system ``/usr/local/bin/uvicorn`` (Python 3.11) and loads the wrong
# site-packages, yielding ``ModuleNotFoundError: duckdb`` at startup.
COPY --from=builder /root/.local/share/uv /root/.local/share/uv

# Copy the environment code (includes the builder's .venv at /app/env/.venv)
COPY --from=builder /app/env /app/env

# Point PATH + PYTHONPATH at the venv that shipped with the env so
# both ``uvicorn ...`` and ``python -m uvicorn ...`` resolve against
# the right interpreter.
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check (hits the stateless /health endpoint exposed by the
# OpenEnv FastAPI factory).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server via the venv's Python interpreter directly
# (bypassing any PATH-lookup ambiguity with the base image's system
# uvicorn).
CMD ["/app/env/.venv/bin/python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/env"]
