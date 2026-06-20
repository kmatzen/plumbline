# syntax=docker/dockerfile:1.7
#
# plumbline GPU-runner image.
#
# Build::
#     docker build -t plumbline .
#
# Or, to bake the heavier research repos (MASt3R, VGGT) in at build time::
#     docker build --build-arg WITH_GIT_DEPS=1 -t plumbline:full .
#
# Run a reproduction (mounts your data + cache from the host)::
#     docker run --rm --gpus all \
#         -v $HOME/data:/data \
#         -v $HOME/.cache/plumbline-docker:/cache \
#         -e NYUV2_ROOT=/data/nyuv2 \
#         plumbline reproduce da-v2-small-nyuv2
#
# Notes
# -----
# - Weight downloads land in /cache/huggingface and /cache/torch, so they
#   survive container restarts and don't re-download when you re-run.
# - Datasets are expected under /data (with env-var pointers per dataset).
# - CUDA 12.4 + cuDNN runtime is sized for modern transformers; bump the
#   base tag if a newer torch needs it.

FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG WITH_GIT_DEPS=0

# System deps: python build tooling, curl for uv, libgl* for PIL/cv2 loads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral's Python package manager). Pin to a specific release
# to keep the image reproducible; bump as needed.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Layer 1: dependency metadata — allows Docker to cache `uv sync` when only
# source changes.
COPY pyproject.toml uv.lock README.md LICENSE ./

# Install Python + dependencies into /app/.venv.
# --no-install-project so the plumbline package itself isn't installed yet
# (source hasn't been copied); we'll do that in the next layer.
RUN uv python install ${PYTHON_VERSION} \
    && uv sync --frozen --no-install-project --no-dev

# Layer 2: source. Changing any .py file busts this layer and below, but
# not the big dep layer above.
COPY src ./src
COPY reproductions ./reproductions

# Install the plumbline package itself into the venv.
RUN uv sync --frozen --no-dev

# Layer 3 (optional): fetch research repos not on PyPI. Skipped by default
# to keep the base image slim; pass --build-arg WITH_GIT_DEPS=1 to include.
RUN if [ "$WITH_GIT_DEPS" = "1" ]; then \
        uv pip install \
            "git+https://github.com/facebookresearch/vggt" ; \
    fi
# MASt3R is not pip-installable from its repo; on a runtime box clone it
# separately and add to PYTHONPATH:
#   git clone https://github.com/naver/mast3r /opt/mast3r
#   export PYTHONPATH=/opt/mast3r

# --- Caches + dataset roots -------------------------------------------
# Point all known caches at /cache so a single volume mount preserves
# weights + prediction cache across container runs.
ENV PLUMBLINE_CACHE_DIR=/cache/plumbline \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    TORCH_HOME=/cache/torch

# Dataset roots are overridable per-invocation via -e flags.
ENV NYUV2_ROOT=/data/nyuv2 \
    SINTEL_ROOT=/data/sintel \
    ETH3D_ROOT=/data/eth3d

# Declare mountable volumes. Docker will create tmpfs if you don't mount
# a host directory; prefer -v $HOME/data:/data -v $HOME/.cache/plumbline:/cache.
VOLUME ["/data", "/cache"]

# Entrypoint hands straight off to the plumbline CLI; `docker run plumbline <args>`
# behaves like running `plumbline <args>` locally.
ENTRYPOINT ["uv", "run", "--no-sync", "plumbline"]
CMD ["--help"]
