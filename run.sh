#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME=/tmp/huggingface

torchrun --standalone --nproc_per_node=8 reproduce.py

