#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lab/letuan/code/speechseparation/code
DATA=/home/lab/letuan/code/speechseparation/data/LibriMix/storage/Libri2Mix
PYTHON="$ROOT/.venv/bin/python"

cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1,2
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

export TMPDIR="$ROOT"
export XDG_CACHE_HOME="$ROOT/.runtime/cache"
export TRITON_CACHE_DIR="$ROOT/.runtime/triton"
export CUDA_CACHE_PATH="$ROOT/.runtime/cuda"

export HF_HOME="$ROOT/.runtime/huggingface"
export TORCH_HOME="$ROOT/.runtime/torch"
export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/extensions"

export MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/.runtime/numba"

export WANDB_DIR="$ROOT/.runtime/wandb"
export WANDB_CACHE_DIR="$ROOT/.runtime/wandb-cache"
export WANDB_CONFIG_DIR="$ROOT/.runtime/wandb-config"
export WANDB_DATA_DIR="$ROOT/.runtime/wandb-data"
export WANDB_MODE=disabled

LOG="$ROOT/train_libri_sep_full_fp16_$(date +%Y%m%d_%H%M%S).log"

printf '[LAUNCH] SepFormer; visible GPUs=1,2,7; precision=fp16; log=%s\n' "$LOG"

"$PYTHON" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train_wsj0mix.py hparams/LibriMix/sepformer.yaml \
    --distributed_backend nccl \
    --device cuda:0 \
    --data_folder "$DATA" \
    --output_folder "$ROOT/results/LibriMix/sepformer/1234" \
    --precision fp32 \
    --limit_training_signal_len False \
    --librimix_validate_audio False \
    --skip_prep False \
    2>&1 | tee "$LOG"