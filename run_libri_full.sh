#!/usr/bin/env bash
set -euo pipefail
#ROOT=/project/anhlt/0607/research/work/ghep_code/Mamba-TasNet
ROOT=/home/lab/letuan/code/speechseparation/code

cd "$ROOT"
# export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1,2,7 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export TMPDIR="$ROOT"
export XDG_CACHE_HOME="$ROOT/.runtime/cache" TRITON_CACHE_DIR="$ROOT/.runtime/triton" CUDA_CACHE_PATH="$ROOT/.runtime/cuda"
export HF_HOME="$ROOT/.runtime/huggingface" TORCH_HOME="$ROOT/.runtime/torch" TORCH_EXTENSIONS_DIR="$ROOT/.runtime/extensions"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib" NUMBA_CACHE_DIR="$ROOT/.runtime/numba"
export WANDB_DIR="$ROOT/.runtime/wandb" WANDB_CACHE_DIR="$ROOT/.runtime/wandb-cache" WANDB_CONFIG_DIR="$ROOT/.runtime/wandb-config" WANDB_DATA_DIR="$ROOT/.runtime/wandb-data" WANDB_MODE=disabled
printf '[LAUNCH] GPU=1,2,7; limit_training_signal_len=False; logging=disabled\n'
/home/lab/letuan/code/speechseparation/code/.venv/bin/python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=3 \
    train_wsj0mix.py hparams/LibriMix/dpmamba_S_jepa.yaml \
    --distributed_backend nccl --device cuda:1 \
    --data_folder /home/lab/letuan/code/speechseparation/data/LibriMix/storage/Libri2Mix \
    --output_folder "$ROOT/results/LibriMix/dpmamba_S_jepa/1234" \
    --precision fp16 --limit_training_signal_len False \
    --librimix_validate_audio False --skip_prep False
