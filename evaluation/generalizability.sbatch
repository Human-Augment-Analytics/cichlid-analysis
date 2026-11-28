#!/bin/bash
# Leave-one-video-out generalizability folds on PACE ICE, one array task per fold.
#   sbatch evaluation/generalizability.sbatch                          # every fold, every model
#   DATA_DIR=/path/to/pose MODELS="xgboost_turns" sbatch evaluation/generalizability.sbatch
# Fold numbers come from `python -m evaluation.generalizability --list`.
#SBATCH --job-name=cichlid-lovo
#SBATCH --partition=ice-gpu
#SBATCH --array=0-6
# The CUDA 13 torch build needs Ampere or newer: V100 (too old) and MI210 (AMD) cannot run ActionFormer.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/generalizability/slurm-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$HOME/scratch/.uv-cache" UV_PYTHON_INSTALL_DIR="$HOME/scratch/.uv-python"
DATA_DIR="${DATA_DIR:-$HOME/scratch/cichlid-data}"
read -r -a ALIASES <<< "${ALIASES:-}"
MODEL_ARGS=()
if [[ -n "${MODELS:-}" ]]; then read -r -a MODEL_LIST <<< "$MODELS"; MODEL_ARGS=(--models "${MODEL_LIST[@]}"); fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Fail before training rather than let ActionFormer fall back to CPU.
uv run --frozen python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; x = torch.ones(1, 1, 8, device='cuda'); torch.nn.functional.conv1d(x, torch.ones(1, 1, 3, device='cuda')); print('CUDA OK:', torch.cuda.get_device_name(0))"
uv run --frozen python -m evaluation.generalizability --fold-index "${SLURM_ARRAY_TASK_ID}" \
    --data-dir "${DATA_DIR}" "${MODEL_ARGS[@]}" ${ALIASES[@]+"${ALIASES[@]/#/--alias=}"}
