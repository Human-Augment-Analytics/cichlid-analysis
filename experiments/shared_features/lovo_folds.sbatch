#!/bin/bash
# Shared-feature XGBoost variants on the issue #19 folds (seven test videos + the 0028 holdout), on PACE ICE.
# One array task per variant and reviewed-negatives setting: tasks 0-11 with, 12-23 without.
# Needs, under outputs/generalizability/shared_features/run_{with,without}/:
#   experiments/model_frames.parquet and experiments/baseline_frames.parquet (dataset.py, baselines.py)
#   video_frames.json (frame counts of the eight videos)
#   sbatch experiments/shared_features/lovo_folds.sbatch
#SBATCH --job-name=sf-lovo
#SBATCH --partition=ice-cpu
#SBATCH --array=0-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/generalizability/shared_features/slurm-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$HOME/scratch/.uv-cache" UV_PYTHON_INSTALL_DIR="$HOME/scratch/.uv-python"
VARIANTS=(xgb_all xgb_all_agg xgb_v1_features xgb_no_distance_proximity xgb_no_orientation_facing
          xgb_no_speed_closing xgb_no_angular_movement xgb_no_temporal_consistency
          xgb_their_lr102 xgb_their_win155 xgb_their_both xgb_mine_plus_their)
task=${SLURM_ARRAY_TASK_ID}
variant=${VARIANTS[$((task % ${#VARIANTS[@]}))]}
if (( task < ${#VARIANTS[@]} )); then setting=with; negatives=(); else setting=without; negatives=(--no-annotated-hard-negatives); fi

root="$PWD/outputs/generalizability/shared_features"
export CIRCLING_OUTPUTS="$root/run_$setting" CIRCLING_DATA_DIR="$root/data" \
       CIRCLING_GROUND_TRUTH="$root/finalized_spans.csv" \
       CIRCLING_VIDEOS=0007_vid,0028_vid,0031_vid,0037_vid,vid3-v13_d13,vid5-v07_d05,vid6-v18_d18,vid7-v02_d00
echo "$variant, reviewed negatives: $setting"
uv run --frozen python -m experiments.shared_features.lovo --holdout 0028_vid --variants "$variant" \
    --out-dir "$CIRCLING_OUTPUTS/experiments/lovo/$variant" ${negatives[@]+"${negatives[@]}"}
