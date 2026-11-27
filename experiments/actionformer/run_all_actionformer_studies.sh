#!/usr/bin/env bash
# Run from repository root or from anywhere inside the git checkout.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
echo "=== ACTIONFORMER CONTROLLED EXPERIMENT BATTERY ==="
echo "Repo commit: $(git rev-parse --short HEAD)"
echo "Started at: $(date)"
echo "Stage 1/2: 63-run feature x hard-negative study"
uv run python -m experiments.actionformer.run_large_actionformer_study --tier full
echo "Stage 1 complete at: $(date)"
echo "Stage 2/2: 6-run full-video leave-one-video-out study"
uv run python -m experiments.actionformer.run_actionformer_lovo
echo "Stage 2 complete at: $(date)"
echo "=== ALL STUDIES COMPLETE ==="
echo "Sampled-segment results: outputs/actionformer_large_controlled_study/summary.md"
echo "LOVO results: outputs/actionformer_lovo_controlled/summary.md"
