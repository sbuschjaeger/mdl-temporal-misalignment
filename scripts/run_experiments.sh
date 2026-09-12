#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

real_datasets=(
  crowdsourced dreamera dreamerv fordchallenge opportunity
  pamap2 skoda stew uciactivity
)
real_configs=()
for dataset in "${real_datasets[@]}"; do
  real_configs+=("configs/${dataset}.yml")
done

# Deployment-lag experiment: train classifiers and SyncNet, evaluate their lag
# curves, and evaluate label-free MDL correction on the same selected group.
for config in configs/*.yml; do
  python scripts/train_models.py \
    --configs "$config" \
    --results-dir results/original \
    --retrain
done
python scripts/compute_metric.py \
  --configs configs/*.yml \
  --results-dir results/original
python scripts/evaluate_syncnet_baselines.py \
  --configs configs/*.yml \
  --results-dir results/original

# Whole-dataset audit: estimate one global correction from every real dataset,
# materialize it, and retrain the four classifiers at zero additional lag.
python scripts/compute_metric.py \
  --configs "${real_configs[@]}" \
  --results-dir results/global_audit \
  --folds 0 \
  --split all
python scripts/prepare_global_corrected_datasets.py \
  --audit-results results/global_audit \
  --data-root data/global_corrected
for config in "${real_configs[@]}"; do
  python scripts/train_models.py \
    --configs "$config" \
    --results-dir results/global_corrected \
    --data-root data/global_corrected \
    --fixed-groups-from results/original \
    --models cnn,resnet,late_fusion,magnitude_cnn \
    --lags 0 \
    --skip-corrections \
    --selected-group-only \
    --retrain
done

python scripts/plot_paper_results.py

