#!/usr/bin/env bash
#
# Phase A sweep at fractions {30%, 50%, 70%} x modes {stable, random, unstable}
# = 9 sequential SimPO runs on princeton-nlp/mistral-instruct-ultrafeedback.
#
# Driven by scripts/run_data_efficiency.sh; see that script for env vars.
#
# Usage:
#   bash scripts/run_dataeff_357.sh                 # foreground
#   nohup bash scripts/run_dataeff_357.sh > logs/dataeff_357.out 2>&1 &
#
# Override at the call site, e.g.:
#   MODES="stable" bash scripts/run_dataeff_357.sh
#   DRY_RUN=1 bash scripts/run_dataeff_357.sh

set -euo pipefail

cd "$(dirname "$0")/.."

export FRACTIONS="${FRACTIONS:-0.30 0.50 0.70}"
export MODES="${MODES:-stable random unstable}"
export TRAJ_NPZ="${TRAJ_NPZ:-experiments/scores/mistral_uf_base_T1.npz}"
export SCORE="${SCORE:-anchor_gap}"
export CONFIG="${CONFIG:-training_configs/mistral-instruct-uf-simpo.yaml}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_configs/deepspeed_zero3.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/data_efficiency}"
export SUBSET_DIR="${SUBSET_DIR:-experiments/subsets}"
export SEED="${SEED:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p logs "${OUTPUT_ROOT}" "${SUBSET_DIR}"

echo "============================================================"
echo "  Phase A sweep"
echo "  fractions: ${FRACTIONS}"
echo "  modes    : ${MODES}"
echo "  score    : ${SCORE}"
echo "  traj_npz : ${TRAJ_NPZ}"
echo "  output   : ${OUTPUT_ROOT}"
echo "============================================================"

bash scripts/run_data_efficiency.sh
