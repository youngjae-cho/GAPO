#!/usr/bin/env bash
#
# Data-efficiency sweep launcher for GAPO/SimPO.
#
# Pipeline per run:
#   (1) generate a subset JSON via scripts/select_subset.py
#   (2) train SimPO on that subset via scripts/run_simpo.py
#
# Usage:
#   bash scripts/run_data_efficiency.sh                 # sweep the default grid
#   bash scripts/run_data_efficiency.sh stable 0.30     # one-off (mode, fraction)
#   FRACTIONS="0.10 0.50" MODES="stable random" \
#     bash scripts/run_data_efficiency.sh               # custom grid via env
#
# Env overrides:
#   TRAJ_NPZ        per-sample scoring npz from score_dataset.py
#                   (default: experiments/scores/mistral_uf_base_T1.npz)
#   SCORE           score function (default: mean_reward_margin)
#   FRACTIONS       space-separated list of fractions (default: "0.10 0.30 0.50")
#   MODES           space-separated list of modes     (default: "stable random unstable")
#   SEED            selection seed (default: 0)
#   CONFIG          training yaml (default: training_configs/mistral-instruct-uf-simpo.yaml)
#   ACCEL_CONFIG    accelerate config (default: accelerate_configs/deepspeed_zero3.yaml)
#   SUBSET_DIR      subset JSONs go here (default: experiments/subsets)
#   OUTPUT_ROOT     parent for run output_dirs (default: outputs/data_efficiency)
#   RUN_TAG         extra tag baked into run_name and output_dir (default: "")
#   PYTHON          python interpreter (default: python)
#   ACCEL           accelerate launcher (default: accelerate)
#   DRY_RUN         if "1", print what would run and exit (default: 0)
#   SKIP_EXISTING   if "1", skip when output_dir already has a model (default: 1)
#
# Notes:
#   * remove_top / remove_random are also valid modes — pass them via MODES env.
#   * "random" mode ignores SCORE (uniform pick).

set -euo pipefail

TRAJ_NPZ="${TRAJ_NPZ:-experiments/scores/mistral_uf_base_T1.npz}"
SCORE="${SCORE:-anchor_gap}"
FRACTIONS="${FRACTIONS:-0.10 0.30 0.50}"
MODES="${MODES:-stable random unstable}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-training_configs/mistral-instruct-uf-simpo.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-accelerate_configs/deepspeed_zero3.yaml}"
SUBSET_DIR="${SUBSET_DIR:-experiments/subsets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/data_efficiency}"
RUN_TAG="${RUN_TAG:-}"
PYTHON="${PYTHON:-python}"
ACCEL="${ACCEL:-accelerate}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Allow positional one-off: $1=mode, $2=fraction
if [[ "$#" -ge 2 ]]; then
  MODES="$1"
  FRACTIONS="$2"
fi

if [[ ! -f "${TRAJ_NPZ}" ]]; then
  echo "ERROR: TRAJ_NPZ not found: ${TRAJ_NPZ}" >&2
  echo "Run scripts/score_dataset.py first." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: CONFIG not found: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${SUBSET_DIR}" "${OUTPUT_ROOT}"

# Tag derived from the trajectory filename so subset JSONs do not collide
# across different scoring runs.
TRAJ_TAG="$(basename "${TRAJ_NPZ}" .npz)"

format_frac() {
  # 0.30 -> 30, 0.05 -> 05
  python -c "f=float('$1'); print(f'{int(round(f*100)):02d}')"
}

run_one() {
  local mode="$1"
  local frac="$2"
  local frac_pct
  frac_pct="$(format_frac "${frac}")"

  local effective_score="${SCORE}"
  if [[ "${mode}" == "random" || "${mode}" == "remove_random" ]]; then
    effective_score="random"
  fi

  local tag="${TRAJ_TAG}__${effective_score}__${mode}__f${frac_pct}__s${SEED}"
  local subset_json="${SUBSET_DIR}/${tag}.json"
  local run_name="dataeff__${mode}__f${frac_pct}__${effective_score}"
  if [[ -n "${RUN_TAG}" ]]; then
    run_name="${run_name}__${RUN_TAG}"
  fi
  local out_dir="${OUTPUT_ROOT}/${run_name}"

  echo
  echo "==========================================================="
  echo "  mode=${mode}  fraction=${frac}  score=${effective_score}"
  echo "  subset_json=${subset_json}"
  echo "  out_dir=${out_dir}"
  echo "==========================================================="

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/config.json" ]]; then
    echo "[skip] ${out_dir} already has a saved model"
    return 0
  fi

  local select_cmd=(
    "${PYTHON}" scripts/select_subset.py
    --trajectory_npz "${TRAJ_NPZ}"
    --score "${effective_score}"
    --mode "${mode}"
    --fraction "${frac}"
    --seed "${SEED}"
    --output "${subset_json}"
    --report_flip_rate
  )

  local train_cmd=(
    "${ACCEL}" launch
    --config_file "${ACCEL_CONFIG}"
    scripts/run_simpo.py
    "${CONFIG}"
    "--data_subset_idx_file=${subset_json}"
    "--output_dir=${out_dir}"
    "--run_name=${run_name}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: ${select_cmd[*]}"
    echo "DRY_RUN: ${train_cmd[*]}"
    return 0
  fi

  "${select_cmd[@]}"
  "${train_cmd[@]}"
}

for frac in ${FRACTIONS}; do
  for mode in ${MODES}; do
    run_one "${mode}" "${frac}"
  done
done

echo
echo "All runs queued. Outputs under ${OUTPUT_ROOT}/"
