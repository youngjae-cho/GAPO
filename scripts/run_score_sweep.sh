#!/usr/bin/env bash
#
# Sweep scripts/score_dataset.py over a list of checkpoints (including the
# raw base model) and write one .npz per checkpoint. Mirrors the shape of
# scripts/run_alpha_iB.sh, but produces Anchor Gap Gamma trajectories
# instead of alpha_{i,B}.
#
# Each run writes:
#   ${OUTPUT_DIR}/${tag}.npz       # per-example margins / margins_perturbed / ...
#   ${OUTPUT_DIR}/${tag}.npz.json  # sidecar with beta/gamma_beta_ratio/rho/etc.
#
# Usage (default sweeps the gapo_noise_0.2_check_2 checkpoints):
#   bash scripts/run_score_sweep.sh
#
# Background:
#   nohup bash scripts/run_score_sweep.sh > logs/score_sweep.out 2>&1 &
#
# Common env overrides:
#   CONFIG          scoring yaml             (default: mistral-instruct-uf_rn0.2-score.yaml)
#   OUTPUT_DIR      where .npz are written   (default: experiments/scores/gapo_noise_0.2_traj)
#   GPU             CUDA_VISIBLE_DEVICES     (default: 0)
#   MAX_SAMPLES     subset size              (default: '' = full split)
#   SUBSET_SEED     deterministic seed       (default: 12345 — matches alpha_iB.py)
#   SCORE_BATCH     per-device batch size    (default: '' = config default)
#   PYTHON          python interpreter       (default: /home/jerome/miniconda3/envs/simpo/bin/python)
#   CKPT_ROOT       prefix for checkpoint paths (default: outputs/mistral-7b-instruct-gapo_noise_0.2_check_2)
#   BASE_MODEL      base model id            (default: mistralai/Mistral-7B-Instruct-v0.2)
#   EXTRA           extra CLI args appended verbatim
#   DRY_RUN         '1' to print and exit
#   SKIP_EXISTING   '1' to skip when target .npz already exists (default: 1)
#
# Override the (tag, path) list via env CKPTS, e.g.:
#   CKPTS='base|.../base ckpt-200|.../checkpoint-200' bash scripts/run_score_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training_configs/mistral-instruct-uf_rn0.2-score.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/scores/gapo_noise_0.2_traj}"
GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBSET_SEED="${SUBSET_SEED:-12345}"
SCORE_BATCH="${SCORE_BATCH:-}"
PYTHON="${PYTHON:-/home/jerome/miniconda3/envs/simpo/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-outputs/mistral-7b-instruct-gapo_noise_0.2_check_2}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
EXTRA="${EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: CONFIG not found: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" logs

# Default sweep: base + 4 main checkpoints from gapo_noise_0.2_check_2.
# (checkpoint-464 omitted by default; pass CKPTS env to include it.)
DEFAULT_CKPTS=(
  "base|${BASE_MODEL}"
  "ckpt-116|${CKPT_ROOT}/checkpoint-116"
  "ckpt-232|${CKPT_ROOT}/checkpoint-232"
  "ckpt-348|${CKPT_ROOT}/checkpoint-348"
  "ckpt-466|${CKPT_ROOT}/checkpoint-466"
)

if [[ -n "${CKPTS:-}" ]]; then
  read -r -a CKPT_LIST <<< "${CKPTS}"
else
  CKPT_LIST=("${DEFAULT_CKPTS[@]}")
fi

echo "============================================================"
echo "  CONFIG       : ${CONFIG}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  GPU          : ${GPU}"
echo "  MAX_SAMPLES  : ${MAX_SAMPLES:-<full>}"
echo "  SUBSET_SEED  : ${SUBSET_SEED}"
echo "  CKPT_LIST    :"
for entry in "${CKPT_LIST[@]}"; do
  echo "    - ${entry}"
done
echo "============================================================"

for entry in "${CKPT_LIST[@]}"; do
  tag="${entry%%|*}"
  path="${entry#*|}"
  out="${OUTPUT_DIR}/${tag}.npz"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out}" ]]; then
    echo "[skip] ${out} already exists"
    continue
  fi

  echo
  echo "==========================================================="
  echo "  scoring tag=${tag}  path=${path}"
  echo "  -> ${out}"
  echo "==========================================================="

  cmd=(
    "${PYTHON}" scripts/score_dataset.py
    "${CONFIG}"
    "--model_name_or_path=${path}"
    "--output_npz=${out}"
    "--subset_seed=${SUBSET_SEED}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=("--max_samples=${MAX_SAMPLES}")
  fi
  if [[ -n "${SCORE_BATCH}" ]]; then
    cmd+=("--score_batch_size=${SCORE_BATCH}")
  fi
  if [[ -n "${EXTRA}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${EXTRA})
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
done

echo
echo "All checkpoints scored. Results in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"
