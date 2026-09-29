#!/usr/bin/env bash
#
# Sweep the 5 GAPO checkpoints (0% / 25% / 50% / 75% / 100%) and write one
# .npz per checkpoint to ${OUTPUT_DIR}/<tag>.npz.
#
# Usage:
#   bash scripts/run_alpha_iB.sh
#
# Env overrides:
#   CONFIG          alpha_iB yaml config (default: experiments/configs/alpha_iB_mistral_gapo.yaml)
#   OUTPUT_DIR      where .npz are written (default: experiments/results/alpha_iB)
#   PYTHON          python interpreter (default: python)
#   GPU             CUDA_VISIBLE_DEVICES value (default: 0)
#
# NOTE: DeepSpeed-saved checkpoints (containing zero_pp_rank_*.pt) must first
# be consolidated to a HF-loadable single state dict, e.g.
#     python <ckpt>/zero_to_fp32.py <ckpt> <ckpt>/pytorch_model.bin
# before being passed in here.

set -euo pipefail

CONFIG="${CONFIG:-experiments/configs/alpha_iB_mistral_gapo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/alpha_iB_batch_16}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-2}"

mkdir -p "${OUTPUT_DIR}"

# (tag, model_name_or_path) pairs.
# Edit these paths to match the actual GAPO run you want to analyze.
# Example below uses the 116/116/233/350/466 checkpoints of mistral-7b-instruct-gapo_len_0.4
# (save_steps=116, total ~466 steps), with the base model standing in for 0%.
CKPTS=(
  # "ckpt-116|/home/jerome/GAPO/outputs/mistral-7b-instruct-gapo_len_0.2_new/checkpoint-116"
  # "ckpt-232|/home/jerome/GAPO/outputs/mistral-7b-instruct-gapo_len_0.2_new/checkpoint-232"
  # "ckpt-348|/home/jerome/GAPO/outputs/mistral-7b-instruct-gapo_len_0.2_new/checkpoint-348"
  "ckpt-466|/home/jerome/GAPO/outputs/mistral-7b-instruct-gapo_checking/checkpoint-466"
)

for entry in "${CKPTS[@]}"; do
  tag="${entry%%|*}"
  path="${entry#*|}"
  out="${OUTPUT_DIR}/${tag}.npz"

  if [[ -f "${out}" ]]; then
    echo "[skip] ${out} already exists"
    continue
  fi

  echo
  echo "==========================================================="
  echo "  alpha_iB :: tag=${tag}  path=${path}"
  echo "==========================================================="

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" experiments/alpha_iB.py "${CONFIG}" \
    --model_name_or_path="${path}" \
    --alpha_ckpt_tag="${tag}" \
    --alpha_output_dir="${OUTPUT_DIR}"
done

echo
echo "All checkpoints done. Results in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"
