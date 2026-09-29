#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
gapo_python="${GAPO_PYTHON:-/home/jerome/miniconda3/envs/simpo/bin/python}"
gapo_run_id="${1:-$(date -u +%Y%m%d_%H%M%S)}"
gapo_output="$(pwd)/outputs/mistral-7b-instruct-gapo-fp32_${gapo_run_id}"

# Refuse to mix a new experiment with an existing output/checkpoint directory.
mkdir "$gapo_output"
mkdir "$gapo_output/source"
cp scripts/gapo_trainer.py scripts/gapo_config.py scripts/run_gapo.py "$gapo_output/source/"
cp training_configs/mistral-7b-instruct-gapo-fp32.yaml "$gapo_output/training_config.yaml"
cp accelerate_configs/deepspeed_zero3_fp32.yaml "$gapo_output/accelerate_config.yaml"
printf '%s\n' "$$" > "$gapo_output/launcher.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$gapo_output/started_at.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/gapo_fp32_triton
export TORCH_EXTENSIONS_DIR=/tmp/gapo_zero3_torch_extensions
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

printf 'Training output: %s\n' "$gapo_output"
set +e
"$gapo_python" -u -m accelerate.commands.launch \
  --config_file accelerate_configs/deepspeed_zero3_fp32.yaml \
  --main_process_port 29834 \
  scripts/run_gapo.py training_configs/mistral-7b-instruct-gapo-fp32.yaml \
  "--output_dir=$gapo_output" \
  "--run_name=mistral-7b-instruct-gapo-fp32_${gapo_run_id}" \
  > "$gapo_output/train.log" 2>&1
gapo_exit_code=$?
printf '%s\n' "$gapo_exit_code" > "$gapo_output/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$gapo_output/finished_at.txt"
exit "$gapo_exit_code"
