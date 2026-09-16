#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "$ROOT"
mkdir -p logs results/main_table

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=20260915
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"

exec > >(tee logs/main_experiment_reproduction.log) 2>&1

printf '[%s] RUN_START\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

printf '[%s] TORONTO_EXPERTS_START\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON_BIN" src/train_toronto_experts.py
printf '[%s] TORONTO_EXPERTS_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '[%s] NEW_YORK_EXPERTS_START\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON_BIN" src/train_new_york_experts.py
printf '[%s] NEW_YORK_EXPERTS_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '[%s] CITER_MOE_START\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON_BIN" src/train_citer_moe.py
printf '[%s] CITER_MOE_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '[%s] MAIN_TABLE_START\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON_BIN" src/build_main_table.py --root . --output results/main_table
printf '[%s] RUN_COMPLETE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

