#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-verify}" in
  verify)
    "$PYTHON" code/verify_main.py --root "$ROOT"
    ;;
  train)
    CITER_OUT="$ROOT/results/checkpoint_run/citer_moe" "$PYTHON" code/v3_sota_checkpoint.py
    ;;
  replay)
    for city in toronto new_york; do
      "$PYTHON" code/citer_interface.py verify-checkpoint \
        --checkpoint "$ROOT/results/checkpoint_run/citer_moe/checkpoints/${city}.pt" \
        --calibrators "$ROOT/results/checkpoint_run/citer_moe/checkpoints/${city}_calibrators.joblib" \
        --features "$ROOT/results/checkpoint_run/citer_moe/checkpoints/${city}_features.parquet" \
        --expected-predictions "$ROOT/results/checkpoint_run/citer_moe/predictions.parquet" \
        --expected-metrics "$ROOT/results/checkpoint_run/citer_moe/metrics.csv"
    done
    ;;
  *)
    echo "usage: $0 {verify|train|replay}" >&2
    exit 2
    ;;
esac
