# CITER main experiment

This repository contains the reproducible main experiment used for the CITER paper. It intentionally excludes auxiliary analyses, development snapshots, partial outputs, process files, and superseded model variants.

## Included material

- `src/`: the expert models, CITER-MoE training, frozen-model inference, and main-table builder.
- `data/`: the New York and Toronto inputs required by the main experiment.
- `artifacts/new_york_experts/` and `artifacts/toronto_experts/`: locked expert predictions and main metrics.
- `artifacts/final_model/`: the final CITER-MoE checkpoints, calibrators, frozen features, predictions, protocol, and metrics.
- `results/main_table/`: exact and paper-rounded main results, provenance, and audit records.
- `logs/`: the retained main-pipeline and main-table logs.

## Environment

The original software environment is recorded in `requirements-lock.txt`. For a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

The experiment was run with a CUDA-capable GPU. Hardware and completion metadata are retained in `reproducibility/`.

## Reproduce the main experiment

From the repository root, run:

```bash
PYTHON=.venv/bin/python bash run_main.sh
```

The script trains the Toronto and New York expert banks, trains the final CITER-MoE models, and rebuilds `results/main_table/` using the same deterministic seed and pipeline order as the reported experiment.

To rebuild only the paper table from the included locked artifacts:

```bash
PYTHONPATH=src .venv/bin/python src/build_main_table.py \
  --root . \
  --output results/main_table
```

## Frozen-model inference

New York example:

```bash
PYTHONPATH=src .venv/bin/python src/predict_citer_moe.py \
  --features artifacts/final_model/checkpoints/new_york_features.parquet \
  --checkpoint artifacts/final_model/checkpoints/new_york.pt \
  --calibrators artifacts/final_model/checkpoints/new_york_calibrators.joblib \
  --output new_york_predictions.parquet
```

Toronto example:

```bash
PYTHONPATH=src .venv/bin/python src/predict_citer_moe.py \
  --features artifacts/final_model/checkpoints/toronto_features.parquet \
  --checkpoint artifacts/final_model/checkpoints/toronto.pt \
  --calibrators artifacts/final_model/checkpoints/toronto_calibrators.joblib \
  --output toronto_predictions.parquet
```

