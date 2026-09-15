# CITER main experiment

This package contains only the final two-city CITER-MoE main experiment.

## Included

- `code/v3_sota_models.py`: original final main-experiment implementation.
- `code/v3_sota_checkpoint.py`: identical model/training protocol with complete checkpoint serialization.
- `code/citer_interface.py`: load, predict, evaluate, and checkpoint replay interface.
- `results/trc_v4/sota_models`: locked publication predictions, metrics, gate weights, and protocol.
- `results/checkpoint_run/citer_moe`: fresh serialized five-seed run, including test features, predictions, metrics, and protocol.
- `models`: Toronto and New York five-seed neural weights plus all fitted calibration layers.
- `logs/checkpoint_training.log`: fresh main-model training record.
- `logs/checkpoint_replay.log`: loaded-model numerical replay record.
- `logs/main_results_verification.log`: main publication metrics recomputed from event-level predictions.

## Inputs

The four compact Parquet files under `results/trc_v3` and `results/trc_v4` are the exact frozen expert and context inputs consumed by the final main experiment. No auxiliary analyses or unrelated raw-data cache are included.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHON=.venv/bin/python bash run_main_experiment.sh verify
PYTHON=.venv/bin/python bash run_main_experiment.sh replay
PYTHON=.venv/bin/python bash run_main_experiment.sh train
```

`verify` recomputes every reported main-experiment metric from event-level publication predictions. `replay` loads both saved models and checks their probabilities, ranking scores, and metrics within `1e-6`. `train` reruns the same two-city five-seed model and writes a complete new checkpoint run.

## Direct prediction

```bash
python code/citer_interface.py predict \
  --checkpoint models/toronto.pt \
  --calibrators models/toronto_calibrators.joblib \
  --features results/checkpoint_run/citer_moe/checkpoints/toronto_features.parquet \
  --output toronto_predictions.parquet
```
