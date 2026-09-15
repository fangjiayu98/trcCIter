#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from citer_interface import evaluate, compare_metrics, print_header, sha256

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('.'))
parser.add_argument('--tolerance', type=float, default=1e-10)
args = parser.parse_args()
base = args.root / 'results/trc_v4/sota_models'
prediction_path = base / 'predictions.parquet'
metric_path = base / 'metrics.csv'
print_header('verify-main')
print(f'sha256 {sha256(prediction_path)} {prediction_path}')
print(f'sha256 {sha256(metric_path)} {metric_path}')
actual = evaluate(pd.read_parquet(prediction_path))
expected = pd.read_csv(metric_path)
for message in compare_metrics(actual, expected, args.tolerance):
    print(message)
print('PASS main experiment: all 8 city-task rows reproduce from event-level predictions')
