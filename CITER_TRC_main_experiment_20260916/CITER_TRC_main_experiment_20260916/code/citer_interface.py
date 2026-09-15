#!/usr/bin/env python3
"""Load, run, and verify the serialized CITER-MoE estimator."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from v3_sota_checkpoint import (
    DEVICE,
    GatedMultiTask,
    capture_at_fraction,
    ece,
    frozen_percentile,
    ndcg_at_fraction,
    predict,
)

FLOAT_COLUMNS = (
    "probability",
    "ranking_score",
)
METRIC_COLUMNS = (
    "AP",
    "Brier",
    "ECE",
    "Capture20",
    "NDCG20",
    "Spearman",
    "MAE_log",
    "RMSE_log",
    "R2_log",
    "Pearson",
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_checkpoint(checkpoint_path: Path, calibrator_path: Path, feature_path: Path) -> pd.DataFrame:
    checkpoint = load_checkpoint(checkpoint_path)
    calibrators = joblib.load(calibrator_path)
    frame = pd.read_parquet(feature_path)
    feature_names = checkpoint["feature_names"]
    raw = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    medians = np.asarray(checkpoint["medians"], dtype=np.float32)
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.take(medians, np.where(missing)[1])
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    x = (raw - mean) / std

    models = []
    for state in checkpoint["state_dicts"]:
        model = GatedMultiTask(checkpoint["input_dim"], checkpoint["n_experts"]).to(DEVICE)
        model.load_state_dict(state)
        models.append(model)
    outputs = [predict(model, x) for model in models]
    detect = np.mean([value[0] for value in outputs], axis=0)
    burden_z = np.mean([value[1] for value in outputs], axis=0)
    rank = np.mean([value[2] for value in outputs], axis=0)
    reference = checkpoint["calibration_reference"]

    detect_fraction = frozen_percentile(reference["detect"], detect).reshape(-1, 1)
    p_detect = calibrators["detect"].predict_proba(detect_fraction)[:, 1]
    burden_log = burden_z * checkpoint["burden_std"] + checkpoint["burden_mean"]
    predicted_log_burden = calibrators["magnitude"].predict(burden_log.reshape(-1, 1))
    predicted_burden = np.maximum(0.0, np.expm1(predicted_log_burden))
    p_burden = calibrators["burden_probability"].predict_proba(burden_log.reshape(-1, 1))[:, 1]
    rank_fraction = frozen_percentile(reference["rank"], rank).reshape(-1, 1)
    p_rank = calibrators["rank"].predict_proba(rank_fraction)[:, 1]
    stack_features = np.column_stack([
        detect_fraction.ravel(),
        frozen_percentile(reference["burden_log"], burden_log),
        rank_fraction.ravel(),
        x[:, : len(checkpoint["signal_cols"])],
    ])
    p_stack = calibrators["stack"].predict_proba(stack_features)[:, 1]

    date_col = checkpoint["date_col"]
    base = frame[["event_id", date_col, "label", "burden"]].reset_index(drop=True)
    blocks = []
    for name, probability, score in (
        ("citer_moe_detect", p_detect, detect),
        ("citer_moe_burden", p_burden, predicted_burden),
        ("citer_moe_rank", p_rank, rank),
        ("citer_moe_stack", p_stack, p_stack),
    ):
        block = base.copy()
        block["city"] = checkpoint["city"]
        block["model"] = name
        block["probability"] = probability
        block["ranking_score"] = score
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, model), block in predictions.groupby(["city", "model"], sort=False):
        y = block["label"].to_numpy(np.int8)
        burden = block["burden"].to_numpy(float)
        probability = block["probability"].to_numpy(float)
        score = block["ranking_score"].to_numpy(float)
        row = {
            "city": city,
            "model": model,
            "n": len(block),
            "prevalence": float(y.mean()),
            "AP": float(average_precision_score(y, probability)),
            "Brier": float(brier_score_loss(y, probability)),
            "ECE": ece(y, probability),
            "Capture20": capture_at_fraction(burden, score),
            "NDCG20": ndcg_at_fraction(burden, score),
            "Spearman": float(spearmanr(burden, score).statistic),
        }
        if model == "citer_moe_burden":
            observed = np.log1p(burden)
            estimated = np.log1p(np.maximum(score, 0))
            row.update(
                MAE_log=float(mean_absolute_error(observed, estimated)),
                RMSE_log=float(mean_squared_error(observed, estimated) ** 0.5),
                R2_log=float(r2_score(observed, estimated)),
                Pearson=float(pearsonr(observed, estimated).statistic),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_metrics(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float) -> list[str]:
    messages = []
    merged = expected.merge(actual, on=["city", "model"], suffixes=("_expected", "_actual"), validate="one_to_one")
    if len(merged) != len(expected) or len(merged) != len(actual):
        raise AssertionError(f"metric row mismatch: expected={len(expected)}, actual={len(actual)}, matched={len(merged)}")
    for _, row in merged.iterrows():
        for column in ("n", "prevalence", *METRIC_COLUMNS):
            left_name = f"{column}_expected"
            right_name = f"{column}_actual"
            if left_name not in row or right_name not in row or pd.isna(row[left_name]):
                continue
            error = abs(float(row[left_name]) - float(row[right_name]))
            if error > tolerance:
                raise AssertionError(f"{row['city']} {row['model']} {column}: abs_error={error:.12g}")
        messages.append(f"PASS metrics {row['city']} {row['model']}")
    return messages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def print_header(command: str) -> None:
    print(f"timestamp_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"command={command}")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"sklearn={sklearn.__version__}")
    print(f"device={DEVICE}")


def verify_checkpoint(args: argparse.Namespace) -> None:
    print_header("verify-checkpoint")
    for path in (args.checkpoint, args.calibrators, args.features, args.expected_predictions, args.expected_metrics):
        print(f"sha256 {sha256(path)} {path}")
    actual_predictions = run_checkpoint(args.checkpoint, args.calibrators, args.features)
    expected_predictions = pd.read_parquet(args.expected_predictions)
    city = load_checkpoint(args.checkpoint)["city"]
    expected_predictions = expected_predictions.loc[expected_predictions["city"].eq(city)].reset_index(drop=True)
    if len(actual_predictions) != len(expected_predictions):
        raise AssertionError(f"prediction row mismatch: expected={len(expected_predictions)}, actual={len(actual_predictions)}")
    for column in FLOAT_COLUMNS:
        error = float(np.max(np.abs(actual_predictions[column].to_numpy(float) - expected_predictions[column].to_numpy(float))))
        print(f"max_abs_error {city} {column} {error:.12g}")
        if error > args.tolerance:
            raise AssertionError(f"{city} {column}: max_abs_error={error:.12g}")
    actual_metrics = evaluate(actual_predictions)
    expected_metrics = pd.read_csv(args.expected_metrics)
    expected_metrics = expected_metrics.loc[expected_metrics["city"].eq(city)]
    for message in compare_metrics(actual_metrics, expected_metrics, args.tolerance):
        print(message)
    print(f"PASS checkpoint replay {city}")


def verify_paper(args: argparse.Namespace) -> None:
    print_header("verify-paper")
    result_dir = args.root / "results/trc_v4/sota_models"
    prediction_path = result_dir / "predictions.parquet"
    metric_path = result_dir / "metrics.csv"
    for path in (prediction_path, metric_path):
        print(f"sha256 {sha256(path)} {path}")
    actual = evaluate(pd.read_parquet(prediction_path))
    expected = pd.read_csv(metric_path)
    for message in compare_metrics(actual, expected, args.tolerance):
        print(message)
    required = [
        args.root / "results/trc_v4/nyc_dynamic_final/cohort_metrics.csv",
        args.root / "results/trc_v4/aligned_dynamic_ablation/paired_group_ablation.csv",
        args.root / "results/trc_v4/revision_experiments/continuous_burden_metrics.csv",
        args.root / "results/trc_v4/revision_experiments/fusion_ablation.csv",
        args.root / "results/trc_v4/revision_experiments/active_ranking_coverage_regret.csv",
        args.root / "results/trc_v5/reconstruction_audit/audit_summary.csv",
    ]
    for path in required:
        frame = pd.read_csv(path)
        if frame.empty:
            raise AssertionError(f"empty required result: {path}")
        print(f"PASS result rows={len(frame)} sha256={sha256(path)} path={path}")
    print("PASS all locked paper results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="run a serialized CITER-MoE checkpoint")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--calibrators", type=Path, required=True)
    predict_parser.add_argument("--features", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-checkpoint", help="replay and compare a saved checkpoint")
    verify_parser.add_argument("--checkpoint", type=Path, required=True)
    verify_parser.add_argument("--calibrators", type=Path, required=True)
    verify_parser.add_argument("--features", type=Path, required=True)
    verify_parser.add_argument("--expected-predictions", type=Path, required=True)
    verify_parser.add_argument("--expected-metrics", type=Path, required=True)
    verify_parser.add_argument("--tolerance", type=float, default=1e-6)

    paper_parser = subparsers.add_parser("verify-paper", help="recompute the locked paper metrics")
    paper_parser.add_argument("--root", type=Path, default=Path("."))
    paper_parser.add_argument("--tolerance", type=float, default=1e-10)

    args = parser.parse_args()
    if args.command == "predict":
        predictions = run_checkpoint(args.checkpoint, args.calibrators, args.features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.output, index=False)
        print(json.dumps({"rows": len(predictions), "output": str(args.output)}, indent=2))
    elif args.command == "verify-checkpoint":
        verify_checkpoint(args)
    else:
        verify_paper(args)


if __name__ == "__main__":
    main()
