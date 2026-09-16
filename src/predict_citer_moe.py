from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_citer_moe import GatedMultiTask


def frozen_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if reference.size == 0:
        raise ValueError("frozen_percentile requires a non-empty reference block")
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return ((left + right + 1.0) / (2.0 * (reference.size + 1.0))).astype(np.float32)


def neural_outputs(
    checkpoint: dict,
    normalized_features: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_outputs = []
    use_cuda = torch.cuda.is_available() and str(checkpoint.get("training_device", "")).startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    for state_dict in checkpoint["state_dicts"]:
        model = GatedMultiTask(checkpoint["input_dim"], checkpoint["n_experts"])
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        blocks = [[], [], [], [], []]
        with torch.no_grad():
            for start in range(0, len(normalized_features), batch_size):
                batch = torch.as_tensor(
                    normalized_features[start : start + batch_size], dtype=torch.float32, device=device
                )
                values = model(batch)
                for target, value in zip(blocks, values):
                    target.append(value.cpu().numpy())
        seed_outputs.append(tuple(np.concatenate(parts, axis=0) for parts in blocks))
    detect = np.mean([output[0] for output in seed_outputs], axis=0)
    burden_z = np.mean([output[1] for output in seed_outputs], axis=0)
    rank = np.mean([output[2] for output in seed_outputs], axis=0)
    gate = np.mean([output[4] for output in seed_outputs], axis=0)
    return detect, burden_z, rank, gate


def run_inference(
    features_path: Path,
    checkpoint_path: Path,
    calibrators_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(features_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    calibrators = joblib.load(calibrators_path)

    required = set(checkpoint["identity_columns"] + checkpoint["feature_names"])
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature file is missing checkpoint columns: {sorted(missing)}")
    if len(checkpoint["state_dicts"]) != 5:
        raise ValueError("The paper model requires exactly five random-seed state dictionaries")

    raw = features[checkpoint["feature_names"]].to_numpy(np.float32)
    medians = np.asarray(checkpoint["medians"], dtype=np.float32)
    if raw.shape[1] != len(medians):
        raise ValueError("Checkpoint feature contract and feature matrix have different widths")
    raw = np.where(np.isfinite(raw), raw, medians)
    normalized = (
        raw - np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    ) / np.asarray(checkpoint["normalization_std"], dtype=np.float32)

    detect, burden_z, rank, gate = neural_outputs(checkpoint, normalized)
    references = checkpoint["calibration_reference"]
    detect_fraction = frozen_percentile(references["detect"], detect).reshape(-1, 1)
    p_detect = calibrators["detect"].predict_proba(detect_fraction)[:, 1]

    burden_log = burden_z * checkpoint["burden_std"] + checkpoint["burden_mean"]
    predicted_log_burden = calibrators["magnitude"].predict(burden_log.reshape(-1, 1))
    predicted_burden = np.maximum(0.0, np.expm1(predicted_log_burden))
    p_burden = calibrators["burden_probability"].predict_proba(
        burden_log.reshape(-1, 1)
    )[:, 1]

    rank_fraction = frozen_percentile(references["rank"], rank).reshape(-1, 1)
    p_rank = calibrators["rank"].predict_proba(rank_fraction)[:, 1]
    stack_features = np.column_stack(
        [
            detect_fraction.ravel(),
            frozen_percentile(references["burden_log"], burden_log),
            rank_fraction.ravel(),
            normalized[:, : len(checkpoint["signal_cols"])],
        ]
    )
    p_stack = calibrators["stack"].predict_proba(stack_features)[:, 1]

    base = features[checkpoint["identity_columns"]].copy().reset_index(drop=True)
    blocks = []
    for model_name, probability, ranking_score in (
        ("citer_moe_detect", p_detect, detect),
        ("citer_moe_burden", p_burden, predicted_burden),
        ("citer_moe_rank", p_rank, rank),
        ("citer_moe_stack", p_stack, p_stack),
    ):
        block = base.copy()
        block["city"] = checkpoint["city"]
        block["model"] = model_name
        block["probability"] = probability
        block["ranking_score"] = ranking_score
        blocks.append(block)
    predictions = pd.concat(blocks, ignore_index=True)
    gate_frame = pd.DataFrame(
        {
            "city": checkpoint["city"],
            "expert": checkpoint["experts"],
            "mean_gate": gate.mean(axis=0),
        }
    )
    return predictions, gate_frame


def compare_with_archive(replayed: pd.DataFrame, archived_path: Path) -> dict:
    archived = pd.read_parquet(archived_path)
    city = str(replayed["city"].iloc[0])
    archived = archived[archived["city"] == city].reset_index(drop=True)
    replayed = replayed.reset_index(drop=True)
    if len(replayed) != len(archived):
        raise ValueError(f"Row-count mismatch: replay={len(replayed)}, archive={len(archived)}")
    for column in ["event_id", "model", "label"]:
        if not np.array_equal(replayed[column].astype(str), archived[column].astype(str)):
            raise ValueError(f"Identity mismatch in column {column}")
    differences = {}
    for column in ["probability", "ranking_score"]:
        differences[column] = float(
            np.max(
                np.abs(
                    replayed[column].to_numpy(float) - archived[column].to_numpy(float)
                )
            )
        )
    differences["status"] = (
        "PASS" if max(differences["probability"], differences["ranking_score"]) <= 1e-6 else "FAIL"
    )
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and run a frozen CITER-MoE checkpoint.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibrators", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    predictions, gates = run_inference(args.features, args.checkpoint, args.calibrators)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output, index=False)
    if args.gate_output:
        args.gate_output.parent.mkdir(parents=True, exist_ok=True)
        gates.to_csv(args.gate_output, index=False)

    report = {"city": str(predictions["city"].iloc[0]), "rows": len(predictions)}
    if args.compare:
        report["archive_comparison"] = compare_with_archive(predictions, args.compare)
        if report["archive_comparison"]["status"] != "PASS":
            raise RuntimeError(f"Checkpoint replay failed: {report}")
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
