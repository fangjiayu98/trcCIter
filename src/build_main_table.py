from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


ROWS = [
    ("Toronto", "CITER-MoE-Detect", "moe", "citer_moe_detect"),
    ("Toronto", "CITER-MoE-Router", "moe", "citer_moe_stack"),
    ("Toronto", "Conditional survival", "toronto", "conditional_survival"),
    ("Toronto", "CITER-Detect", "toronto", "citer_dynamic_exposure"),
    ("Toronto", "CatBoost", "toronto", "catboost"),
    ("Toronto", "Fine-tuned MiniLM", "toronto", "finetuned_minilm"),
    ("Toronto", "Duration x demand", "toronto", "btm_lgbm_duration"),
    ("Toronto", "CITER-Burden", "toronto", "direct_exposure_lgbm"),
    ("Toronto", "CITER-Rank", "toronto", "exposure_lambdamart"),
    ("New York", "CITER-MoE-Router", "moe", "citer_moe_stack"),
    ("New York", "CITER-Burden", "new_york", "direct_exposure_lgbm"),
    ("New York", "CITER-Detect", "new_york", "citer_detect"),
    ("New York", "CITER-Rank", "new_york", "exposure_lambdamart"),
    ("New York", "Native CatBoost", "new_york", "native_catboost"),
    ("New York", "Duration x demand", "new_york", "text_structured_lgbm_duration"),
    ("New York", "CITER-MoE-Detect", "moe", "citer_moe_detect"),
    ("New York", "Multi-task survival", "new_york", "mtdnn_survival"),
    ("New York", "Fine-tuned MiniLM", "new_york", "finetuned_minilm"),
]


PAPER = {
    ("Toronto", "CITER-MoE-Detect"): (0.635, 0.104, 0.060, 0.577),
    ("Toronto", "CITER-MoE-Router"): (0.630, 0.111, 0.079, 0.634),
    ("Toronto", "Conditional survival"): (0.612, 0.104, 0.036, 0.609),
    ("Toronto", "CITER-Detect"): (0.609, 0.102, 0.025, 0.606),
    ("Toronto", "CatBoost"): (0.591, 0.105, 0.030, 0.580),
    ("Toronto", "Fine-tuned MiniLM"): (0.590, 0.105, 0.027, 0.579),
    ("Toronto", "Duration x demand"): (0.546, 0.111, 0.025, 0.615),
    ("Toronto", "CITER-Burden"): (0.520, 0.117, 0.032, 0.542),
    ("Toronto", "CITER-Rank"): (0.495, 0.114, 0.035, 0.602),
    ("New York", "CITER-MoE-Router"): (0.621, 0.057, 0.010, 0.616),
    ("New York", "CITER-Burden"): (0.615, 0.059, 0.014, 0.635),
    ("New York", "CITER-Detect"): (0.614, 0.060, 0.028, 0.633),
    ("New York", "CITER-Rank"): (0.612, 0.059, 0.014, 0.619),
    ("New York", "Native CatBoost"): (0.601, 0.061, 0.038, 0.606),
    ("New York", "Duration x demand"): (0.599, 0.060, 0.017, 0.631),
    ("New York", "CITER-MoE-Detect"): (0.588, 0.072, 0.058, 0.602),
    ("New York", "Multi-task survival"): (0.489, 0.068, 0.012, 0.574),
    ("New York", "Fine-tuned MiniLM"): (0.413, 0.134, 0.181, 0.523),
}


EXPECTED_N = {"Toronto": 25_534, "New York": 5_875}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for bin_index in range(bins):
        mask = ids == bin_index
        if mask.any():
            value += mask.mean() * abs(
                float(labels[mask].mean()) - float(probability[mask].mean())
            )
    return float(value)


def ndcg20(burden: np.ndarray, score: np.ndarray) -> float:
    k = max(1, int(math.ceil(len(score) * 0.20)))
    order = np.argsort(-score, kind="stable")[:k]
    ideal = np.argsort(-burden, kind="stable")[:k]
    discount = 1.0 / np.log2(np.arange(k) + 2.0)
    denominator = float(np.sum(burden[ideal] * discount))
    return float(np.sum(burden[order] * discount) / denominator) if denominator else 0.0


def select_rows(frame: pd.DataFrame, source: str, city: str, model_key: str) -> pd.DataFrame:
    if source == "moe":
        selected = frame[(frame["city"] == city) & (frame["model"] == model_key)].copy()
    else:
        selected = frame[(frame["split"] == "test_2024") & (frame["model"] == model_key)].copy()
    selected["_event_occurrence"] = selected.groupby("event_id", sort=False).cumcount()
    return selected.sort_values(
        ["event_id", "_event_occurrence"], kind="stable"
    ).reset_index(drop=True)


def validate(selected: pd.DataFrame, city: str, display_name: str) -> None:
    required = {"event_id", "label", "probability", "ranking_score", "burden"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"{city}/{display_name}: missing columns {sorted(missing)}")
    if len(selected) != EXPECTED_N[city]:
        raise ValueError(
            f"{city}/{display_name}: expected {EXPECTED_N[city]} rows, found {len(selected)}"
        )
    event_key = selected[["event_id", "_event_occurrence"]].astype(str)
    if event_key.duplicated().any():
        raise ValueError(f"{city}/{display_name}: duplicate event-occurrence keys")
    numeric = selected[["label", "probability", "ranking_score", "burden"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{city}/{display_name}: non-finite prediction values")
    probability = selected["probability"].to_numpy(float)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{city}/{display_name}: probability outside [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the locked-test TRC main table.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/main_table"))
    parser.add_argument(
        "--moe-predictions",
        type=Path,
        default=Path("artifacts/final_model/predictions.parquet"),
        help="Predictions emitted by the serialized CITER-MoE checkpoint run.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "toronto": root / "artifacts/toronto_experts/predictions.parquet",
        "new_york": root / "artifacts/new_york_experts/predictions.parquet",
        "moe": root / args.moe_predictions,
    }
    for source, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source} prediction artifact: {path}")

    frames = {source: pd.read_parquet(path) for source, path in paths.items()}
    records = []
    references: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    provenance = []

    for city, display_name, source, model_key in ROWS:
        selected = select_rows(frames[source], source, city, model_key)
        validate(selected, city, display_name)
        labels = selected["label"].to_numpy(int)
        probability = selected["probability"].to_numpy(float)
        ranking_score = selected["ranking_score"].to_numpy(float)
        burden = selected["burden"].to_numpy(float)
        event_ids = (
            selected["event_id"].astype(str)
            + "#"
            + selected["_event_occurrence"].astype(str)
        ).to_numpy()

        if city not in references:
            references[city] = (event_ids, labels, burden)
        else:
            ref_ids, ref_labels, ref_burden = references[city]
            if not np.array_equal(event_ids, ref_ids):
                raise ValueError(f"{city}/{display_name}: event IDs do not match the city reference")
            if not np.array_equal(labels, ref_labels):
                raise ValueError(f"{city}/{display_name}: labels do not match the city reference")
            if not np.allclose(burden, ref_burden, rtol=0.0, atol=1e-9):
                raise ValueError(f"{city}/{display_name}: burdens do not match the city reference")

        values = {
            "city": city,
            "model": display_name,
            "artifact_model_key": model_key,
            "n_test": len(selected),
            "prevalence": float(labels.mean()),
            "AP": float(average_precision_score(labels, probability)),
            "Brier": float(brier_score_loss(labels, probability)),
            "ECE_10": ece(labels, probability),
            "NDCG_top20pct": ndcg20(burden, ranking_score),
        }
        records.append(values)
        provenance.append(
            {
                "city": city,
                "model": display_name,
                "artifact_model_key": model_key,
                "prediction_file": str(paths[source].relative_to(root)),
                "prediction_sha256": sha256(paths[source]),
            }
        )

    table = pd.DataFrame(records)
    table.to_csv(output / "main_performance_exact.csv", index=False)
    rounded = table.copy()
    for column in ["prevalence", "AP", "Brier", "ECE_10", "NDCG_top20pct"]:
        rounded[column] = rounded[column].round(3)
    rounded.to_csv(output / "main_performance_paper_rounded.csv", index=False)

    audit_rows = []
    for row in records:
        paper = PAPER[(row["city"], row["model"])]
        current = (row["AP"], row["Brier"], row["ECE_10"], row["NDCG_top20pct"])
        item = {"city": row["city"], "model": row["model"]}
        for metric, paper_value, current_value in zip(
            ["AP", "Brier", "ECE_10", "NDCG_top20pct"], paper, current
        ):
            item[f"paper_{metric}"] = paper_value
            item[f"artifact_{metric}"] = current_value
            item[f"rounded_match_{metric}"] = round(current_value, 3) == paper_value
        item["all_rounded_metrics_match"] = all(
            item[f"rounded_match_{metric}"]
            for metric in ["AP", "Brier", "ECE_10", "NDCG_top20pct"]
        )
        audit_rows.append(item)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "paper_vs_artifact_audit.csv", index=False)
    pd.DataFrame(provenance).drop_duplicates().to_csv(output / "provenance.csv", index=False)

    manifest = {
        "status": "PASS" if bool(audit["all_rounded_metrics_match"].all()) else "MISMATCH",
        "rows": len(table),
        "fully_matching_rows": int(audit["all_rounded_metrics_match"].sum()),
        "input_sha256": {source: sha256(path) for source, path in paths.items()},
        "metric_contract": {
            "AP": "sklearn.metrics.average_precision_score",
            "Brier": "sklearn.metrics.brier_score_loss",
            "ECE": "10 equal-width probability bins",
            "NDCG": "raw burden relevance at top ceil(0.20*n), stable descending sort",
        },
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
