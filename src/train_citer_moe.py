#!/usr/bin/env python3
"""Context-gated multi-task stacking for the three CITER decision outputs."""
from __future__ import annotations

import json
import math
import os
import random

import joblib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20260915
ROOT = Path(".")
OUT = Path(os.environ.get("CITER_OUT", str(ROOT / "artifacts/final_model")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CitySpec:
    city: str
    predictions: Path
    context: Path
    experts: tuple[str, ...]
    date_col: str
    validation_cutoff: pd.Timestamp
    duration_col: str
    context_cols: tuple[str, ...]


TORONTO_CONTEXT = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "weekend", "peak", "station_score", "demand_rank", "degree_rank", "route_rank",
    "transfer", "line_count_scaled", "temperature_scaled", "wind_scaled",
    "wet_weather", "weather_missing", "kw_signal", "kw_track", "kw_train",
    "kw_passenger", "kw_police", "kw_power", "kw_weather", "kw_medical",
    "kw_fire", "kw_operations", "demand_rate_v2",
)
NYC_CONTEXT = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "weekend", "peak", "elapsed_minutes", "observed_updates", "minutes_since_update",
    "initial_line_count", "current_line_count", "union_line_count", "line_expansion",
    "distinct_statuses", "initial_header_chars", "observed_text_chars", "severe_status",
    "mentions_alternative", "mentions_time_commitment", "matched_station_rows",
    "matched_complexes", "cbd_share", "transfer_share", "route_count_mean",
    "route_count_max", "graph_degree_mean", "graph_degree_max", "demand_rate_sum",
    "demand_rate_max", "temperature_c", "precipitation_mm", "wind_kmh",
    "kw_signal", "kw_track", "kw_train", "kw_passenger", "kw_police", "kw_power",
    "kw_weather",
)


class GatedMultiTask(nn.Module):
    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(160, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.residual = nn.Sequential(nn.Linear(input_dim, 96), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, n_experts))
        self.detect_residual = nn.Linear(96, 1)
        self.burden_head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        self.rank_head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        self.duration_head = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x) + self.residual(x)
        weights = torch.softmax(self.gate(hidden), dim=1)
        expert_logits = x[:, : self.n_experts]
        gated_logit = torch.sum(weights * expert_logits, dim=1)
        detect = gated_logit + self.detect_residual(hidden).squeeze(1)
        burden = self.burden_head(hidden).squeeze(1)
        rank = burden + 0.35 * self.rank_head(hidden).squeeze(1)
        duration = self.duration_head(hidden).squeeze(1)
        return detect, burden, rank, duration, weights


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    result = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            result += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(result)


def ndcg_at_fraction(burden: np.ndarray, score: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(math.ceil(fraction * len(score))))
    order = np.argsort(-score, kind="stable")[:n]
    ideal = np.argsort(-burden, kind="stable")[:n]
    discount = 1.0 / np.log2(np.arange(n) + 2.0)
    denominator = float(np.sum(burden[ideal] * discount))
    return float(np.sum(burden[order] * discount) / denominator) if denominator else 0.0


def capture_at_fraction(burden: np.ndarray, score: np.ndarray, fraction: float = 0.20) -> float:
    n = max(1, int(math.ceil(fraction * len(score))))
    order = np.argsort(-score, kind="stable")[:n]
    return float(burden[order].sum() / max(burden.sum(), 1e-12))


def rank_fraction(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def frozen_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values through an empirical CDF fitted only on a pre-test reference block."""
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if reference.size == 0:
        raise ValueError("frozen_percentile requires a non-empty reference block")
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return ((left + right + 1.0) / (2.0 * (reference.size + 1.0))).astype(np.float32)


def prepare(spec: CitySpec):
    pred = pd.read_parquet(spec.predictions)
    pred = pred.loc[pred["model"].isin(spec.experts)].copy()
    key = ["event_id", spec.date_col]
    if {"row_hash", "duplicate_index"}.issubset(pred.columns):
        key.extend(["row_hash", "duplicate_index"])
    key.append("split")
    base = pred.drop_duplicates(key)[key + ["label", "burden"]].copy()
    probability = pred.pivot_table(
        index=key, columns="model", values="probability", aggfunc="first"
    ).reset_index()
    score = pred.pivot_table(
        index=key, columns="model", values="ranking_score", aggfunc="first"
    ).reset_index()
    probability.columns = [*key, *[f"p__{c}" for c in probability.columns[len(key):]]]
    score.columns = [*key, *[f"r__{c}" for c in score.columns[len(key):]]]
    frame = base.merge(probability, on=key).merge(score, on=key)
    if spec.city == "Toronto":
        context = pd.read_parquet(spec.context)
        context = context.drop_duplicates("event_id")
        keep = ["event_id", spec.duration_col, *[c for c in spec.context_cols if c in context]]
        frame = frame.merge(context[keep], on="event_id", how="left")
    else:
        context = pd.read_parquet(spec.context)
        context = context.loc[context["checkpoint"].eq(0)].drop_duplicates("event_id")
        keep = ["event_id", spec.duration_col, *[c for c in spec.context_cols if c in context]]
        frame = frame.merge(context[keep], on="event_id", how="left")
    frame[spec.date_col] = pd.to_datetime(frame[spec.date_col])
    logit_cols, rank_cols = [], []
    for expert in spec.experts:
        pcol, rcol = f"p__{expert}", f"r__{expert}"
        frame[f"logit__{expert}"] = np.log(
            np.clip(frame[pcol], 1e-5, 1 - 1e-5) / np.clip(1 - frame[pcol], 1e-5, 1)
        )
        reference_mask = frame["split"].ne("test_2024")
        frame[f"rank__{expert}"] = frozen_percentile(
            frame.loc[reference_mask, rcol].to_numpy(float),
            frame[rcol].to_numpy(float),
        )
        logit_cols.append(f"logit__{expert}")
        rank_cols.append(f"rank__{expert}")
    context_cols = [c for c in spec.context_cols if c in frame]
    for col in context_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame, logit_cols + rank_cols, context_cols


def train_seed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    burden_train: np.ndarray,
    duration_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    burden_mean: float,
    burden_std: float,
    duration_mean: float,
    duration_std: float,
    n_experts: int,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GatedMultiTask(x_train.shape[1], n_experts).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor((np.log1p(burden_train) - burden_mean) / burden_std, dtype=torch.float32),
        torch.tensor((np.log1p(duration_train) - duration_mean) / duration_std, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(768, len(dataset)), shuffle=True, drop_last=False)
    x_cal_t = torch.tensor(x_cal, dtype=torch.float32, device=DEVICE)
    best_state = None
    best_ap = -np.inf
    stale = 0
    for _ in range(180):
        model.train()
        for xb, yb, bb, db in loader:
            xb, yb, bb, db = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE), db.to(DEVICE)
            detect, burden, rank, duration, _ = model(xb)
            perm = torch.randperm(len(xb), device=DEVICE)
            sign = torch.sign(bb - bb[perm])
            valid = sign.ne(0)
            pair_loss = torch.nn.functional.softplus(-sign[valid] * (rank[valid] - rank[perm][valid])).mean() if valid.any() else rank.sum() * 0
            loss = (
                nn.functional.binary_cross_entropy_with_logits(detect, yb)
                + 0.45 * nn.functional.smooth_l1_loss(burden, bb)
                + 0.12 * nn.functional.smooth_l1_loss(duration, db)
                + 0.18 * pair_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_cal_t)[0].cpu().numpy()
        ap = average_precision_score(y_cal, logits)
        if ap > best_ap + 1e-5:
            best_ap = ap
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 22:
            break
    model.load_state_dict(best_state)
    return model


def predict(model: GatedMultiTask, x: np.ndarray):
    model.eval()
    outputs = [[], [], [], [], []]
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = torch.tensor(x[start:start + 4096], dtype=torch.float32, device=DEVICE)
            values = model(batch)
            for target, value in zip(outputs, values):
                target.append(value.cpu().numpy())
    return tuple(np.concatenate(parts, axis=0) for parts in outputs)


def run_city(spec: CitySpec):
    frame, signal_cols, context_cols = prepare(spec)
    meta = frame["split"].eq("validation_2023")
    meta_train = meta & frame[spec.date_col].lt(spec.validation_cutoff)
    calibration = meta & frame[spec.date_col].ge(spec.validation_cutoff)
    test = frame["split"].eq("test_2024")
    if calibration.sum() < 100:
        raise RuntimeError(f"{spec.city}: calibration segment too small")
    all_features = signal_cols + context_cols
    medians = frame.loc[meta_train, all_features].median()
    raw = frame[all_features].fillna(medians).to_numpy(np.float32)
    mean = raw[meta_train].mean(axis=0)
    std = raw[meta_train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = (raw - mean) / std
    y = frame["label"].to_numpy(np.int8)
    burden = frame["burden"].to_numpy(float)
    duration = np.maximum(frame[spec.duration_col].to_numpy(float), 0.1)
    burden_mean = float(np.log1p(burden[meta_train]).mean())
    burden_std = float(np.log1p(burden[meta_train]).std()) or 1.0
    duration_mean = float(np.log1p(duration[meta_train]).mean())
    duration_std = float(np.log1p(duration[meta_train]).std()) or 1.0
    models = [
        train_seed(
            x[meta_train], y[meta_train], burden[meta_train], duration[meta_train],
            x[calibration], y[calibration], burden_mean, burden_std,
            duration_mean, duration_std, len(spec.experts), SEED + seed,
        )
        for seed in range(5)
    ]
    cal_outputs = [predict(model, x[calibration]) for model in models]
    test_outputs = [predict(model, x[test]) for model in models]
    cal_detect = np.mean([o[0] for o in cal_outputs], axis=0)
    test_detect = np.mean([o[0] for o in test_outputs], axis=0)
    cal_burden_z = np.mean([o[1] for o in cal_outputs], axis=0)
    test_burden_z = np.mean([o[1] for o in test_outputs], axis=0)
    cal_rank = np.mean([o[2] for o in cal_outputs], axis=0)
    test_rank = np.mean([o[2] for o in test_outputs], axis=0)
    cal_gate = np.mean([o[4] for o in cal_outputs], axis=0)
    test_gate = np.mean([o[4] for o in test_outputs], axis=0)

    detect_calibrator = LogisticRegression(C=0.5, max_iter=3000, random_state=SEED)
    cal_detect_rank = frozen_percentile(cal_detect, cal_detect).reshape(-1, 1)
    test_detect_rank = frozen_percentile(cal_detect, test_detect).reshape(-1, 1)
    detect_calibrator.fit(cal_detect_rank, y[calibration])
    p_detect = detect_calibrator.predict_proba(test_detect_rank)[:, 1]
    burden_log_cal = cal_burden_z * burden_std + burden_mean
    burden_log_test = test_burden_z * burden_std + burden_mean
    magnitude_calibrator = LinearRegression().fit(burden_log_cal.reshape(-1, 1), np.log1p(burden[calibration]))
    predicted_log_burden = magnitude_calibrator.predict(burden_log_test.reshape(-1, 1))
    predicted_burden = np.maximum(0.0, np.expm1(predicted_log_burden))
    burden_probability_calibrator = LogisticRegression(C=0.5, max_iter=3000, random_state=SEED)
    burden_probability_calibrator.fit(burden_log_cal.reshape(-1, 1), y[calibration])
    p_burden = burden_probability_calibrator.predict_proba(burden_log_test.reshape(-1, 1))[:, 1]
    rank_calibrator = LogisticRegression(C=0.5, max_iter=3000, random_state=SEED)
    cal_rank_fraction = frozen_percentile(cal_rank, cal_rank).reshape(-1, 1)
    test_rank_fraction = frozen_percentile(cal_rank, test_rank).reshape(-1, 1)
    rank_calibrator.fit(cal_rank_fraction, y[calibration])
    p_rank = rank_calibrator.predict_proba(test_rank_fraction)[:, 1]

    # A strongly regularized temporal stacking layer learns how much confidence
    # to place in the three multi-task heads and the original expert signals.
    # It is fitted only on the late-2023 calibration segment; 2024 remains locked.
    stack_cal = np.column_stack([
        cal_detect_rank.ravel(),
        frozen_percentile(burden_log_cal, burden_log_cal),
        cal_rank_fraction.ravel(),
        x[calibration, :len(signal_cols)],
    ])
    stack_test = np.column_stack([
        test_detect_rank.ravel(),
        frozen_percentile(burden_log_cal, burden_log_test),
        test_rank_fraction.ravel(),
        x[test, :len(signal_cols)],
    ])
    stacker = LogisticRegression(C=0.05, max_iter=4000, random_state=SEED)
    stacker.fit(stack_cal, y[calibration])
    p_stack = stacker.predict_proba(stack_test)[:, 1]

    base = frame.loc[test, ["event_id", spec.date_col, "label", "burden"]].reset_index(drop=True)
    blocks = []
    for name, probability, score in (
        ("citer_moe_detect", p_detect, test_detect),
        ("citer_moe_burden", p_burden, predicted_burden),
        ("citer_moe_rank", p_rank, test_rank),
        ("citer_moe_stack", p_stack, p_stack),
    ):
        block = base.copy()
        block["city"] = spec.city
        block["model"] = name
        block["probability"] = probability
        block["ranking_score"] = score
        blocks.append(block)
    predictions = pd.concat(blocks, ignore_index=True)
    metrics = []
    y_test = y[test]
    burden_test = burden[test]
    for name, block in predictions.groupby("model", sort=False):
        probability = block["probability"].to_numpy(float)
        score = block["ranking_score"].to_numpy(float)
        row = {
            "city": spec.city,
            "model": name,
            "n": len(block),
            "prevalence": float(y_test.mean()),
            "AP": float(average_precision_score(y_test, probability)),
            "Brier": float(brier_score_loss(y_test, probability)),
            "ECE": ece(y_test, probability),
            "Capture20": capture_at_fraction(burden_test, score),
            "NDCG20": ndcg_at_fraction(burden_test, score),
            "Spearman": float(spearmanr(burden_test, score).statistic),
        }
        if name == "citer_moe_burden":
            observed_log = np.log1p(burden_test)
            predicted_log = np.log1p(np.maximum(score, 0))
            row.update(
                MAE_log=float(mean_absolute_error(observed_log, predicted_log)),
                RMSE_log=float(mean_squared_error(observed_log, predicted_log) ** 0.5),
                R2_log=float(r2_score(observed_log, predicted_log)),
                Pearson=float(pearsonr(observed_log, predicted_log).statistic),
            )
        metrics.append(row)
    gate = pd.DataFrame(
        {
            "city": spec.city,
            "expert": spec.experts,
            "mean_gate_calibration": cal_gate.mean(axis=0),
            "mean_gate_test": test_gate.mean(axis=0),
        }
    )
    protocol = {
        "city": spec.city,
        "meta_train_n": int(meta_train.sum()),
        "calibration_n": int(calibration.sum()),
        "test_n": int(test.sum()),
        "validation_cutoff": str(spec.validation_cutoff),
        "features": all_features,
        "experts": list(spec.experts),
        "device": str(DEVICE),
    }
    identity_columns = ["event_id", spec.date_col, "split", "label", "burden"]
    for optional in ("row_hash", "duplicate_index"):
        if optional in frame.columns:
            identity_columns.append(optional)
    feature_export = frame.loc[test, identity_columns + all_features].copy()
    checkpoint = {
        "format_version": 1,
        "city": spec.city,
        "date_col": spec.date_col,
        "duration_col": spec.duration_col,
        "experts": list(spec.experts),
        "signal_cols": signal_cols,
        "context_cols": context_cols,
        "feature_names": all_features,
        "identity_columns": identity_columns,
        "input_dim": int(x.shape[1]),
        "n_experts": int(len(spec.experts)),
        "medians": medians.to_numpy(np.float32),
        "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32),
        "burden_mean": burden_mean,
        "burden_std": burden_std,
        "duration_mean": duration_mean,
        "duration_std": duration_std,
        "calibration_reference": {
            "detect": cal_detect.astype(np.float32),
            "burden_log": burden_log_cal.astype(np.float32),
            "rank": cal_rank.astype(np.float32),
        },
        "state_dicts": [
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
            for model in models
        ],
        "training_device": str(DEVICE),
        "seed_base": SEED,
    }
    calibrators = {
        "detect": detect_calibrator,
        "magnitude": magnitude_calibrator,
        "burden_probability": burden_probability_calibrator,
        "rank": rank_calibrator,
        "stack": stacker,
    }
    return predictions, pd.DataFrame(metrics), gate, protocol, checkpoint, calibrators, feature_export


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        CitySpec(
            city="Toronto",
            predictions=ROOT / "artifacts/toronto_experts/predictions.parquet",
            context=ROOT / "data/toronto_context.parquet",
            experts=(
                "native_catboost", "finetuned_minilm", "conditional_survival",
                "direct_exposure_lgbm", "exposure_lambdamart", "extra_trees",
            ),
            date_col="timestamp",
            validation_cutoff=pd.Timestamp("2023-10-01"),
            duration_col="min_delay",
            context_cols=TORONTO_CONTEXT,
        ),
        CitySpec(
            city="New York",
            predictions=ROOT / "artifacts/new_york_experts/predictions.parquet",
            context=ROOT / "data/new_york_context.parquet",
            experts=(
                "native_catboost", "finetuned_minilm", "mtdnn_survival",
                "direct_exposure_lgbm", "exposure_lambdamart", "extra_trees",
            ),
            date_col="start_time",
            validation_cutoff=pd.Timestamp("2023-12-01"),
            duration_col="duration_minutes",
            context_cols=NYC_CONTEXT,
        ),
    ]
    prediction_blocks, metric_blocks, gate_blocks, protocols = [], [], [], []
    for spec in specs:
        print(f"training {spec.city}", flush=True)
        predictions, metrics, gates, protocol, checkpoint, calibrators, feature_export = run_city(spec)
        slug = spec.city.lower().replace(" ", "_")
        checkpoint_dir = OUT / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{slug}.pt"
        calibrator_path = checkpoint_dir / f"{slug}_calibrators.joblib"
        feature_path = checkpoint_dir / f"{slug}_features.parquet"
        torch.save(checkpoint, checkpoint_path)
        joblib.dump(calibrators, calibrator_path, compress=3)
        feature_export.to_parquet(feature_path, index=False)
        protocol["checkpoint"] = str(checkpoint_path)
        protocol["calibrators"] = str(calibrator_path)
        protocol["feature_export"] = str(feature_path)
        prediction_blocks.append(predictions)
        metric_blocks.append(metrics)
        gate_blocks.append(gates)
        protocols.append(protocol)
        print(metrics.to_string(index=False), flush=True)
        print(f"saved {checkpoint_path}, {calibrator_path}, and {feature_path}", flush=True)
    pd.concat(prediction_blocks, ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)
    pd.concat(metric_blocks, ignore_index=True).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(gate_blocks, ignore_index=True).to_csv(OUT / "gate_weights.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps(protocols, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
