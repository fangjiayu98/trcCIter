#!/usr/bin/env python3
"""Demand-conditioned monotone survival model for potential exposure triage."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score


ROOT = Path("/root/autodl-tmp/rail2road")
HERE = ROOT / "experiments_v2"
OUT = ROOT / "results/trc_v2/conditional_survival"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import pipeline as bp  # noqa: E402
import toronto_baselines as base  # noqa: E402


SEED = 20260915


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy(float)


def frozen_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * (len(reference) + 1.0))


def expand_landmarks(
    x: np.ndarray,
    duration: np.ndarray,
    event_threshold: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_thresholds = np.concatenate(
        [np.broadcast_to(grid[None, :], (len(x), len(grid))), event_threshold[:, None]],
        axis=1,
    )
    threshold_flat = all_thresholds.reshape(-1).astype(np.float32)
    x_expanded = np.empty((len(x) * all_thresholds.shape[1], x.shape[1] + 1), dtype=np.float32)
    x_expanded[:, :-1] = np.repeat(x, all_thresholds.shape[1], axis=0)
    x_expanded[:, -1] = np.log1p(threshold_flat)
    y_expanded = (np.repeat(duration, all_thresholds.shape[1]) > threshold_flat).astype(np.int8)
    weights = np.ones_like(threshold_flat, dtype=np.float32)
    weights.reshape(len(x), -1)[:, -1] = 4.0
    return x_expanded, y_expanded, weights


def survival_at(model: LGBMClassifier, x: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    if np.isscalar(threshold):
        threshold = np.full(len(x), float(threshold), dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)
    query = np.empty((len(x), x.shape[1] + 1), dtype=np.float32)
    query[:, :-1] = x
    query[:, -1] = np.log1p(np.clip(threshold, 0.0, 300.0))
    return model.predict_proba(query)[:, 1]


def expected_duration(model: LGBMClassifier, x: np.ndarray) -> np.ndarray:
    grid = np.asarray(
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180, 240, 300],
        dtype=np.float32,
    )
    survival = np.column_stack([np.ones(len(x))] + [survival_at(model, x, t) for t in grid[1:]])
    survival = np.minimum.accumulate(survival, axis=1)
    return np.trapezoid(survival, grid, axis=1)


def main() -> None:
    np.random.seed(SEED)
    data, fused, _, _, meta = base.prepare()
    train = data.year.le(2022).to_numpy()
    val = data.year.eq(2023).to_numpy()
    test = data.year.eq(2024).to_numpy()
    duration = data.min_delay.to_numpy(float)
    rate = np.maximum(data.demand_rate.to_numpy(float), 1e-4)
    burden = data.exposed_demand.to_numpy(float)
    label = data.high_burden.to_numpy(int)
    burden_threshold = float(meta["thresholds"]["high_burden"])
    event_threshold = np.clip(burden_threshold / rate, 0.1, 300.0)

    demand_position = 48 + list(bp.NUMERIC_COLUMNS).index("demand_rank")
    mechanism_columns = [i for i in range(fused.shape[1]) if i != demand_position]
    mechanism = fused[:, mechanism_columns].astype(np.float32)
    log_rate = np.log1p(rate).astype(np.float32)
    log_required = np.log1p(event_threshold).astype(np.float32)

    rows = []
    predictions = []
    score_store = {}

    def record(name: str, val_score: np.ndarray, test_score: np.ndarray, elapsed: float, probabilities: bool = True) -> None:
        if probabilities:
            val_prob = np.clip(val_score, 1e-6, 1 - 1e-6)
            test_prob = base.platt(val_score, label[val], test_score)
        else:
            val_prob = base.platt(val_score, label[val], val_score)
            test_prob = base.platt(val_score, label[val], test_score)
        rows.append(base.evaluate("validation_2023", "high_burden", name, label[val], val_prob, burden[val], elapsed))
        rows.append(base.evaluate("test_2024", "high_burden", name, label[test], test_prob, burden[test], elapsed))
        score_store[name] = (np.asarray(val_score), np.asarray(test_score))
        for split_name, mask, probability, score in [
            ("validation_2023", val, val_prob, val_score),
            ("test_2024", test, test_prob, test_score),
        ]:
            predictions.append(pd.DataFrame({
                "event_id": data.loc[mask, "event_id"].to_numpy(),
                "split": split_name,
                "target": "high_burden",
                "model": name,
                "label": label[mask],
                "probability": probability,
                "ranking_score": score,
                "burden": burden[mask],
                "required_duration": event_threshold[mask],
            }))

    print("fit fair threshold-aware direct LightGBM", flush=True)
    direct_x = np.column_stack([fused, log_rate, log_required]).astype(np.float32)
    direct_lgbm = LGBMClassifier(
        objective="binary", n_estimators=1600, learning_rate=0.025, num_leaves=47,
        min_child_samples=35, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        class_weight="balanced", n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    start = time.time()
    direct_lgbm.fit(
        direct_x[train], label[train], eval_set=[(direct_x[val], label[val])],
        eval_metric="average_precision", callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    direct_lgbm_val = direct_lgbm.predict_proba(direct_x[val])[:, 1]
    direct_lgbm_test = direct_lgbm.predict_proba(direct_x[test])[:, 1]
    record("threshold_aware_lightgbm", direct_lgbm_val, direct_lgbm_test, time.time() - start)

    print("fit fair threshold-aware native CatBoost", flush=True)
    cat_cols = [c for c in ["code", "station", "line", "text"] if c in data.columns]
    native = data[list(bp.NUMERIC_COLUMNS) + cat_cols].copy()
    native["log_demand_rate"] = log_rate
    native["log_required_duration"] = log_required
    for c in cat_cols:
        native[c] = native[c].fillna("UNK").astype(str)
    native[list(bp.NUMERIC_COLUMNS)] = native[list(bp.NUMERIC_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    direct_cat = CatBoostClassifier(
        iterations=1800, learning_rate=0.03, depth=9, loss_function="Logloss",
        eval_metric="PRAUC", auto_class_weights="Balanced", random_seed=SEED,
        verbose=False, allow_writing_files=False, thread_count=-1,
    )
    start = time.time()
    direct_cat.fit(
        native.loc[train], label[train], cat_features=cat_cols,
        eval_set=(native.loc[val], label[val]), early_stopping_rounds=150, verbose=False,
    )
    direct_cat_val = direct_cat.predict_proba(native.loc[val])[:, 1]
    direct_cat_test = direct_cat.predict_proba(native.loc[test])[:, 1]
    record("threshold_aware_catboost", direct_cat_val, direct_cat_test, time.time() - start)

    print("expand conditional-survival landmarks", flush=True)
    landmark_grid = np.asarray([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 120], dtype=np.float32)
    x_train, y_train, w_train = expand_landmarks(
        mechanism[train], duration[train], event_threshold[train], landmark_grid
    )
    x_val, y_val, w_val = expand_landmarks(
        mechanism[val], duration[val], event_threshold[val], landmark_grid
    )
    print(f"expanded train={len(y_train)} validation={len(y_val)} positive={y_train.mean():.4f}", flush=True)
    monotone = [0] * mechanism.shape[1] + [-1]
    survival = LGBMClassifier(
        objective="binary", n_estimators=1800, learning_rate=0.025, num_leaves=63,
        min_child_samples=100, max_bin=127, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, n_jobs=-1, random_state=SEED, verbosity=-1,
        monotone_constraints=monotone,
    )
    start = time.time()
    survival.fit(
        x_train, y_train, sample_weight=w_train,
        eval_set=[(x_val, y_val)], eval_sample_weight=[w_val], eval_metric="binary_logloss",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    fit_seconds = time.time() - start
    del x_train, y_train, w_train, x_val, y_val, w_val

    val_exceedance = survival_at(survival, mechanism[val], event_threshold[val])
    test_exceedance = survival_at(survival, mechanism[test], event_threshold[test])
    record("conditional_survival_exceedance", val_exceedance, test_exceedance, fit_seconds)

    print("integrate conditional survival for expected exposure", flush=True)
    val_expected_duration = expected_duration(survival, mechanism[val])
    test_expected_duration = expected_duration(survival, mechanism[test])
    val_expected_burden = val_expected_duration * rate[val]
    test_expected_burden = test_expected_duration * rate[test]
    record("conditional_expected_exposure", val_expected_burden, test_expected_burden, fit_seconds, probabilities=False)

    print("select one locked exceedance-exposure decision score", flush=True)
    val_components = np.column_stack([
        rank01(val_exceedance), rank01(val_expected_burden), rank01(direct_cat_val), rank01(direct_lgbm_val)
    ])
    test_components = np.column_stack([
        frozen_percentile(val_exceedance, test_exceedance),
        frozen_percentile(val_expected_burden, test_expected_burden),
        frozen_percentile(direct_cat_val, direct_cat_test),
        frozen_percentile(direct_lgbm_val, direct_lgbm_test),
    ])
    component_names = [
        "conditional_survival_exceedance", "conditional_expected_exposure",
        "threshold_aware_catboost", "threshold_aware_lightgbm",
    ]
    rng = np.random.default_rng(SEED)
    candidates = [np.eye(len(component_names))[i] for i in range(len(component_names))]
    candidates.extend(rng.dirichlet(np.ones(len(component_names)), size=6000))
    best = None
    for weights in candidates:
        score = val_components @ weights
        ap = average_precision_score(label[val], score)
        _, capture, _ = base.top_fraction_metrics(label[val], score, burden[val])
        objective = 0.70 * ap + 0.30 * capture
        if best is None or objective > best[0]:
            best = (objective, weights.copy(), ap, capture)
    final_val = val_components @ best[1]
    final_test = test_components @ best[1]
    record("citer_v3_conditional_survival", final_val, final_test, 0.0, probabilities=False)

    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame({"component": component_names, "weight": best[1]}).to_csv(OUT / "weights.csv", index=False)
    import joblib
    joblib.dump(
        {"survival": survival, "direct_lgbm": direct_lgbm, "direct_catboost": direct_cat},
        OUT / "models.joblib", compress=3,
    )
    summary = {
        "burden_threshold": burden_threshold,
        "mechanism_feature_count": int(mechanism.shape[1]),
        "landmark_grid": landmark_grid.tolist(),
        "validation_objective": float(best[0]),
        "validation_ap": float(best[2]),
        "validation_capture20": float(best[3]),
        "weights": dict(zip(component_names, map(float, best[1]))),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).sort_values(["split", "ap"], ascending=[True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
