#!/usr/bin/env python3
"""Full same-setting benchmark on the redesigned TTC exposure target."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lifelines import WeibullAFTFitter
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor, early_stopping, log_evaluation
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments_v2"
OUT = ROOT / "artifacts/toronto_experts"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import pipeline as bp  # noqa: E402
import conditional_survival as cs  # noqa: E402
import semantic_experts as stm  # noqa: E402
import toronto_baselines as base  # noqa: E402


SEED = 20260915
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FILE = ROOT / "data/toronto_context.parquet"


def rank01(values):
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy(float)


def frozen_percentile(reference, values):
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * (len(reference) + 1.0))


def build_features(data: pd.DataFrame, train: np.ndarray):
    base_columns = [c for c in bp.NUMERIC_COLUMNS if c != "demand_rank"]
    raw_numeric = data[base_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    engineered = np.column_stack([
        np.log1p(np.maximum(data.daily_station_usage_v2.to_numpy(float), 0.0)),
        np.log1p(np.maximum(data.demand_rate_v2.to_numpy(float), 0.0)),
        np.log1p(np.clip(data.required_duration_v2.to_numpy(float), 0.0, 1440.0)),
    ]).astype(np.float32)
    all_numeric_raw = np.concatenate([raw_numeric, engineered], axis=1)
    scaler = StandardScaler()
    all_numeric = np.empty_like(all_numeric_raw)
    all_numeric[train] = scaler.fit_transform(all_numeric_raw[train]).astype(np.float32)
    all_numeric[~train] = scaler.transform(all_numeric_raw[~train]).astype(np.float32)

    mechanism_scaler = StandardScaler()
    mechanism_numeric = np.empty_like(raw_numeric)
    mechanism_numeric[train] = mechanism_scaler.fit_transform(raw_numeric[train]).astype(np.float32)
    mechanism_numeric[~train] = mechanism_scaler.transform(raw_numeric[~train]).astype(np.float32)

    texts = data.text.fillna("").astype(str).to_numpy()
    unique_text, inverse = np.unique(texts, return_inverse=True)
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda", local_files_only=True
    )
    unique_embedding = encoder.encode(
        unique_text.tolist(), batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    embedding = unique_embedding[inverse]
    pca = PCA(n_components=48, random_state=SEED)
    semantic = np.empty((len(data), 48), dtype=np.float32)
    semantic[train] = pca.fit_transform(embedding[train]).astype(np.float32)
    semantic[~train] = pca.transform(embedding[~train]).astype(np.float32)
    fused = np.concatenate([semantic, all_numeric], axis=1).astype(np.float32)
    mechanism = np.concatenate([semantic, mechanism_numeric], axis=1).astype(np.float32)
    state = {
        "base_columns": base_columns,
        "engineered_columns": ["log_daily_usage_v2", "log_demand_rate_v2", "log_required_duration_v2"],
        "numeric_scaler": scaler,
        "mechanism_scaler": mechanism_scaler,
        "pca": pca,
        "semantic_dim": 48,
    }
    return fused, mechanism, all_numeric, texts.tolist(), state


def mtdnn_survival_at(model, thresholds, x, event_threshold):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    survival = np.minimum.accumulate(np.concatenate(outputs), axis=1)
    result = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        result[i] = np.interp(event_threshold[i], thresholds, survival[i], left=1.0, right=survival[i, -1])
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data = pd.read_parquet(TARGET_FILE)
    data = data[data.ridership_matched.eq(1)].reset_index(drop=True)
    train = data.year.le(2022).to_numpy()
    val = data.year.eq(2023).to_numpy()
    test = data.year.eq(2024).to_numpy()
    fused, mechanism, numeric, texts, feature_state = build_features(data, train)
    label = data.high_exposure_v2.to_numpy(int)
    burden = data.potential_exposure_v2.to_numpy(float)
    duration = data.min_delay.to_numpy(float)
    required = np.clip(data.required_duration_v2.to_numpy(float), 0.1, 1440.0)
    start_rate = np.maximum(data.demand_rate_v2.to_numpy(float), 1e-6)
    rows = []
    prediction_rows = []
    score_store = {}
    fitted = {"feature_state": feature_state}

    stable = data[["event_id", "timestamp", "station_key", "line_std", "code", "min_delay"]].copy()
    stable["burden_key"] = np.round(burden, 8)
    stable["row_hash"] = pd.util.hash_pandas_object(stable, index=False).astype(str)
    stable["duplicate_index"] = stable.groupby("row_hash", sort=False).cumcount()

    def record(name, val_score, test_score, elapsed, probabilities=True):
        val_score = np.asarray(val_score, dtype=float)
        test_score = np.asarray(test_score, dtype=float)
        if probabilities:
            val_prob = np.clip(val_score, 1e-6, 1 - 1e-6)
            test_prob = base.platt(val_score, label[val], test_score)
        else:
            val_prob = base.platt(val_score, label[val], val_score)
            test_prob = base.platt(val_score, label[val], test_score)
        rows.append(base.evaluate("validation_2023", "high_exposure_v2", name, label[val], val_prob, burden[val], elapsed))
        rows.append(base.evaluate("test_2024", "high_exposure_v2", name, label[test], test_prob, burden[test], elapsed))
        score_store[name] = (val_score, test_score)
        for split_name, mask, probability, score in [
            ("validation_2023", val, val_prob, val_score),
            ("test_2024", test, test_prob, test_score),
        ]:
            part = stable.loc[mask, ["event_id", "timestamp", "row_hash", "duplicate_index"]].reset_index(drop=True)
            part["split"] = split_name
            part["target"] = "high_exposure_v2"
            part["model"] = name
            part["label"] = label[mask]
            part["probability"] = probability
            part["ranking_score"] = score
            part["burden"] = burden[mask]
            prediction_rows.append(part)
        pd.DataFrame(rows).to_csv(OUT / "metrics_partial.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "predictions_partial.parquet", index=False)

    prior = float(label[train].mean())
    record("prior", np.full(val.sum(), prior), np.full(test.sum(), prior), 0.0)

    for name, model in base.candidate_models(SEED).items():
        print(f"fit general model={name}", flush=True)
        val_score, test_score, elapsed = base.fit_scores(
            name, model, fused[train], label[train], fused[val], label[val], fused[test]
        )
        record(name, val_score, test_score, elapsed)
        fitted[name] = model

    print("fit native categorical CatBoost", flush=True)
    cat_cols = [c for c in ["code", "station", "line", "text"] if c in data.columns]
    native_columns = [c for c in bp.NUMERIC_COLUMNS if c != "demand_rank"]
    native = data[native_columns + cat_cols].copy()
    native["log_daily_usage_v2"] = np.log1p(np.maximum(data.daily_station_usage_v2, 0.0))
    native["log_demand_rate_v2"] = np.log1p(np.maximum(data.demand_rate_v2, 0.0))
    native["log_required_duration_v2"] = np.log1p(required)
    for c in cat_cols:
        native[c] = native[c].fillna("UNK").astype(str)
    native[native_columns] = native[native_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    native_cat = CatBoostClassifier(
        iterations=1800, learning_rate=0.03, depth=9, loss_function="Logloss",
        eval_metric="PRAUC", auto_class_weights="Balanced", random_seed=SEED,
        verbose=False, allow_writing_files=False, thread_count=-1,
    )
    start = time.time()
    native_cat.fit(
        native.loc[train], label[train], cat_features=cat_cols,
        eval_set=(native.loc[val], label[val]), early_stopping_rounds=150, verbose=False,
    )
    record(
        "native_catboost", native_cat.predict_proba(native.loc[val])[:, 1],
        native_cat.predict_proba(native.loc[test])[:, 1], time.time() - start,
    )
    fitted["native_catboost"] = native_cat

    print("fit BTM-LightGBM duration baseline", flush=True)
    topics, topic_state = stm.build_btm_topics(texts)
    topic_mechanism = np.concatenate([topics, mechanism[:, 48:]], axis=1).astype(np.float32)
    btm_model = LGBMRegressor(
        objective="huber", n_estimators=1200, learning_rate=0.025, num_leaves=31,
        min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    start = time.time()
    btm_model.fit(
        topic_mechanism[train], np.log1p(duration[train]),
        eval_set=[(topic_mechanism[val], np.log1p(duration[val]))],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    btm_val_duration = np.clip(np.expm1(btm_model.predict(topic_mechanism[val])), 0, 1440)
    btm_test_duration = np.clip(np.expm1(btm_model.predict(topic_mechanism[test])), 0, 1440)
    record("btm_lgbm_duration", btm_val_duration - required[val], btm_test_duration - required[test], time.time() - start, False)
    fitted["btm_lgbm_duration"] = {"model": btm_model, "topic_state": topic_state}

    print("fit Weibull AFT duration baseline", flush=True)
    aft_names = [f"topic_{i}" for i in range(topics.shape[1])] + [f"x_{i}" for i in range(mechanism.shape[1] - 48)]
    aft_train = pd.DataFrame(topic_mechanism[train], columns=aft_names)
    aft_train["duration"] = np.maximum(duration[train], 0.25)
    aft_train["observed"] = 1
    aft = WeibullAFTFitter(penalizer=0.05)
    start = time.time()
    aft.fit(aft_train, duration_col="duration", event_col="observed", show_progress=False)
    aft_val_duration = np.clip(np.asarray(aft.predict_median(pd.DataFrame(topic_mechanism[val], columns=aft_names))), 0, 1440)
    aft_test_duration = np.clip(np.asarray(aft.predict_median(pd.DataFrame(topic_mechanism[test], columns=aft_names))), 0, 1440)
    record("weibull_aft", aft_val_duration - required[val], aft_test_duration - required[test], time.time() - start, False)
    fitted["weibull_aft"] = aft

    print("fit MTDNN survival baseline", flush=True)
    start = time.time()
    mtdnn, thresholds = stm.train_survival_mtdnn(mechanism[train], duration[train], mechanism[val], duration[val])
    val_mtdnn = mtdnn_survival_at(mtdnn, thresholds, mechanism[val], required[val])
    test_mtdnn = mtdnn_survival_at(mtdnn, thresholds, mechanism[test], required[test])
    record("mtdnn_survival", val_mtdnn, test_mtdnn, time.time() - start)
    fitted["mtdnn_survival"] = {"state": mtdnn.cpu().state_dict(), "thresholds": thresholds}
    mtdnn.to(DEVICE)

    print("fit fine-tuned MiniLM", flush=True)
    multitask_labels = np.column_stack([data.severe_delay.to_numpy(int), label])
    start = time.time()
    minilm, token_state, _ = stm.train_finetuned_minilm(texts, numeric, train, val, multitask_labels)
    val_minilm = stm.minilm_predict(minilm, token_state, numeric, val)[:, 1]
    test_minilm = stm.minilm_predict(minilm, token_state, numeric, test)[:, 1]
    record("finetuned_minilm", val_minilm, test_minilm, time.time() - start)
    fitted["finetuned_minilm"] = minilm.cpu().state_dict()

    print("fit monotone conditional-survival model", flush=True)
    landmark_grid = np.asarray([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 120, 180, 240], dtype=np.float32)
    x_train, y_train, w_train = cs.expand_landmarks(mechanism[train], duration[train], required[train], landmark_grid)
    x_val, y_val, w_val = cs.expand_landmarks(mechanism[val], duration[val], required[val], landmark_grid)
    survival = LGBMClassifier(
        objective="binary", n_estimators=1800, learning_rate=0.025, num_leaves=63,
        min_child_samples=100, max_bin=127, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, n_jobs=-1, random_state=SEED, verbosity=-1,
        monotone_constraints=[0] * mechanism.shape[1] + [-1],
    )
    start = time.time()
    survival.fit(
        x_train, y_train, sample_weight=w_train,
        eval_set=[(x_val, y_val)], eval_sample_weight=[w_val], eval_metric="binary_logloss",
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    survival_seconds = time.time() - start
    del x_train, y_train, w_train, x_val, y_val, w_val
    val_survival = cs.survival_at(survival, mechanism[val], required[val])
    test_survival = cs.survival_at(survival, mechanism[test], required[test])
    record("conditional_survival", val_survival, test_survival, survival_seconds)
    fitted["conditional_survival"] = survival

    print("fit continuous exposure regression", flush=True)
    burden_reg = LGBMRegressor(
        objective="huber", n_estimators=1400, learning_rate=0.025, num_leaves=47,
        min_child_samples=35, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    start = time.time()
    burden_reg.fit(
        fused[train], np.log1p(burden[train]),
        eval_set=[(fused[val], np.log1p(burden[val]))],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    val_burden_reg = np.expm1(burden_reg.predict(fused[val]))
    test_burden_reg = np.expm1(burden_reg.predict(fused[test]))
    record("direct_exposure_lgbm", val_burden_reg, test_burden_reg, time.time() - start, False)
    fitted["direct_exposure_lgbm"] = burden_reg

    print("fit exposure-aware LambdaMART", flush=True)
    dates = data.timestamp.dt.strftime("%Y-%m-%d").to_numpy()
    train_order = np.argsort(dates[train], kind="stable")
    val_order = np.argsort(dates[val], kind="stable")
    _, train_group = np.unique(dates[train][train_order], return_counts=True)
    _, val_group = np.unique(dates[val][val_order], return_counts=True)
    train_log_burden = np.log1p(burden[train])
    cuts = np.unique(np.quantile(train_log_burden[train_log_burden > 0], np.linspace(0, 1, 16)[1:-1]))
    rel_train = np.digitize(train_log_burden, cuts).astype(int)
    rel_val = np.digitize(np.log1p(burden[val]), cuts).astype(int)
    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=1400, learning_rate=0.025,
        num_leaves=31, min_child_samples=40, reg_lambda=2.0, n_jobs=-1,
        random_state=SEED, verbosity=-1,
    )
    start = time.time()
    ranker.fit(
        fused[train][train_order], rel_train[train_order], group=train_group,
        eval_set=[(fused[val][val_order], rel_val[val_order])], eval_group=[val_group], eval_at=[20],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    val_rank = ranker.predict(fused[val])
    test_rank = ranker.predict(fused[test])
    record("exposure_lambdamart", val_rank, test_rank, time.time() - start, False)
    fitted["exposure_lambdamart"] = ranker

    print("select validation-locked CITER risk score", flush=True)
    components = ["native_catboost", "finetuned_minilm", "conditional_survival", "direct_exposure_lgbm", "exposure_lambdamart"]
    val_matrix = np.column_stack([rank01(score_store[name][0]) for name in components])
    test_matrix = np.column_stack([
        frozen_percentile(score_store[name][0], score_store[name][1]) for name in components
    ])
    rng = np.random.default_rng(SEED)
    candidates = [np.eye(len(components))[i] for i in range(len(components))]
    candidates.extend(rng.dirichlet(np.ones(len(components)), size=6000))
    best = None
    for weights in candidates:
        score = val_matrix @ weights
        ap = average_precision_score(label[val], score)
        if best is None or ap > best[0]:
            best = (ap, weights.copy())
    record("citer_dynamic_exposure", val_matrix @ best[1], test_matrix @ best[1], 0.0, False)
    pd.DataFrame({"component": components, "weight": best[1]}).to_csv(OUT / "citer_weights.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)
    joblib.dump(fitted, OUT / "models.joblib", compress=3)

    curve_rows = []
    for name, (val_score, test_score) in score_store.items():
        for split_name, y, b, score in [
            ("validation_2023", label[val], burden[val], val_score),
            ("test_2024", label[test], burden[test], test_score),
        ]:
            for fraction in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
                recall, capture, ndcg = base.top_fraction_metrics(y, score, b, fraction)
                curve_rows.append({
                    "split": split_name, "model": name, "budget_fraction": fraction,
                    "high_risk_recall": recall, "exposure_capture": capture, "exposure_ndcg": ndcg,
                })
    pd.DataFrame(curve_rows).to_csv(OUT / "budget_curves.csv", index=False)
    summary = {
        "rows": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum())},
        "prevalence": {"train": float(label[train].mean()), "validation": float(label[val].mean()), "test": float(label[test].mean())},
        "fused_dim": int(fused.shape[1]),
        "mechanism_dim": int(mechanism.shape[1]),
        "citer_validation_ap": float(best[0]),
        "citer_weights": dict(zip(components, map(float, best[1]))),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.sort_values(["split", "ap"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nBUDGET LEADERS", flush=True)
    curves = pd.DataFrame(curve_rows)
    print(
        curves[curves.split.eq("test_2024")]
        .sort_values(["budget_fraction", "exposure_capture"], ascending=[True, False])
        .groupby("budget_fraction", as_index=False).first().to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
