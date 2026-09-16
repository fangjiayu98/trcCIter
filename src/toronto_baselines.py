#!/usr/bin/env python3
"""Same-setting Toronto benchmark for disruption persistence and demand exposure.

The script deliberately separates model comparison from target redesign. It uses
the current locked target construction, selects every learner on Toronto 2023,
and evaluates the selected specification on Toronto 2024. All methods receive
the same event rows, split, onset information, labels, and evaluation metrics.
"""

from __future__ import annotations

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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor, early_stopping, log_evaluation
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("/root/autodl-tmp/rail2road")
HERE = ROOT / "experiments_v2"
OUT = ROOT / "results" / "trc_v2" / "baseline_screen"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import pipeline as bp  # noqa: E402

SEED = 20260915
TARGETS = {
    "severe_delay": "min_delay",
    "high_burden": "exposed_demand",
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = ids == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def top_fraction_metrics(
    y: np.ndarray,
    p: np.ndarray,
    burden: np.ndarray,
    fraction: float = 0.20,
) -> tuple[float, float, float]:
    n_top = max(1, int(np.ceil(len(y) * fraction)))
    order = np.argsort(-p, kind="stable")
    selected = order[:n_top]
    recall = float(y[selected].sum() / max(1.0, y.sum()))
    capture = float(burden[selected].sum() / max(1e-12, burden.sum()))
    gains = np.maximum(burden[order], 0.0)
    discounts = 1.0 / np.log2(np.arange(2, n_top + 2))
    dcg = float(np.sum(gains[:n_top] * discounts))
    ideal = np.sort(np.maximum(burden, 0.0))[::-1]
    idcg = float(np.sum(ideal[:n_top] * discounts))
    ndcg = dcg / max(idcg, 1e-12)
    return recall, capture, ndcg


def platt(val_score: np.ndarray, y_val: np.ndarray, test_score: np.ndarray) -> np.ndarray:
    val_score = np.asarray(val_score, dtype=float).reshape(-1, 1)
    test_score = np.asarray(test_score, dtype=float).reshape(-1, 1)
    if np.nanstd(val_score) < 1e-10:
        return np.full(len(test_score), float(np.mean(y_val)))
    calibrator = LogisticRegression(C=1e4, max_iter=1000, random_state=SEED)
    calibrator.fit(val_score, y_val)
    return calibrator.predict_proba(test_score)[:, 1]


def evaluate(split: str, target: str, model: str, y: np.ndarray, p: np.ndarray, burden: np.ndarray, elapsed: float) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    recall20, capture20, ndcg20 = top_fraction_metrics(y, p, burden)
    return {
        "split": split,
        "target": target,
        "model": model,
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece10": ece(y, p),
        "top20_recall": recall20,
        "top20_burden_capture": capture20,
        "ndcg20": ndcg20,
        "fit_seconds": float(elapsed),
    }


def class_weights(y: np.ndarray) -> np.ndarray:
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg)).astype(np.float32)


def prepare() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict]:
    data_dir = ROOT / "raw" / "toronto"
    registry_frame, registry = bp.build_station_registry(data_dir)
    code_desc = bp.parse_code_descriptions(data_dir / "ttc-subway-delay-codes.xlsx")
    data, thresholds = bp.prepare_toronto(
        data_dir,
        set(range(2014, 2025)),
        registry,
        code_desc,
        thresholds=None,
    )
    data = data.sort_values(["timestamp", "event_id"]).reset_index(drop=True)

    train = data.year.le(2022).to_numpy()
    val = data.year.eq(2023).to_numpy()
    test = data.year.eq(2024).to_numpy()

    numeric = data[bp.NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    scaler = StandardScaler()
    numeric_scaled = np.empty_like(numeric, dtype=np.float32)
    numeric_scaled[train] = scaler.fit_transform(numeric[train]).astype(np.float32)
    numeric_scaled[~train] = scaler.transform(numeric[~train]).astype(np.float32)

    texts = data.text.fillna("").astype(str).to_numpy()
    unique_text, inverse = np.unique(texts, return_inverse=True)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda", local_files_only=True)
    unique_embedding = encoder.encode(
        unique_text.tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding = unique_embedding[inverse]
    pca = PCA(n_components=48, random_state=SEED)
    semantic = np.empty((len(data), 48), dtype=np.float32)
    semantic[train] = pca.fit_transform(embedding[train]).astype(np.float32)
    semantic[~train] = pca.transform(embedding[~train]).astype(np.float32)
    fused = np.concatenate([semantic, numeric_scaled], axis=1).astype(np.float32)

    meta = {
        "rows": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum())},
        "thresholds": thresholds,
        "numeric_columns": list(bp.NUMERIC_COLUMNS),
        "semantic_dim": 48,
        "fused_dim": int(fused.shape[1]),
        "unique_texts": int(len(unique_text)),
    }
    return data, fused, semantic, texts.tolist(), meta


def candidate_models(seed: int = SEED) -> dict:
    models = {
        "fused_logistic": LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=20,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=900,
            learning_rate=0.04,
            depth=8,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=2e-4,
            learning_rate_init=8e-4,
            batch_size=1024,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=900,
            learning_rate=0.035,
            max_depth=7,
            min_child_weight=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            device="cuda",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"xgboost unavailable: {exc}", flush=True)
    return models


def fit_scores(model_name: str, model, x_train, y_train, x_val, y_val, x_test):
    weights = class_weights(y_train)
    start = time.time()
    if model_name == "lightgbm":
        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
            eval_set=[(x_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )
    elif model_name == "catboost":
        model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=80, verbose=False)
    elif model_name == "hist_gradient_boosting":
        model.fit(x_train, y_train, sample_weight=weights)
    elif model_name == "mlp":
        model.fit(x_train, y_train, sample_weight=weights)
    elif model_name == "xgboost":
        model.set_params(scale_pos_weight=float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train)
    elapsed = time.time() - start
    val_score = model.predict_proba(x_val)[:, 1]
    test_score = model.predict_proba(x_test)[:, 1]
    return val_score, test_score, elapsed


def ranker_scores(x_train, y_train, date_train, x_val, y_val, date_val, x_test):
    train_order = np.argsort(date_train, kind="stable")
    val_order = np.argsort(date_val, kind="stable")
    train_dates = date_train[train_order]
    val_dates = date_val[val_order]
    train_group = pd.Series(train_dates).value_counts(sort=False).to_numpy()
    val_group = pd.Series(val_dates).value_counts(sort=False).to_numpy()
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    start = time.time()
    ranker.fit(
        x_train[train_order],
        y_train[train_order],
        group=train_group,
        eval_set=[(x_val[val_order], y_val[val_order])],
        eval_group=[val_group],
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    return ranker.predict(x_val), ranker.predict(x_test), time.time() - start, ranker


def factorized_scores(data, fused, train, val, test):
    demand_positions = [bp.NUMERIC_COLUMNS.index(c) for c in ["demand_rank", "degree_rank", "route_rank", "transfer"]]
    structured_offset = 48
    excluded = {structured_offset + i for i in demand_positions}
    mechanism_columns = [i for i in range(fused.shape[1]) if i not in excluded]
    regressor = LGBMRegressor(
        objective="huber",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    start = time.time()
    y_duration = np.log1p(data.min_delay.to_numpy(float))
    regressor.fit(
        fused[train][:, mechanism_columns],
        y_duration[train],
        eval_set=[(fused[val][:, mechanism_columns], y_duration[val])],
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    val_delay = np.maximum(0.0, np.expm1(regressor.predict(fused[val][:, mechanism_columns])))
    test_delay = np.maximum(0.0, np.expm1(regressor.predict(fused[test][:, mechanism_columns])))
    val_rate = data.loc[val, "demand_rate"].to_numpy(float)
    test_rate = data.loc[test, "demand_rate"].to_numpy(float)
    return {
        "severe_delay": (val_delay, test_delay),
        "high_burden": (val_delay * val_rate, test_delay * test_rate),
    }, time.time() - start, regressor


def main() -> None:
    np.random.seed(SEED)
    data, fused, semantic, texts, meta = prepare()
    train = data.year.le(2022).to_numpy()
    val = data.year.eq(2023).to_numpy()
    test = data.year.eq(2024).to_numpy()
    dates = data.timestamp.dt.strftime("%Y-%m-%d").to_numpy()

    joblib.dump(
        {"meta": meta, "fused_shape": fused.shape},
        OUT / "preparation_state.joblib",
        compress=3,
    )
    (OUT / "data_summary.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    rows = []
    prediction_rows = []
    fitted = {}
    factor_scores, factor_seconds, factor_model = factorized_scores(data, fused, train, val, test)
    fitted["factorized_citer"] = factor_model

    for target, burden_col in TARGETS.items():
        y = data[target].to_numpy(int)
        burden = data[burden_col].to_numpy(float)
        y_train, y_val, y_test = y[train], y[val], y[test]
        b_val, b_test = burden[val], burden[test]
        prior = float(y_train.mean())
        for split_name, ys, bs in [("validation_2023", y_val, b_val), ("test_2024", y_test, b_test)]:
            rows.append(evaluate(split_name, target, "prior", ys, np.full(len(ys), prior), bs, 0.0))

        models = candidate_models(SEED)
        for name, model in models.items():
            print(f"fit target={target} model={name}", flush=True)
            val_score, test_score, elapsed = fit_scores(
                name,
                model,
                fused[train],
                y_train,
                fused[val],
                y_val,
                fused[test],
            )
            calibrated_test = platt(val_score, y_val, test_score)
            rows.append(evaluate("validation_2023", target, name, y_val, val_score, b_val, elapsed))
            rows.append(evaluate("test_2024", target, name, y_test, calibrated_test, b_test, elapsed))
            prediction_rows.append(pd.DataFrame({
                "event_id": data.loc[test, "event_id"].to_numpy(),
                "target": target,
                "model": name,
                "label": y_test,
                "probability": calibrated_test,
                "burden": b_test,
            }))
            fitted[f"{target}:{name}"] = model

        print(f"fit target={target} model=lambdamart", flush=True)
        val_score, test_score, elapsed, ranker = ranker_scores(
            fused[train], y_train, dates[train], fused[val], y_val, dates[val], fused[test]
        )
        calibrated_test = platt(val_score, y_val, test_score)
        rows.append(evaluate("validation_2023", target, "lambdamart", y_val, expit(val_score), b_val, elapsed))
        rows.append(evaluate("test_2024", target, "lambdamart", y_test, calibrated_test, b_test, elapsed))
        prediction_rows.append(pd.DataFrame({
            "event_id": data.loc[test, "event_id"].to_numpy(),
            "target": target,
            "model": "lambdamart",
            "label": y_test,
            "probability": calibrated_test,
            "burden": b_test,
        }))
        fitted[f"{target}:lambdamart"] = ranker

        val_score, test_score = factor_scores[target]
        calibrated_test = platt(val_score, y_val, test_score)
        rows.append(evaluate("validation_2023", target, "factorized_citer", y_val, expit(val_score), b_val, factor_seconds))
        rows.append(evaluate("test_2024", target, "factorized_citer", y_test, calibrated_test, b_test, factor_seconds))
        prediction_rows.append(pd.DataFrame({
            "event_id": data.loc[test, "event_id"].to_numpy(),
            "target": target,
            "model": "factorized_citer",
            "label": y_test,
            "probability": calibrated_test,
            "burden": b_test,
        }))

        # A strong sparse text baseline using the exact same incident description.
        print(f"fit target={target} model=tfidf_logistic", flush=True)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
        start = time.time()
        xtr = tfidf.fit_transform(np.asarray(texts, dtype=object)[train])
        xva = tfidf.transform(np.asarray(texts, dtype=object)[val])
        xte = tfidf.transform(np.asarray(texts, dtype=object)[test])
        text_model = LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced", random_state=SEED)
        text_model.fit(xtr, y_train)
        elapsed = time.time() - start
        val_score = text_model.predict_proba(xva)[:, 1]
        test_score = text_model.predict_proba(xte)[:, 1]
        calibrated_test = platt(val_score, y_val, test_score)
        rows.append(evaluate("validation_2023", target, "tfidf_logistic", y_val, val_score, b_val, elapsed))
        rows.append(evaluate("test_2024", target, "tfidf_logistic", y_test, calibrated_test, b_test, elapsed))
        prediction_rows.append(pd.DataFrame({
            "event_id": data.loc[test, "event_id"].to_numpy(),
            "target": target,
            "model": "tfidf_logistic",
            "label": y_test,
            "probability": calibrated_test,
            "burden": b_test,
        }))
        fitted[f"{target}:tfidf_logistic"] = (tfidf, text_model)

        pd.DataFrame(rows).to_csv(OUT / "metrics_partial.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "predictions_partial.parquet", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    joblib.dump(fitted, OUT / "models.joblib", compress=3)
    print(metrics.sort_values(["target", "split", "ap"], ascending=[True, True, False]).to_string(index=False), flush=True)
    print(f"outputs={OUT}", flush=True)


if __name__ == "__main__":
    main()
