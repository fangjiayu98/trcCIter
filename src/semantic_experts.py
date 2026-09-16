#!/usr/bin/env python3
"""Same-setting literature and general baselines for Toronto disruption triage."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import bitermplus as btm
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lifelines import WeibullAFTFitter
from lightgbm import LGBMRanker, LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path("/root/autodl-tmp/rail2road")
HERE = ROOT / "experiments_v2"
OUT = ROOT / "results/trc_v2/same_task_models"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import pipeline as bp  # noqa: E402
import toronto_baselines as base  # noqa: E402


SEED = 20260915
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGETS = {"severe_delay": "min_delay", "high_burden": "exposed_demand"}


def class_pos_weight(y: np.ndarray) -> float:
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy(float)


def frozen_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right + 1.0) / (2.0 * (len(reference) + 1.0))


def build_btm_topics(texts: list[str], n_topics: int = 16) -> tuple[np.ndarray, dict]:
    unique_text, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    x_words, vocabulary, _ = btm.get_words_freqs(unique_text.tolist())
    docs_vec = btm.get_vectorized_docs(unique_text.tolist(), vocabulary)
    biterms = btm.get_biterms(docs_vec)
    model = btm.BTM(
        x_words,
        vocabulary,
        T=n_topics,
        M=20,
        alpha=50.0 / n_topics,
        beta=0.01,
        seed=SEED,
    )
    model.fit(biterms, iterations=200, verbose=False)
    unique_topics = np.asarray(model.transform(docs_vec), dtype=np.float32)
    return unique_topics[inverse], {
        "model": model,
        "vocabulary": vocabulary,
        "unique_text": unique_text.tolist(),
        "n_topics": n_topics,
    }


class SurvivalMTDNN(nn.Module):
    def __init__(self, n_features: int, n_thresholds: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.head = nn.Linear(96, n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def train_survival_mtdnn(
    x_train: np.ndarray,
    duration_train: np.ndarray,
    x_val: np.ndarray,
    duration_val: np.ndarray,
) -> tuple[SurvivalMTDNN, np.ndarray]:
    thresholds = np.asarray([2, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240], dtype=np.float32)
    y_train = (duration_train[:, None] > thresholds[None, :]).astype(np.float32)
    y_val = (duration_val[:, None] > thresholds[None, :]).astype(np.float32)
    positives = y_train.sum(axis=0)
    pos_weight = np.clip((len(y_train) - positives) / np.maximum(positives, 1.0), 0.25, 30.0)

    model = SurvivalMTDNN(x_train.shape[1], len(thresholds)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=2048,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    x_val_t = torch.from_numpy(x_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)
    best_loss = np.inf
    best_state = None
    stale = 0
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        print(f"mtdnn epoch={epoch + 1} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    return model, thresholds


def mtdnn_duration(model: SurvivalMTDNN, thresholds: np.ndarray, x: np.ndarray) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    survival = np.concatenate(outputs, axis=0)
    survival = np.minimum.accumulate(survival, axis=1)
    widths = np.diff(np.concatenate([[0.0], thresholds])).astype(np.float32)
    expected = widths[0] + np.sum(survival[:, :-1] * widths[1:][None, :], axis=1)
    expected += survival[:, -1] * 60.0
    return np.clip(expected, 0.0, 300.0)


class FineTunedMiniLM(nn.Module):
    def __init__(self, n_numeric: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        hidden = int(self.encoder.config.hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + n_numeric, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        mask_f = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return self.head(self.fusion(torch.cat([pooled, numeric], dim=1)))


def train_finetuned_minilm(
    texts: list[str],
    numeric: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    labels: np.ndarray,
) -> tuple[FineTunedMiniLM, dict, np.ndarray]:
    unique_text, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True,
    )
    tokens = tokenizer(
        unique_text.tolist(),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    ids_all = tokens["input_ids"]
    mask_all = tokens["attention_mask"]
    model = FineTunedMiniLM(numeric.shape[1]).to(DEVICE)
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.fusion.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": 2e-5}, {"params": head_params, "lr": 8e-4}],
        weight_decay=1e-4,
    )
    pos_weight = torch.tensor(
        [class_pos_weight(labels[train, 0]), class_pos_weight(labels[train, 1])],
        device=DEVICE,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_ds = TensorDataset(
        torch.from_numpy(inverse[train].astype(np.int64)),
        torch.from_numpy(numeric[train]),
        torch.from_numpy(labels[train].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(inverse[val].astype(np.int64)),
        torch.from_numpy(numeric[val]),
        torch.from_numpy(labels[val].astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_objective = -np.inf
    best_state = None
    for epoch in range(3):
        model.train()
        for text_idx, xb, yb in train_loader:
            batch_ids = ids_all[text_idx].to(DEVICE, non_blocking=True)
            batch_mask = mask_all[text_idx].to(DEVICE, non_blocking=True)
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(batch_ids, batch_mask, xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_prob = []
        with torch.no_grad():
            for text_idx, xb, _ in val_loader:
                logits = model(
                    ids_all[text_idx].to(DEVICE),
                    mask_all[text_idx].to(DEVICE),
                    xb.to(DEVICE),
                )
                val_prob.append(torch.sigmoid(logits).cpu().numpy())
        val_prob = np.concatenate(val_prob)
        objective = average_precision_score(labels[val, 0], val_prob[:, 0]) + average_precision_score(labels[val, 1], val_prob[:, 1])
        print(f"minilm epoch={epoch + 1} validation_ap_sum={objective:.6f}", flush=True)
        if objective > best_objective:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    token_state = {"ids": ids_all, "mask": mask_all, "inverse": inverse, "unique_text": unique_text.tolist()}
    return model, token_state, labels


def minilm_predict(model: FineTunedMiniLM, token_state: dict, numeric: np.ndarray, rows: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(rows)
    ds = TensorDataset(
        torch.from_numpy(token_state["inverse"][indices].astype(np.int64)),
        torch.from_numpy(numeric[indices]),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    outputs = []
    model.eval()
    with torch.no_grad():
        for text_idx, xb in loader:
            logits = model(
                token_state["ids"][text_idx].to(DEVICE),
                token_state["mask"][text_idx].to(DEVICE),
                xb.to(DEVICE),
            )
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data, fused, _, texts, meta = base.prepare()
    train = data.year.le(2022).to_numpy()
    val = data.year.eq(2023).to_numpy()
    test = data.year.eq(2024).to_numpy()
    numeric = fused[:, 48:].astype(np.float32)
    duration = data.min_delay.to_numpy(float)
    rate = data.demand_rate.to_numpy(float)
    labels = np.column_stack([data.severe_delay.to_numpy(int), data.high_burden.to_numpy(int)])
    rows = []
    prediction_rows = []
    score_store = {target: {} for target in TARGETS}
    fitted = {}

    def add_scores(target: str, model_name: str, val_score: np.ndarray, test_score: np.ndarray, elapsed: float, probabilities: bool = False) -> None:
        y = data[target].to_numpy(int)
        burden = data[TARGETS[target]].to_numpy(float)
        if probabilities:
            val_prob = np.clip(val_score, 1e-6, 1 - 1e-6)
            test_prob = base.platt(val_score, y[val], test_score)
        else:
            val_prob = base.platt(val_score, y[val], val_score)
            test_prob = base.platt(val_score, y[val], test_score)
        rows.append(base.evaluate("validation_2023", target, model_name, y[val], val_prob, burden[val], elapsed))
        rows.append(base.evaluate("test_2024", target, model_name, y[test], test_prob, burden[test], elapsed))
        score_store[target][model_name] = (np.asarray(val_score), np.asarray(test_score))
        for split_name, mask, prob, score in [
            ("validation_2023", val, val_prob, val_score),
            ("test_2024", test, test_prob, test_score),
        ]:
            prediction_rows.append(pd.DataFrame({
                "event_id": data.loc[mask, "event_id"].to_numpy(),
                "split": split_name,
                "target": target,
                "model": model_name,
                "label": y[mask],
                "probability": prob,
                "ranking_score": score,
                "burden": burden[mask],
            }))
        pd.DataFrame(rows).to_csv(OUT / "metrics_partial.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "predictions_partial.parquet", index=False)

    print("fit BTM topics", flush=True)
    topics, topic_state = build_btm_topics(texts)
    topic_numeric = np.concatenate([topics, numeric], axis=1).astype(np.float32)
    btm_reg = LGBMRegressor(
        objective="huber", n_estimators=1200, learning_rate=0.025, num_leaves=31,
        min_child_samples=40, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    start = time.time()
    btm_reg.fit(
        topic_numeric[train], np.log1p(duration[train]),
        eval_set=[(topic_numeric[val], np.log1p(duration[val]))],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    btm_val_duration = np.clip(np.expm1(btm_reg.predict(topic_numeric[val])), 0, 300)
    btm_test_duration = np.clip(np.expm1(btm_reg.predict(topic_numeric[test])), 0, 300)
    elapsed = time.time() - start
    add_scores("severe_delay", "btm_lgbm_duration", btm_val_duration, btm_test_duration, elapsed)
    add_scores("high_burden", "btm_lgbm_duration", btm_val_duration * rate[val], btm_test_duration * rate[test], elapsed)
    fitted["btm_lgbm_duration"] = btm_reg

    print("fit Weibull AFT", flush=True)
    aft_names = [f"topic_{i}" for i in range(topics.shape[1])] + [f"x_{i}" for i in range(numeric.shape[1])]
    aft_train = pd.DataFrame(topic_numeric[train], columns=aft_names)
    aft_train["duration"] = np.maximum(duration[train], 0.25)
    aft_train["observed"] = 1
    aft = WeibullAFTFitter(penalizer=0.05)
    start = time.time()
    aft.fit(aft_train, duration_col="duration", event_col="observed", show_progress=False)
    aft_val_duration = np.clip(np.asarray(aft.predict_median(pd.DataFrame(topic_numeric[val], columns=aft_names))), 0, 300)
    aft_test_duration = np.clip(np.asarray(aft.predict_median(pd.DataFrame(topic_numeric[test], columns=aft_names))), 0, 300)
    elapsed = time.time() - start
    add_scores("severe_delay", "weibull_aft", aft_val_duration, aft_test_duration, elapsed)
    add_scores("high_burden", "weibull_aft", aft_val_duration * rate[val], aft_test_duration * rate[test], elapsed)
    fitted["weibull_aft"] = aft

    print("fit MTDNN survival", flush=True)
    start = time.time()
    mtdnn, thresholds = train_survival_mtdnn(fused[train], duration[train], fused[val], duration[val])
    mtdnn_val_duration = mtdnn_duration(mtdnn, thresholds, fused[val])
    mtdnn_test_duration = mtdnn_duration(mtdnn, thresholds, fused[test])
    elapsed = time.time() - start
    add_scores("severe_delay", "mtdnn_survival", mtdnn_val_duration, mtdnn_test_duration, elapsed)
    add_scores("high_burden", "mtdnn_survival", mtdnn_val_duration * rate[val], mtdnn_test_duration * rate[test], elapsed)
    fitted["mtdnn_survival"] = {"state": mtdnn.cpu().state_dict(), "thresholds": thresholds}
    mtdnn.to(DEVICE)

    print("fit native categorical CatBoost", flush=True)
    cat_cols = [c for c in ["code", "station", "line", "text"] if c in data.columns]
    native = data[list(bp.NUMERIC_COLUMNS) + cat_cols].copy()
    for c in cat_cols:
        native[c] = native[c].fillna("UNK").astype(str)
    native[list(bp.NUMERIC_COLUMNS)] = native[list(bp.NUMERIC_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for target in TARGETS:
        cat_model = CatBoostClassifier(
            iterations=1400, learning_rate=0.035, depth=9, loss_function="Logloss",
            eval_metric="PRAUC", auto_class_weights="Balanced", random_seed=SEED,
            verbose=False, allow_writing_files=False, thread_count=-1,
        )
        start = time.time()
        cat_model.fit(
            native.loc[train], data.loc[train, target].to_numpy(int),
            cat_features=cat_cols,
            eval_set=(native.loc[val], data.loc[val, target].to_numpy(int)),
            early_stopping_rounds=120,
            verbose=False,
        )
        val_score = cat_model.predict_proba(native.loc[val])[:, 1]
        test_score = cat_model.predict_proba(native.loc[test])[:, 1]
        add_scores(target, "native_catboost", val_score, test_score, time.time() - start, probabilities=True)
        fitted[f"{target}:native_catboost"] = cat_model

    print("fit fine-tuned MiniLM", flush=True)
    start = time.time()
    minilm, token_state, _ = train_finetuned_minilm(texts, numeric, train, val, labels)
    val_minilm = minilm_predict(minilm, token_state, numeric, val)
    test_minilm = minilm_predict(minilm, token_state, numeric, test)
    elapsed = time.time() - start
    add_scores("severe_delay", "finetuned_minilm", val_minilm[:, 0], test_minilm[:, 0], elapsed, probabilities=True)
    add_scores("high_burden", "finetuned_minilm", val_minilm[:, 1], test_minilm[:, 1], elapsed, probabilities=True)
    fitted["finetuned_minilm"] = minilm.cpu().state_dict()

    print("fit direct continuous-burden LightGBM", flush=True)
    burden_reg = LGBMRegressor(
        objective="huber", n_estimators=1400, learning_rate=0.025, num_leaves=47,
        min_child_samples=35, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    start = time.time()
    burden_reg.fit(
        fused[train], np.log1p(data.loc[train, "exposed_demand"].to_numpy(float)),
        eval_set=[(fused[val], np.log1p(data.loc[val, "exposed_demand"].to_numpy(float)))],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    burden_val = np.expm1(burden_reg.predict(fused[val]))
    burden_test = np.expm1(burden_reg.predict(fused[test]))
    elapsed = time.time() - start
    add_scores("high_burden", "direct_burden_lgbm", burden_val, burden_test, elapsed)
    fitted["direct_burden_lgbm"] = burden_reg

    print("fit burden-aware LambdaMART", flush=True)
    dates = data.timestamp.dt.strftime("%Y-%m-%d").to_numpy()
    train_order = np.argsort(dates[train], kind="stable")
    val_order = np.argsort(dates[val], kind="stable")
    train_dates = dates[train][train_order]
    val_dates = dates[val][val_order]
    _, train_group = np.unique(train_dates, return_counts=True)
    _, val_group = np.unique(val_dates, return_counts=True)
    log_burden_train = np.log1p(data.loc[train, "exposed_demand"].to_numpy(float))
    positive = log_burden_train[log_burden_train > 0]
    cuts = np.unique(np.quantile(positive, np.linspace(0.0, 1.0, 16)[1:-1]))
    relevance_train = np.digitize(log_burden_train, cuts).astype(int)
    log_burden_val = np.log1p(data.loc[val, "exposed_demand"].to_numpy(float))
    relevance_val = np.digitize(log_burden_val, cuts).astype(int)
    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=1200, learning_rate=0.025,
        num_leaves=31, min_child_samples=40, reg_lambda=2.0, n_jobs=-1,
        random_state=SEED, verbosity=-1,
    )
    start = time.time()
    ranker.fit(
        fused[train][train_order], relevance_train[train_order], group=train_group,
        eval_set=[(fused[val][val_order], relevance_val[val_order])], eval_group=[val_group],
        eval_at=[20], callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    rank_val = ranker.predict(fused[val])
    rank_test = ranker.predict(fused[test])
    elapsed = time.time() - start
    add_scores("high_burden", "burden_lambdamart", rank_val, rank_test, elapsed)
    fitted["burden_lambdamart"] = ranker

    print("select CITER survival-exposure fusion on validation year", flush=True)
    candidate_names = [
        "native_catboost", "finetuned_minilm", "mtdnn_survival", "btm_lgbm_duration",
        "weibull_aft", "direct_burden_lgbm", "burden_lambdamart",
    ]
    candidate_names = [name for name in candidate_names if name in score_store["high_burden"]]
    val_rank = np.column_stack([rank01(score_store["high_burden"][name][0]) for name in candidate_names])
    test_rank = np.column_stack([
        frozen_percentile(
            score_store["high_burden"][name][0],
            score_store["high_burden"][name][1],
        )
        for name in candidate_names
    ])
    y_val = data.loc[val, "high_burden"].to_numpy(int)
    b_val = data.loc[val, "exposed_demand"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    candidates = [np.eye(len(candidate_names))[i] for i in range(len(candidate_names))]
    candidates.extend(rng.dirichlet(np.ones(len(candidate_names)), size=8000))
    best = None
    for weights in candidates:
        score = val_rank @ weights
        ap = average_precision_score(y_val, score)
        _, capture, _ = base.top_fraction_metrics(y_val, score, b_val)
        objective = 0.75 * ap + 0.25 * capture
        if best is None or objective > best[0]:
            best = (objective, weights.copy(), ap, capture)
    final_val = val_rank @ best[1]
    final_test = test_rank @ best[1]
    add_scores("high_burden", "citer_v2_survival_exposure", final_val, final_test, 0.0)
    weights_frame = pd.DataFrame({"model": candidate_names, "weight": best[1]})
    weights_frame.to_csv(OUT / "citer_weights.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    joblib_path = OUT / "models.joblib"
    import joblib
    joblib.dump(fitted, joblib_path, compress=3)
    summary = {
        "device": str(DEVICE),
        "data": meta,
        "citer_validation_objective": float(best[0]),
        "citer_validation_ap": float(best[2]),
        "citer_validation_capture20": float(best[3]),
        "citer_weights": dict(zip(candidate_names, map(float, best[1]))),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(metrics.sort_values(["target", "split", "ap"], ascending=[True, True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
