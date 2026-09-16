from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lifelines import WeibullAFTFitter
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xgboost import XGBClassifier


SEED = 20260915
SOURCE = Path("data/new_york_context.parquet")
OUT = Path("artifacts/new_york_experts")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for bin_id in range(bins):
        mask = ids == bin_id
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def exposure_capture(burden: np.ndarray, score: np.ndarray, fraction: float) -> float:
    selected = max(1, int(math.ceil(fraction * len(score))))
    order = np.lexsort((np.arange(len(score)), -score))[:selected]
    return float(burden[order].sum() / burden.sum())


def exposure_ndcg(burden: np.ndarray, score: np.ndarray, fraction: float) -> float:
    selected = max(1, int(math.ceil(fraction * len(score))))
    discounts = 1.0 / np.log2(np.arange(selected) + 2.0)
    order = np.lexsort((np.arange(len(score)), -score))[:selected]
    ideal = np.argsort(-burden, kind="stable")[:selected]
    denominator = float(np.sum(burden[ideal] * discounts))
    return float(np.sum(burden[order] * discounts) / denominator) if denominator else 0.0


def calibrate_score(
    score_val: np.ndarray, score_test: np.ndarray, y_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    score_val = np.asarray(score_val, dtype=float)
    score_test = np.asarray(score_test, dtype=float)
    signed = bool(np.nanmin(score_val) < 0)
    if signed:
        transformed_val = np.sign(score_val) * np.log1p(np.abs(score_val))
        transformed_test = np.sign(score_test) * np.log1p(np.abs(score_test))
    else:
        transformed_val = np.log1p(np.maximum(score_val, 0.0))
        transformed_test = np.log1p(np.maximum(score_test, 0.0))
    mean = float(np.nanmean(transformed_val))
    scale = float(np.nanstd(transformed_val)) or 1.0
    transformed_val = np.nan_to_num((transformed_val - mean) / scale).reshape(-1, 1)
    transformed_test = np.nan_to_num((transformed_test - mean) / scale).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    model.fit(transformed_val, y_val)
    return (
        model.predict_proba(transformed_val)[:, 1],
        model.predict_proba(transformed_test)[:, 1],
        model,
    )


def metric_row(
    split: str,
    family: str,
    model: str,
    y: np.ndarray,
    probability: np.ndarray,
    ranking_score: np.ndarray,
    burden: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "split": split,
        "family": family,
        "model": model,
        "n": len(y),
        "prevalence": float(y.mean()),
        "ap": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "ece15": ece(y, probability),
    }
    for fraction in BUDGETS:
        tag = int(round(100 * fraction))
        row[f"top{tag}_high_risk_recall"] = float(
            y[np.lexsort((np.arange(len(y)), -ranking_score))[: max(1, math.ceil(fraction * len(y)))]].sum()
            / max(1, y.sum())
        )
        row[f"top{tag}_exposure_capture"] = exposure_capture(burden, ranking_score, fraction)
        row[f"top{tag}_exposure_ndcg"] = exposure_ndcg(burden, ranking_score, fraction)
    return row


class MultiThresholdSurvival(nn.Module):
    def __init__(self, width: int, outputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_survival_network(
    x_train: np.ndarray,
    duration_train: np.ndarray,
    x_val: np.ndarray,
    duration_val: np.ndarray,
    thresholds: np.ndarray,
) -> MultiThresholdSurvival:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MultiThresholdSurvival(x_train.shape[1], len(thresholds)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=1e-4)
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(duration_train[:, None] > thresholds[None, :], dtype=torch.float32)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(duration_val[:, None] > thresholds[None, :], dtype=torch.float32, device=device)
    positives = train_y.sum(0)
    negatives = train_y.shape[0] - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 0.5, 12.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = float("inf")
    best_state = None
    patience = 0
    for _ in range(80):
        network.train()
        order = torch.randperm(len(train_x))
        for start in range(0, len(order), 256):
            idx = order[start : start + 256]
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(bx), by)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(network(val_x), val_y).cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(network.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return network.cpu()


def survival_score(
    network: MultiThresholdSurvival,
    x: np.ndarray,
    required_duration: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    network.eval()
    with torch.no_grad():
        survival = torch.sigmoid(network(torch.as_tensor(x, dtype=torch.float32))).numpy()
    survival = np.minimum.accumulate(survival, axis=1)
    result = np.empty(len(x), dtype=float)
    for i, required in enumerate(required_duration):
        result[i] = np.interp(required, thresholds, survival[i], left=survival[i, 0], right=survival[i, -1])
    return result


def finetune_minilm(
    train_text: list[str],
    y_train: np.ndarray,
    val_text: list[str],
    y_val: np.ndarray,
    test_text: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    def tokenize(text: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(text, padding=True, truncation=True, max_length=160, return_tensors="pt")

    train_tokens = tokenize(train_text)
    val_tokens = tokenize(val_text)
    test_tokens = tokenize(test_text)
    labels = torch.as_tensor(y_train, dtype=torch.long)
    class_weights = torch.tensor(
        [1.0, max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_ap = -1.0
    best_state = None

    def predict(tokens: dict[str, torch.Tensor]) -> np.ndarray:
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(tokens["input_ids"]), 96):
                batch = {k: v[start : start + 96].to(device) for k, v in tokens.items()}
                values.append(torch.softmax(model(**batch).logits, dim=1)[:, 1].cpu().numpy())
        return np.concatenate(values)

    for _ in range(3):
        model.train()
        order = torch.randperm(len(labels))
        for start in range(0, len(order), 32):
            idx = order[start : start + 32]
            batch = {k: v[idx].to(device) for k, v in train_tokens.items()}
            target = labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(**batch).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_probability = predict(val_tokens)
        validation_ap = average_precision_score(y_val, validation_probability)
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(val_tokens), predict(test_tokens)


def main() -> None:
    seed_everything(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    frame = frame.loc[frame["checkpoint"].eq(0)].sort_values(["start_time", "event_id"]).reset_index(drop=True)
    frame = frame.loc[frame["high_burden"].notna() & frame["exposed_demand"].notna()].reset_index(drop=True)
    cutoff = pd.Timestamp("2023-10-01")
    train = frame["year"].eq(2023) & frame["start_time"].lt(cutoff)
    validation = frame["year"].eq(2023) & frame["start_time"].ge(cutoff)
    test = frame["year"].eq(2024)
    print({"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())}, flush=True)

    y = frame["high_burden"].to_numpy(np.int8)
    duration = frame["duration_minutes"].to_numpy(float)
    burden = frame["exposed_demand"].to_numpy(float)
    y_train, y_val, y_test = y[train], y[validation], y[test]
    duration_train, duration_val = duration[train], duration[validation]
    burden_val, burden_test = burden[validation], burden[test]

    excluded = {
        "event_id", "start_time", "year", "checkpoint", "duration_minutes", "remaining_minutes",
        "exposed_demand", "long60", "high_burden", "text", "matched_complex_ids",
    }
    categorical = [
        "initial_status", "current_status", "initial_lines", "current_lines", "borough_mode", "structure_mode"
    ]
    categorical = [column for column in categorical if column in frame.columns]
    numeric = [
        column for column in frame.columns
        if column not in excluded and column not in categorical and pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric_imputer = SimpleImputer(strategy="median")
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    numeric_values = numeric_imputer.fit(frame.loc[train, numeric]).transform(frame[numeric]).astype(np.float32)
    category_values = ordinal.fit(frame.loc[train, categorical].fillna("missing").astype(str)).transform(
        frame[categorical].fillna("missing").astype(str)
    ).astype(np.float32)

    texts = frame["text"].fillna("").astype(str)
    unique_texts, inverse = np.unique(texts.to_numpy(), return_inverse=True)
    encoder = SentenceTransformer(TEXT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    unique_embeddings = encoder.encode(
        unique_texts.tolist(), batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = unique_embeddings[inverse]
    pca = PCA(n_components=48, svd_solver="randomized", random_state=SEED)
    pca.fit(embeddings[train])
    text_values = pca.transform(embeddings).astype(np.float32)
    x = np.column_stack([numeric_values, category_values, text_values]).astype(np.float32)
    x_train, x_val, x_test = x[train], x[validation], x[test]

    probabilities_val: dict[str, np.ndarray] = {}
    probabilities_test: dict[str, np.ndarray] = {}
    scores_val: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}
    families: dict[str, str] = {}
    metric_rows: list[dict[str, float | int | str]] = []

    def register(
        name: str,
        family: str,
        probability_val: np.ndarray,
        probability_test: np.ndarray,
        score_val: np.ndarray | None = None,
        score_test: np.ndarray | None = None,
    ) -> None:
        probabilities_val[name] = np.asarray(probability_val, dtype=float)
        probabilities_test[name] = np.asarray(probability_test, dtype=float)
        scores_val[name] = np.asarray(score_val if score_val is not None else probability_val, dtype=float)
        scores_test[name] = np.asarray(score_test if score_test is not None else probability_test, dtype=float)
        families[name] = family
        metric_rows.extend(
            [
                metric_row("validation_2023", family, name, y_val, probabilities_val[name], scores_val[name], burden_val),
                metric_row("test_2024", family, name, y_test, probabilities_test[name], scores_test[name], burden_test),
            ]
        )
        pd.DataFrame(metric_rows).to_csv(OUT / "metrics_partial.csv", index=False)
        print(f"finished {name}", flush=True)

    prior = float(y_train.mean())
    register("prior", "reference", np.full(len(y_val), prior), np.full(len(y_test), prior))

    positive_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    general_models = {
        "fused_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=4000, random_state=SEED)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=350, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=700, learning_rate=0.035, num_leaves=31, max_depth=-1, min_child_samples=30,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, verbosity=-1, n_jobs=-1, random_state=SEED,
        ),
        "catboost": CatBoostClassifier(
            iterations=700, depth=7, learning_rate=0.04, loss_function="Logloss", eval_metric="PRAUC",
            l2_leaf_reg=5.0, verbose=False, random_seed=SEED, allow_writing_files=False,
        ),
        "mlp": make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=8e-4, max_iter=300,
                early_stopping=True, validation_fraction=0.15, random_state=SEED,
            )
        ),
        "xgboost": XGBClassifier(
            n_estimators=700, learning_rate=0.035, max_depth=6, min_child_weight=5, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, eval_metric="aucpr", tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=-1, random_state=SEED,
        ),
    }
    for name, model in general_models.items():
        print(f"fit {name}", flush=True)
        model.fit(x_train, y_train)
        register(name, "general", model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1])

    native = frame[numeric + categorical].copy()
    for column in numeric:
        native[column] = pd.to_numeric(native[column], errors="coerce").fillna(float(frame.loc[train, column].median()))
    for column in categorical:
        native[column] = native[column].fillna("missing").astype(str)
    for index in range(text_values.shape[1]):
        native[f"text_pc_{index:02d}"] = text_values[:, index]
    native_model = CatBoostClassifier(
        iterations=900, depth=8, learning_rate=0.035, loss_function="Logloss", eval_metric="PRAUC",
        l2_leaf_reg=6.0, verbose=False, random_seed=SEED, allow_writing_files=False,
    )
    native_model.fit(native.loc[train], y_train, cat_features=categorical)
    register(
        "native_catboost", "general_native_categorical",
        native_model.predict_proba(native.loc[validation])[:, 1], native_model.predict_proba(native.loc[test])[:, 1],
    )

    duration_model = LGBMRegressor(
        objective="huber", n_estimators=700, learning_rate=0.035, num_leaves=31, min_child_samples=30,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, verbosity=-1, n_jobs=-1, random_state=SEED,
    )
    duration_model.fit(x_train, np.log1p(duration_train))
    duration_val_score = np.expm1(duration_model.predict(x_val)) * np.maximum(frame.loc[validation, "demand_rate_sum"], 0) / 60.0
    duration_test_score = np.expm1(duration_model.predict(x_test)) * np.maximum(frame.loc[test, "demand_rate_sum"], 0) / 60.0
    p_val, p_test, _ = calibrate_score(duration_val_score, duration_test_score, y_val)
    register("text_structured_lgbm_duration", "same_task_duration", p_val, p_test, duration_val_score, duration_test_score)

    aft_width = min(40, x_train.shape[1])
    aft_scaler = StandardScaler()
    aft_train = aft_scaler.fit_transform(x_train[:, :aft_width])
    aft_val = aft_scaler.transform(x_val[:, :aft_width])
    aft_test = aft_scaler.transform(x_test[:, :aft_width])
    aft_columns = [f"x{i}" for i in range(aft_width)]
    aft_frame = pd.DataFrame(aft_train, columns=aft_columns)
    aft_frame["duration"] = np.maximum(duration_train, 0.5)
    aft_frame["observed"] = 1
    aft = WeibullAFTFitter(penalizer=0.20)
    aft.fit(aft_frame, duration_col="duration", event_col="observed")
    aft_duration_val = np.asarray(aft.predict_median(pd.DataFrame(aft_val, columns=aft_columns))).reshape(-1)
    aft_duration_test = np.asarray(aft.predict_median(pd.DataFrame(aft_test, columns=aft_columns))).reshape(-1)
    aft_val_score = aft_duration_val * np.maximum(frame.loc[validation, "demand_rate_sum"], 0) / 60.0
    aft_test_score = aft_duration_test * np.maximum(frame.loc[test, "demand_rate_sum"], 0) / 60.0
    p_val, p_test, _ = calibrate_score(aft_val_score, aft_test_score, y_val)
    register("weibull_aft", "same_task_survival", p_val, p_test, aft_val_score, aft_test_score)

    survival_scaler = StandardScaler()
    survival_x_train = survival_scaler.fit_transform(x_train).astype(np.float32)
    survival_x_val = survival_scaler.transform(x_val).astype(np.float32)
    survival_x_test = survival_scaler.transform(x_test).astype(np.float32)
    thresholds = np.array([5, 10, 15, 20, 30, 45, 60, 90, 120, 180], dtype=float)
    survival_network = train_survival_network(
        survival_x_train, duration_train, survival_x_val, duration_val, thresholds
    )
    burden_threshold = float(np.quantile(burden[train], 0.90))
    required_val = 60.0 * burden_threshold / np.maximum(frame.loc[validation, "demand_rate_sum"].to_numpy(float), 1e-3)
    required_test = 60.0 * burden_threshold / np.maximum(frame.loc[test, "demand_rate_sum"].to_numpy(float), 1e-3)
    survival_val_score = survival_score(survival_network, survival_x_val, required_val, thresholds)
    survival_test_score = survival_score(survival_network, survival_x_test, required_test, thresholds)
    p_val, p_test, _ = calibrate_score(survival_val_score, survival_test_score, y_val)
    register("mtdnn_survival", "same_task_deep_survival", p_val, p_test, survival_val_score, survival_test_score)

    exposure_model = LGBMRegressor(
        objective="huber", n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=25,
        subsample=0.85, colsample_bytree=0.90, reg_lambda=1.0, verbosity=-1, n_jobs=-1, random_state=SEED,
    )
    exposure_model.fit(x_train, np.log1p(burden[train]))
    exposure_val_score = np.expm1(exposure_model.predict(x_val))
    exposure_test_score = np.expm1(exposure_model.predict(x_test))
    p_val, p_test, _ = calibrate_score(exposure_val_score, exposure_test_score, y_val)
    register("direct_exposure_lgbm", "same_task_exposure", p_val, p_test, exposure_val_score, exposure_test_score)

    relevance_edges = np.unique(np.quantile(burden[train], np.linspace(0.0, 1.0, 11)))
    relevance_train = np.digitize(burden[train], relevance_edges[1:-1], right=True).astype(int)
    train_weeks = frame.loc[train, "start_time"].dt.to_period("W").astype(str).to_numpy()
    validation_weeks = frame.loc[validation, "start_time"].dt.to_period("W").astype(str).to_numpy()
    train_order = np.argsort(train_weeks, kind="stable")
    validation_order = np.argsort(validation_weeks, kind="stable")
    train_group = pd.Series(train_weeks[train_order]).value_counts(sort=False).to_numpy()
    validation_group = pd.Series(validation_weeks[validation_order]).value_counts(sort=False).to_numpy()
    ranker = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=700, learning_rate=0.035, num_leaves=31,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.90, reg_lambda=1.0,
        verbosity=-1, n_jobs=-1, random_state=SEED,
    )
    ranker.fit(
        x_train[train_order], relevance_train[train_order], group=train_group,
        eval_set=[(x_val[validation_order], np.digitize(burden_val, relevance_edges[1:-1], right=True)[validation_order])],
        eval_group=[validation_group],
    )
    rank_val_score = ranker.predict(x_val)
    rank_test_score = ranker.predict(x_test)
    p_val, p_test, _ = calibrate_score(rank_val_score, rank_test_score, y_val)
    register("exposure_lambdamart", "same_task_learning_to_rank", p_val, p_test, rank_val_score, rank_test_score)

    print("fine-tune MiniLM", flush=True)
    ft_val, ft_test = finetune_minilm(
        texts.loc[train].tolist(), y_train, texts.loc[validation].tolist(), y_val, texts.loc[test].tolist()
    )
    register("finetuned_minilm", "general_finetuned_transformer", ft_val, ft_test)

    stack_names = [
        "native_catboost", "xgboost", "finetuned_minilm", "mtdnn_survival",
        "text_structured_lgbm_duration", "direct_exposure_lgbm",
    ]
    stack_val = np.column_stack([probabilities_val[name] for name in stack_names])
    stack_test = np.column_stack([probabilities_test[name] for name in stack_names])
    stacker = LogisticRegression(C=0.15, max_iter=4000, random_state=SEED)
    stacker.fit(stack_val, y_val)
    register(
        "citer_detect", "proposed_detection",
        stacker.predict_proba(stack_val)[:, 1], stacker.predict_proba(stack_test)[:, 1],
    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    prediction_rows = []
    for split_name, mask, labels, burdens, probability_dict, score_dict in (
        ("validation_2023", validation, y_val, burden_val, probabilities_val, scores_val),
        ("test_2024", test, y_test, burden_test, probabilities_test, scores_test),
    ):
        identifiers = frame.loc[mask, ["event_id", "start_time"]].reset_index(drop=True)
        for model in probability_dict:
            block = identifiers.copy()
            block["split"] = split_name
            block["target"] = "high_exposure"
            block["model"] = model
            block["label"] = labels
            block["probability"] = probability_dict[model]
            block["ranking_score"] = score_dict[model]
            block["burden"] = burdens
            prediction_rows.append(block)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "predictions.parquet", index=False)

    budget_rows = []
    for split_name, labels, burdens, score_dict in (
        ("validation_2023", y_val, burden_val, scores_val),
        ("test_2024", y_test, burden_test, scores_test),
    ):
        for model, score in score_dict.items():
            for fraction in BUDGETS:
                selected = max(1, int(math.ceil(fraction * len(score))))
                order = np.lexsort((np.arange(len(score)), -score))[:selected]
                budget_rows.append(
                    {
                        "split": split_name,
                        "model": model,
                        "budget_fraction": fraction,
                        "high_risk_recall": float(labels[order].sum() / max(1, labels.sum())),
                        "exposure_capture": exposure_capture(burdens, score, fraction),
                        "exposure_ndcg": exposure_ndcg(burdens, score, fraction),
                    }
                )
    pd.DataFrame(budget_rows).to_csv(OUT / "budget_curves.csv", index=False)
    print(metrics.sort_values(["split", "ap"], ascending=[True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
