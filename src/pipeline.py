#!/usr/bin/env python3
"""Locked cross-city early triage experiment for NYC and Toronto rail incidents.

Development phase:
  * NYC 2023 onset snapshots are the source domain.
  * Toronto 2014-2022 is the target training set.
  * Toronto 2023 is the only development/validation set.
  * Toronto 2024 is neither required nor read.

Final phase:
  * Loads the immutable development state.
  * Reads Toronto 2024 once and reports the locked external test.

Realized delay and gap are outcomes only. They never enter model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260914
PCA_DIM = 48
LAMBDA_GRID = [0.0003, 0.003, 0.03, 0.3]
FEWSHOT_BUDGETS = [200, 1000, 5000, 20000, -1]
FEWSHOT_SEEDS = [17, 31, 53]
TARGETS = ["severe_delay", "high_burden"]
SOURCE_LABEL = {"severe_delay": "long60", "high_burden": "high_burden"}
TORONTO_2024_URL = (
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/"
    "996cfe8d-fb35-40ce-b569-698d51fc683b/resource/"
    "2ee1a65c-da06-4ad1-bdfb-b1a57701e46a/download/ttc-subway-delay-2024.xlsx"
)

NUMERIC_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "peak",
    "line_count_scaled",
    "station_matched",
    "demand_rank",
    "degree_rank",
    "route_rank",
    "transfer",
    "temperature_scaled",
    "wet_weather",
    "wind_scaled",
    "weather_missing",
    "kw_signal",
    "kw_track",
    "kw_train",
    "kw_passenger",
    "kw_police",
    "kw_power",
    "kw_weather",
    "kw_medical",
    "kw_fire",
    "kw_operations",
]

MECHANISM_PATTERNS = {
    "kw_signal": r"signal|switch|interlocking|atc|control system|wayside",
    "kw_track": r"track|rail|roadbed|work zone|right.of.way",
    "kw_train": r"train|vehicle|car |door|brake|propulsion|equipment|mechanical",
    "kw_passenger": r"passenger|customer|patron|person|crowd|assistance alarm",
    "kw_police": r"police|security|unauthori[sz]ed|trespass|disorder|investigat",
    "kw_power": r"power|electric|voltage|current|collector|panto",
    "kw_weather": r"weather|rain|snow|ice|storm|wind|flood|heat",
    "kw_medical": r"medical|injur|illness|sick|emergency medical|ems",
    "kw_fire": r"fire|smoke|burn",
    "kw_operations": r"operator|crew|staff|schedule|dispatch|operat|late relief",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def quantile_higher(values: pd.Series | np.ndarray, q: float) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q, method="higher"))


def empirical_rank(values: pd.Series | np.ndarray, reference: pd.Series | np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    val = np.asarray(values, dtype=float)
    if len(ref) == 0:
        return np.full(len(val), 0.5, dtype=float)
    fill = float(np.median(ref))
    val = np.where(np.isfinite(val), val, fill)
    return np.searchsorted(ref, val, side="right") / len(ref)


def normalize_station(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value).upper().replace("&", " AND ")
    raw = raw.replace("’", "'").replace("-", " ")
    raw = re.sub(r"\bST[.]?\s+", "ST ", raw)
    raw = re.sub(r"\b(VIA|AT|TO)\b.*$", "", raw)
    raw = re.sub(
        r"\b(STATION|STN|SUBWAY|PLATFORM|TRACK LEVEL|TRACK|NORTHBOUND|SOUTHBOUND|"
        r"EASTBOUND|WESTBOUND|N/B|S/B|E/B|W/B)\b",
        " ",
        raw,
    )
    raw = re.sub(r"\b(YUS|YU|BD|SRT|SHP)\b", " ", raw)
    raw = re.sub(r"\bLINE\s*[12346]\b", " ", raw)
    raw = re.sub(r"[^A-Z0-9 ]+", " ", raw)
    key = re.sub(r"\s+", " ", raw).strip()
    aliases = {
        "BLOOR": "BLOOR YONGE",
        "YONGE": "BLOOR YONGE",
        "BLOOR YONGE": "BLOOR YONGE",
        "SHEPPARD": "SHEPPARD YONGE",
        "YONGE SHEPPARD": "SHEPPARD YONGE",
        "SHEPPARD YONGE": "SHEPPARD YONGE",
        "VAUGHAN MC": "VAUGHAN METROPOLITAN CENTRE",
        "VAUGHAN METRO CENTRE": "VAUGHAN METROPOLITAN CENTRE",
        "SCARBOROUGH CTR": "SCARBOROUGH CENTRE",
        "SCARBOROUGH CENTER": "SCARBOROUGH CENTRE",
        "MAIN": "MAIN STREET",
        "DOWNSVIEW": "SHEPPARD WEST",
        "QUEENS PARK": "QUEEN S PARK",
        "ST PATRICK": "ST PATRICK",
        "ST GEORGE": "ST GEORGE",
    }
    return aliases.get(key, key)


def station_similarity(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    qt, ct = set(query.split()), set(candidate.split())
    jac = len(qt & ct) / max(1, len(qt | ct))
    seq = SequenceMatcher(None, query, candidate).ratio()
    contained = 0.94 if min(len(query), len(candidate)) >= 5 and (query in candidate or candidate in query) else 0.0
    return max(0.55 * seq + 0.45 * jac, contained)


def parse_code_descriptions(path: Path) -> dict[str, str]:
    raw = pd.read_excel(path, header=None)
    mapping: dict[str, str] = {}
    for _, row in raw.iterrows():
        vals = row.tolist()
        for j in range(len(vals) - 1):
            code = "" if pd.isna(vals[j]) else str(vals[j]).strip().upper()
            desc = "" if pd.isna(vals[j + 1]) else str(vals[j + 1]).strip()
            if re.fullmatch(r"[A-Z]{2,10}", code) and len(desc) >= 3 and "CODE" not in desc.upper():
                mapping.setdefault(code, desc)
    return mapping


def load_station_usage(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, header=4)
    d.columns = [str(c).strip() for c in d.columns]
    station_col = next(c for c in d.columns if c.lower() == "station")
    total_col = next(c for c in d.columns if c.lower() == "total")
    line_col = next(c for c in d.columns if "line" in c.lower())
    d = d[[station_col, line_col, total_col]].copy()
    d.columns = ["station", "usage_line", "daily_volume"]
    d["station_key"] = d.station.map(normalize_station)
    d["daily_volume"] = to_float(d.daily_volume)
    d = d[(d.station_key != "") & d.daily_volume.notna()]
    out = d.groupby("station_key", as_index=False).agg(
        daily_volume=("daily_volume", "sum"),
        route_count=("usage_line", "nunique"),
    )
    return out


def build_gtfs_graph(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with zipfile.ZipFile(path) as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        routes["route_type_num"] = pd.to_numeric(routes.route_type, errors="coerce")
        subway_routes = set(routes.loc[routes.route_type_num == 1, "route_id"].astype(str))
        trips = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "trip_id", "direction_id"])
        trips = trips[trips.route_id.isin(subway_routes)].copy()
        trip_ids = set(trips.trip_id)
        stops = pd.read_csv(
            z.open("stops.txt"),
            dtype=str,
            usecols=["stop_id", "stop_name", "parent_station"],
        ).fillna("")
        stop_times = pd.read_csv(
            z.open("stop_times.txt"),
            dtype={"trip_id": str, "stop_id": str},
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
    stop_times = stop_times[stop_times.trip_id.isin(trip_ids)].copy()
    stop_times["stop_sequence"] = pd.to_numeric(stop_times.stop_sequence, errors="coerce")
    parent = dict(zip(stops.stop_id, stops.parent_station))
    names = dict(zip(stops.stop_id, stops.stop_name))

    def node_for(stop_id: str) -> str:
        root = parent.get(stop_id, "") or stop_id
        return normalize_station(names.get(root, names.get(stop_id, "")))

    stop_times["node"] = stop_times.stop_id.map(node_for)
    counts = stop_times.groupby("trip_id").size()
    trips = trips.assign(n_stops=trips.trip_id.map(counts).fillna(0))
    chosen = trips.sort_values("n_stops").groupby(["route_id", "direction_id"], as_index=False).tail(1)
    for trip_id in chosen.trip_id:
        seq = stop_times.loc[stop_times.trip_id == trip_id].sort_values("stop_sequence").node.tolist()
        seq = [x for i, x in enumerate(seq) if x and (i == 0 or x != seq[i - 1])]
        graph.add_edges_from(zip(seq[:-1], seq[1:]))
    srt = ["KENNEDY", "LAWRENCE EAST", "ELLESMERE", "MIDLAND", "SCARBOROUGH CENTRE", "MCCOWAN"]
    graph.add_edges_from(zip(srt[:-1], srt[1:]))
    return graph


def build_station_registry(data_dir: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    usage = load_station_usage(data_dir / "ttc-subway-station-usage-2017.xlsx")
    graph = build_gtfs_graph(data_dir / "opendata_ttc_schedules.zip")
    degree = nx.degree_centrality(graph) if len(graph) > 1 else {}
    between = nx.betweenness_centrality(graph, normalized=True) if len(graph) > 1 else {}
    closeness = nx.closeness_centrality(graph) if len(graph) > 1 else {}
    metrics = defaultdict(dict)
    for row in usage.itertuples(index=False):
        metrics[row.station_key].update(
            daily_volume=float(row.daily_volume),
            route_count=float(row.route_count),
        )
    for node in graph.nodes:
        metrics[node].update(
            graph_degree=float(degree.get(node, 0.0)),
            betweenness=float(between.get(node, 0.0)),
            closeness=float(closeness.get(node, 0.0)),
        )
    registry = pd.DataFrame([{"station_key": k, **v} for k, v in metrics.items()])
    for c in ["daily_volume", "route_count", "graph_degree", "betweenness", "closeness"]:
        registry[c] = to_float(registry.get(c, pd.Series(index=registry.index, dtype=float)))
        registry[c] = registry[c].fillna(registry[c].median())
        registry[c + "_rank"] = registry[c].rank(pct=True, method="average")
    registry["transfer"] = (registry.route_count > 1).astype(float)
    records = registry.set_index("station_key").to_dict(orient="index")
    return registry, records


def match_station(raw: object, registry: dict[str, dict]) -> tuple[str, float]:
    query = normalize_station(raw)
    if query in registry:
        return query, 1.0
    if not query:
        return "", 0.0
    candidates = list(registry)
    contained = [c for c in candidates if len(c) >= 5 and c in query]
    if contained:
        best = max(contained, key=len)
        return best, 0.94
    scores = [(station_similarity(query, c), c) for c in candidates]
    score, best = max(scores, default=(0.0, ""))
    return (best, score) if score >= 0.76 else ("", score)


def parse_time_minutes(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour * 60 + value.minute + getattr(value, "second", 0) / 60)
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x * 1440 if 0 <= x < 1 else x
    text = str(value).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3) or 0) / 60)
    return np.nan


def normalize_line(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNK"
    s = re.sub(r"\s+", "", str(value).upper())
    has_1 = any(x in s for x in ["YU", "YUS", "LINE1"])
    has_2 = any(x in s for x in ["BD", "LINE2", "BLOORDANFORTH"])
    if has_1 and has_2:
        return "1+2"
    if has_1:
        return "1"
    if has_2:
        return "2"
    if "SRT" in s or "LINE3" in s or "SCARBOROUGH" in s:
        return "3"
    if "SHP" in s or "LINE4" in s or "SHEPPARD" in s:
        return "4"
    return "UNK"


def load_toronto_delays(data_dir: Path, years: set[int]) -> pd.DataFrame:
    pieces = []
    for path in sorted(data_dir.glob("*.xlsx")):
        name = path.name.lower()
        if "delay" not in name or "codes" in name:
            continue
        explicit = re.findall(r"20\d{2}", name)
        if explicit and not any(int(y) in years for y in explicit):
            continue
        for sheet, frame in pd.read_excel(path, sheet_name=None).items():
            cols = {str(c).strip().lower(): c for c in frame.columns}
            required = ["date", "time", "station", "code", "min delay", "min gap", "line"]
            if not all(c in cols for c in required):
                continue
            f = frame[[cols[c] for c in required] + ([cols["bound"]] if "bound" in cols else [])].copy()
            f.columns = required + (["bound"] if "bound" in cols else [])
            f["source_file"] = path.name
            f["source_sheet"] = str(sheet)
            pieces.append(f)
    if not pieces:
        raise RuntimeError(f"No Toronto delay records found for {sorted(years)}")
    d = pd.concat(pieces, ignore_index=True)
    d["date"] = pd.to_datetime(d.date, errors="coerce").dt.normalize()
    d["year"] = d.date.dt.year
    d = d[d.year.isin(years)].copy()
    minutes = d.time.map(parse_time_minutes)
    d["timestamp"] = d.date + pd.to_timedelta(minutes.fillna(0), unit="m")
    d["hour"] = d.timestamp.dt.hour
    d["dow"] = d.timestamp.dt.dayofweek
    d["month"] = d.timestamp.dt.month
    d["min_delay"] = to_float(d["min delay"]).fillna(0).clip(lower=0, upper=360)
    d["min_gap"] = to_float(d["min gap"]).fillna(0).clip(lower=0, upper=360)
    d["code"] = d.code.fillna("UNKNOWN").astype(str).str.strip().str.upper()
    d["line_std"] = d.line.map(normalize_line)
    d["event_id"] = [f"toronto-{y}-{i:07d}" for i, y in enumerate(d.year.astype(int))]
    return d.reset_index(drop=True)


def load_weather(data_dir: Path, through_year: int) -> pd.DataFrame:
    raw_path = data_dir / "meteostat_71624_hourly.csv.gz"
    cols = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]
    w = pd.read_csv(raw_path, header=None, names=cols, compression="gzip")
    utc = pd.to_datetime(w.date.astype(str) + " " + w.hour.astype(str).str.zfill(2) + ":00", utc=True)
    w["timestamp_hour"] = utc.dt.tz_convert("America/Toronto").dt.tz_localize(None)
    w = w[w.timestamp_hour.dt.year <= through_year].copy()
    wet_codes = set(range(7, 27))
    w["wet_weather"] = w.coco.map(lambda x: float(x in wet_codes) if pd.notna(x) else np.nan)
    observed_precip = to_float(w.prcp)
    w.loc[observed_precip > 0, "wet_weather"] = 1.0
    return w[["timestamp_hour", "temp", "wspd", "wet_weather"]]


def add_temporal_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out.dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out.month - 1) / 12)
    out["weekend"] = (out.dow >= 5).astype(float)
    out["peak"] = (((out.hour >= 7) & (out.hour < 10)) | ((out.hour >= 16) & (out.hour < 19))).astype(float)
    return out


def add_mechanism_flags(d: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = d.copy()
    text = out[text_col].fillna("").astype(str)
    for col, pattern in MECHANISM_PATTERNS.items():
        out[col] = text.str.contains(pattern, case=False, regex=True, na=False).astype(float)
    return out


def prepare_toronto(
    data_dir: Path,
    years: set[int],
    registry: dict[str, dict],
    code_desc: dict[str, str],
    thresholds: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    d = load_toronto_delays(data_dir, years)
    unique_stations = d.station.fillna("").astype(str).unique()
    station_map = {s: match_station(s, registry) for s in unique_stations}
    matched = d.station.fillna("").astype(str).map(station_map)
    d["station_key"] = matched.map(lambda x: x[0])
    d["station_score"] = matched.map(lambda x: x[1])
    d["station_matched"] = (d.station_key != "").astype(float)
    median_demand = float(np.median([v["daily_volume"] for v in registry.values()]))
    median_degree = float(np.median([v["graph_degree_rank"] for v in registry.values()]))
    median_route = float(np.median([v["route_count_rank"] for v in registry.values()]))
    d["daily_volume"] = d.station_key.map(lambda k: registry.get(k, {}).get("daily_volume", median_demand))
    d["demand_rank"] = d.station_key.map(lambda k: registry.get(k, {}).get("daily_volume_rank", 0.5))
    d["degree_rank"] = d.station_key.map(lambda k: registry.get(k, {}).get("graph_degree_rank", median_degree))
    d["route_rank"] = d.station_key.map(lambda k: registry.get(k, {}).get("route_count_rank", median_route))
    d["transfer"] = d.station_key.map(lambda k: registry.get(k, {}).get("transfer", 0.0))
    d["line_count_scaled"] = d.line_std.map({"1+2": 0.5, "1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25, "UNK": 0.0})
    d["description"] = d.code.map(code_desc).fillna("")
    d["text"] = np.where(
        d.description.str.len() > 0,
        "Transit incident: " + d.description,
        "Transit incident category " + d.code,
    )
    d = add_temporal_features(d)
    weather = load_weather(data_dir, max(years))
    d["timestamp_hour"] = d.timestamp.dt.floor("h")
    d = d.merge(weather, on="timestamp_hour", how="left")
    d["temperature_scaled"] = to_float(d.temp) / 20.0
    d["wind_scaled"] = to_float(d.wspd) / 50.0
    d["weather_missing"] = d[["temp", "wspd", "wet_weather"]].isna().any(axis=1).astype(float)
    d["wet_weather"] = to_float(d.wet_weather).fillna(0.0)
    d["temperature_scaled"] = d.temperature_scaled.fillna(0.0)
    d["wind_scaled"] = d.wind_scaled.fillna(0.0)
    d = add_mechanism_flags(d)
    d["demand_rate"] = d.daily_volume / (18 * 60)
    d["exposed_demand"] = d.min_delay * d.demand_rate

    train = d.year <= 2022
    if thresholds is None:
        nominal = (
            d.loc[train & (d.min_gap > 0)]
            .groupby(["line_std", "peak"], dropna=False)
            .min_gap.quantile(0.25)
            .to_dict()
        )
        d["nominal_gap"] = [float(nominal.get((line, peak), 5.0)) for line, peak in zip(d.line_std, d.peak)]
        d["gap_excess"] = (d.min_gap - d.nominal_gap).clip(lower=0)
        d["gap_burden"] = d.gap_excess * d.demand_rate
        thresholds = {
            "severe_delay": quantile_higher(d.loc[train, "min_delay"], 0.90),
            "high_burden": quantile_higher(d.loc[train, "exposed_demand"], 0.90),
            "gap_burden": quantile_higher(d.loc[train, "gap_burden"], 0.90),
            "nominal_gap": {f"{k[0]}|{int(k[1])}": float(v) for k, v in nominal.items()},
        }
    else:
        nominal = {}
        for k, v in thresholds["nominal_gap"].items():
            line, peak = k.rsplit("|", 1)
            nominal[(line, float(peak))] = float(v)
        d["nominal_gap"] = [float(nominal.get((line, float(peak)), 5.0)) for line, peak in zip(d.line_std, d.peak)]
        d["gap_excess"] = (d.min_gap - d.nominal_gap).clip(lower=0)
        d["gap_burden"] = d.gap_excess * d.demand_rate
    d["severe_delay"] = (d.min_delay >= float(thresholds["severe_delay"])).astype(int)
    d["high_burden"] = (d.exposed_demand >= float(thresholds["high_burden"])).astype(int)
    d["high_gap_burden"] = (d.gap_burden >= float(thresholds["gap_burden"])).astype(int)
    return d, thresholds


def prepare_nyc(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[(d.checkpoint == 0) & (d.year == 2023)].copy()
    d["timestamp"] = pd.to_datetime(d.start_time)
    d["hour"] = d.timestamp.dt.hour
    d["dow"] = d.timestamp.dt.dayofweek
    d["month"] = d.timestamp.dt.month
    d["line_count_scaled"] = to_float(d.initial_line_count).fillna(0).clip(0, 4) / 4
    d["station_matched"] = (to_float(d.matched_complexes).fillna(0) > 0).astype(float)
    d["demand_rank"] = empirical_rank(d.demand_rate_sum, d.demand_rate_sum)
    d["degree_rank"] = empirical_rank(d.graph_degree_mean, d.graph_degree_mean)
    d["route_rank"] = empirical_rank(d.route_count_mean, d.route_count_mean)
    d["transfer"] = to_float(d.transfer_share).fillna(0).clip(0, 1)
    d["temperature_scaled"] = to_float(d.temperature_c).fillna(0) / 20.0
    d["wet_weather"] = (to_float(d.precipitation_mm).fillna(0) > 0).astype(float)
    d["wind_scaled"] = to_float(d.wind_kmh).fillna(0) / 50.0
    d["weather_missing"] = d[["temperature_c", "precipitation_mm", "wind_kmh"]].isna().any(axis=1).astype(float)
    d = add_mechanism_flags(d)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend", "peak"]:
        d[col] = to_float(d[col]).fillna(0)
    return d.reset_index(drop=True)


def encode_texts(model: SentenceTransformer, texts: pd.Series | list[str], batch_size: int = 256) -> np.ndarray:
    vals = pd.Series(texts).fillna("").astype(str).tolist()
    unique = list(dict.fromkeys(vals))
    emb = model.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    lookup = {t: emb[i] for i, t in enumerate(unique)}
    return np.asarray([lookup[t] for t in vals], dtype=np.float32)


def fit_representation(
    nyc: pd.DataFrame,
    toronto: pd.DataFrame,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model = SentenceTransformer(model_name, device="cuda")
    all_text = pd.concat([nyc.text, toronto.text], ignore_index=True)
    all_emb = encode_texts(model, all_text)
    n_source = len(nyc)
    src_emb, tgt_emb = all_emb[:n_source], all_emb[n_source:]
    train_rows = np.r_[np.arange(n_source), n_source + np.flatnonzero((toronto.year <= 2022).to_numpy())]
    rng = np.random.default_rng(SEED)
    if len(train_rows) > 30000:
        train_rows = rng.choice(train_rows, 30000, replace=False)
    pca = PCA(n_components=min(PCA_DIM, all_emb.shape[1], len(train_rows) - 1), random_state=SEED)
    pca.fit(all_emb[train_rows])
    src_text = pca.transform(src_emb).astype(np.float32)
    tgt_text = pca.transform(tgt_emb).astype(np.float32)
    src_num = nyc[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    tgt_num = toronto[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    scaler = StandardScaler()
    scaler.fit(np.vstack([src_num, tgt_num[toronto.year.to_numpy() <= 2022]]))
    src_num = scaler.transform(src_num).astype(np.float32)
    tgt_num = scaler.transform(tgt_num).astype(np.float32)
    state = {
        "pca": pca,
        "scaler": scaler,
        "model_name": model_name,
        "pca_dim": src_text.shape[1],
        "numeric_columns": NUMERIC_COLUMNS,
    }
    return np.hstack([src_text, src_num]), np.hstack([tgt_text, tgt_num]), state


def transform_representation(frame: pd.DataFrame, state: dict) -> np.ndarray:
    model = SentenceTransformer(state["model_name"], device="cuda")
    emb = encode_texts(model, frame.text)
    text_x = state["pca"].transform(emb).astype(np.float32)
    num = frame[state["numeric_columns"]].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    num_x = state["scaler"].transform(num).astype(np.float32)
    return np.hstack([text_x, num_x])


def logistic_fit(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    anchor: np.ndarray | None = None,
    maxiter: int = 220,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    d = x64.shape[1]
    prevalence = np.clip(y64.mean(), 1e-5, 1 - 1e-5)
    if anchor is None:
        anchor = np.r_[np.zeros(d), math.log(prevalence / (1 - prevalence))]
        penalize_to = np.zeros(d)
    else:
        anchor = np.asarray(anchor, dtype=np.float64).copy()
        penalize_to = anchor[:d].copy()

    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:d], theta[d]
        z = x64 @ w + b
        p = expit(z)
        loss = np.mean(np.logaddexp(0.0, z) - y64 * z)
        diff = w - penalize_to
        loss += 0.5 * lam * np.mean(diff * diff)
        residual = p - y64
        grad_w = x64.T @ residual / len(y64) + (lam / d) * diff
        grad_b = residual.mean()
        return float(loss), np.r_[grad_w, grad_b]

    result = minimize(fun, anchor, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6})
    if not result.success and result.nit < 2:
        raise RuntimeError(f"Logistic optimization failed: {result.message}")
    return result.x.astype(np.float32)


def predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return expit(np.asarray(x) @ theta[:-1] + theta[-1])


def stratified_sample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n < 0 or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), max(2, round(n * len(pos) / len(y))))
    n_neg = min(len(neg), n - n_pos)
    if n_pos + n_neg < n:
        extra_pool = np.setdiff1d(np.arange(len(y)), np.r_[pos[:0], neg[:0]])
        selected = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        remaining = np.setdiff1d(extra_pool, selected)
        selected = np.r_[selected, rng.choice(remaining, n - len(selected), replace=False)]
    else:
        selected = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    rng.shuffle(selected)
    return selected


def ece10(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.linspace(0, 1, 11)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            out += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(out)


def metric_row(
    y: np.ndarray,
    p: np.ndarray,
    burden: np.ndarray,
    model: str,
    target: str,
    split: str,
) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    burden = np.asarray(burden, dtype=float)
    order = np.argsort(-p)
    row = {
        "split": split,
        "target": target,
        "model": model,
        "n": len(y),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece10": ece10(y, p),
    }
    for frac in [0.10, 0.20]:
        k = max(1, int(math.ceil(frac * len(y))))
        chosen = order[:k]
        row[f"top{int(frac * 100)}_positive_recall"] = float(y[chosen].sum() / max(1, y.sum()))
        row[f"top{int(frac * 100)}_precision"] = float(y[chosen].mean())
        row[f"top{int(frac * 100)}_burden_capture"] = float(burden[chosen].sum() / max(1e-9, burden.sum()))
    return row


def bootstrap_ap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, reps: int = 600) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    deltas = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta_ap": float(np.mean(deltas)), "ci_low": float(lo), "ci_high": float(hi), "reps": len(deltas)}


def select_source_lambda(x: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    cut = int(0.8 * len(y))
    rows = []
    for lam in LAMBDA_GRID:
        theta = logistic_fit(x[:cut], y[:cut], lam)
        p = predict(theta, x[cut:])
        rows.append({"lambda": lam, "ap": average_precision_score(y[cut:], p)})
    table = pd.DataFrame(rows)
    best = float(table.sort_values(["ap", "lambda"], ascending=[False, True]).iloc[0]["lambda"])
    return best, table


def select_target_lambdas(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    source_theta: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    idx = stratified_sample(y_train, min(5000, len(y_train)), SEED)
    rows = []
    for method, anchor in [("target_only", None), ("anchored_transfer", source_theta)]:
        for lam in LAMBDA_GRID:
            theta = logistic_fit(x_train[idx], y_train[idx], lam, anchor=anchor)
            p = predict(theta, x_val)
            rows.append({"method": method, "lambda": lam, "ap": average_precision_score(y_val, p)})
    table = pd.DataFrame(rows)
    selected = {}
    for method, g in table.groupby("method"):
        selected[method] = float(g.sort_values(["ap", "lambda"], ascending=[False, True]).iloc[0]["lambda"])
    return selected, table


def feature_masks(pca_dim: int) -> dict[str, np.ndarray]:
    d = pca_dim + len(NUMERIC_COLUMNS)
    all_idx = np.arange(d)
    text_idx = np.arange(pca_dim)
    num_index = {name: pca_dim + i for i, name in enumerate(NUMERIC_COLUMNS)}
    weather = [num_index[x] for x in ["temperature_scaled", "wet_weather", "wind_scaled", "weather_missing"]]
    demand_network = [num_index[x] for x in ["station_matched", "demand_rank", "degree_rank", "route_rank", "transfer"]]
    mechanism = [num_index[x] for x in MECHANISM_PATTERNS]
    time = [num_index[x] for x in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "weekend", "peak"]]
    return {
        "full": all_idx,
        "semantic_only": text_idx,
        "context_no_text": np.arange(pca_dim, d),
        "no_weather": np.setdiff1d(all_idx, weather),
        "no_demand_network": np.setdiff1d(all_idx, demand_network),
        "no_mechanism": np.setdiff1d(all_idx, np.r_[text_idx, mechanism]),
        "time_only": np.asarray(time),
        "demand_network_only": np.asarray(demand_network),
    }


def choose_ensemble_alpha(y: np.ndarray, p_target: np.ndarray, p_transfer: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in np.linspace(0, 1, 11):
        p = alpha * p_transfer + (1 - alpha) * p_target
        rows.append({"alpha_transfer": float(alpha), "ap": average_precision_score(y, p)})
    table = pd.DataFrame(rows)
    best = float(table.sort_values(["ap", "alpha_transfer"], ascending=[False, True]).iloc[0].alpha_transfer)
    return best, table


def build_development_report(
    out_dir: Path,
    data_quality: dict,
    metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    fewshot: pd.DataFrame,
    config: dict,
) -> None:
    lines = [
        "# NYC to Toronto locked cross-city disruption triage",
        "",
        "## Design",
        "",
        "NYC 2023 onset alerts form the source domain. Toronto 2014-2022 forms the target training set; Toronto 2023 is used for model selection. Toronto 2024 remained unread during this phase.",
        "",
        "Realized TTC delay and gap are outcomes only. Inputs are incident mechanism text, calendar/time, line scope, station demand and graph context, and contemporaneous weather.",
        "",
        "## Data quality",
        "",
        f"- NYC source events: {data_quality['nyc_n']:,}",
        f"- Toronto train events: {data_quality['toronto_train_n']:,}",
        f"- Toronto 2023 validation events: {data_quality['toronto_val_n']:,}",
        f"- Toronto station match rate: {data_quality['station_match_rate']:.1%}",
        f"- TTC code-description coverage: {data_quality['description_coverage']:.1%}",
        f"- Severe-delay threshold learned on 2014-2022: {config['thresholds']['severe_delay']:.1f} min",
        f"- High-burden threshold: {config['thresholds']['high_burden']:.2f} expected exposed passengers",
        "",
        "## 2023 development results",
        "",
    ]
    for target in TARGETS:
        lines.append(f"### {target}")
        sub = metrics[metrics.target == target].sort_values("average_precision", ascending=False)
        for r in sub.itertuples(index=False):
            lines.append(
                f"- {r.model}: AP {r.average_precision:.3f}, AUC {r.roc_auc:.3f}, "
                f"Top-20% positive recall {r.top20_positive_recall:.1%}, burden capture {r.top20_burden_capture:.1%}."
            )
        lines.append("")
    lines += [
        "## Interpretation before the locked test",
        "",
        "The development result is used only to lock hyperparameters and the target/transfer ensemble. Claims about external validity are deferred until Toronto 2024 is evaluated once.",
        "",
        "The ablation and few-shot CSV files contain the mechanism, network-demand, weather, and transfer comparisons used to test the paper hypotheses.",
    ]
    (out_dir / "development_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_development(out_dir: Path, ablations: pd.DataFrame, fewshot: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for target, marker in [("severe_delay", "o"), ("high_burden", "s")]:
        sub = fewshot[fewshot.target == target]
        for method, ls in [("target_only", "--"), ("anchored_transfer", "-")]:
            g = sub[sub.method == method].groupby("budget_label").agg(budget=("budget", "first"), ap=("ap", "mean"), sd=("ap", "std")).sort_values("budget")
            axes[0].plot(g.budget, g.ap, marker=marker, linestyle=ls, label=f"{target}: {method}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Labeled Toronto training events")
    axes[0].set_ylabel("2023 average precision")
    axes[0].set_title("Few-shot transfer value")
    axes[0].legend(fontsize=8)
    high = ablations[ablations.target == "high_burden"].sort_values("average_precision")
    axes[1].barh(high.variant, high.average_precision, color="#247BA0")
    axes[1].axvline(high.prevalence.iloc[0], color="#B33F40", linestyle=":", label="prevalence")
    axes[1].set_xlabel("2023 average precision")
    axes[1].set_title("High-burden feature ablation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "development_transfer_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def development(args) -> None:
    data_dir = Path(args.toronto_data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "ttc-subway-delay-2024.xlsx").exists():
        raise RuntimeError("Toronto 2024 file is present. Move it out before the locked development phase.")
    registry_frame, registry = build_station_registry(data_dir)
    code_desc = parse_code_descriptions(data_dir / "ttc-subway-delay-codes.xlsx")
    toronto, thresholds = prepare_toronto(data_dir, set(range(2014, 2024)), registry, code_desc)
    assert toronto.year.max() == 2023 and 2024 not in set(toronto.year)
    nyc = prepare_nyc(Path(args.nyc_snapshots))
    x_source, x_target, representation = fit_representation(nyc, toronto, args.embedding_model)
    train_mask = toronto.year.to_numpy() <= 2022
    val_mask = toronto.year.to_numpy() == 2023
    x_train, x_val = x_target[train_mask], x_target[val_mask]
    burden_val = toronto.loc[val_mask, "exposed_demand"].to_numpy(float)
    metrics_rows = []
    tuning_rows = []
    fewshot_rows = []
    ablation_rows = []
    model_state = {}
    masks = feature_masks(representation["pca_dim"])

    for target in TARGETS:
        y_source = nyc[SOURCE_LABEL[target]].to_numpy(int)
        y_train = toronto.loc[train_mask, target].to_numpy(int)
        y_val = toronto.loc[val_mask, target].to_numpy(int)
        source_lam, source_table = select_source_lambda(x_source, y_source)
        source_table["target"] = target
        source_table["stage"] = "source"
        source_table["method"] = "nyc_source"
        tuning_rows.append(source_table)
        source_theta = logistic_fit(x_source, y_source, source_lam)
        target_lams, target_table = select_target_lambdas(x_train, y_train, x_val, y_val, source_theta)
        target_table["target"] = target
        target_table["stage"] = "target"
        tuning_rows.append(target_table)
        theta_target = logistic_fit(x_train, y_train, target_lams["target_only"])
        theta_transfer = logistic_fit(x_train, y_train, target_lams["anchored_transfer"], anchor=source_theta)
        p_source = predict(source_theta, x_val)
        p_target = predict(theta_target, x_val)
        p_transfer = predict(theta_transfer, x_val)
        alpha, alpha_table = choose_ensemble_alpha(y_val, p_target, p_transfer)
        alpha_table["target"] = target
        alpha_table["stage"] = "ensemble"
        alpha_table["method"] = "ensemble"
        tuning_rows.append(alpha_table)
        p_ensemble = alpha * p_transfer + (1 - alpha) * p_target
        prior = np.full(len(y_val), y_train.mean())
        for name, pred in [
            ("prior", prior),
            ("nyc_zero_shot", p_source),
            ("toronto_target_only", p_target),
            ("nyc_anchored_transfer", p_transfer),
            ("locked_ensemble", p_ensemble),
        ]:
            metrics_rows.append(metric_row(y_val, pred, burden_val, name, target, "toronto_2023_validation"))

        for variant, idx in masks.items():
            theta = logistic_fit(x_train[:, idx], y_train, target_lams["target_only"])
            pred = predict(theta, x_val[:, idx])
            row = metric_row(y_val, pred, burden_val, variant, target, "toronto_2023_validation")
            row["variant"] = variant
            ablation_rows.append(row)

        for budget in FEWSHOT_BUDGETS:
            actual_budget = len(y_train) if budget < 0 else min(budget, len(y_train))
            label = "all" if budget < 0 else str(actual_budget)
            for seed in FEWSHOT_SEEDS:
                idx = stratified_sample(y_train, actual_budget, seed)
                for method, anchor, lam in [
                    ("target_only", None, target_lams["target_only"]),
                    ("anchored_transfer", source_theta, target_lams["anchored_transfer"]),
                ]:
                    theta = logistic_fit(x_train[idx], y_train[idx], lam, anchor=anchor)
                    pred = predict(theta, x_val)
                    fewshot_rows.append(
                        {
                            "target": target,
                            "method": method,
                            "budget": actual_budget,
                            "budget_label": label,
                            "seed": seed,
                            "ap": average_precision_score(y_val, pred),
                            "auc": roc_auc_score(y_val, pred),
                        }
                    )

        combined_mask = toronto.year.to_numpy() <= 2023
        x_combined = x_target[combined_mask]
        y_combined = toronto.loc[combined_mask, target].to_numpy(int)
        final_target = logistic_fit(x_combined, y_combined, target_lams["target_only"])
        final_transfer = logistic_fit(x_combined, y_combined, target_lams["anchored_transfer"], anchor=source_theta)
        final_ablations = {}
        for variant, idx in masks.items():
            final_ablations[variant] = logistic_fit(x_combined[:, idx], y_combined, target_lams["target_only"])
        model_state[target] = {
            "source_lambda": source_lam,
            "target_lambdas": target_lams,
            "ensemble_alpha": alpha,
            "source_theta": source_theta,
            "target_theta": final_target,
            "transfer_theta": final_transfer,
            "ablation_thetas": final_ablations,
            "train_prevalence": float(y_combined.mean()),
        }

    gap_y_train = toronto.loc[train_mask, "high_gap_burden"].to_numpy(int)
    gap_y_val = toronto.loc[val_mask, "high_gap_burden"].to_numpy(int)
    gap_lams, gap_tune = select_target_lambdas(x_train, gap_y_train, x_val, gap_y_val, model_state["high_burden"]["source_theta"])
    gap_tune["target"] = "high_gap_burden"
    gap_tune["stage"] = "target"
    tuning_rows.append(gap_tune)
    gap_theta = logistic_fit(x_train, gap_y_train, gap_lams["target_only"])
    gap_pred = predict(gap_theta, x_val)
    gap_burden_val = toronto.loc[val_mask, "gap_burden"].to_numpy(float)
    metrics_rows.append(metric_row(gap_y_val, gap_pred, gap_burden_val, "toronto_target_only", "high_gap_burden", "toronto_2023_validation"))
    combined_mask = toronto.year.to_numpy() <= 2023
    model_state["high_gap_burden"] = {
        "target_lambdas": gap_lams,
        "target_theta": logistic_fit(
            x_target[combined_mask],
            toronto.loc[combined_mask, "high_gap_burden"].to_numpy(int),
            gap_lams["target_only"],
        ),
        "train_prevalence": float(toronto.loc[combined_mask, "high_gap_burden"].mean()),
    }

    metrics = pd.DataFrame(metrics_rows)
    ablations = pd.DataFrame(ablation_rows)
    fewshot = pd.DataFrame(fewshot_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    data_quality = {
        "nyc_n": len(nyc),
        "toronto_train_n": int(train_mask.sum()),
        "toronto_val_n": int(val_mask.sum()),
        "station_match_rate": float(toronto.station_matched.mean()),
        "station_exact_rate": float((toronto.station_score == 1).mean()),
        "description_coverage": float((toronto.description.str.len() > 0).mean()),
        "n_station_registry": len(registry),
        "n_gtfs_graph_nodes": int((registry_frame.graph_degree > 0).sum()),
        "years": toronto.groupby("year").size().astype(int).to_dict(),
        "line_distribution": toronto.line_std.value_counts(normalize=True).to_dict(),
    }
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "development_policy": "NYC 2023 source; Toronto <=2022 train; Toronto 2023 validation; Toronto 2024 unread",
        "thresholds": thresholds,
        "numeric_columns": NUMERIC_COLUMNS,
        "pca_dim": representation["pca_dim"],
        "models": {
            target: {
                "source_lambda": state.get("source_lambda"),
                "target_lambdas": state["target_lambdas"],
                "ensemble_alpha": state.get("ensemble_alpha"),
            }
            for target, state in model_state.items()
        },
        "toronto_2024_url": TORONTO_2024_URL,
    }
    state = {
        "representation": representation,
        "registry": registry,
        "code_desc": code_desc,
        "thresholds": thresholds,
        "models": model_state,
        "feature_masks": masks,
        "train_codes": sorted(toronto.loc[toronto.year <= 2023, "code"].unique().tolist()),
        "data_quality": data_quality,
        "config": config,
    }
    state_path = out_dir / "locked_state.joblib"
    joblib.dump(state, state_path, compress=3)
    config["locked_state_sha256"] = sha256_file(state_path)
    json_dump(config, out_dir / "locked_config.json")
    json_dump(data_quality, out_dir / "data_quality.json")
    metrics.to_csv(out_dir / "development_metrics.csv", index=False)
    ablations.to_csv(out_dir / "development_ablations.csv", index=False)
    fewshot.to_csv(out_dir / "development_fewshot.csv", index=False)
    tuning.to_csv(out_dir / "development_tuning.csv", index=False)
    build_development_report(out_dir, data_quality, metrics, ablations, fewshot, config)
    plot_development(out_dir, ablations, fewshot)
    print("DEVELOPMENT_LOCKED")
    print(json.dumps({
        "state_sha256": config["locked_state_sha256"],
        "data_quality": data_quality,
        "metrics": metrics[["target", "model", "average_precision", "top20_positive_recall", "top20_burden_capture"]].to_dict(orient="records"),
    }, indent=2, default=str))


def plot_final(out_dir: Path, predictions: pd.DataFrame, metrics: pd.DataFrame, ablations: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"nyc_zero_shot": "#F28F3B", "toronto_target_only": "#247BA0", "nyc_anchored_transfer": "#2E933C", "locked_ensemble": "#8B5E3C"}
    for target, ls in [("severe_delay", "--"), ("high_burden", "-")]:
        sub = metrics[(metrics.target == target) & metrics.model.isin(colors)]
        axes[0].bar(
            np.arange(len(sub)) + (0.2 if target == "high_burden" else -0.2),
            sub.average_precision,
            width=0.38,
            label=target,
        )
    axes[0].set_xticks(np.arange(4), ["zero-shot", "target", "transfer", "ensemble"], rotation=20)
    axes[0].set_ylabel("Average precision")
    axes[0].set_title("Locked Toronto 2024 test")
    axes[0].legend()
    high = ablations[ablations.target == "high_burden"].sort_values("average_precision")
    axes[1].barh(high.variant, high.average_precision, color="#247BA0")
    axes[1].set_xlabel("Average precision")
    axes[1].set_title("External feature ablation")
    for target, marker in [("severe_delay", "o"), ("high_burden", "s")]:
        sub = predictions[(predictions.target == target) & (predictions.model == "locked_ensemble")].sort_values("probability", ascending=False)
        y = sub.label.to_numpy()
        burden = sub.realized_burden.to_numpy(float)
        x = np.linspace(0, 1, len(sub), endpoint=True)
        capture = np.cumsum(burden) / max(1e-9, burden.sum())
        axes[2].plot(x, capture, marker=marker, markevery=[max(0, int(0.2 * len(x)) - 1)], label=target)
    axes[2].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[2].set_xlabel("Fraction of incidents prioritized")
    axes[2].set_ylabel("Fraction of realized burden captured")
    axes[2].set_title("Resource-constrained triage")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "locked_external_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_final_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ablations: pd.DataFrame,
    quality: dict,
) -> None:
    lines = [
        "# Locked external test: Toronto 2024",
        "",
        "## Integrity",
        "",
        f"- Locked state SHA-256 verified: `{quality['locked_state_sha256']}`",
        f"- Toronto 2024 events: {quality['n_test']:,}",
        f"- Station match rate: {quality['station_match_rate']:.1%}",
        f"- Cause-description coverage: {quality['description_coverage']:.1%}",
        f"- Previously unseen cause-code share: {quality['unseen_code_share']:.1%}",
        "",
        "## Primary results",
        "",
    ]
    for target in ["severe_delay", "high_burden", "high_gap_burden"]:
        sub = metrics[metrics.target == target].sort_values("average_precision", ascending=False)
        lines.append(f"### {target}")
        for r in sub.itertuples(index=False):
            lines.append(
                f"- {r.model}: AP {r.average_precision:.3f} versus prevalence {r.prevalence:.3f}; "
                f"AUC {r.roc_auc:.3f}; Top-20% positive recall {r.top20_positive_recall:.1%}; "
                f"realized burden capture {r.top20_burden_capture:.1%}."
            )
        lines.append("")
    lines += ["## Paired bootstrap AP differences", ""]
    for r in bootstrap.itertuples(index=False):
        lines.append(f"- {r.target}, {r.comparison}: ΔAP {r.delta_ap:+.3f}, 95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}].")
    lines += ["", "## Feature interpretation", ""]
    for target in TARGETS:
        sub = ablations[ablations.target == target].sort_values("average_precision", ascending=False)
        full = sub[sub.variant == "full"].average_precision.iloc[0]
        no_net = sub[sub.variant == "no_demand_network"].average_precision.iloc[0]
        no_weather = sub[sub.variant == "no_weather"].average_precision.iloc[0]
        no_mech = sub[sub.variant == "no_mechanism"].average_precision.iloc[0]
        lines.append(
            f"- {target}: full AP {full:.3f}; removing network-demand changes AP by {full-no_net:+.3f}, "
            f"removing weather by {full-no_weather:+.3f}, and removing mechanism information by {full-no_mech:+.3f}."
        )
    lines += [
        "",
        "## Paper-level conclusion rule",
        "",
        "A cross-city generalization claim is supported only if zero-shot AP exceeds prevalence and anchored transfer improves target-only learning at constrained label budgets. The operational claim is supported if the locked ensemble concentrates substantially more than 20% of realized burden in the top 20% of incidents.",
    ]
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def final_test(args) -> None:
    data_dir = Path(args.toronto_data)
    out_dir = Path(args.output)
    state_path = out_dir / "locked_state.joblib"
    config_path = out_dir / "locked_config.json"
    if not state_path.exists() or not config_path.exists():
        raise RuntimeError("Run --phase develop before the locked final test.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(state_path)
    if actual_hash != config["locked_state_sha256"]:
        raise RuntimeError("Locked state hash mismatch; refusing final evaluation.")
    test_path = data_dir / "ttc-subway-delay-2024.xlsx"
    if not test_path.exists():
        raise RuntimeError(f"Locked test file missing: {test_path}")
    state = joblib.load(state_path)
    toronto, _ = prepare_toronto(data_dir, {2024}, state["registry"], state["code_desc"], state["thresholds"])
    if set(toronto.year) != {2024}:
        raise RuntimeError("Final phase received non-2024 outcomes.")
    x = transform_representation(toronto, state["representation"])
    metrics_rows = []
    pred_rows = []
    ablation_rows = []
    masks = state["feature_masks"]
    burden_by_target = {
        "severe_delay": toronto.exposed_demand.to_numpy(float),
        "high_burden": toronto.exposed_demand.to_numpy(float),
        "high_gap_burden": toronto.gap_burden.to_numpy(float),
    }
    prediction_lookup = {}
    for target in TARGETS:
        m = state["models"][target]
        y = toronto[target].to_numpy(int)
        burden = burden_by_target[target]
        p_source = predict(m["source_theta"], x)
        p_target = predict(m["target_theta"], x)
        p_transfer = predict(m["transfer_theta"], x)
        alpha = float(m["ensemble_alpha"])
        p_ensemble = alpha * p_transfer + (1 - alpha) * p_target
        models = {
            "prior": np.full(len(y), m["train_prevalence"]),
            "nyc_zero_shot": p_source,
            "toronto_target_only": p_target,
            "nyc_anchored_transfer": p_transfer,
            "locked_ensemble": p_ensemble,
        }
        for name, p in models.items():
            metrics_rows.append(metric_row(y, p, burden, name, target, "toronto_2024_locked_test"))
            prediction_lookup[(target, name)] = p
            pred_rows.extend(
                {
                    "event_id": eid,
                    "timestamp": ts,
                    "target": target,
                    "model": name,
                    "label": int(label),
                    "probability": float(prob),
                    "realized_burden": float(b),
                    "code": code,
                    "station": station,
                    "line": line,
                }
                for eid, ts, label, prob, b, code, station, line in zip(
                    toronto.event_id, toronto.timestamp, y, p, burden, toronto.code, toronto.station, toronto.line_std
                )
            )
        for variant, idx in masks.items():
            theta = m["ablation_thetas"][variant]
            p = predict(theta, x[:, idx])
            row = metric_row(y, p, burden, variant, target, "toronto_2024_locked_test")
            row["variant"] = variant
            ablation_rows.append(row)

    gap_model = state["models"]["high_gap_burden"]
    gap_y = toronto.high_gap_burden.to_numpy(int)
    gap_p = predict(gap_model["target_theta"], x)
    metrics_rows.append(metric_row(gap_y, gap_p, toronto.gap_burden.to_numpy(float), "toronto_target_only", "high_gap_burden", "toronto_2024_locked_test"))
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    ablations = pd.DataFrame(ablation_rows)
    boot_rows = []
    for target in TARGETS:
        y = toronto[target].to_numpy(int)
        selected = prediction_lookup[(target, "locked_ensemble")]
        for baseline in ["nyc_zero_shot", "toronto_target_only"]:
            delta = bootstrap_ap_delta(y, selected, prediction_lookup[(target, baseline)])
            boot_rows.append({"target": target, "comparison": f"ensemble_minus_{baseline}", **delta})
    bootstrap = pd.DataFrame(boot_rows)

    quarter_rows = []
    quarters = toronto.timestamp.dt.quarter.to_numpy()
    for target in TARGETS:
        y = toronto[target].to_numpy(int)
        p = prediction_lookup[(target, "locked_ensemble")]
        for q in sorted(np.unique(quarters)):
            mask = quarters == q
            quarter_rows.append(
                metric_row(y[mask], p[mask], burden_by_target[target][mask], "locked_ensemble", target, f"2024_Q{q}")
            )
    quarters_df = pd.DataFrame(quarter_rows)
    seen_codes = set(state["train_codes"])
    quality = {
        "locked_state_sha256": actual_hash,
        "test_file_sha256": sha256_file(test_path),
        "n_test": len(toronto),
        "station_match_rate": float(toronto.station_matched.mean()),
        "description_coverage": float((toronto.description.str.len() > 0).mean()),
        "unseen_code_share": float((~toronto.code.isin(seen_codes)).mean()),
        "test_prevalence": {t: float(toronto[t].mean()) for t in ["severe_delay", "high_burden", "high_gap_burden"]},
    }
    metrics.to_csv(out_dir / "final_metrics.csv", index=False)
    predictions.to_parquet(out_dir / "final_predictions.parquet", index=False)
    ablations.to_csv(out_dir / "final_ablations.csv", index=False)
    bootstrap.to_csv(out_dir / "final_bootstrap.csv", index=False)
    quarters_df.to_csv(out_dir / "final_quarters.csv", index=False)
    json_dump(quality, out_dir / "final_data_quality.json")
    build_final_report(out_dir, metrics, bootstrap, ablations, quality)
    plot_final(out_dir, predictions, metrics, ablations)
    print("LOCKED_FINAL_COMPLETE")
    print(json.dumps({
        "quality": quality,
        "metrics": metrics[["target", "model", "average_precision", "roc_auc", "top20_positive_recall", "top20_burden_capture"]].to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
    }, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["develop", "final"])
    parser.add_argument("--toronto-data", default="/root/autodl-tmp/rail2road/raw/toronto")
    parser.add_argument("--nyc-snapshots", default="/root/autodl-tmp/rail2road/results/nyc_disruption_triage_screen/snapshots.parquet")
    parser.add_argument("--output", default="/root/autodl-tmp/rail2road/results/crosscity_triage_nyc_toronto")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    np.random.seed(SEED)
    if args.phase == "develop":
        development(args)
    else:
        final_test(args)


if __name__ == "__main__":
    main()
