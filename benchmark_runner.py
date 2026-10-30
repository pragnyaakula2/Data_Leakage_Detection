#!/usr/bin/env python3
"""Run the shared Jackfruit API on clean benchmark splits from the four notebooks.

Example:
  python3 benchmark_runner.py --data-dir ./data --models rf --seed 0

The runner reads real source datasets, reproduces each notebook's clean data
preparation/splitting pipeline, and writes per-dataset JSON plus one comparison
CSV under outputs/benchmark by default. Missing datasets are downloaded from
the notebook sources and cached under data/raw. It does not execute or edit
notebooks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
except ImportError as exc:  # pragma: no cover - environment-specific guidance
    raise SystemExit(
        f"Missing dependency: {exc}. Install pandas, NumPy, scikit-learn, "
        "and Jackfruit package dependencies."
    ) from exc

from leakage_detection import DetectorConfig, main as jackfruit_main


DATASET_ORDER = ("Adult Income", "Credit Card Fraud", "California Housing", "Rossmann")
MODEL_NAMES = ("logreg", "rf", "xgb", "mlp")


@dataclass
class DatasetSplit:
    name: str
    source: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    row_id_train: np.ndarray
    row_id_test: np.ndarray
    ts_train: Any = None
    ts_test: Any = None


def find_local_file(explicit: Path | None, data_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    hits = sorted({p for pattern in patterns for p in data_dir.glob(pattern) if p.is_file()})
    return hits[0] if hits else None


def download_cached(url: str, destination: Path, *, timeout: int = 300) -> Path:
    """Download to a temporary sibling, atomically cache, and reuse thereafter."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached {destination}")
        return destination
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        print(f"Downloading {url} -> {destination}")
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise IOError(f"Downloaded file was empty: {url}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def acquire_adult(explicit: Path | None, data_dir: Path, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    from run_adult_jackfruit import load_adult_frame

    local = find_local_file(explicit, data_dir,
                            ("adult.csv", "adult*.csv", "adult*.data", "adult*.test", "adult*.txt"))
    if local is not None:
        frame, source = load_adult_frame(local, data_dir)
        return frame, source

    cached_csv = raw_dir / "adult-all.csv"
    if cached_csv.is_file():
        return load_adult_frame(cached_csv, raw_dir)

    openml_cache = raw_dir / "adult-openml.csv"
    if openml_cache.is_file() and openml_cache.stat().st_size > 0:
        return load_adult_frame(openml_cache, raw_dir)

    # The notebook's first network fallback is OpenML adult version 2.
    try:
        from sklearn.datasets import fetch_openml

        frame = fetch_openml("adult", version=2, as_frame=True).frame.copy()
        temporary = openml_cache.with_name(openml_cache.name + ".part")
        frame.to_csv(temporary, index=False)
        temporary.replace(openml_cache)
        return frame, "OpenML adult (version 2)"
    except Exception:
        pass

    # The notebook's UCI fallback downloads and combines both official splits.
    uci_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    train_file, test_file = raw_dir / "adult.data", raw_dir / "adult.test"
    try:
        download_cached(uci_root + "adult.data", train_file, timeout=90)
        download_cached(uci_root + "adult.test", test_file, timeout=90)
        from run_adult_jackfruit import _read_adult_file
        return (pd.concat([_read_adult_file(train_file), _read_adult_file(test_file)], ignore_index=True),
                "UCI Adult adult.data + adult.test")
    except Exception as uci_error:
        github_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
        try:
            download_cached(github_url, cached_csv, timeout=90)
            frame, _ = load_adult_frame(cached_csv, raw_dir)
            return frame, "GitHub mirror adult-all.csv"
        except Exception as github_error:
            raise RuntimeError(
                "Could not acquire Adult Income from the notebook's UCI/GitHub sources: "
                f"UCI={type(uci_error).__name__}, GitHub={type(github_error).__name__}"
            ) from github_error


def _adult_split(frame: pd.DataFrame, seed: int) -> DatasetSplit:
    # Import the single-dataset script's notebook-extracted helpers to keep its
    # exact Adult cleaning, MCAR mask, stratified split and fit-scope pipeline.
    from run_adult_jackfruit import (
        AdultNotebookPipeline,
        prepare_adult_data,
        split_adult,
    )

    data = prepare_adult_data(frame)
    train, val, test = split_adult(data, seed)
    pipeline = AdultNotebookPipeline().fit(train, test)
    return DatasetSplit(
        name="Adult Income", source="Adult Income source file",
        X_train=pipeline.transform(train), y_train=train["y"].to_numpy(),
        X_val=pipeline.transform(val), y_val=val["y"].to_numpy(),
        X_test=pipeline.transform(test), y_test=test["y"].to_numpy(),
        feature_names=list(pipeline.feature_names),
        row_id_train=train["row_id"].to_numpy(), row_id_test=test["row_id"].to_numpy(),
    )


def load_adult(frame: pd.DataFrame, source: str, seed: int) -> DatasetSplit:
    split = _adult_split(frame, seed)
    split.source = source
    return split


def _frequency_pipeline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    output_categorical_first: bool,
    categorical_as_string: bool = False,
    zero_if_all_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Notebook fit scope: train-only frequency maps, means, then scaling."""
    frequency_maps = {}
    for column in categorical_columns:
        values = train[column].astype(str) if categorical_as_string else train[column]
        frequency_maps[column] = (
            values.value_counts(normalize=True).to_dict() if categorical_as_string
            else (values.value_counts(dropna=True) / len(train)).to_dict()
        )
    means = {
        column: float(np.nanmean(train[column].to_numpy(dtype=float)))
        for column in numeric_columns
    }
    if zero_if_all_missing:
        means = {column: (0.0 if not np.isfinite(value) else value)
                 for column, value in means.items()}
    train_numeric = _impute_numeric(train, numeric_columns, means)
    mean = train_numeric.mean(axis=0)
    std = train_numeric.std(axis=0, ddof=0)
    std[std == 0] = 1.0

    def transform(frame: pd.DataFrame) -> np.ndarray:
        numeric = (_impute_numeric(frame, numeric_columns, means) - mean) / std
        categorical = [
            (frame[column].astype(str) if categorical_as_string else frame[column])
            .map(frequency_maps[column]).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
            for column in categorical_columns
        ]
        categorical_matrix = np.hstack(categorical) if categorical else np.empty((len(frame), 0))
        result = np.hstack([categorical_matrix, numeric]) if output_categorical_first else np.hstack([numeric, categorical_matrix])
        return result.astype(np.float64)

    names = ([f"{column}_freq" for column in categorical_columns] + numeric_columns
             if output_categorical_first else numeric_columns + [f"{column}_freq" for column in categorical_columns])
    return transform(train), transform(val), transform(test), names


def _impute_numeric(frame: pd.DataFrame, columns: list[str], means: dict[str, float]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float).copy()
    for index, column in enumerate(columns):
        values[np.isnan(values[:, index]), index] = means[column]
    return values


def load_housing(path: Path, seed: int) -> DatasetSplit:
    with path.open("rb") as stream:
        is_zip = stream.read(4).startswith(b"PK")
    frame = pd.read_excel(path) if is_zip or path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    lower = {column.lower(): column for column in frame.columns}
    kaggle_map = {
        "longitude": "longitude", "latitude": "latitude", "housing_median_age": "housing_age",
        "total_rooms": "rooms", "total_bedrooms": "bedrooms", "population": "population",
        "households": "households_or_occupancy", "median_income": "median_income",
        "median_house_value": "price", "ocean_proximity": "ocean_proximity",
    }
    sklearn_map = {
        "longitude": "longitude", "latitude": "latitude", "houseage": "housing_age",
        "averooms": "rooms", "avebedrms": "bedrooms", "population": "population",
        "aveoccup": "households_or_occupancy", "medinc": "median_income",
        "medhouseval": "price",
    }
    if "median_house_value" in lower or "ocean_proximity" in lower:
        mapping = kaggle_map
    elif "medhouseval" in lower:
        mapping = sklearn_map
    else:
        raise ValueError("Housing file must use the notebook's Kaggle or scikit-learn column schema")
    missing = [key for key in mapping if key != "ocean_proximity" and key not in lower]
    if missing:
        raise ValueError(f"Housing file is missing required columns: {missing}")
    rename = {lower[key]: canonical for key, canonical in mapping.items() if key in lower}
    raw = frame[list(rename)].rename(columns=rename).copy()
    has_category = "ocean_proximity" in raw
    if has_category:
        raw["ocean_proximity"] = raw["ocean_proximity"].astype("object")
    raw = raw.dropna(subset=["price"]).reset_index(drop=True)

    data = raw.copy()
    data.insert(0, "row_id", np.arange(len(data)))
    price_median = float(data["price"].median())
    data["y"] = (data["price"] > price_median).astype(int)
    categorical = ["ocean_proximity"] if has_category else []
    numeric = [column for column in data.columns
               if column not in ("row_id", "price", "y", *categorical)]
    mask_rng = np.random.default_rng(2024)
    for column in ("rooms", "bedrooms", "population"):
        target_missing = int(math.ceil(0.05 * len(data)))
        missing_now = int(data[column].isna().sum())
        if missing_now < target_missing:
            observed = np.flatnonzero(data[column].notna().to_numpy())
            chosen = mask_rng.choice(observed, size=target_missing - missing_now, replace=False)
            data.loc[chosen, column] = np.nan
    data = data.drop(columns=["price"])

    indices = np.arange(len(data))
    train_idx, rest_idx = train_test_split(indices, test_size=0.4, stratify=data["y"].to_numpy(), random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5,
                                         stratify=data["y"].to_numpy()[rest_idx], random_state=seed)
    take = lambda idx: data.iloc[np.sort(idx)].reset_index(drop=True)
    train, val, test = take(train_idx), take(val_idx), take(test_idx)
    X_train, X_val, X_test, names = _frequency_pipeline(
        train, val, test, numeric_columns=numeric, categorical_columns=categorical,
        output_categorical_first=True,
    )
    return DatasetSplit(
        name="California Housing", source=str(path),
        X_train=X_train, y_train=train["y"].to_numpy(), X_val=X_val, y_val=val["y"].to_numpy(),
        X_test=X_test, y_test=test["y"].to_numpy(), feature_names=names,
        row_id_train=train["row_id"].to_numpy(), row_id_test=test["row_id"].to_numpy(),
    )


def acquire_housing(explicit: Path | None, data_dir: Path, raw_dir: Path) -> Path:
    local = find_local_file(explicit, data_dir,
                            ("*housing*.csv", "*housing*.xlsx", "*housing*.xls"))
    if local is not None:
        return local
    cached = raw_dir / "california_housing.csv"
    if cached.is_file() and cached.stat().st_size > 0:
        print(f"Using cached {cached}")
        return cached
    # The notebook supports scikit-learn's California Housing column schema.
    # fetch_california_housing downloads the official sklearn archive into raw_dir.
    try:
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing(data_home=str(raw_dir), as_frame=True)
        frame = dataset.frame
        frame.to_csv(cached, index=False)
        print(f"Cached scikit-learn California Housing table at {cached}")
        return cached
    except Exception as exc:
        raise RuntimeError(
            "Could not acquire California Housing using the notebook-supported "
            f"scikit-learn dataset source: {type(exc).__name__}: {exc}"
        ) from exc


def load_credit(path: Path, seed: int) -> DatasetSplit:
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Credit Card file is missing required columns: {missing}")
    raw = frame[columns].apply(pd.to_numeric, errors="raise").astype(float)
    raw["Class"] = raw["Class"].astype(int)
    if not set(raw["Class"].unique()) <= {0, 1}:
        raise ValueError("Credit Card Class labels must be 0/1")
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    raw = raw.drop_duplicates(subset=features + ["Class"]).sort_values("Time", kind="stable").reset_index(drop=True)
    data = raw.rename(columns={"Class": "y"}).copy()
    data.insert(0, "row_id", np.arange(len(data)))
    mask_rng = np.random.default_rng(2024)
    for column in ("V1", "V5", "V8"):
        target_missing = int(math.ceil(0.05 * len(data)))
        missing_now = int(data[column].isna().sum())
        if missing_now < target_missing:
            observed = np.flatnonzero(data[column].notna().to_numpy())
            chosen = mask_rng.choice(observed, size=target_missing - missing_now, replace=False)
            data.loc[chosen, column] = np.nan

    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    positive, negative = data[data.y == 1], data[data.y == 0]
    if len(negative) > 40_000:
        negative = negative.iloc[np.sort(rng.choice(len(negative), size=40_000, replace=False))]
    sampled = pd.concat([positive, negative]).sort_values(["Time", "row_id"], kind="stable").reset_index(drop=True)
    n = len(sampled)
    c1, c2, c3 = int(round(0.60 * n)), int(round(0.70 * n)), int(round(0.85 * n))
    train, val, test = (block.reset_index(drop=True) for block in
                        (sampled.iloc[:c1], sampled.iloc[c1:c2], sampled.iloc[c2:c3]))
    if min(int(part.y.sum()) for part in (train, val, test)) < 20:
        raise ValueError("Credit Card time block has fewer than 20 fraud cases; the notebook split is not viable")
    X_train, X_val, X_test, names = _frequency_pipeline(
        train, val, test, numeric_columns=features, categorical_columns=[],
        output_categorical_first=False,
    )
    return DatasetSplit(
        name="Credit Card Fraud", source=str(path),
        X_train=X_train, y_train=train["y"].to_numpy(), X_val=X_val, y_val=val["y"].to_numpy(),
        X_test=X_test, y_test=test["y"].to_numpy(), feature_names=names,
        row_id_train=train["row_id"].to_numpy(), row_id_test=test["row_id"].to_numpy(),
        ts_train=train["Time"].to_numpy(), ts_test=test["Time"].to_numpy(),
    )


def acquire_credit(explicit: Path | None, data_dir: Path, raw_dir: Path) -> Path:
    local = find_local_file(explicit, data_dir, ("creditcard.csv", "creditcard*.csv"))
    if local is not None:
        return local
    cached = raw_dir / "creditcard.csv"
    if cached.is_file() and cached.stat().st_size > 0:
        print(f"Using cached {cached}")
        return cached
    # Primary URL is the exact public mirror used by the notebook.
    credit_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    try:
        return download_cached(credit_url, cached)
    except Exception as mirror_error:
        # Match the notebook's OpenML fallback, while writing its table into raw_dir.
        try:
            from sklearn.datasets import fetch_openml

            frame = fetch_openml(data_id=1597, as_frame=True).frame.copy()
            frame.to_csv(cached, index=False)
            return cached
        except Exception as openml_error:
            raise RuntimeError(
                "Could not acquire Credit Card Fraud from the notebook's public "
                f"mirror or OpenML: mirror={type(mirror_error).__name__}, "
                f"OpenML={type(openml_error).__name__}"
            ) from openml_error


ROSS_NUMERIC = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear",
    "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Weeks", "PromoMonthActive",
]
ROSS_CATEGORICAL = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]


def load_rossmann(train_path: Path, store_path: Path, seed: int,
                  validation_path: Path | None = None, test_path: Path | None = None) -> DatasetSplit:
    def read_sales(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype={"StateHoliday": str}, low_memory=False, parse_dates=["Date"])

    train_raw = read_sales(train_path)
    store = pd.read_csv(store_path)
    labelled_frames = [train_raw]
    for optional in (validation_path, test_path):
        if optional is not None:
            candidate = read_sales(optional)
            if "Sales" in candidate.columns:
                labelled_frames.append(candidate)
    pool_raw = pd.concat(labelled_frames, ignore_index=True).drop_duplicates(
        subset=["Store", "Date"], keep="first"
    ).reset_index(drop=True)

    required = ["Store", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday",
                "SchoolHoliday", "DayOfWeek"]
    missing = [column for column in required if column not in pool_raw]
    if missing:
        raise ValueError(f"Rossmann sales file is missing required columns: {missing}")
    if "Store" not in store:
        raise ValueError("Rossmann store metadata must have a Store column")
    d = pool_raw[(pool_raw["Open"] == 1) & (pool_raw["Sales"] > 0)].copy()
    d["StateHoliday"] = d["StateHoliday"].astype(str).str.strip().replace({"0.0": "0", "nan": "0"})
    d = d.merge(store, on="Store", how="left")
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["Day"] = d["Date"].dt.day
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["CompetitionOpenMonths"] = ((d["Year"] - d["CompetitionOpenSinceYear"]) * 12
                                  + (d["Month"] - d["CompetitionOpenSinceMonth"]))
    d["Promo2Weeks"] = np.where(
        d["Promo2"] == 1,
        (d["Year"] - d["Promo2SinceYear"]) * 52 + (d["WeekOfYear"] - d["Promo2SinceWeek"]),
        np.nan,
    )
    d["PromoInterval"] = d["PromoInterval"].fillna("none").astype(str)
    month_abbr = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                  7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_names = d["Month"].map(month_abbr).to_numpy()
    d["PromoMonthActive"] = np.fromiter(
        (int(month in interval.split(",")) for month, interval in
         zip(month_names, d["PromoInterval"].to_numpy())), dtype=int, count=len(d)
    )
    pool = d[["Date", "Sales", "Customers"] + ROSS_NUMERIC + ROSS_CATEGORICAL].reset_index(drop=True)
    pool["row_id"] = np.arange(len(pool))

    unique_dates = np.sort(pool["Date"].unique())
    cut_points = np.round(np.cumsum((0.55, 0.15, 0.15, 0.15)) * len(unique_dates)).astype(int)
    cut_points[-1] = len(unique_dates)
    block_of_date = np.searchsorted(cut_points, np.arange(len(unique_dates)), side="right")
    block_names = np.array(["B1", "B2", "B3", "B4"])
    pool["block"] = block_names[block_of_date[np.searchsorted(unique_dates, pool["Date"].values)]]
    store_median = pool.loc[pool["block"] == "B1"].groupby("Store")["Sales"].median()
    pool["store_median"] = pool["Store"].map(store_median)
    if pool["store_median"].isna().any():
        pool = pool.dropna(subset=["store_median"]).reset_index(drop=True)
        pool["row_id"] = np.arange(len(pool))
    pool["y"] = (pool["Sales"] > pool["store_median"]).astype(int)
    pool = pool.drop(columns=["store_median"])

    rng = np.random.default_rng(seed)
    block_positions = {block: np.flatnonzero((pool["block"] == block).to_numpy())
                       for block in block_names}
    def sample(block: str, n_rows: int) -> pd.DataFrame:
        positions = block_positions[block]
        selected = rng.choice(positions, min(n_rows, len(positions)), replace=False)
        return pool.iloc[selected][["row_id", "Date", "y"] + ROSS_NUMERIC + ROSS_CATEGORICAL].reset_index(drop=True)
    train, val, test = sample("B1", 30_000), sample("B2", 8_000), sample("B3", 10_000)
    if any(train.empty or part.empty for part in (train, val, test)):
        raise ValueError("Rossmann clean time blocks are empty; check train/store source files")
    X_train, X_val, X_test, names = _frequency_pipeline(
        train, val, test, numeric_columns=ROSS_NUMERIC, categorical_columns=ROSS_CATEGORICAL,
        output_categorical_first=False, categorical_as_string=True, zero_if_all_missing=True,
    )
    return DatasetSplit(
        name="Rossmann", source=f"{train_path}; {store_path}",
        X_train=X_train, y_train=train["y"].to_numpy(), X_val=X_val, y_val=val["y"].to_numpy(),
        X_test=X_test, y_test=test["y"].to_numpy(), feature_names=names,
        row_id_train=train["row_id"].to_numpy(), row_id_test=test["row_id"].to_numpy(),
        ts_train=train["Date"].to_numpy(), ts_test=test["Date"].to_numpy(),
    )


def acquire_rossmann(train_arg: Path | None, store_arg: Path | None,
                     data_dir: Path, raw_dir: Path) -> tuple[Path, Path]:
    train_local = find_local_file(train_arg, data_dir, ("train.csv", "train*.csv"))
    store_local = find_local_file(store_arg, data_dir, ("store.csv", "store*.csv"))
    train_cache, store_cache = raw_dir / "rossmann_train.csv", raw_dir / "rossmann_store.csv"
    train_path = train_local or (train_cache if train_cache.is_file() and train_cache.stat().st_size else None)
    store_path = store_local or (store_cache if store_cache.is_file() and store_cache.stat().st_size else None)

    # The notebook asks for Kaggle's train.csv/store.csv and contains no public
    # download URL. Use the same public GitHub mirror used for project runs.
    urls = {
        "train": "https://raw.githubusercontent.com/Muosvr/rossmann-store-sales/master/data/train.csv",
        "store": "https://raw.githubusercontent.com/Muosvr/rossmann-store-sales/master/data/store.csv",
    }
    try:
        if train_path is None:
            train_path = download_cached(urls["train"], train_cache)
        if store_path is None:
            store_path = download_cached(urls["store"], store_cache)
    except Exception as exc:
        raise RuntimeError(
            "Could not acquire Rossmann train/store files from the public mirror. "
            "The original notebook requires Kaggle competition files; supply "
            f"--rossmann-train and --rossmann-store to use local copies. ({type(exc).__name__}: {exc})"
        ) from exc
    return train_path, store_path


def make_model(dataset: str, name: str, seed: int):
    rossmann = dataset == "Rossmann"
    if name == "logreg":
        kwargs = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
        if rossmann:
            kwargs["random_state"] = seed
        return LogisticRegression(**kwargs)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200 if rossmann else 300,
            min_samples_leaf=3 if rossmann else 2,
            n_jobs=-1, random_state=seed,
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit("Selected xgb model requires xgboost; install it or use --models rf.") from exc
        return XGBClassifier(
            n_estimators=500 if rossmann else 600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
            early_stopping_rounds=30, random_state=seed, n_jobs=-1,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, max_iter=200 if rossmann else 300, random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}")


def fit_model(split: DatasetSplit, model_name: str, seed: int):
    model = make_model(split.name, model_name, seed)
    if model_name == "xgb":
        model.fit(split.X_train, split.y_train,
                  eval_set=[(split.X_val, split.y_val)], verbose=False)
    else:
        model.fit(split.X_train, split.y_train)
    return model


def make_detector_config() -> DetectorConfig:
    """One identical detector configuration instance is used for all datasets."""
    return DetectorConfig()


def run_dataset(split: DatasetSplit, models: list[str], seed: int,
                detector_config: DetectorConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = {
        "dataset": split.name,
        "source": split.source,
        "split_sizes": {"train": len(split.y_train), "validation": len(split.y_val), "test": len(split.y_test)},
        "n_features": len(split.feature_names),
        "feature_names": split.feature_names,
        "detector_config": {
            "target": asdict(detector_config.target),
            "contamination": asdict(detector_config.contamination),
            "temporal": asdict(detector_config.temporal),
            "preprocessing": asdict(detector_config.preprocessing),
        },
        "runs": [],
    }
    rows = []
    for model_name in models:
        current = split
        model = fit_model(current, model_name, seed)
        probabilities = model.predict_proba(current.X_test)[:, 1]
        auc = float(roc_auc_score(current.y_test, probabilities))
        ap = float(average_precision_score(current.y_test, probabilities))
        result = jackfruit_main(
            current.X_train, current.y_train, current.X_test, current.y_test, model,
            ts_train=current.ts_train, ts_test=current.ts_test,
            feature_names=current.feature_names,
            row_id_train=current.row_id_train, row_id_test=current.row_id_test,
            seed=seed, detector_config=detector_config,
        )
        run = {"seed": seed, "model": model_name, "test_auc": auc,
               "test_average_precision": ap, "jackfruit": result}
        report["runs"].append(run)
        row = {
            "dataset": current.name, "seed": seed, "model": model_name,
            "n_train": len(current.y_train), "n_validation": len(current.y_val),
            "n_test": len(current.y_test), "n_features": len(current.feature_names),
            "test_auc": auc, "test_average_precision": ap,
            "verdict": result["verdict"],
        }
        for detector in ("target", "contamination", "temporal", "preprocessing"):
            row[f"score_{detector}"] = result["scores"][detector]
            row[f"flag_{detector}"] = result["flags"][detector]
        rows.append(row)
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument("--adult-data", type=Path)
    parser.add_argument("--credit-card-data", type=Path)
    parser.add_argument("--housing-data", type=Path)
    parser.add_argument("--rossmann-train", type=Path)
    parser.add_argument("--rossmann-store", type=Path)
    parser.add_argument("--rossmann-validation", type=Path)
    parser.add_argument("--rossmann-test", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["rf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="cache directory for downloaded source datasets (default: ./data/raw)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    adult_frame, adult_source = acquire_adult(args.adult_data, data_dir, raw_dir)
    cc_path = acquire_credit(args.credit_card_data, data_dir, raw_dir)
    housing_path = acquire_housing(args.housing_data, data_dir, raw_dir)
    ross_train, ross_store = acquire_rossmann(
        args.rossmann_train, args.rossmann_store, data_dir, raw_dir
    )
    ross_val = args.rossmann_validation.expanduser().resolve() if args.rossmann_validation else None
    ross_test = args.rossmann_test.expanduser().resolve() if args.rossmann_test else None

    # All four benchmark datasets receive this same configuration instance.
    detector_config = make_detector_config()
    splits = [
        load_adult(adult_frame, adult_source, args.seed),
        load_credit(cc_path, args.seed),
        load_housing(housing_path, args.seed),
        load_rossmann(ross_train, ross_store, args.seed, ross_val, ross_test),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for split in splits:
        print(f"Running {split.name}: {len(split.y_train):,} train / {len(split.y_test):,} test, "
              f"{len(split.feature_names)} features; models={args.models}, seed={args.seed}")
        dataset_report, rows = run_dataset(split, args.models, args.seed, detector_config)
        safe_name = re.sub(r"[^a-z0-9]+", "_", split.name.lower()).strip("_")
        report_path = args.output_dir / f"{safe_name}.json"
        report_path.write_text(json.dumps(dataset_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for row in rows:
            row["report_file"] = report_path.name
        comparison_rows.extend(rows)

    csv_path = args.output_dir / "combined_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(csv_path, index=False)
    print("Individual reports:")
    for split in splits:
        safe_name = re.sub(r"[^a-z0-9]+", "_", split.name.lower()).strip("_")
        print(f"  {args.output_dir / (safe_name + '.json')}")
    print(f"Combined comparison: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
