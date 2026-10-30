#!/usr/bin/env python3
"""Recreate the Adult Income notebook's clean split and run Jackfruit.

Run from the project directory, optionally passing --data /path/to/adult.csv.
The source file must be a real Adult dataset; when no local file is found, the
loader follows the notebook's OpenML, UCI, then GitHub fallback order.
"""

from __future__ import annotations

import argparse
import glob
import io
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
except ImportError as exc:  # pragma: no cover - environment-specific guidance
    raise SystemExit(
        f"Missing dependency: {exc}. Install pandas, NumPy, scikit-learn, "
        "and the Jackfruit package dependencies."
    ) from exc

from leakage_detection import main as run_jackfruit


ADULT_RAW_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
CAT_COLS = [
    "workclass", "marital_status", "occupation", "relationship", "race",
    "sex", "native_country",
]
NUM_COLS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
MCAR_COLS = ["age", "fnlwgt"]
MASK_SEED = 2024
MCAR_RATE = 0.05
SPLIT = (0.6, 0.2, 0.2)


def _normalise_column(name: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _read_adult_file(path: Path) -> pd.DataFrame:
    """Read the notebook's headerless UCI files or a headed Adult CSV."""
    with path.open("r", errors="ignore") as stream:
        first_lines = [stream.readline() for _ in range(3)]
    skip = 1 if first_lines[0].startswith("|") else 0
    first = first_lines[skip].split(",")[0].strip()
    if first.replace(".", "", 1).isdigit():
        return pd.read_csv(path, header=None, names=ADULT_RAW_COLS,
                           skiprows=skip, skipinitialspace=True)
    return pd.read_csv(path, skiprows=skip, skipinitialspace=True)


def _read_adult_bytes(contents: bytes, source_name: str) -> pd.DataFrame:
    """Same schema detection as _read_adult_file, for network fallback data."""
    text = contents.decode("utf-8", errors="ignore")
    lines = text.splitlines()[:3]
    skip = 1 if lines and lines[0].startswith("|") else 0
    first = lines[skip].split(",")[0].strip()
    buffer = io.BytesIO(contents)
    if first.replace(".", "", 1).isdigit():
        return pd.read_csv(buffer, header=None, names=ADULT_RAW_COLS,
                           skiprows=skip, skipinitialspace=True)
    frame = pd.read_csv(buffer, skiprows=skip, skipinitialspace=True)
    if not len(frame.columns):
        raise ValueError(f"No columns found in {source_name}")
    return frame


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def load_adult_frame(data_path: Path | None, root: Path) -> tuple[pd.DataFrame, str]:
    """Use a local file first, then the notebook's real-data source fallbacks."""
    candidates = [data_path] if data_path else []
    if not data_path:
        default = root / "adult.csv"
        candidates.append(default)
        candidates.extend(
            Path(path) for path in sorted(glob.glob(str(root / "*")))
            if "adult" in Path(path).name.lower()
            and Path(path).suffix.lower() in {".csv", ".data", ".txt", ".test"}
            and Path(path) != default
        )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return _read_adult_file(candidate), f"local file {candidate}"
    if data_path is not None:
        raise FileNotFoundError(f"Adult data file not found: {data_path}")

    errors = []
    try:
        from sklearn.datasets import fetch_openml

        frame = fetch_openml("adult", version=2, as_frame=True).frame.copy()
        return frame, "OpenML adult (version 2)"
    except Exception as exc:
        errors.append(f"OpenML: {type(exc).__name__}")

    try:
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
        train = _read_adult_bytes(_download(base + "adult.data"), "UCI adult.data")
        test = _read_adult_bytes(_download(base + "adult.test"), "UCI adult.test")
        # UCI adult.test begins with a comment line; _read_adult_bytes skips it.
        return pd.concat([train, test], ignore_index=True), "UCI adult.data + adult.test"
    except Exception as exc:
        errors.append(f"UCI: {type(exc).__name__}")

    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
        return _read_adult_bytes(_download(url), "GitHub adult-all.csv"), "GitHub adult-all.csv"
    except Exception as exc:
        errors.append(f"GitHub mirror: {type(exc).__name__}")

    raise RuntimeError(
        "Could not load real Adult Income data. Place adult.csv/adult.data in "
        f"{root}, or pass --data PATH. Source errors: " + "; ".join(errors)
    )


def prepare_adult_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's label cleanup, deduplication, and MCAR mask."""
    frame = frame.copy()
    frame.columns = [_normalise_column(name) for name in frame.columns]
    if "income" not in frame.columns and "class" in frame.columns:
        frame = frame.rename(columns={"class": "income"})
    missing = [name for name in ADULT_RAW_COLS if name not in frame.columns]
    if missing:
        raise ValueError(f"Adult schema missing columns {missing}; got {list(frame.columns)}")

    raw = frame[ADULT_RAW_COLS].copy()
    for column in raw.columns:
        if not pd.api.types.is_numeric_dtype(raw[column]):
            raw[column] = raw[column].astype("object").map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    raw = raw.replace("?", np.nan)
    raw["income"] = raw["income"].astype(str).str.strip().str.rstrip(".")
    unexpected = set(raw["income"].unique()) - {"<=50K", ">50K"}
    if unexpected:
        raise ValueError(f"Unexpected Adult income labels: {sorted(unexpected)}")
    raw[NUM_COLS] = raw[NUM_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
    raw = raw.drop_duplicates().reset_index(drop=True)
    raw = raw.drop(columns=["education"])

    data = raw.copy().reset_index(drop=True)
    data.insert(0, "row_id", np.arange(len(data)))
    data["y"] = (data["income"] == ">50K").astype(int)
    data = data[["row_id"] + FEATURE_COLS + ["y"]]

    # Notebook's fixed MCAR mask, applied once before all seed-specific splits.
    mask_rng = np.random.default_rng(MASK_SEED)
    for column in MCAR_COLS:
        target_n = int(np.ceil(MCAR_RATE * len(data)))
        already_missing = int(data[column].isna().sum())
        if already_missing < target_n:
            observed = np.flatnonzero(data[column].notna().to_numpy())
            chosen = mask_rng.choice(observed, size=target_n - already_missing,
                                     replace=False)
            data.loc[chosen, column] = np.nan
    return data


def split_adult(data: pd.DataFrame, seed: int):
    """Return notebook-compatible train, validation, and test frames."""
    indices = np.arange(len(data))
    y_all = data["y"].to_numpy()
    train_frac, val_frac, test_frac = SPLIT
    train_idx, remainder_idx = train_test_split(
        indices, test_size=val_frac + test_frac, stratify=y_all, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=test_frac / (val_frac + test_frac),
        stratify=y_all[remainder_idx],
        random_state=seed,
    )

    def take(indices_):
        return data.iloc[np.sort(indices_)].reset_index(drop=True)

    return take(train_idx), take(val_idx), take(test_idx)


class AdultNotebookPipeline:
    """Notebook's frequency encode -> mean impute -> standardise pipeline."""

    def fit(self, train: pd.DataFrame, test: pd.DataFrame):
        self.cat_cols = [column for column in FEATURE_COLS if column in CAT_COLS]
        self.num_cols = [column for column in FEATURE_COLS if column not in self.cat_cols]
        self.enc_maps = {
            column: (train[column].value_counts(dropna=True) / len(train)).to_dict()
            for column in self.cat_cols
        }
        self.impute_means = {
            column: float(np.nanmean(train[column].to_numpy(dtype=float)))
            for column in self.num_cols
        }
        numeric = self._imputed_numeric(train)
        self.mu = numeric.mean(axis=0)
        std = numeric.std(axis=0, ddof=0)
        self.sd = np.where(std == 0, 1.0, std)
        self.feature_names = [f"{column}_freq" for column in self.cat_cols] + self.num_cols
        return self

    def _imputed_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.num_cols].to_numpy(dtype=float).copy()
        for index, column in enumerate(self.num_cols):
            values[np.isnan(values[:, index]), index] = self.impute_means[column]
        return values

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = [
            frame[column].map(self.enc_maps[column]).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
            for column in self.cat_cols
        ]
        encoded.append((self._imputed_numeric(frame) - self.mu) / self.sd)
        return np.hstack(encoded).astype(np.float64)


def make_model(name: str, seed: int):
    """Construct a notebook-configured estimator (full, non-quick mode)."""
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit("The notebook's xgb model requires xgboost; install it or choose --model rf.") from exc
        return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="auc", early_stopping_rounds=30,
                             random_state=seed, n_jobs=-1)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=10,
                             max_iter=300, random_state=seed)
    raise ValueError(f"Unknown model {name!r}")


def fit_model(name: str, X_train, y_train, X_val, y_val, seed: int):
    model = make_model(name, seed)
    if name == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Real Adult CSV/data/test file; default searches the current directory")
    parser.add_argument("--model", choices=("logreg", "rf", "xgb", "mlp"), default="rf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("adult_leakage_report.json"))
    args = parser.parse_args()
    root = Path.cwd()

    frame, source = load_adult_frame(args.data, root)
    data = prepare_adult_data(frame)
    train, val, test = split_adult(data, args.seed)
    pipeline = AdultNotebookPipeline().fit(train, test)
    X_train, X_test = pipeline.transform(train), pipeline.transform(test)
    y_train, y_test = train["y"].to_numpy(), test["y"].to_numpy()
    X_val, y_val = pipeline.transform(val), val["y"].to_numpy()
    model = fit_model(args.model, X_train, y_train, X_val, y_val, args.seed)

    report = run_jackfruit(
        X_train, y_train, X_test, y_test, model,
        feature_names=pipeline.feature_names,
        row_id_train=train["row_id"].to_numpy(),
        row_id_test=test["row_id"].to_numpy(),
        seed=args.seed,
        report_path=args.report,
    )
    print(f"Data source: {source}")
    print(f"Split sizes: train={len(train):,}, validation={len(val):,}, test={len(test):,}")
    print(f"Model: {args.model} | features: {len(pipeline.feature_names)}")
    print(f"Jackfruit verdict: {report['verdict']}")
    print(f"Scores: {report['scores']}")
    print(f"JSON report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
