"""Timestamp-based temporal leakage detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def _numeric_time(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64).astype(float)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    try:
        return arr.astype("datetime64[ns]").astype(np.int64).astype(float)
    except (TypeError, ValueError):
        return arr.astype(float)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


@dataclass(frozen=True)
class TemporalConfig:
    threshold: float = 0.5
    overlap_scale: float = 0.05
    auc_delta_scale: float = 0.05
    overlap_weight: float = 0.7
    auc_delta_weight: float = 0.3
    max_probe_rows: int = 20_000
    seed: int = 0


class TemporalLeakageDetector:
    """Detect training records within or after a test time window.

    The score combines timestamp overlap with the AUC difference between the
    supplied split and a chronological surrogate resplit. If timestamps are
    absent, the result abstains with a NaN score.
    """

    def __init__(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any,
                 ts_train: Any = None, ts_test: Any = None, *,
                 exclude_train_positions: Any = None, config: TemporalConfig | None = None):
        self.X_train, self.X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
        self.y_train, self.y_test = np.asarray(y_train), np.asarray(y_test)
        self.ts_train, self.ts_test = ts_train, ts_test
        self.exclude_train_positions = np.asarray([] if exclude_train_positions is None else exclude_train_positions, dtype=int)
        self.config = config or TemporalConfig()

    def run(self) -> dict[str, Any]:
        if self.ts_train is None or self.ts_test is None:
            return {"score": float("nan"), "flag": False,
                    "evidence": {"abstained": "timestamps not provided"}, "signals": {}}
        t_train, t_test = _numeric_time(self.ts_train), _numeric_time(self.ts_test)
        if len(t_train) != len(self.X_train) or len(t_test) != len(self.X_test):
            raise ValueError("Timestamp arrays must match their corresponding feature arrays")
        keep = np.ones(len(t_train), dtype=bool)
        keep[self.exclude_train_positions] = False
        if not keep.any() or not len(t_test):
            return {"score": float("nan"), "flag": False,
                    "evidence": {"abstained": "empty train or test time window"}, "signals": {}}

        lo, hi = float(np.min(t_test)), float(np.max(t_test))
        overlap = float(np.mean(t_train[keep] >= lo))
        future = float(np.mean(t_train[keep] > hi))
        cfg = self.config
        auc_given, auc_chronological = float("nan"), float("nan")
        auc_delta = 0.0
        if len(self.X_train) + len(self.X_test) <= cfg.max_probe_rows and len(np.unique(self.y_train[keep])) > 1:
            model = HistGradientBoostingClassifier(max_iter=100, random_state=cfg.seed)
            try:
                auc_given = _auc(self.y_test, model.fit(self.X_train[keep], self.y_train[keep]).predict_proba(self.X_test)[:, 1])
                Xall = np.vstack([self.X_train[keep], self.X_test])
                yall = np.r_[self.y_train[keep], self.y_test]
                tall = np.r_[t_train[keep], t_test]
                order = np.argsort(tall, kind="stable")
                n_tr = int(keep.sum())
                train_idx, test_idx = order[:n_tr], order[n_tr:]
                chrono = HistGradientBoostingClassifier(max_iter=100, random_state=cfg.seed)
                auc_chronological = _auc(yall[test_idx], chrono.fit(Xall[train_idx], yall[train_idx]).predict_proba(Xall[test_idx])[:, 1])
                if np.isfinite(auc_given) and np.isfinite(auc_chronological):
                    auc_delta = float(auc_given - auc_chronological)
            except (ValueError, IndexError):
                pass
        score = cfg.overlap_weight * np.clip(overlap / cfg.overlap_scale, 0.0, 1.0)
        score += cfg.auc_delta_weight * np.clip(auc_delta / cfg.auc_delta_scale, 0.0, 1.0)
        return {"score": float(score), "flag": score >= cfg.threshold,
                "evidence": {"train_at_or_after_test_start_fraction": overlap,
                             "train_after_test_end_fraction": future, "test_start": lo, "test_end": hi,
                             "auc_given_split": auc_given, "auc_chronological": auc_chronological,
                             "auc_delta": auc_delta},
                "signals": {"overlap_fraction": overlap, "future_fraction": future,
                            "chronological_auc_delta": auc_delta}}
