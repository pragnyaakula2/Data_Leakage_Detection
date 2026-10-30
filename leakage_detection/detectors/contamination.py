"""Exact and near-duplicate train/test contamination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import Counter

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ContaminationConfig:
    threshold: float = 0.5
    decimals: int = 6
    near_duplicate_quantile: float = 0.05
    exact_excess_scale: float = 0.02
    near_excess_scale: float = 0.05
    significance_level: float = 0.01


class ContaminationLeakageDetector:
    """Compare test rows with train rows using exact hashes and nearest neighbours.

    ``run`` returns the per-type report and a boolean mask of matched test rows.
    Row IDs are carried through into the evidence when supplied.
    """

    def __init__(self, X_train: Any, X_test: Any, *, y_test: Any = None, model: Any = None,
                 row_id_train: Any = None, row_id_test: Any = None,
                 config: ContaminationConfig | None = None):
        self.X_train = np.asarray(X_train, dtype=float)
        self.X_test = np.asarray(X_test, dtype=float)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2 or self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must be 2D arrays with equal feature counts")
        self.row_id_train = np.arange(len(self.X_train)) if row_id_train is None else np.asarray(row_id_train)
        self.row_id_test = np.arange(len(self.X_test)) if row_id_test is None else np.asarray(row_id_test)
        self.y_test, self.model = (None if y_test is None else np.asarray(y_test)), model
        if len(self.row_id_train) != len(self.X_train) or len(self.row_id_test) != len(self.X_test):
            raise ValueError("Row ID arrays must match their corresponding feature arrays")
        self.config = config or ContaminationConfig()

    def run(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """Return ``(report, matched_test_mask, matched_train_mask)``."""
        cfg = self.config
        train_keys = [np.round(row, cfg.decimals).tobytes() for row in self.X_train]
        test_keys = [np.round(row, cfg.decimals).tobytes() for row in self.X_test]
        train_key_set = set(train_keys)
        exact = np.array([key in train_key_set for key in test_keys], dtype=bool)
        counts = Counter(train_keys)
        baseline_exact = float(np.mean([counts[k] > 1 for k in train_keys])) if train_keys else 0.0
        exact_rate = float(exact.mean()) if len(exact) else 0.0
        exact_p = float(binomtest(int(exact.sum()), len(exact), min(baseline_exact, 1.0),
                                  alternative="greater").pvalue) if len(exact) else 1.0
        excess_exact = max(0.0, exact_rate - baseline_exact) if exact_p < cfg.significance_level else 0.0

        near = np.zeros(len(self.X_test), dtype=bool)
        near_excess = 0.0
        near_p = 1.0
        epsilon = float("nan")
        if len(self.X_train) > 1 and len(self.X_test):
            nn = NearestNeighbors(n_neighbors=1).fit(self.X_train)
            d_test = nn.kneighbors(self.X_test, return_distance=True)[0][:, 0]
            d_train = nn.kneighbors(self.X_train, n_neighbors=2, return_distance=True)[0][:, 1]
            epsilon = max(float(np.quantile(d_train, cfg.near_duplicate_quantile)), 1e-12)
            near = d_test <= epsilon
            baseline_near = float(np.mean(d_train <= epsilon))
            observed_near = float(np.mean(near))
            near_p = float(binomtest(int(near.sum()), len(near), min(baseline_near, 1.0),
                                     alternative="greater").pvalue)
            if near_p < cfg.significance_level:
                near_excess = max(0.0, (observed_near - baseline_near) / max(1.0 - baseline_near, 1e-12))

        exact_score = float(np.clip(excess_exact / cfg.exact_excess_scale, 0.0, 1.0))
        near_score = float(np.clip(near_excess / cfg.near_excess_scale, 0.0, 1.0))
        exact_matched = exact if excess_exact > 0 else np.zeros_like(exact)
        near_matched = near if near_excess > 0 else np.zeros_like(near)
        matched_test = exact_matched | near_matched
        matched_train = np.zeros(len(self.X_train), dtype=bool)
        if matched_test.any() and len(self.X_train):
            if exact_matched.any():
                matched_keys = {test_keys[i] for i in np.flatnonzero(exact_matched)}
                matched_train |= np.array([key in matched_keys for key in train_keys])
            if near_matched.any():
                near_idx = NearestNeighbors(n_neighbors=1).fit(self.X_train).kneighbors(
                    self.X_test[near_matched], return_distance=False
                )[:, 0]
                matched_train[near_idx] = True

        mem_gap, mem_score = 0.0, 0.0
        if self.model is not None and self.y_test is not None and matched_test.sum() >= 30 and (~matched_test).sum() >= 30:
            probs = np.asarray(self.model.predict_proba(self.X_test))[:, 1]
            if len(np.unique(self.y_test[matched_test])) > 1 and len(np.unique(self.y_test[~matched_test])) > 1:
                mem_gap = float(roc_auc_score(self.y_test[matched_test], probs[matched_test])
                                - roc_auc_score(self.y_test[~matched_test], probs[~matched_test]))
                mem_score = float(np.clip(mem_gap / 0.2, 0.0, 1.0))
        score = 0.8 * max(exact_score, near_score) + 0.2 * mem_score
        evidence = {
            "exact_match_count": int(exact_matched.sum()),
            "near_match_count": int((near_matched & ~exact_matched).sum()),
            "matched_test_row_ids": self.row_id_test[matched_test].tolist(),
            "matched_train_row_ids": self.row_id_train[matched_train].tolist(),
            "exact_overlap_rate": exact_rate,
            "exact_train_baseline": baseline_exact,
            "exact_excess": excess_exact,
            "exact_p_value": exact_p,
            "near_excess": near_excess,
            "near_p_value": near_p,
            "near_distance_threshold": epsilon,
        }
        report = {"score": float(score), "flag": score >= cfg.threshold, "evidence": evidence,
                  "signals": {"exact_score": exact_score, "near_score": near_score,
                              "memorization_gap": mem_gap, "memorization_score": mem_score}}
        return report, matched_test, matched_train
