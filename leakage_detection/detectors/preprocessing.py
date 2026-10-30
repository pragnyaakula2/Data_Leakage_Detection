"""Processed-array fingerprints for pooled-fit preprocessing leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    threshold: float = 0.5
    tolerance: float = 1e-7
    scaler_min_unique: int = 20
    frequency_max_unique: int = 10


class PreprocessingLeakageDetector:
    """Find scaler, imputation, and frequency-encoding fit-scope fingerprints."""

    def __init__(self, X_train: Any, X_test: Any, feature_names: Any = None, *,
                 config: PreprocessingConfig | None = None):
        self.X_train, self.X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2 or self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must be 2D arrays with equal feature counts")
        self.feature_names = ([f"feature_{i}" for i in range(self.X_train.shape[1])]
                              if feature_names is None else list(feature_names))
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError("feature_names length must match the number of columns")
        self.config = config or PreprocessingConfig()

    @staticmethod
    def _pooled_share(pooled: int, train: int) -> float:
        return pooled / (pooled + train) if pooled + train else 0.0

    def run(self) -> dict[str, Any]:
        tol, Xtr = self.config.tolerance, self.X_train
        Xpool = np.vstack([Xtr, self.X_test])
        evidence: dict[str, dict[str, Any]] = {"scaler": {}, "imputation": {}, "encoding": {}}
        n_unique = np.array([len(np.unique(Xpool[:, j])) for j in range(Xtr.shape[1])])

        train_count = pooled_count = 0
        for j in np.flatnonzero(n_unique > self.config.scaler_min_unique):
            tr, po = Xtr[:, j], Xpool[:, j]
            train_fit = ((abs(tr.mean()) < tol and abs(tr.std() - 1) < tol)
                         or (abs(tr.min()) < tol and abs(tr.max() - 1) < tol))
            pooled_fit = ((abs(po.mean()) < tol and abs(po.std() - 1) < tol)
                          or (abs(po.min()) < tol and abs(po.max() - 1) < tol)) and not train_fit
            train_count += int(train_fit)
            pooled_count += int(pooled_fit)
            evidence["scaler"][self.feature_names[j]] = ("train-consistent" if train_fit else
                                                          "POOLED-consistent" if pooled_fit else "neither")
        scale_score = self._pooled_share(pooled_count, train_count)

        train_count = pooled_count = 0
        for j in range(Xtr.shape[1]):
            vals, counts = np.unique(Xtr[:, j], return_counts=True)
            share = counts.max() / len(Xtr) if len(Xtr) else 0.0
            if not 0.01 <= share <= 0.80:
                continue
            spike = vals[counts.argmax()]
            train_rest = Xtr[Xtr[:, j] != spike, j]
            pool_rest = Xpool[Xpool[:, j] != spike, j]
            if not len(train_rest) or not len(pool_rest):
                continue
            train_fit = abs(spike - train_rest.mean()) < tol
            pooled_fit = abs(spike - pool_rest.mean()) < tol and abs(spike - train_rest.mean()) > 10 * tol
            train_count += int(train_fit)
            pooled_count += int(pooled_fit)
            evidence["imputation"][self.feature_names[j]] = {
                "spike_share": float(share),
                "verdict": "train-consistent" if train_fit else "POOLED-consistent" if pooled_fit else "neither",
            }
        imputation_score = self._pooled_share(pooled_count, train_count)

        train_count = pooled_count = 0
        for j in np.flatnonzero((n_unique >= 2) & (n_unique <= self.config.frequency_max_unique)):
            pooled_vals = np.unique(Xpool[:, j])
            vals = pooled_vals[pooled_vals != 0.0]
            if not len(vals) or vals.min() <= 0 or vals.max() > 1:
                continue
            train_fit = all(abs(v - np.mean(Xtr[:, j] == v)) < tol for v in vals)
            pooled_fit = all(abs(v - np.mean(Xpool[:, j] == v)) < tol for v in vals) and not train_fit
            train_count += int(train_fit)
            pooled_count += int(pooled_fit)
            evidence["encoding"][self.feature_names[j]] = "train-consistent" if train_fit else "POOLED-consistent" if pooled_fit else "neither"
        encoding_score = self._pooled_share(pooled_count, train_count)

        score = float(max(scale_score, imputation_score, encoding_score))
        if not any(evidence.values()):
            evidence["note"] = "no fingerprintable column"
        return {"score": score, "flag": score >= self.config.threshold, "evidence": evidence,
                "signals": {"scaler_fingerprint": scale_score, "imputation_fingerprint": imputation_score,
                            "encoding_fingerprint": encoding_score}}
