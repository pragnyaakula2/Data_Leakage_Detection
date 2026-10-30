"""Pipeline-agnostic checks over processed train/test arrays and a fitted model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from .target import TargetLeakageConfig, TargetLeakageDetector
from .contamination import ContaminationConfig, ContaminationLeakageDetector
from .preprocessing import PreprocessingConfig, PreprocessingLeakageDetector
from .temporal import TemporalConfig, TemporalLeakageDetector


@dataclass(frozen=True)
class DetectorConfig:
    """Optional settings for the shared four-detector system.

    Each section accepts the existing detector config type. Defaults preserve
    the common Jackfruit behavior; callers can tune any section without using
    a dataset-specific detector class.
    """

    target: TargetLeakageConfig = field(
        default_factory=lambda: TargetLeakageConfig(threshold=0.65)
    )
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def _probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.reshape(-1)
    return np.asarray(model.predict(X)).reshape(-1)


@dataclass
class LeakageDetector:
    """Run four evidence checks using processed arrays and a fitted black-box model.

    Parameters are row-major numeric arrays. Timestamps are optional; temporal
    leakage is reported as not assessed when either timestamp vector is absent.
    """

    model: Any
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    feature_names: Any = None
    ts_train: Any = None
    ts_test: Any = None
    row_id_train: Any = None
    row_id_test: Any = None
    thresholds: Any = None
    detector_config: DetectorConfig | None = None
    near_duplicate_quantile: float | None = None
    seed: int = 0
    contamination_decimals: int | None = None

    def __post_init__(self) -> None:
        self.X_train = np.asarray(self.X_train, dtype=float)
        self.X_test = np.asarray(self.X_test, dtype=float)
        self.y_train = np.asarray(self.y_train)
        self.y_test = np.asarray(self.y_test)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError("X_train and X_test must be two-dimensional arrays")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Train and test arrays must have the same number of features")
        if len(self.X_train) != len(self.y_train) or len(self.X_test) != len(self.y_test):
            raise ValueError("Each feature array must have the same row count as its labels")
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        else:
            self.feature_names = list(self.feature_names)
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError("feature_names length must match the number of columns")
        self.row_id_train = np.arange(len(self.X_train)) if self.row_id_train is None else np.asarray(self.row_id_train)
        self.row_id_test = np.arange(len(self.X_test)) if self.row_id_test is None else np.asarray(self.row_id_test)
        self.thresholds = {} if self.thresholds is None else dict(self.thresholds)
        if self.detector_config is None:
            self.detector_config = DetectorConfig()
        elif not isinstance(self.detector_config, DetectorConfig):
            raise TypeError("detector_config must be a DetectorConfig instance")

    def run(self) -> dict[str, Any]:
        """Return per-type scores, flags, evidence, and an overall verdict."""
        cfg = self.detector_config
        target_detector = TargetLeakageDetector(
            self.X_train, self.y_train, self.X_test, self.y_test, self.feature_names,
            config=replace(cfg.target, threshold=self.thresholds.get("target", cfg.target.threshold)),
            seed=self.seed,
        )
        target = target_detector.run(self.model)
        contamination_detector = ContaminationLeakageDetector(
            self.X_train, self.X_test, y_test=self.y_test, model=self.model,
            row_id_train=self.row_id_train, row_id_test=self.row_id_test,
            config=replace(
                cfg.contamination,
                threshold=self.thresholds.get("contamination", cfg.contamination.threshold),
                **({} if self.contamination_decimals is None else {"decimals": self.contamination_decimals}),
                **({} if self.near_duplicate_quantile is None else {
                    "near_duplicate_quantile": self.near_duplicate_quantile
                }),
            ),
        )
        contamination, _, matched_train = contamination_detector.run()
        temporal = TemporalLeakageDetector(
            self.X_train, self.y_train, self.X_test, self.y_test, self.ts_train, self.ts_test,
            exclude_train_positions=np.flatnonzero(matched_train),
            config=replace(
                cfg.temporal,
                threshold=self.thresholds.get("temporal", cfg.temporal.threshold),
                seed=self.seed,
            ),
        ).run()
        preprocessing = PreprocessingLeakageDetector(
            self.X_train, self.X_test, self.feature_names,
            config=replace(
                cfg.preprocessing,
                threshold=self.thresholds.get("preprocessing", cfg.preprocessing.threshold),
            ),
        ).run()
        p_train, p_test = _probabilities(self.model, self.X_train), _probabilities(self.model, self.X_test)
        model_signals = {
            "train_auc": _auc(self.y_train, p_train),
            "test_auc": _auc(self.y_test, p_test),
            "train_test_auc_gap": _auc(self.y_train, p_train) - _auc(self.y_test, p_test),
            "train_test_confidence_gap": float(np.mean(np.maximum(p_train, 1 - p_train)) - np.mean(np.maximum(p_test, 1 - p_test))),
        }
        checks = {"target": target, "contamination": contamination, "temporal": temporal, "preprocessing": preprocessing}
        flags = [item["flag"] for item in checks.values()]
        return {**checks, "model_signals": model_signals, "verdict": "leaky" if any(flags) else "clean"}
