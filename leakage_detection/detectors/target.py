"""Reusable implementation of the notebooks' target-leakage detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def _clip01(x: Any) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def _robust_z(values: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    return (values - med) / (1.4826 * mad + 1e-9)


def _stratified_subsample(y: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if n <= limit:
        return np.arange(n)
    selected = []
    for cls in np.unique(y):
        class_idx = np.flatnonzero(y == cls)
        size = max(1, int(round(limit * len(class_idx) / n)))
        selected.append(rng.choice(class_idx, size=min(size, len(class_idx)), replace=False))
    return np.sort(np.concatenate(selected))


@dataclass(frozen=True)
class TargetLeakageConfig:
    mi_subsample: int = 20_000
    top_k: int = 5
    permutation_rows: int = 4_000
    permutation_repeats: int = 3
    permutation_seed_offset: int = 17
    surrogate_rows: int = 20_000
    threshold: float = 0.5
    seed_offset: int = 0
    surrogate_early_stopping: bool | None = None


class TargetLeakageDetector:
    """Notebook-style target detector: MI, single-feature strength, ablation, permutation.

    ``data_level`` can be cached per split. ``run`` computes model-specific
    permutation importance and returns a score/flag/evidence report.
    """

    def __init__(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any,
                 feature_names: Any = None, *, config: TargetLeakageConfig | None = None,
                 seed: int = 0, surrogate_factory: Any = None):
        self.X_train, self.X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
        self.y_train, self.y_test = np.asarray(y_train), np.asarray(y_test)
        if self.X_train.ndim != 2 or self.X_test.ndim != 2 or self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must be 2D arrays with the same feature count")
        if len(self.X_train) != len(self.y_train) or len(self.X_test) != len(self.y_test):
            raise ValueError("Feature arrays and label arrays must have matching row counts")
        self.feature_names = ([f"feature_{i}" for i in range(self.X_train.shape[1])]
                              if feature_names is None else list(feature_names))
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError("feature_names length must match the feature count")
        self.config = config or TargetLeakageConfig()
        self.base_seed = seed
        self.seed = seed + self.config.seed_offset
        if surrogate_factory is None:
            def make_surrogate():
                kwargs = {"max_iter": 100, "random_state": self.seed}
                if self.config.surrogate_early_stopping is not None:
                    kwargs["early_stopping"] = self.config.surrogate_early_stopping
                return HistGradientBoostingClassifier(**kwargs)

            self.surrogate_factory = make_surrogate
        else:
            self.surrogate_factory = surrogate_factory

    def _single_feature_strength(self, x: np.ndarray, y: np.ndarray) -> float:
        rank_auc = _auc(y, x)
        rank_auc = max(rank_auc, 1 - rank_auc) if np.isfinite(rank_auc) else 0.0
        if len(np.unique(y)) < 2 or len(y) < 6:
            return float(rank_auc)
        cv = StratifiedKFold(3, shuffle=True, random_state=self.seed)
        try:
            pred = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=self.seed),
                                     x.reshape(-1, 1), y, cv=cv, method="predict_proba")[:, 1]
            tree_auc = _auc(y, pred)
        except (ValueError, IndexError):
            tree_auc = float("nan")
        return float(max(rank_auc, tree_auc)) if np.isfinite(tree_auc) else float(rank_auc)

    def data_level(self) -> dict[str, Any]:
        """Compute MI, feature strength, and surrogate-ablation signals once per split."""
        cfg = self.config
        rng = np.random.default_rng(self.seed)
        n_features = self.X_train.shape[1]
        idx = _stratified_subsample(self.y_train, cfg.mi_subsample, rng)
        Xs, ys = self.X_train[idx], self.y_train[idx]
        discrete = np.array([len(np.unique(Xs[:, j])) <= 10 for j in range(n_features)])
        # Mutual-information treats each marked feature as categorical. Encode
        # low-cardinality values as integer category IDs so sklearn's clustering
        # metric validation does not misclassify fractional frequency codes as
        # continuous labels. This one-to-one recoding preserves the MI values.
        X_mi = Xs.copy()
        for j in np.flatnonzero(discrete):
            X_mi[:, j] = np.unique(Xs[:, j], return_inverse=True)[1]
        mi = mutual_info_classif(X_mi, ys, discrete_features=discrete,
                                 n_neighbors=3, random_state=self.seed)
        mi_z = _robust_z(mi)
        top = np.argsort(-mi)[:min(cfg.top_k, n_features)]
        strength = np.full(n_features, np.nan)
        for j in top:
            strength[j] = 0.5 * (self._single_feature_strength(Xs[:, j], ys)
                                 + self._single_feature_strength(self.X_test[:, j], self.y_test))

        drop = np.zeros(n_features)
        if len(np.unique(ys)) >= 2 and len(ys) >= 6:
            cv = StratifiedKFold(3, shuffle=True, random_state=self.seed)
            sub = rng.choice(len(Xs), min(cfg.surrogate_rows, len(Xs)), replace=False)
            Xa, ya = Xs[sub], ys[sub]
            auc_all = float(cross_val_score(self.surrogate_factory(), Xa, ya, cv=cv, scoring="roc_auc").mean())
            for j in top:
                keep = [k for k in range(n_features) if k != j]
                if keep:
                    without = cross_val_score(self.surrogate_factory(), Xa[:, keep], ya, cv=cv, scoring="roc_auc").mean()
                    drop[j] = auc_all - float(without)
        else:
            auc_all = float("nan")
        return {"mi": mi, "mi_z": mi_z, "top": top, "single_feature_strength": strength,
                "ablation_drop": drop, "surrogate_auc": auc_all}

    def model_level(self, model: Any) -> dict[str, np.ndarray]:
        """Calculate AUC-loss permutation importance using the trained black-box model."""
        if not hasattr(model, "predict_proba"):
            raise TypeError("model must implement predict_proba")
        rng = np.random.default_rng(self.base_seed + self.config.permutation_seed_offset)
        idx = _stratified_subsample(self.y_test, self.config.permutation_rows, rng)
        Xp, yp = self.X_test[idx], self.y_test[idx]
        importance = np.zeros(self.X_test.shape[1])
        # AUC is undefined for a single-class test subset. In that case the
        # permutation signal contributes no evidence, so keep zero importance.
        if len(np.unique(yp)) < 2:
            return {"permutation_importance": importance,
                    "permutation_share": np.zeros_like(importance)}

        def positive_probability(X: np.ndarray) -> np.ndarray:
            probabilities = np.asarray(model.predict_proba(X))
            if probabilities.ndim == 1:
                return probabilities
            if probabilities.shape[1] > 1:
                return probabilities[:, 1]
            # Some classifiers return one probability column when fitted on
            # only one class. Map it to the conventional positive class (1).
            classes = np.asarray(getattr(model, "classes_", []))
            return (probabilities[:, 0] if len(classes) == 1 and classes[0] == 1
                    else np.zeros(len(probabilities)))

        base = _auc(yp, positive_probability(Xp))
        for j in range(self.X_test.shape[1]):
            losses = []
            for _ in range(self.config.permutation_repeats):
                Xq = Xp.copy()
                Xq[:, j] = rng.permutation(Xq[:, j])
                losses.append(base - _auc(yp, positive_probability(Xq)))
            importance[j] = float(np.mean(losses))
        positive = np.clip(importance, 0, None)
        share = positive / positive.sum() if positive.sum() > 0 else np.zeros_like(positive)
        return {"permutation_importance": importance, "permutation_share": share}

    def score(self, data: dict[str, Any], model_signals: dict[str, Any]) -> dict[str, Any]:
        """Apply the notebook's weighted feature scores and return ranked evidence."""
        z = data["mi_z"]
        strength = np.nan_to_num(data["single_feature_strength"], nan=0.0)
        drop, share = data["ablation_drop"], model_signals["permutation_share"]
        s_mi = _clip01(np.log10(np.maximum(z, 1.0)) / 2.0)
        s_auc = _clip01((strength - 0.85) / 0.15)
        s_abl = _clip01(drop / 0.05)
        s_perm = _clip01((share - 0.3) / 0.5)
        values = 0.35 * s_mi + 0.30 * s_auc + 0.20 * s_abl + 0.15 * s_perm
        ranked = [{"feature": self.feature_names[j], "mi": float(data["mi"][j]), "mi_z": float(z[j]),
                   "single_feature_strength": (None if np.isnan(data["single_feature_strength"][j])
                                               else float(data["single_feature_strength"][j])),
                   "ablation_drop": float(drop[j]), "permutation_share": float(share[j]), "score": float(values[j])}
                  for j in range(len(self.feature_names))]
        ranked.sort(key=lambda row: row["score"], reverse=True)
        score = float(values.max()) if len(values) else 0.0
        return {"score": score, "flag": score >= self.config.threshold,
                "evidence": {"suspect_features": [r for r in ranked if r["score"] >= self.config.threshold],
                             "ranked_features": ranked},
                "signals": {"mi_z_max": float(z.max()), "single_feature_strength_max": float(np.max(strength)),
                            "ablation_drop_max": float(drop.max()), "permutation_share_max": float(share.max())}}

    def run(self, model: Any, data_level: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.score(data_level if data_level is not None else self.data_level(), self.model_level(model))


