"""Convenience entry point for running all four leakage detectors."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .detectors.base import DetectorConfig, LeakageDetector

DETECTOR_NAMES = ("target", "contamination", "temporal", "preprocessing")


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def generate_json_report(report: dict[str, Any], output_path: str | Path | None = None) -> str:
    """Serialize a detector report as strict JSON and optionally write it to disk."""
    serialized = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    if output_path is not None:
        Path(output_path).write_text(serialized + "\n", encoding="utf-8")
    return serialized


def main(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    model: Any,
    ts_train: Any = None,
    ts_test: Any = None,
    *,
    feature_names: Any = None,
    row_id_train: Any = None,
    row_id_test: Any = None,
    thresholds: Any = None,
    detector_config: DetectorConfig | None = None,
    seed: int = 0,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run target, contamination, temporal, and preprocessing checks.

    ``model`` must be a fitted classifier with ``predict_proba``.
    Timestamps are optional; when omitted, the temporal detector returns an
    abstaining result. The returned JSON-safe mapping contains scores, binary
    flags, per-detector evidence, model-level signals, and an overall verdict.
    If ``report_path`` is provided, the report is also written there as JSON.
    """
    result = LeakageDetector(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        ts_train=ts_train,
        ts_test=ts_test,
        row_id_train=row_id_train,
        row_id_test=row_id_test,
        thresholds=thresholds,
        detector_config=detector_config,
        seed=seed,
    ).run()
    result["scores"] = {name: result[name]["score"] for name in DETECTOR_NAMES}
    result["flags"] = {name: bool(result[name]["flag"]) for name in DETECTOR_NAMES}
    result["evidence"] = {name: result[name]["evidence"] for name in DETECTOR_NAMES}
    safe_result = _json_safe(result)
    generate_json_report(safe_result, report_path)
    return safe_result
