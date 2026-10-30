# Jackfruit leakage detection

Jackfruit is one dataset-independent leakage detection system. Its shared `main()` function runs the target, contamination, temporal, and preprocessing detectors over processed numeric arrays and a fitted binary classifier. The same interface works for Adult Income, California Housing, Credit Card Fraud, Rossmann, and other compatible datasets. Dataset loading, feature engineering, train/test splitting, and model fitting stay with the caller or the original notebooks.

The original notebooks remain unchanged and are still the full benchmark record. Jackfruit is a reusable baseline for the notebooks' principal checks; scenario generation, calibration, adversarial validation, performance-inflation analysis, and full experiment reporting remain notebook work.

## Install

From this directory:

```bash
python -m pip install -e .
```

Requires Python 3.10+, NumPy, SciPy, and scikit-learn.

## Common API

Prepare numeric, row-major `X_train` and `X_test` arrays with the same feature order; corresponding `y_train` and `y_test` labels; and a fitted binary classifier with `predict_proba`. Optionally supply aligned timestamps, feature names, row IDs, score thresholds, or detector-specific configuration. Jackfruit does not load or preprocess raw files.

```python
from leakage_detection import main

report = main(
    X_train, y_train, X_test, y_test, fitted_model,
    ts_train=train_timestamps,   # optional; omit if unavailable
    ts_test=test_timestamps,     # optional; omit if unavailable
    feature_names=feature_names,
    row_id_train=train_ids,      # optional; evidence only
    row_id_test=test_ids,        # optional; evidence only
    report_path="leakage_report.json",
)
print(report["verdict"])
print(report["scores"])
```

The result contains `target`, `contamination`, `temporal`, and `preprocessing` reports with scores, flags, signals, and evidence. It also contains aggregate `scores`, `flags`, and `evidence`, model-level signals, and the overall `verdict`. When timestamps are omitted, temporal detection abstains; its score is serialized as JSON `null`.

## Optional detector configuration

Use one `DetectorConfig` to tune any shared detector. This does not select dataset-specific implementations; each section configures the same detector used by `main()`.

```python
from leakage_detection import (
    DetectorConfig,
    TargetLeakageConfig,
    ContaminationConfig,
    TemporalConfig,
    PreprocessingConfig,
    main,
)

config = DetectorConfig(
    target=TargetLeakageConfig(permutation_rows=5_000, permutation_repeats=3),
    contamination=ContaminationConfig(decimals=6),
    temporal=TemporalConfig(max_probe_rows=20_000),
    preprocessing=PreprocessingConfig(tolerance=1e-7),
)

report = main(
    X_train, y_train, X_test, y_test, fitted_model,
    feature_names=feature_names,
    detector_config=config,
)
```

Thresholds can be overridden separately with `thresholds={"target": 0.7}`. A threshold override takes precedence over the threshold in `DetectorConfig`.

Notebook-specific tuning is still expressible through this shared configuration without dataset adapters. For example, the previous Credit Card target settings are `TargetLeakageConfig(permutation_rows=3_000, permutation_repeats=2)`. The previous Rossmann target settings are `permutation_rows=5_000`, `surrogate_rows=10_000`, `permutation_seed_offset=41`, `seed_offset=1`, and `surrogate_early_stopping=False`; its contamination precision remains the shared default of six decimals.

## JSON reports

Pass `report_path` to `main()` to write a strict JSON report. For a report already returned by `main()`, `generate_json_report(report, "leakage_report.json")` returns its formatted JSON and writes the same content to disk.
