import json
import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from leakage_detection import (
    ContaminationLeakageDetector,
    PreprocessingLeakageDetector,
    TargetLeakageConfig,
    TargetLeakageDetector,
    TemporalLeakageDetector,
    generate_json_report,
    main,
)


def make_binary_data(seed=12):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(120, 4))
    y_train = (X_train[:, 0] + 0.25 * rng.normal(size=120) > 0).astype(int)
    X_test = rng.normal(size=(50, 4))
    y_test = (X_test[:, 0] + 0.25 * rng.normal(size=50) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=12, random_state=seed).fit(X_train, y_train)
    return X_train, y_train, X_test, y_test, model


def test_target_detector_returns_ranked_feature_evidence():
    X_train, y_train, X_test, y_test, model = make_binary_data()
    config = TargetLeakageConfig(mi_subsample=120, surrogate_rows=120,
                                 permutation_rows=50, permutation_repeats=1)
    result = TargetLeakageDetector(X_train, y_train, X_test, y_test,
                                   config=config, seed=4).run(model)

    assert 0 <= result["score"] <= 1
    assert isinstance(result["flag"], (bool, np.bool_))
    assert len(result["evidence"]["ranked_features"]) == X_train.shape[1]
    assert "permutation_share_max" in result["signals"]


def test_fractional_discrete_features_are_encoded_for_mi_without_metric_warnings():
    rng = np.random.default_rng(22)
    categories = np.tile(np.array([0.125, 0.375, 0.625, 0.875]), 30)
    second_categories = np.tile(np.array([0.2, 0.4, 0.8]), 40)
    X_train = np.column_stack([categories, second_categories, rng.normal(size=120)])
    y_train = ((np.arange(120) % 4) == 0).astype(int)
    X_test = X_train[:40].copy()
    y_test = y_train[:40].copy()
    config = TargetLeakageConfig(mi_subsample=120, surrogate_rows=120)
    detector = TargetLeakageDetector(X_train, y_train, X_test, y_test, config=config)

    discrete = np.array([True, True, False])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_mi = mutual_info_classif(
            X_train, y_train, discrete_features=discrete, n_neighbors=3,
            random_state=detector.seed,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        signals = detector.data_level()

    assert not any("Clustering metrics expects discrete values" in str(w.message)
                   for w in caught)
    np.testing.assert_allclose(signals["mi"], expected_mi)


def test_contamination_detector_returns_masks_and_row_ids():
    X_train, _, X_test, _, _ = make_binary_data()
    X_test[:35] = X_train[:35]
    report, test_mask, train_mask = ContaminationLeakageDetector(
        X_train, X_test,
        row_id_train=np.arange(1000, 1120),
        row_id_test=np.arange(2000, 2050),
    ).run()

    assert test_mask.shape == (50,)
    assert train_mask.shape == (120,)
    assert report["evidence"]["exact_overlap_rate"] >= 0.7
    assert len(report["evidence"]["matched_test_row_ids"]) == int(test_mask.sum())


def test_temporal_detector_scores_timestamp_overlap_and_abstains_without_times():
    X_train, y_train, X_test, y_test, _ = make_binary_data()
    train_times = np.arange(120)
    test_times = np.arange(60, 110)
    result = TemporalLeakageDetector(
        X_train, y_train, X_test, y_test, train_times, test_times
    ).run()
    abstained = TemporalLeakageDetector(X_train, y_train, X_test, y_test).run()

    assert result["evidence"]["train_at_or_after_test_start_fraction"] > 0
    assert 0 <= result["score"] <= 1
    assert np.isnan(abstained["score"])
    assert not abstained["flag"]
    assert "abstained" in abstained["evidence"]


def test_preprocessing_detector_returns_fingerprints():
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(100, 3))
    X_test = rng.normal(size=(40, 3))
    result = PreprocessingLeakageDetector(
        X_train, X_test, feature_names=["a", "b", "c"]
    ).run()

    assert 0 <= result["score"] <= 1
    assert set(result["evidence"]) >= {"scaler", "imputation", "encoding"}
    assert set(result["signals"]) == {
        "scaler_fingerprint", "imputation_fingerprint", "encoding_fingerprint"
    }


def test_main_runs_all_detectors_and_generates_json_report(tmp_path):
    X_train, y_train, X_test, y_test, model = make_binary_data()
    report_path = tmp_path / "leakage-report.json"
    result = main(X_train, y_train, X_test, y_test, model,
                  ts_train=np.arange(120), ts_test=np.arange(120, 170),
                  report_path=report_path)
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(result["scores"]) == {"target", "contamination", "temporal", "preprocessing"}
    assert set(result["flags"]) == set(result["scores"])
    assert set(result["evidence"]) == set(result["scores"])
    assert result["verdict"] in {"clean", "leaky"}
    assert on_disk["verdict"] == result["verdict"]
    assert json.loads(generate_json_report(result))["scores"] == result["scores"]


def test_main_handles_single_class_test_labels():
    X_train, y_train, X_test, _, model = make_binary_data()
    single_class_test_labels = np.ones(len(X_test), dtype=int)

    result = main(X_train, y_train, X_test, single_class_test_labels, model)

    assert 0 <= result["target"]["score"] <= 1
    assert result["target"]["signals"]["permutation_share_max"] == 0
    assert result["flags"]["target"] == result["target"]["flag"]
    json.dumps(result, allow_nan=False)


def test_target_detector_handles_single_class_probability_output():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(80, 3))
    y_train = np.zeros(80, dtype=int)
    X_test = rng.normal(size=(40, 3))
    y_test = (X_test[:, 0] > 0).astype(int)
    one_class_model = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        X_train, y_train
    )
    config = TargetLeakageConfig(mi_subsample=80, surrogate_rows=80,
                                 permutation_rows=40, permutation_repeats=1)

    result = TargetLeakageDetector(X_train, y_train, X_test, y_test,
                                   config=config).run(one_class_model)

    assert np.all(result["signals"]["permutation_share_max"] == 0)


def test_main_detects_deliberately_injected_target_leakage():
    rng = np.random.default_rng(101)
    y_train = rng.integers(0, 2, size=240)
    y_test = rng.integers(0, 2, size=100)
    X_train = np.column_stack([y_train.astype(float), rng.normal(size=(240, 4))])
    X_test = np.column_stack([y_test.astype(float), rng.normal(size=(100, 4))])
    model = RandomForestClassifier(n_estimators=40, random_state=4).fit(X_train, y_train)

    result = main(X_train, y_train, X_test, y_test, model)

    target = result["target"]
    strongest = target["evidence"]["ranked_features"][0]
    assert strongest["feature"] == "feature_0"
    assert strongest["single_feature_strength"] == 1.0
    assert strongest["permutation_share"] > 0.9
    assert target["score"] >= 0.65
    assert target["flag"] is True
    assert result["verdict"] == "leaky"


def test_main_detects_exact_train_test_duplicate_contamination():
    rng = np.random.default_rng(102)
    X_train = rng.normal(size=(160, 5))
    y_train = (X_train[:, 0] + rng.normal(size=160) * 0.2 > 0).astype(int)
    X_test = rng.normal(size=(80, 5))
    y_test = (X_test[:, 0] + rng.normal(size=80) * 0.2 > 0).astype(int)
    X_test[:35] = X_train[:35]
    y_test[:35] = y_train[:35]
    train_ids = np.arange(1_000, 1_160)
    test_ids = np.arange(5_000, 5_080)
    model = RandomForestClassifier(n_estimators=30, random_state=5).fit(X_train, y_train)

    result = main(X_train, y_train, X_test, y_test, model,
                  row_id_train=train_ids, row_id_test=test_ids)

    contamination = result["contamination"]
    evidence = contamination["evidence"]
    assert evidence["exact_match_count"] == 35
    assert evidence["exact_overlap_rate"] == 35 / 80
    assert evidence["exact_p_value"] < 0.01
    assert set(test_ids[:35]) <= set(evidence["matched_test_row_ids"])
    assert evidence["exact_match_count"] <= len(evidence["matched_test_row_ids"])
    assert contamination["score"] > 0.75
    assert contamination["flag"] is True
    assert result["verdict"] == "leaky"


def test_main_detects_temporal_overlap_and_future_training_rows():
    rng = np.random.default_rng(103)
    X_train = rng.normal(loc=-2, scale=1, size=(120, 5))
    y_train = (rng.random(120) > 0.5).astype(int)
    X_test = rng.normal(loc=2, scale=1, size=(60, 5))
    y_test = (rng.random(60) > 0.5).astype(int)
    train_times = np.linspace(0, 150, len(X_train))
    test_times = np.linspace(100, 140, len(X_test))
    model = RandomForestClassifier(n_estimators=30, random_state=6).fit(X_train, y_train)

    result = main(X_train, y_train, X_test, y_test, model,
                  ts_train=train_times, ts_test=test_times)

    temporal = result["temporal"]
    evidence = temporal["evidence"]
    assert evidence["train_at_or_after_test_start_fraction"] >= 0.3
    assert evidence["train_after_test_end_fraction"] > 0.05
    assert evidence["auc_given_split"] is not None
    assert temporal["score"] > 0.7
    assert temporal["flag"] is True
    assert result["verdict"] == "leaky"


def test_main_detects_pooled_scaler_preprocessing_fingerprint():
    rng = np.random.default_rng(104)
    raw = rng.normal(loc=2, scale=3, size=(300, 6))
    # Simulate scaler statistics fitted on the pooled train+test rows, one of
    # the preprocessing detector's documented fingerprint signals.
    pooled_scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    X_train, X_test = pooled_scaled[:200], pooled_scaled[200:]
    y_train = (rng.random(len(X_train)) > 0.5).astype(int)
    y_test = (rng.random(len(X_test)) > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=30, random_state=7).fit(X_train, y_train)

    result = main(X_train, y_train, X_test, y_test, model)

    preprocessing = result["preprocessing"]
    scaler_evidence = preprocessing["evidence"]["scaler"]
    assert len(scaler_evidence) == X_train.shape[1]
    assert all(value == "POOLED-consistent" for value in scaler_evidence.values())
    assert preprocessing["signals"]["scaler_fingerprint"] == 1.0
    assert preprocessing["score"] == 1.0
    assert preprocessing["flag"] is True
    assert result["verdict"] == "leaky"
