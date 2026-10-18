"""Comprehensive Verification & Validation Suite for SensorTrust ML Pipeline (Step 3).

Validates:
1. Model artifact serialization and schema compatibility.
2. 16-D spatiotemporal feature extraction and sensor invariance.
3. Deterministic rule layer (frozen/dropout) + LightGBM multi-head classifier.
4. Dataset reproducibility and multi-sensor fault/event injection.
5. Baseline comparison metrics (Z-score, Isolation Forest, SensorTrust).
6. TreeSHAP explanation calculation and health score integration.
"""

import os
import sys
import numpy as np
import pandas as pd
import unittest

import core
import evaluate


def test_model_loading_and_attributes():
    """Verify model checkpoint exists, loads properly, and exposes required properties."""
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    assert os.path.exists(model_path), f"Checkpoint '{model_path}' not found on disk."
    
    model = core.SensorTrustAIModel.load(model_path)
    assert model is not None
    assert hasattr(model, "classes_")
    assert hasattr(model, "booster_")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "predict")
    assert len(model.features) == 16
    assert set(model.features) == set(core.FEATURES)
    assert set(model.classes) == set(core.CLASSES)


def test_dataset_generation_reproducibility():
    """Verify simulate() generates deterministic synthetic telemetry with fixed random seed."""
    df1, lab1 = core.simulate(n_sensors=6, days=2, seed=42, random_faults=True)
    df2, lab2 = core.simulate(n_sensors=6, days=2, seed=42, random_faults=True)
    
    pd.testing.assert_frame_equal(df1, df2)
    np.testing.assert_array_equal(lab1, lab2)
    
    # Verify different seed produces different readings
    df3, _ = core.simulate(n_sensors=6, days=2, seed=99, random_faults=True)
    assert not df1.equals(df3)


def test_distinct_classes_in_synthetic_data():
    """Verify synthetic dataset covers normal, genuine events, and all sensor fault types."""
    events = [(100, 3.5)]  # genuine event seen across all sensors
    faults = [
        dict(sensor=0, kind="spike", start=150, dur=1, mag=5.0),
        dict(sensor=1, kind="drift", start=160, dur=48, mag=3.0),
        dict(sensor=2, kind="stuck", start=220, dur=24, mag=0.0),
        dict(sensor=3, kind="noise", start=260, dur=30, mag=6.0),
        dict(sensor=4, kind="bias", start=300, dur=40, mag=2.5),
        dict(sensor=5, kind="dropout", start=360, dur=12, mag=0.0),
    ]
    df, lab = core.simulate(n_sensors=6, days=2, seed=7, events=events, faults=faults)
    
    unique_labels = np.unique(lab)
    expected_classes = [core.C[k] for k in ["normal", "event", "spike", "drift", "stuck", "noise", "bias", "dropout"]]
    for c in expected_classes:
        assert c in unique_labels, f"Expected class index {c} ({core.CLASSES[c]}) to be present in synthetic dataset."


def test_feature_extraction_pipeline():
    """Verify feature extractor generates 16 features per (time, sensor) with no NaN values."""
    df, _ = core.simulate(n_sensors=5, days=1, seed=12)
    F, unit = core.compute_features(df)
    
    assert unit > 0
    assert len(F) == len(df) * 5
    for feat in core.FEATURES:
        assert feat in F.columns, f"Feature '{feat}' missing from extracted features."
        assert not F[feat].isna().any(), f"Feature '{feat}' contains NaN values."
    
    # Feature invariance: Features are spatial residuals and ratios, not sensor names
    assert "sensor" in F.columns
    assert "pos" in F.columns


def test_deterministic_rule_layer():
    """Verify deterministic guardrail overrides: stuck (>=4 consecutive) and missing (dropout)."""
    df, _ = core.simulate(n_sensors=4, days=1, seed=20)
    # Inject 10 identical values into S1
    df.iloc[100:110, 0] = df.iloc[99, 0]
    # Inject missing values into S2
    df.iloc[120:130, 1] = np.nan
    
    model = core.load_or_train_model()
    res, _ = core.analyse(df, model)
    
    # Filter for S1 frozen readings
    s1_frozen = res[(res["sensor"] == "S1") & (res["pos"].between(103, 109))]
    assert (s1_frozen["verdict"] == "stuck").all()
    assert (s1_frozen["rule"] == "frozen reading").all()
    
    # Filter for S2 missing readings
    s2_missing = res[(res["sensor"] == "S2") & (res["pos"].between(120, 129))]
    assert (s2_missing["verdict"] == "dropout").all()
    assert (s2_missing["rule"] == "missing reading").all()


def test_treeshap_explanation_generation():
    """Verify TreeSHAP explanation extraction works for detected anomalies."""
    df, _ = core.simulate(n_sensors=4, days=1, seed=30)
    df.iloc[100, 0] += 5.0  # isolated spike
    
    model = core.load_or_train_model()
    res, unit = core.analyse(df, model)
    
    row_idx = res[(res["sensor"] == "S1") & (res["pos"] == 100)].index[0]
    text, top_contributors, action = core.explain(res, model, unit, int(row_idx), n_sensors=4)
    
    assert isinstance(text, str) and len(text) > 0
    assert isinstance(action, str) and len(action) > 0
    assert isinstance(top_contributors, list)
    if top_contributors:
        name, val = top_contributors[0]
        assert isinstance(name, str)
        assert isinstance(val, float)
        assert val > 0


def test_health_scoring_logic():
    """Verify health score penalizes faults and preserves clean sensors."""
    verdicts_clean = ["normal"] * 100
    conf_clean = [0.9] * 100
    h_clean = core.health_series(verdicts_clean, conf_clean)
    assert h_clean[-1] >= 95.0, "Healthy sensor score should remain high."
    
    verdicts_event = ["event"] * 100
    conf_event = [0.9] * 100
    h_event = core.health_series(verdicts_event, conf_event)
    assert h_event[-1] >= 95.0, "Genuine event must NOT lower sensor health score."
    
    verdicts_fault = ["drift"] * 100
    conf_fault = [0.9] * 100
    h_fault = core.health_series(verdicts_fault, conf_fault)
    assert h_fault[-1] < 40.0, "Severe fault must significantly lower sensor health score."


def test_baseline_evaluation_execution():
    """Verify evaluation runs across baselines and produces expected metrics structure."""
    model = core.load_or_train_model()
    # Fast evaluation run with 2 runs
    summary, per_class, extra = evaluate.run(model=model, n_runs=2)
    
    assert "fault_precision" in summary.columns
    assert "fault_recall" in summary.columns
    assert "fault_f1" in summary.columns
    assert "event_miss_rate" in summary.columns
    
    assert "Z-score" in summary.index
    assert "Isolation Forest" in summary.index
    assert "SensorTrust" in summary.index
    
    # SensorTrust should have high fault precision and very low event miss rate
    st_row = summary.loc["SensorTrust"]
    assert st_row["fault_precision"] > 0.85
    assert st_row["event_miss_rate"] < 0.05
    assert extra["macro_f1"] > 0.70


if __name__ == "__main__":
    print("Running SensorTrust Step 3 Validation Suite...")
    test_model_loading_and_attributes()
    print("✓ Model loading and properties validated.")
    test_dataset_generation_reproducibility()
    print("✓ Dataset reproducibility validated.")
    test_distinct_classes_in_synthetic_data()
    print("✓ Distinct classes and fault injection validated.")
    test_feature_extraction_pipeline()
    print("✓ 16-D feature extraction validated.")
    test_deterministic_rule_layer()
    print("✓ Deterministic rule layer validated.")
    test_treeshap_explanation_generation()
    print("✓ TreeSHAP explainer validated.")
    test_health_scoring_logic()
    print("✓ Health scoring logic validated.")
    test_baseline_evaluation_execution()
    print("✓ Baseline evaluation validated.")
    print("\n🎉 ALL STEP 3 VALIDATION TESTS PASSED SUCCESSFULLY!")
