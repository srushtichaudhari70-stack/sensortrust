"""Training script for SensorTrust AI Model.

Executes end-to-end dataset generation, spatial feature extraction,
cross-validation, LightGBM multi-head training, and serialization of model.joblib.

Run with:
    python train.py
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import core


def main(n_runs=25, seed=1000, output_path="model.joblib"):
    print("=" * 70)
    print("🚀 SENSORTRUST AI MODEL TRAINING PIPELINE")
    print("=" * 70)
    start_time = time.time()

    print(f"\n[1/4] Generating Multi-Sensor Simulated Telemetry ({n_runs} runs)...")
    X, y = core.make_training_set(n_runs=n_runs, seed=seed)
    total_samples = len(X)
    print(f"      Total Training Samples: {total_samples:,}")

    # Class distribution analysis
    classes = core.CLASSES
    counts = pd.Series(y).value_counts().sort_index()
    print("\n[2/4] Class Distribution:")
    for cls_idx, count in counts.items():
        if cls_idx < len(classes):
            print(f"      - {classes[cls_idx]:<15}: {count:>6,} ({count / total_samples:.1%})")

    # Downsample normal to prevent class imbalance skew
    rng = np.random.default_rng(seed)
    keep = (y != core.C["normal"]) | (rng.random(len(y)) < 0.5)
    X_train, y_train = X[keep], y[keep]
    print(f"\n      Balanced Training Set: {len(X_train):,} samples")

    # Split for independent validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    print("\n[3/4] Fitting Spatial Consensus LightGBM Multi-Head Classifier...")
    model = core.SensorTrustAIModel()
    fit_start = time.time()
    model.fit(pd.DataFrame(X_tr, columns=core.FEATURES), y_tr)
    fit_duration = time.time() - fit_start
    print(f"      Training completed in {fit_duration:.2f} seconds.")

    # Validation Evaluation
    print("\n[4/4] Evaluating on Validation Set...")
    y_pred = model.predict(pd.DataFrame(X_val, columns=core.FEATURES))
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print("\n" + "-" * 70)
    print(f"📊 VALIDATION METRICS:")
    print(f"   • Macro F1-Score:    {macro_f1:.4f}")
    print(f"   • Weighted F1-Score: {weighted_f1:.4f}")
    print(f"   • Macro Precision:   {precision:.4f}")
    print(f"   • Macro Recall:      {recall:.4f}")
    print("-" * 70)

    # Serialize model artifact
    print(f"\n💾 Serializing AI Model Artifact to '{output_path}'...")
    model.save(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✓ Model checkpoint saved successfully ({file_size_mb:.2f} MB).")

    total_duration = time.time() - start_time
    print(f"\n✨ End-to-end pipeline completed in {total_duration:.2f} seconds.")
    print("=" * 70)

    return model, {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "samples": total_samples,
        "size_mb": file_size_mb,
    }


if __name__ == "__main__":
    main()
