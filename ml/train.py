import os
import sys
import json
import time
import hashlib
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from ml.config import (
    TRAIN_DATASET_PATH, TEST_DATASET_PATH,
    BUNDLES_DIR, ACTIVE_MANIFEST_PATH,
    FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
)
from ml.memory_tracker import WindowsMemoryTracker, get_adaptive_n_jobs, get_system_memory_status
from ml.preprocessor import create_preprocessor, prepare_features, downcast_matrix_to_float32
from ml.evaluate import evaluate_model_performance

def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_dataset_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Computes genuine distribution analysis from the real UNSW-NB15 dataset partitions.
    """
    train_normal = int((train_df["label"] == 0).sum())
    train_attack = int((train_df["label"] == 1).sum())
    test_normal = int((test_df["label"] == 0).sum())
    test_attack = int((test_df["label"] == 1).sum())

    total_records = len(train_df) + len(test_df)
    total_normal = train_normal + test_normal
    total_attack = train_attack + test_attack

    # Attack category breakdown from training partition
    attack_cats = {}
    if "attack_cat" in train_df.columns:
        cats = train_df["attack_cat"].fillna("Normal").astype(str).str.strip().value_counts()
        attack_cats = {k: int(v) for k, v in cats.items()}

    # Top protocol distribution
    proto_dist = {}
    if "proto" in train_df.columns:
        top_protos = train_df["proto"].astype(str).str.upper().value_counts().head(5)
        proto_dist = {k: int(v) for k, v in top_protos.items()}

    return {
        "dataset_name": "UNSW-NB15 Benchmark",
        "total_records": total_records,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "traffic_distribution": {
            "total_normal": total_normal,
            "total_attack": total_attack,
            "normal_percentage": round((total_normal / total_records) * 100, 2),
            "attack_percentage": round((total_attack / total_records) * 100, 2),
            "train_normal": train_normal,
            "train_attack": train_attack,
            "test_normal": test_normal,
            "test_attack": test_attack
        },
        "attack_categories": attack_cats,
        "top_protocols": proto_dist
    }

def train_and_activate_model():
    """
    Complete model training pipeline:
    - Measures actual peak RSS and peak working set (peak_wset)
    - Validates float32 vs float64 parity (100% classification match, ΔP < 1e-4)
    - Trains Random Forest with adaptive n_jobs
    - Evaluates on held-out test partition
    - Builds versioned immutable bundle and atomically activates it
    """
    print("=" * 70)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Start Memory Monitoring
    mem_tracker = WindowsMemoryTracker(sample_interval_sec=0.05)
    mem_tracker.start()

    sys_mem = get_system_memory_status()
    print(f"[SYSTEM] Total RAM: {sys_mem['total_mb']} MB, Available: {sys_mem['available_mb']} MB ({sys_mem['percent_used']}% used)")
    adaptive_jobs = get_adaptive_n_jobs()
    print(f"[CONFIG] Adaptive n_jobs selected: {adaptive_jobs}")

    # 2. Load Datasets
    print("\nLoading dataset partitions from disk...")
    t0 = time.time()
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH)
    print(f"Loaded Train: {train_df.shape} and Test: {test_df.shape} in {time.time() - t0:.2f}s")

    # 3. Extract Features and Target
    X_train_df, y_train = prepare_features(train_df, is_training=True)
    X_test_df, y_test = prepare_features(test_df, is_training=False)

    print(f"Features: {X_train_df.shape[1]} input columns (excluding id, attack_cat, label)")

    # 4. Fit Preprocessor
    print("\nFitting ColumnTransformer on training features...")
    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_df)
    X_test_trans = preprocessor.transform(X_test_df)

    # 5. Float32 vs Float64 Parity Verification Suite
    print("\nExecuting Float32 vs Float64 Parity Verification Suite...")
    sample_size = min(10000, len(X_test_trans))
    indices = np.random.RandomState(42).choice(len(X_test_trans), size=sample_size, replace=False)
    
    X_sample_f64 = X_test_trans[indices].astype(np.float64)
    X_sample_f32 = downcast_matrix_to_float32(X_sample_f64)

    # Baseline test with 20 trees on random training sample
    train_sample_idx = np.random.RandomState(42).choice(len(X_train_trans), size=min(20000, len(X_train_trans)), replace=False)
    rf_baseline = RandomForestClassifier(n_estimators=20, max_depth=15, random_state=42, n_jobs=adaptive_jobs)
    rf_baseline.fit(X_train_trans[train_sample_idx], y_train[train_sample_idx])

    preds_f64 = rf_baseline.predict(X_sample_f64)
    preds_f32 = rf_baseline.predict(X_sample_f32)
    proba_f64 = rf_baseline.predict_proba(X_sample_f64)[:, 1] if rf_baseline.classes_.shape[0] > 1 else np.zeros(len(preds_f64))
    proba_f32 = rf_baseline.predict_proba(X_sample_f32)[:, 1] if rf_baseline.classes_.shape[0] > 1 else np.zeros(len(preds_f32))


    match_rate = float(np.mean(preds_f64 == preds_f32))
    max_prob_diff = float(np.max(np.abs(proba_f64 - proba_f32)))

    print(f"  Parity Results on {sample_size} validation samples:")
    print(f"  Classification Match Rate: {match_rate * 100:.4f}% (Required: 100.0000%)")
    print(f"  Max Probability Divergence: {max_prob_diff:.6e} (Required: < 1.0e-04)")

    use_float32 = False
    if match_rate == 1.0 and max_prob_diff < 1e-4:
        print("  [PASS] Float32 downcasting verified with zero prediction divergence. Enabling float32 optimization.")
        use_float32 = True
        X_train_final = downcast_matrix_to_float32(X_train_trans)
        X_test_final = downcast_matrix_to_float32(X_test_trans)
    else:
        print("  [ROLLBACK] Divergence exceeded threshold. Retaining standard float64 matrix.")
        X_train_final = X_train_trans
        X_test_final = X_test_trans

    del rf_baseline, X_sample_f64, X_sample_f32
    gc.collect()

    # 6. Full Model Training
    print(f"\nTraining Production Random Forest Classifier (n_estimators=100, max_depth=20, n_jobs={adaptive_jobs})...")
    t_train_start = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=adaptive_jobs,
        verbose=0
    )
    model.fit(X_train_final, y_train)
    train_duration = time.time() - t_train_start
    print(f"Training completed successfully in {train_duration:.2f} seconds.")

    # 7. Held-Out Test Evaluation
    print("\nEvaluating on held-out test partition (82,332 rows)...")
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    metrics = evaluate_model_performance(y_test, y_pred, y_prob)
    print("\n--- Genuine Test Metrics ---")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # 8. Compute Dataset Analysis
    print("\nComputing dataset distribution analytics...")
    dataset_analysis = compute_dataset_analysis(train_df, test_df)

    # 9. Stop Memory Tracker & Capture Measured Stats
    mem_tracker.stop()
    mem_summary = mem_tracker.get_summary()
    print("\n--- Measured Memory Usage ---")
    print(f"Start Process RSS:        {mem_summary['start_rss_mb']} MB")
    print(f"Peak Process RSS:         {mem_summary['peak_rss_mb']} MB")
    print(f"Peak Windows Working Set: {mem_summary['peak_wset_mb']} MB")
    print(f"Net RSS Increase:         {mem_summary['net_rss_increase_mb']} MB")

    # 10. Staging Bundle Creation
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"bundle_{timestamp}"
    bundle_dir = BUNDLES_DIR / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStaging immutable bundle at {bundle_dir.name}...")
    model_path = bundle_dir / "model.joblib"
    preprocessor_path = bundle_dir / "preprocessor.joblib"
    metrics_path = bundle_dir / "metrics.json"
    dataset_analysis_path = bundle_dir / "dataset_analysis.json"
    manifest_path = bundle_dir / "manifest.json"

    # Save components
    joblib.dump(model, model_path, compress=3)
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(dataset_analysis_path, "w", encoding="utf-8") as f:
        json.dump(dataset_analysis, f, indent=2)

    # Compute checksums
    checksums = {
        "model.joblib": compute_file_sha256(model_path),
        "preprocessor.joblib": compute_file_sha256(preprocessor_path),
        "metrics.json": compute_file_sha256(metrics_path),
        "dataset_analysis.json": compute_file_sha256(dataset_analysis_path)
    }

    manifest_data = {
        "bundle_id": bundle_name,
        "created_at_utc": datetime.utcnow().isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "float32_optimization": use_float32
        },
        "features": {
            "total_input_features": len(FEATURE_COLUMNS),
            "feature_names": FEATURE_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES
        },
        "metrics_summary": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        },
        "measured_memory": mem_summary,
        "checksums": checksums
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 11. Atomic Activation
    print("\nPerforming atomic bundle activation...")
    active_data = {
        "active_bundle_id": bundle_name,
        "bundle_dir": bundle_dir.relative_to(BUNDLES_DIR.parent).as_posix(),
        "activated_at_utc": datetime.utcnow().isoformat(),
        "manifest_path": manifest_path.relative_to(BUNDLES_DIR.parent).as_posix(),
        "metrics": metrics_summary_for_active(metrics)
    }

    active_tmp = ACTIVE_MANIFEST_PATH.with_suffix(".json.tmp")
    with open(active_tmp, "w", encoding="utf-8") as f:
        json.dump(active_data, f, indent=2)

    if ACTIVE_MANIFEST_PATH.exists():
        ACTIVE_MANIFEST_PATH.unlink()
    os.replace(str(active_tmp), str(ACTIVE_MANIFEST_PATH))

    print(f"[ACTIVATED] Active bundle successfully pointed to: {bundle_name}")
    print("=" * 70)
    return bundle_dir, metrics

def metrics_summary_for_active(m: dict) -> dict:
    return {
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1_score": m["f1_score"],
        "roc_auc": m["roc_auc"],
        "confusion_matrix": m["confusion_matrix"]
    }

if __name__ == "__main__":
    train_and_activate_model()
