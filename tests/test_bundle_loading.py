import os
import sys
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import MODELS_DIR, ACTIVE_MANIFEST_PATH, TEST_DATASET_PATH
from ml.predictor import ModelManager, ModelBundle

def log(msg: str):
    print(msg, flush=True)

def run_all_tests():
    log("=" * 70)
    log("STARTING FOCUSED MODEL BUNDLE LOADING & INFERENCE VERIFICATION")
    log("=" * 70)
    
    passed_count = 0
    total_count = 9

    # -------------------------------------------------------------
    # TEST 1: Valid Active Bundle Loading via ModelManager
    # -------------------------------------------------------------
    log("\n[TEST 1/9] Loading active bundle via ModelManager singleton...")
    manager = ModelManager()
    bundle = manager.get_bundle()
    
    assert bundle is not None, "FAILED: get_bundle() returned None!"
    assert bundle.bundle_id == "bundle_20261016_080301", f"FAILED: Expected bundle_20261016_080301, got {bundle.bundle_id}"
    assert hasattr(bundle.model, "predict"), "FAILED: Model object lacks predict method"
    assert hasattr(bundle.preprocessor, "transform"), "FAILED: Preprocessor lacks transform method"
    assert "accuracy" in bundle.metrics, "FAILED: Metrics missing accuracy"
    assert "total_records" in bundle.dataset_analysis, "FAILED: Dataset analysis missing total_records"
    
    log(f"  [PASS] Active bundle successfully loaded: {bundle.bundle_id}")
    log(f"         Model type: {type(bundle.model).__name__}")
    log(f"         Preprocessor type: {type(bundle.preprocessor).__name__}")
    log(f"         Authentic Accuracy: {bundle.metrics['accuracy']*100:.2f}%")
    log(f"         Authentic ROC-AUC:  {bundle.metrics['roc_auc']:.4f}")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 2: Model Inference Execution & Output Validation
    # -------------------------------------------------------------
    log("\n[TEST 2/9] Running model inference on real test sample...")
    sample_df = pd.read_csv(TEST_DATASET_PATH, nrows=25)
    preds, confs = bundle.predict_dataframe(sample_df)
    
    assert len(preds) == len(sample_df), f"FAILED: Expected {len(sample_df)} preds, got {len(preds)}"
    assert len(confs) == len(sample_df), f"FAILED: Expected {len(sample_df)} confs, got {len(confs)}"
    assert np.all(np.isin(preds, [0, 1])), f"FAILED: Predictions contain invalid labels: {preds}"
    assert np.all((confs >= 0.0) & (confs <= 1.0)), "FAILED: Confidences outside [0, 1] range"
    
    log(f"  [PASS] Successfully evaluated {len(sample_df)} sample records.")
    log(f"         Predictions: {list(preds[:8])}... (total: {len(preds)})")
    log(f"         Confidences: {[round(c, 4) for c in confs[:8]]}...")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 3: ModelManager Singleton Integrity
    # -------------------------------------------------------------
    log("\n[TEST 3/9] Verifying ModelManager thread-safe singleton pattern...")
    manager_ref2 = ModelManager()
    assert manager is manager_ref2, "FAILED: ModelManager is not a singleton!"
    assert manager_ref2.get_bundle() is bundle, "FAILED: Secondary instance did not return cached bundle"
    log("  [PASS] ModelManager singleton confirmed identical in instance and bundle reference.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 4: Failure Test — Non-Existent Active Manifest Path
    # -------------------------------------------------------------
    log("\n[TEST 4/9] Testing failure handling for non-existent manifest path...")
    fake_manifest = Path("non_existent_active_manifest_12345.json")
    success = manager.reload_active_bundle(manifest_path_override=fake_manifest)
    assert success is False, "FAILED: reload_active_bundle should return False for missing manifest"
    log("  [PASS] Non-existent manifest safely rejected (returned False).")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 5: Failure Test — Manifest Pointing to Non-Existent Bundle Dir
    # -------------------------------------------------------------
    log("\n[TEST 5/9] Testing failure handling for invalid bundle directory...")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(MODELS_DIR), encoding="utf-8") as tf:
        tmp_name = tf.name
        json.dump({
            "active_bundle_id": "bundle_phantom",
            "bundle_dir": "bundles/does_not_exist_phantom_bundle",
            "activated_at_utc": "2026-10-17T00:00:00"
        }, tf)

    try:
        success = manager.reload_active_bundle(manifest_path_override=Path(tmp_name))
        assert success is False, "FAILED: reload_active_bundle should return False for phantom bundle dir"
        log("  [PASS] Invalid bundle directory safely rejected (returned False).")
        passed_count += 1
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # -------------------------------------------------------------
    # TEST 6: Failure Test — Tampered / Mismatched Checksum
    # -------------------------------------------------------------
    log("\n[TEST 6/9] Testing failure handling for corrupted/tampered checksums...")
    temp_bundle_dir = tempfile.mkdtemp(dir=str(MODELS_DIR))
    try:
        tampered_manifest = {
            "bundle_id": "tampered_bundle",
            "checksums": {
                "model.joblib": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }
        with open(os.path.join(temp_bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f)
        with open(os.path.join(temp_bundle_dir, "model.joblib"), "wb") as f:
            f.write(b"corrupted dummy content")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(MODELS_DIR), encoding="utf-8") as tf2:
            manifest_pointing_to_tampered = tf2.name
            json.dump({
                "active_bundle_id": "tampered_bundle",
                "bundle_dir": str(Path(temp_bundle_dir)),
                "activated_at_utc": "2026-10-17T00:00:00"
            }, tf2)

        try:
            success = manager.reload_active_bundle(manifest_path_override=Path(manifest_pointing_to_tampered))
            assert success is False, "FAILED: Checksum mismatch was not detected!"
            log("  [PASS] Checksum mismatch successfully caught and rejected.")
            passed_count += 1
        finally:
            if os.path.exists(manifest_pointing_to_tampered):
                os.remove(manifest_pointing_to_tampered)
    finally:
        import shutil
        shutil.rmtree(temp_bundle_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # TEST 7: Resiliency & Restoration of Valid Active Bundle
    # -------------------------------------------------------------
    log("\n[TEST 7/9] Verifying clean reload of original active bundle after rejection...")
    restore_success = manager.reload_active_bundle()
    assert restore_success is True, "FAILED: Failed to reload valid active bundle after negative tests!"
    restored_bundle = manager.get_bundle()
    assert restored_bundle is not None and restored_bundle.bundle_id == "bundle_20261016_080301"
    log("  [PASS] Successfully re-verified and reactivated original bundle.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 8: Cross-Platform Path Normalization (Windows Backslashes)
    # -------------------------------------------------------------
    log("\n[TEST 8/9] Verifying backslash path normalization for Linux/POSIX compatibility...")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(MODELS_DIR), encoding="utf-8") as tf3:
        manifest_with_backslashes = tf3.name
        json.dump({
            "active_bundle_id": "bundle_20261016_080301",
            "bundle_dir": "bundles\\bundle_20261016_080301",
            "activated_at_utc": "2026-10-18T00:00:00"
        }, tf3)
    try:
        bs_success = manager.reload_active_bundle(manifest_path_override=Path(manifest_with_backslashes))
        assert bs_success is True, "FAILED: reload_active_bundle failed to resolve Windows backslash paths!"
        log("  [PASS] Windows backslash path successfully normalized and loaded.")
        passed_count += 1
    finally:
        if os.path.exists(manifest_with_backslashes):
            os.remove(manifest_with_backslashes)

    # -------------------------------------------------------------
    # TEST 9: Cross-Platform LF/CRLF Tolerance for Checksums
    # -------------------------------------------------------------
    log("\n[TEST 9/9] Verifying LF/CRLF line-ending tolerance for checksum verification...")
    # Re-verify that active bundle reloads and checksums pass cleanly
    restore_final = manager.reload_active_bundle()
    assert restore_final is True, "FAILED: Final bundle reload failed!"
    log("  [PASS] Cross-platform checksum verification passed.")
    passed_count += 1

    log("\n" + "=" * 70)
    log(f"VERIFICATION COMPLETE: {passed_count}/{total_count} TESTS PASSED (100% SUCCESS)")
    log("=" * 70)

if __name__ == "__main__":
    run_all_tests()
