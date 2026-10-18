import os
import sys
import time
import socket
import threading
import tempfile
from pathlib import Path
import pandas as pd
import requests
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import TEST_DATASET_PATH, MAX_UPLOAD_SIZE_BYTES
from app import app

def log(msg: str):
    print(msg, flush=True)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def run_backend_tests():
    log("=" * 70)
    log("STARTING COMPREHENSIVE FASTAPI BACKEND API VERIFICATION SUITE")
    log("=" * 70)

    # 1. Start live uvicorn server on an ephemeral loopback port
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log(f"\n[SERVER SETUP] Launching in-process FastAPI server on {base_url}...")

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to accept connections
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/api/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    passed_count = 0
    total_count = 10
    active_session_id = None

    try:
        # -------------------------------------------------------------
        # TEST 1: Serve Static index.html Root
        # -------------------------------------------------------------
        log("\n[TEST 1/10] Verifying GET / serves preserved index.html...")
        res = requests.get(f"{base_url}/", timeout=5)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "Network Anomaly Detection Dashboard" in res.text, "Index HTML title not found in response"
        log("  [PASS] GET / successfully served index.html with preserved dashboard title.")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 2: GET /api/health
        # -------------------------------------------------------------
        log("\n[TEST 2/10] Verifying GET /api/health...")
        res = requests.get(f"{base_url}/api/health", timeout=5)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert data.get("status") == "healthy", f"Status not healthy: {data}"
        assert data.get("model_active") is True, "model_active is not True"
        assert data.get("active_bundle_id") == "bundle_20261016_080301", f"Unexpected bundle: {data}"
        assert abs(data.get("accuracy", 0) - 0.9074) < 0.001, f"Unexpected accuracy: {data}"
        log(f"  [PASS] Server healthy. Bundle: {data['active_bundle_id']}, Model Accuracy: {data['accuracy']*100:.2f}%")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 3: GET /api/model-performance
        # -------------------------------------------------------------
        log("\n[TEST 3/10] Verifying GET /api/model-performance...")
        res = requests.get(f"{base_url}/api/model-performance", timeout=5)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        perf = res.json()
        assert perf.get("bundle_id") == "bundle_20261016_080301"
        metrics = perf.get("metrics", {})
        assert abs(metrics.get("accuracy", 0) - 0.9074) < 0.001
        assert abs(metrics.get("precision", 0) - 0.8768) < 0.001
        assert abs(metrics.get("recall", 0) - 0.9679) < 0.001
        assert abs(metrics.get("f1_score", 0) - 0.9201) < 0.001
        assert abs(metrics.get("roc_auc", 0) - 0.9845) < 0.001
        assert "confusion_matrix" in metrics
        log(f"  [PASS] Authentic metrics served: Accuracy={metrics['accuracy']*100:.2f}%, F1={metrics['f1_score']*100:.2f}%, ROC-AUC={metrics['roc_auc']:.4f}")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 4: GET /api/dataset-analysis
        # -------------------------------------------------------------
        log("\n[TEST 4/10] Verifying GET /api/dataset-analysis...")
        res = requests.get(f"{base_url}/api/dataset-analysis", timeout=5)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        analysis = res.json()
        assert analysis.get("dataset_name") == "UNSW-NB15 Benchmark"
        assert analysis.get("total_records") == 257673
        assert analysis.get("train_records") == 175341
        assert analysis.get("test_records") == 82332
        assert "attack_categories" in analysis
        assert len(analysis["attack_categories"]) == 10
        log(f"  [PASS] Dataset analysis served: {analysis['dataset_name']} ({analysis['total_records']:,} total records, {len(analysis['attack_categories'])} attack categories).")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 5: POST /api/predict (Valid CSV Upload & Chunked Inference)
        # -------------------------------------------------------------
        log("\n[TEST 5/10] Verifying POST /api/predict with valid sample CSV (50 rows)...")
        sample_df = pd.read_csv(TEST_DATASET_PATH, nrows=50)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
            sample_df.to_csv(tf.name, index=False)
            valid_csv_path = tf.name

        try:
            with open(valid_csv_path, "rb") as f:
                res = requests.post(f"{base_url}/api/predict", files={"file": ("test_traffic.csv", f, "text/csv")}, timeout=15)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            pred_summary = res.json()
            assert "session_id" in pred_summary
            active_session_id = pred_summary["session_id"]
            assert pred_summary.get("total_rows") == 50
            assert pred_summary.get("attack_count") + pred_summary.get("normal_count") == 50
            assert len(pred_summary.get("preview", [])) == 10
            log(f"  [PASS] Prediction processed successfully: Session ID={active_session_id}, Rows=50, Attack Rate={pred_summary['attack_rate']}%, Preview Count={len(pred_summary['preview'])}.")
            passed_count += 1
        finally:
            if os.path.exists(valid_csv_path):
                os.remove(valid_csv_path)

        # -------------------------------------------------------------
        # TEST 6: GET /api/predictions/{session_id} (Pagination & Filters)
        # -------------------------------------------------------------
        log(f"\n[TEST 6/10] Verifying GET /api/predictions/{active_session_id} pagination & filtering...")
        # Page 1 (20 records)
        p1_res = requests.get(f"{base_url}/api/predictions/{active_session_id}?page=1&page_size=20", timeout=5)
        assert p1_res.status_code == 200, f"Expected 200, got {p1_res.status_code}: {p1_res.text}"
        p1_data = p1_res.json()
        assert p1_data["page"] == 1
        assert p1_data["page_size"] == 20
        assert p1_data["total_results"] == 50
        assert p1_data["total_pages"] == 3
        assert len(p1_data["results"]) == 20

        # Page 2 (20 records)
        p2_res = requests.get(f"{base_url}/api/predictions/{active_session_id}?page=2&page_size=20", timeout=5)
        assert p2_res.status_code == 200
        p2_data = p2_res.json()
        assert len(p2_data["results"]) == 20
        assert p1_data["results"][0]["id"] != p2_data["results"][0]["id"], "Page 1 and Page 2 should not have identical first IDs"

        # Filter by Protocol (TCP)
        tcp_res = requests.get(f"{base_url}/api/predictions/{active_session_id}?proto=TCP", timeout=5)
        assert tcp_res.status_code == 200
        tcp_data = tcp_res.json()
        for rec in tcp_data["results"]:
            assert rec["proto"] == "TCP", f"Expected TCP, got {rec['proto']}"

        # Filter by Status (Attack)
        atk_res = requests.get(f"{base_url}/api/predictions/{active_session_id}?status=Attack", timeout=5)
        assert atk_res.status_code == 200
        atk_data = atk_res.json()
        for rec in atk_data["results"]:
            assert rec["prediction"] == "Attack", f"Expected Attack, got {rec['prediction']}"

        log(f"  [PASS] SQLite pagination verified: Total Results=50, Total Pages=3, Protocol & Status filters confirmed.")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 7: Error Case — 0-byte Malformed Upload
        # -------------------------------------------------------------
        log("\n[TEST 7/10] Verifying error handling for 0-byte CSV upload...")
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tf:
            empty_path = tf.name

        try:
            with open(empty_path, "rb") as f:
                res = requests.post(f"{base_url}/api/predict", files={"file": ("empty.csv", f, "text/csv")}, timeout=5)
            assert res.status_code == 422, f"Expected 422 Unprocessable Entity, got {res.status_code}"
            assert "empty" in res.json().get("detail", "").lower()
            log(f"  [PASS] 0-byte upload correctly rejected with HTTP 422: {res.json()['detail']}")
            passed_count += 1
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

        # -------------------------------------------------------------
        # TEST 8: Error Case — Missing Required Feature Columns
        # -------------------------------------------------------------
        log("\n[TEST 8/10] Verifying error handling for missing feature schema...")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
            tf.write("id,source,destination,misc_note\n1,192.168.1.1,10.0.0.1,dummy\n")
            invalid_schema_path = tf.name

        try:
            with open(invalid_schema_path, "rb") as f:
                res = requests.post(f"{base_url}/api/predict", files={"file": ("invalid_schema.csv", f, "text/csv")}, timeout=5)
            assert res.status_code == 422, f"Expected 422 Unprocessable Entity, got {res.status_code}"
            detail = res.json().get("detail", "")
            assert "missing required feature columns" in detail.lower()
            log(f"  [PASS] Missing required features correctly rejected with HTTP 422: {detail}")
            passed_count += 1
        finally:
            if os.path.exists(invalid_schema_path):
                os.remove(invalid_schema_path)

        # -------------------------------------------------------------
        # TEST 9: Error Case — File Size > 10 MiB Rejection
        # -------------------------------------------------------------
        log("\n[TEST 9/10] Verifying error handling for oversized file (> 10 MiB)...")
        # Stream a generator of 10.5 MB without generating a huge file on disk
        def oversized_generator():
            chunk = b"A" * (64 * 1024)  # 64 KB
            for _ in range(165):        # ~10.56 MB
                yield chunk

        res = requests.post(
            f"{base_url}/api/predict",
            data=oversized_generator(),
            headers={"Content-Type": "text/csv"},
            timeout=10
        )
        assert res.status_code == 413, f"Expected 413 Payload Too Large, got {res.status_code}"
        assert "exceeds maximum limit" in res.json().get("detail", "").lower()
        log(f"  [PASS] Oversized file aborted early with HTTP 413: {res.json()['detail']}")
        passed_count += 1

        # -------------------------------------------------------------
        # TEST 10: Error Case — Path Traversal & Missing Session ID
        # -------------------------------------------------------------
        log("\n[TEST 10/10] Verifying path traversal rejection and 404 on missing session...")
        traversal_res = requests.get(f"{base_url}/api/predictions/..%2F..%2Fetc%2Fpasswd", timeout=5)
        assert traversal_res.status_code in (404, 400), f"Expected 404 or 400 for path traversal, got {traversal_res.status_code}"

        missing_res = requests.get(f"{base_url}/api/predictions/00000000000000000000000000000000", timeout=5)
        assert missing_res.status_code == 404, f"Expected 404 for missing session, got {missing_res.status_code}"
        assert "not found" in missing_res.json().get("detail", "").lower()
        log(f"  [PASS] Path traversal and non-existent session safely rejected with HTTP 404.")
        passed_count += 1

    finally:
        log("\n[SERVER TEARDOWN] Stopping in-process server...")
        server.should_exit = True
        server_thread.join(timeout=3)
        log("  Server stopped.")

    log("\n" + "=" * 70)
    log(f"BACKEND VERIFICATION COMPLETE: {passed_count}/{total_count} TESTS PASSED (100% SUCCESS)")
    log("=" * 70)

if __name__ == "__main__":
    run_backend_tests()
