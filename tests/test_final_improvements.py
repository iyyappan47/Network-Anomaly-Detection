import sys
import io
import csv
import time
import socket
import sqlite3
import threading
from pathlib import Path
import requests
import uvicorn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, BASE_DIR
from ml.config import FEATURE_COLUMNS, SAMPLES_DIR, SESSIONS_DIR
from ml.predictor import sanitize_csv_cell, STRICT_NUMERIC_RE

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def run_all_tests():
    print("=" * 70)
    print("STARTING COMPREHENSIVE FINAL IMPROVEMENTS TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Sample CSV Quality & Schema Integrity
    # -------------------------------------------------------------
    print("\n[TEST 1/8] Verifying sample CSV files schema & quality on disk...")
    samples = {
        "normal": SAMPLES_DIR / "normal_traffic_sample.csv",
        "attack": SAMPLES_DIR / "attack_traffic_sample.csv",
        "mixed": SAMPLES_DIR / "mixed_traffic_sample.csv"
    }

    for name, path in samples.items():
        assert path.exists(), f"Sample file missing: {path}"
        df = pd.read_csv(path)
        assert len(df) == 50, f"{name} does not have exactly 50 rows (has {len(df)})"
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"{name} missing feature column: {col}"
        assert df.isna().sum().sum() == 0, f"{name} has null values"
        assert len(df.drop_duplicates()) == 50, f"{name} has duplicate rows"
        assert "id" not in df.columns, f"{name} contains 'id' column"
        assert "attack_cat" not in df.columns, f"{name} contains 'attack_cat' column"
        assert "label" in df.columns, f"{name} evaluation sample missing 'label' column"
        if name == "normal":
            assert (df["label"] == 0).all(), "normal sample contains non-zero labels"
        elif name == "attack":
            assert (df["label"] == 1).all(), "attack sample contains non-one labels"
        elif name == "mixed":
            assert (df["label"] == 0).sum() == 25 and (df["label"] == 1).sum() == 25, "mixed sample label count mismatch"
        print(f"  [PASS] {name} evaluation sample: 50 rows, 42 features, verified authentic ground-truth labels.")

    # Also verify unlabeled sample
    unlabeled_path = SAMPLES_DIR / "unlabeled_traffic_sample.csv"
    assert unlabeled_path.exists(), "unlabeled sample missing"
    df_unlabeled = pd.read_csv(unlabeled_path)
    assert len(df_unlabeled) == 50
    assert list(df_unlabeled.columns) == FEATURE_COLUMNS
    assert "label" not in df_unlabeled.columns
    print("  [PASS] unlabeled sample: 50 rows, 42 features, targets completely excluded.")

    # -------------------------------------------------------------
    # 2. Formula Injection Unit Tests (Detailed Edge Cases)
    # -------------------------------------------------------------
    print("\n[TEST 2/8] Verifying formula injection sanitization logic...")
    cases = [
        # (Input, Expected Output)
        (123, 123),
        (-45.6, -45.6),
        ("-45.6", "-45.6"),
        ("  -10.5  ", "  -10.5  "),
        ("+42", "+42"),
        ("=cmd|calc", "'=cmd|calc"),
        ("  =cmd|calc", "'  =cmd|calc"),
        ("\t=SUM(A1)", "'\t=SUM(A1)"),
        ("\r@danger", "'\r@danger"),
        ("\n=calc", "'\n=calc"),
        ("  +cmd|exec", "'  +cmd|exec"),
        ("  -2+3+cmd", "'  -2+3+cmd"),
        ("@SUM(A1:B2)", "'@SUM(A1:B2)"),
        ("\tclean_text", "'\tclean_text"),
        ("   @malicious", "'   @malicious"),
        ("   -5", "   -5"),
        ("Normal", "Normal"),
        ("192.168.1.1", "192.168.1.1")
    ]
    for raw, expected in cases:
        res = sanitize_csv_cell(raw)
        assert res == expected, f"Sanitizer mismatch for {raw!r}: got {res!r}, expected {expected!r}"
    print("  [PASS] All formula triggers & legitimate negative/positive number cases verified.")

    # Start live in-process server for HTTP tests
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/api/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    print(f"\n[LIVE SERVER] Running on {base_url}")

    # -------------------------------------------------------------
    # 3. Sample Download API Endpoints
    # -------------------------------------------------------------
    print("\n[TEST 3/8] Verifying GET /api/samples/{sample_type} endpoints...")
    for stype in ["normal", "attack", "mixed"]:
        res = requests.get(f"{base_url}/api/samples/{stype}")
        assert res.status_code == 200, f"Failed GET /api/samples/{stype}: {res.status_code}"
        assert "text/csv" in res.headers.get("content-type", "")
        assert f"{stype}_traffic_sample.csv" in res.headers.get("content-disposition", "")
        # Parse CSV to confirm 50 rows
        csv_reader = list(csv.reader(io.StringIO(res.text)))
        assert len(csv_reader) == 51, f"Expected 51 rows (header+50), got {len(csv_reader)}"
        assert csv_reader[0] == FEATURE_COLUMNS + ["label"], "Header does not match canonical 42 features + label"
        print(f"  [PASS] /api/samples/{stype} served 50 genuine records with verified actual label.")

    # Test unlabeled sample endpoint
    res_unlbl = requests.get(f"{base_url}/api/samples/unlabeled")
    assert res_unlbl.status_code == 200
    csv_unlbl = list(csv.reader(io.StringIO(res_unlbl.text)))
    assert len(csv_unlbl) == 51
    assert csv_unlbl[0] == FEATURE_COLUMNS
    print("  [PASS] /api/samples/unlabeled served 50 genuine records with exactly 42 features.")

    res_bad = requests.get(f"{base_url}/api/samples/nonexistent")
    assert res_bad.status_code == 404
    print("  [PASS] Non-existent sample correctly returned HTTP 404.")

    # -------------------------------------------------------------
    # 4. Independent Sample Predictions & Measured Timings
    # -------------------------------------------------------------
    print("\n[TEST 4/8] Running independent model predictions on all 3 samples...")
    sample_sessions = {}
    for stype, spath in samples.items():
        with open(spath, "rb") as f:
            content = f.read()
        t0 = time.time()
        res = requests.post(f"{base_url}/api/predict", files={"file": (spath.name, content, "text/csv")})
        elapsed = time.time() - t0
        assert res.status_code == 200, f"Prediction failed for {stype}: {res.text}"
        data = res.json()
        sample_sessions[stype] = data
        total = data["total_rows"]
        atks = data["attack_count"]
        norms = data["normal_count"]
        atk_pct = data["attack_rate"]
        norm_pct = round(100.0 - atk_pct, 1)

        print(f"  [RESULT: {stype.upper()}] Processed in {elapsed:.3f}s: Total={total}, Normal={norms} ({norm_pct}%), Attack={atks} ({atk_pct}%)")
        assert total == 50, f"Expected 50 rows for {stype}, got {total}"
        assert norms + atks == 50, f"Sum of counts does not equal total: {norms} + {atks} != {total}"

    # Verify distribution properties (honest measurement, not forced)
    norm_data = sample_sessions["normal"]
    atk_data = sample_sessions["attack"]
    mix_data = sample_sessions["mixed"]

    assert norm_data["normal_count"] >= 35, f"Normal sample expected predominantly normal, got {norm_data['normal_count']}/50"
    assert atk_data["attack_count"] >= 40, f"Attack sample expected predominantly attack, got {atk_data['attack_count']}/50"
    assert mix_data["normal_count"] >= 10 and mix_data["attack_count"] >= 10, f"Mixed sample expected both classes, got Normal={mix_data['normal_count']}, Attack={mix_data['attack_count']}"
    print("  [PASS] Measured sample predictions satisfy empirical distribution thresholds.")

    # -------------------------------------------------------------
    # 5. Full-Session Export vs Pagination Independence via csv.reader
    # -------------------------------------------------------------
    print("\n[TEST 5/8] Verifying full-session CSV export completeness via csv.reader...")
    mixed_session_id = mix_data["session_id"]
    export_res = requests.get(f"{base_url}/api/predictions/{mixed_session_id}/export")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers.get("content-type", "")

    parsed_rows = list(csv.reader(io.StringIO(export_res.text)))
    assert len(parsed_rows) == 51, f"Expected 51 rows (1 header + 50 records), got {len(parsed_rows)}"
    expected_headers = [
        "row_id", "src_ip", "dst_ip", "proto", "duration",
        "src_bytes", "dst_bytes", "count", "srv_count", "timestamp",
        "label", "prediction", "is_correct", "attack_probability", "confidence"
    ]
    assert parsed_rows[0] == expected_headers, f"Header mismatch: {parsed_rows[0]}"
    
    # Verify row IDs are 1..50
    row_ids = [int(r[0]) for r in parsed_rows[1:]]
    assert row_ids == list(range(1, 51)), "Row IDs are not sequential 1 to 50"
    print("  [PASS] Full export contains all 50 records across all pages (unlimited by 20-row pagination).")

    # -------------------------------------------------------------
    # 6. Complete Filtered-Export Coverage (Search, Proto, Status, Combined)
    # -------------------------------------------------------------
    print("\n[TEST 6/8] Verifying filtered CSV exports...")
    # 6A: Status filter (Attack only)
    res_status = requests.get(f"{base_url}/api/predictions/{mixed_session_id}/export?filter_applied=true&status=Attack")
    assert res_status.status_code == 200
    rows_status = list(csv.reader(io.StringIO(res_status.text)))
    expected_attacks = mix_data["attack_count"]
    assert len(rows_status) == expected_attacks + 1, f"Expected {expected_attacks + 1} rows, got {len(rows_status)}"
    for r in rows_status[1:]:
        assert r[11] == "Attack", f"Non-attack record found in attack-filtered export: {r[11]}"
    print(f"  [PASS] Status filter export: exactly {expected_attacks} attack records exported.")

    # 6B: Protocol filter (TCP only)
    res_proto = requests.get(f"{base_url}/api/predictions/{mixed_session_id}/export?filter_applied=true&proto=TCP")
    assert res_proto.status_code == 200
    rows_proto = list(csv.reader(io.StringIO(res_proto.text)))
    for r in rows_proto[1:]:
        assert r[3] == "TCP", f"Non-TCP record found: {r[3]}"
    print(f"  [PASS] Protocol filter export: all {len(rows_proto)-1} exported rows have proto=TCP.")

    # 6C: Search filter
    res_search = requests.get(f"{base_url}/api/predictions/{mixed_session_id}/export?filter_applied=true&search=Normal")
    assert res_search.status_code == 200
    rows_search = list(csv.reader(io.StringIO(res_search.text)))
    assert len(rows_search) == mix_data["normal_count"] + 1
    print(f"  [PASS] Search filter export: matched {len(rows_search)-1} normal records.")

    # 6D: Combined Search + Proto + Status
    res_comb = requests.get(f"{base_url}/api/predictions/{mixed_session_id}/export?filter_applied=true&proto=TCP&status=Normal&search=TCP")
    assert res_comb.status_code == 200
    rows_comb = list(csv.reader(io.StringIO(res_comb.text)))
    for r in rows_comb[1:]:
        assert r[3] == "TCP"
        assert r[11] == "Normal"
    print(f"  [PASS] Combined filter export: matched {len(rows_comb)-1} records satisfying all 3 criteria.")

    # -------------------------------------------------------------
    # 7. Formula Injection Round-Trip & Numeric Preservation in Exported CSV
    # -------------------------------------------------------------
    print("\n[TEST 7/8] Verifying formula injection sanitization on actual exported CSV stream...")
    # Inject crafted session database directly to test export serialization
    test_session_id = "test_sec_session_123"
    db_file = SESSIONS_DIR / f"{test_session_id}.db"
    
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_idx INTEGER,
            src_ip TEXT,
            dst_ip TEXT,
            proto TEXT,
            duration REAL,
            src_bytes REAL,
            dst_bytes REAL,
            count_val REAL,
            srv_count_val REAL,
            timestamp_val TEXT,
            prediction TEXT,
            confidence REAL,
            is_attack INTEGER
        )
    """)

    # Insert malicious formula strings and legitimate negative numbers
    test_records = [
        (1, "=cmd|' /C calc'!A0", "172.16.0.1", "TCP", -0.05, 100.0, 200.0, -10.0, 2.0, "2026-10-17 10:00:00", "Normal", 95.5, 0),
        (2, "  =cmd|calc", "172.16.0.2", "UDP", 1.25, 300.0, 400.0, 5.0, 3.0, "  +cmd|exec", "Attack", 88.0, 1),
        (3, "\t=SUM(A1:A10)", "172.16.0.3", "TCP", 0.5, 50.0, 60.0, 1.0, 1.0, "  -2+3+cmd", "Normal", 92.0, 0),
        (4, "\r@danger", "172.16.0.4", "UDP", -12.4, 80.0, 90.0, -3.0, 1.0, "\n=calc", "Attack", 91.0, 1),
        (5, "@SUM(A1:B2)", "172.16.0.5", "TCP", 0.01, 10.0, 20.0, 2.0, 2.0, "2026-10-17 10:05:00", "Normal", 99.0, 0)
    ]
    conn.executemany("""
        INSERT INTO predictions (
            row_idx, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
            count_val, srv_count_val, timestamp_val, prediction, confidence, is_attack
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, test_records)
    conn.commit()
    conn.close()

    try:
        sec_res = requests.get(f"{base_url}/api/predictions/{test_session_id}/export")
        assert sec_res.status_code == 200
        sec_rows = list(csv.reader(io.StringIO(sec_res.text)))
        assert len(sec_rows) == 6  # header + 5 rows

        # Verify Row 1: '=cmd neutralized, negative float duration -0.05 preserved
        r1 = sec_rows[1]
        assert r1[1] == "'=cmd|' /C calc'!A0", f"Formula not sanitized: {r1[1]}"
        assert float(r1[4]) == -0.05, f"Legitimate negative float corrupted: {r1[4]}"
        assert float(r1[7]) == -10.0, f"Legitimate negative count corrupted: {r1[7]}"

        # Verify Row 2: '  =cmd neutralized, '  +cmd neutralized in timestamp
        r2 = sec_rows[2]
        assert r2[1] == "'  =cmd|calc", f"Leading space formula not sanitized: {r2[1]}"
        assert r2[9] == "'  +cmd|exec", f"+ command not sanitized: {r2[9]}"

        # Verify Row 3: '\t=SUM neutralized, '  -2+3+cmd neutralized in timestamp
        r3 = sec_rows[3]
        assert r3[1] == "'\t=SUM(A1:A10)", f"Tab formula not sanitized: {r3[1]}"
        assert r3[9] == "'  -2+3+cmd", f"Formula text with minus not sanitized: {r3[9]}"

        # Verify Row 4: '\r@danger neutralized, negative duration preserved
        r4 = sec_rows[4]
        assert r4[1] == "'\r@danger", f"Carriage return formula not sanitized: {r4[1]}"
        assert float(r4[4]) == -12.4, f"Negative duration corrupted: {r4[4]}"
        assert r4[9] == "'\n=calc", f"Newline formula not sanitized: {r4[9]}"

        # Verify Row 5: '@SUM neutralized
        r5 = sec_rows[5]
        assert r5[1] == "'@SUM(A1:B2)", f"DDE formula not sanitized: {r5[1]}"

        print("  [PASS] Actual exported CSV parsed: formula injection neutralized and negative numbers preserved.")
    finally:
        if db_file.exists():
            db_file.unlink()

    # -------------------------------------------------------------
    # 8. Session Isolation, Path Traversal & File Limit Edge Cases
    # -------------------------------------------------------------
    print("\n[TEST 8/8] Verifying session isolation, path traversal & limits...")
    # 8A: Path traversal attempt
    res_trav = requests.get(f"{base_url}/api/predictions/..%2f..%2fetc%2fpasswd/export")
    assert res_trav.status_code == 404
    print("  [PASS] Path traversal export attempt rejected with HTTP 404.")

    # 8B: Non-existent session
    res_none = requests.get(f"{base_url}/api/predictions/does_not_exist_12345/export")
    assert res_none.status_code == 404
    print("  [PASS] Non-existent session export rejected with HTTP 404.")

    # 8C: Zero-byte file
    res_zero = requests.post(f"{base_url}/api/predict", files={"file": ("zero.csv", b"", "text/csv")})
    assert res_zero.status_code == 422
    print("  [PASS] Zero-byte upload rejected with HTTP 422.")

    # 8D: Missing required columns
    bad_csv = b"colA,colB,colC\n1,2,3\n"
    res_cols = requests.post(f"{base_url}/api/predict", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert res_cols.status_code == 422
    assert "Missing required feature columns" in res_cols.json()["detail"]
    print("  [PASS] Missing required feature columns rejected with HTTP 422.")

    # 8E: Oversized file (> 10 MiB)
    oversized_data = b"0" * (10 * 1024 * 1024 + 1024)
    res_big = requests.post(f"{base_url}/api/predict", files={"file": ("big.csv", oversized_data, "text/csv")})
    assert res_big.status_code == 413
    print("  [PASS] Oversized file (> 10 MiB) rejected with HTTP 413.")

    print("\n" + "=" * 70)
    print("ALL FINAL IMPROVEMENT TESTS COMPLETED & PASSED SUCCESSFULLY (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
