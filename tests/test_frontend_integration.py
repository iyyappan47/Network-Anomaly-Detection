import os
import sys
import io
import time
import socket
import threading
from pathlib import Path
import requests
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, BASE_DIR

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def test_offline_readiness_no_cdns():
    """Verify index.html contains no external CDN links or font dependencies."""
    index_path = BASE_DIR / "index.html"
    assert index_path.exists(), "index.html does not exist"
    content = index_path.read_text(encoding="utf-8")

    # Disallowed CDNs and external font services
    assert "cdn.tailwindcss.com" not in content, "Found external tailwind CDN in index.html"
    assert "fonts.googleapis.com" not in content, "Found external google fonts in index.html"
    assert "fonts.gstatic.com" not in content, "Found external gstatic fonts in index.html"
    assert "cdnjs.cloudflare.com" not in content, "Found external cdnjs in index.html"
    assert "unpkg.com" not in content, "Found external unpkg in index.html"

    # Must contain local stylesheet link
    assert '<link rel="stylesheet" href="/static/css/app.css">' in content
    print("PASS: test_offline_readiness_no_cdns")

def test_local_assets_exist():
    """Verify static css and all font files are present on disk."""
    css_path = BASE_DIR / "static" / "css" / "app.css"
    assert css_path.exists(), "static/css/app.css does not exist"
    assert css_path.stat().st_size > 10000, "static/css/app.css seems incomplete"

    fonts_dir = BASE_DIR / "static" / "fonts"
    expected_fonts = [
        "inter-400.woff2",
        "inter-500.woff2",
        "inter-600.woff2",
        "inter-700.woff2",
        "inter-800.woff2"
    ]
    for font_file in expected_fonts:
        fpath = fonts_dir / font_file
        assert fpath.exists(), f"Font file missing: {font_file}"
        assert fpath.stat().st_size > 10000, f"Font file seems empty: {font_file}"
    print("PASS: test_local_assets_exist")

def run_integration_tests():
    test_offline_readiness_no_cdns()
    test_local_assets_exist()

    # Start live in-process server
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

    print("\n--- Running Live HTTP Verification Tests ---")

    # 1. Static asset serving
    css_res = requests.get(f"{base_url}/static/css/app.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers.get("content-type", "")
    assert len(css_res.content) > 10000
    print("PASS: /static/css/app.css served successfully (200 OK, text/css)")

    font_res = requests.get(f"{base_url}/static/fonts/inter-400.woff2")
    assert font_res.status_code == 200
    assert len(font_res.content) > 10000
    print("PASS: /static/fonts/inter-400.woff2 served successfully (200 OK)")

    # 2. Root index.html check
    res = requests.get(f"{base_url}/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    html = res.text

    # Verify all 5 screen containers exist
    assert 'id="screen-overview"' in html
    assert 'id="screen-model-performance"' in html
    assert 'id="screen-detection-results"' in html
    assert 'id="screen-dataset-analysis"' in html
    assert 'id="screen-csv-upload-prediction"' in html

    # Verify all 5 sidebar navigation anchors exist
    assert 'href="#/overview"' in html
    assert 'href="#/model-performance"' in html
    assert 'href="#/detection-results"' in html
    assert 'href="#/dataset-analysis"' in html
    assert 'href="#/csv-upload-prediction"' in html

    # Verify key interactive IDs exist
    assert 'id="csv-file-input"' in html
    assert 'id="start-preprocess-btn"' in html
    assert 'id="detection-prev-btn"' in html
    assert 'id="detection-next-btn"' in html
    assert 'id="detection-page-num"' in html
    assert 'id="detection-search"' in html
    assert 'id="detection-protocol-filter"' in html
    assert 'id="detection-status-filter"' in html
    assert 'id="upload-feedback-banner"' in html
    print("PASS: GET / serves index.html with all 5 screens, routes, and interactive controls")

    # 3. Model performance endpoint
    perf_res = requests.get(f"{base_url}/api/model-performance")
    assert perf_res.status_code == 200
    perf = perf_res.json()
    assert "metrics" in perf
    assert "confusion_matrix" in perf["metrics"]
    assert perf["metrics"]["accuracy"] > 0.85
    print(f"PASS: /api/model-performance (Accuracy: {perf['metrics']['accuracy']}, ROC-AUC: {perf['metrics']['roc_auc']})")

    # 4. Dataset analysis endpoint
    da_res = requests.get(f"{base_url}/api/dataset-analysis")
    assert da_res.status_code == 200
    da = da_res.json()
    assert "traffic_distribution" in da
    assert "attack_categories" in da
    assert da["test_records"] == 82332
    print(f"PASS: /api/dataset-analysis (Training: {da['train_records']}, Testing: {da['test_records']})")

    # 5. CSV prediction upload
    test_csv_path = BASE_DIR / "data" / "raw" / "UNSW_NB15_testing-set.csv"
    with open(test_csv_path, "r", encoding="utf-8") as f:
        sample_rows = [f.readline() for _ in range(51)]
    sample_csv_bytes = "".join(sample_rows).encode("utf-8")

    upload_res = requests.post(
        f"{base_url}/api/predict",
        files={"file": ("sample_test.csv", sample_csv_bytes, "text/csv")}
    )
    assert upload_res.status_code == 200
    pred_summary = upload_res.json()
    assert pred_summary["total_rows"] == 50
    session_id = pred_summary["session_id"]
    assert session_id
    print(f"PASS: /api/predict (50 rows, Session ID: {session_id}, Anomalies: {pred_summary['attack_count']})")

    # 6. Fetch prediction page
    session_res = requests.get(f"{base_url}/api/predictions/{session_id}?page=1&page_size=20")
    assert session_res.status_code == 200
    page_data = session_res.json()
    assert len(page_data["results"]) == 20
    assert page_data["total_results"] == 50
    assert page_data["page"] == 1
    assert page_data["total_pages"] == 3

    first = page_data["results"][0]
    assert "id" in first
    assert "proto" in first
    assert "prediction" in first
    assert "confidence" in first
    print("PASS: /api/predictions/{session_id} pagination & record schema verified")

    # 7. Invalid CSV handling (empty file)
    bad_res = requests.post(
        f"{base_url}/api/predict",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert bad_res.status_code == 422
    assert "detail" in bad_res.json()
    print(f"PASS: /api/predict correctly rejected empty CSV with HTTP 422: {bad_res.json()['detail']}")

    print("\n=======================================================")
    print("ALL FRONTEND/API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    run_integration_tests()
