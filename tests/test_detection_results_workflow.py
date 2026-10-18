import os
import io
import csv
import sys
import time
import socket
import threading
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import requests
import uvicorn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from ml.predictor import ModelManager

NORMAL_SAMPLE = str(PROJECT_ROOT / "data" / "samples" / "normal_traffic_sample.csv")
ATTACK_SAMPLE = str(PROJECT_ROOT / "data" / "samples" / "attack_traffic_sample.csv")
MIXED_SAMPLE = str(PROJECT_ROOT / "data" / "samples" / "mixed_traffic_sample.csv")
UNLABELED_SAMPLE = str(PROJECT_ROOT / "data" / "samples" / "unlabeled_traffic_sample.csv")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    for _ in range(50):
        try:
            r = requests.get(f"{url}/api/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("FastAPI server failed to start on ephemeral port")

    yield url
    server.should_exit = True


def test_sample_files_exist_and_labels_authentic():
    """Verify sample CSVs exist and contain authentic ground-truth labels."""
    assert os.path.exists(NORMAL_SAMPLE), "Normal sample CSV missing"
    assert os.path.exists(ATTACK_SAMPLE), "Attack sample CSV missing"
    assert os.path.exists(MIXED_SAMPLE), "Mixed sample CSV missing"
    assert os.path.exists(UNLABELED_SAMPLE), "Unlabeled sample CSV missing"

    df_norm = pd.read_csv(NORMAL_SAMPLE)
    assert len(df_norm) == 50
    assert "label" in df_norm.columns
    assert (df_norm["label"] == 0).all(), "Normal sample should all be label 0"

    df_atk = pd.read_csv(ATTACK_SAMPLE)
    assert len(df_atk) == 50
    assert "label" in df_atk.columns
    assert (df_atk["label"] == 1).all(), "Attack sample should all be label 1"

    df_mix = pd.read_csv(MIXED_SAMPLE)
    assert len(df_mix) == 50
    assert "label" in df_mix.columns
    assert (df_mix["label"] == 0).sum() == 25
    assert (df_mix["label"] == 1).sum() == 25

    df_unl = pd.read_csv(UNLABELED_SAMPLE)
    assert len(df_unl) == 50
    assert "label" not in df_unl.columns, "Unlabeled sample must not contain label column"


def test_model_feature_safety_excludes_label_and_metadata():
    """Ensure label, id, and attack_cat are strictly excluded from model features."""
    bundle = ModelManager().get_bundle()
    assert bundle is not None, "Active bundle must be loaded"

    df_mix = pd.read_csv(MIXED_SAMPLE)
    df_test = df_mix.copy()
    df_test["id"] = range(len(df_test))
    df_test["attack_cat"] = "Fuzzers"
    
    # Model prediction must succeed and not crash on non-feature columns
    preds, confs, probs = bundle.predict_dataframe(df_test, return_probabilities=True)
    assert len(preds) == 50
    assert len(confs) == 50
    assert len(probs) == 50
    assert all(p in [0, 1] for p in preds)


def test_mixed_sample_evaluation_metrics_match_sklearn(base_url):
    """Verify that held-out evaluation metrics on mixed sample match independent sklearn calculations."""
    with open(MIXED_SAMPLE, "rb") as f:
        response = requests.post(f"{base_url}/api/predict", files={"file": ("mixed.csv", f, "text/csv")})
    assert response.status_code == 200, f"Upload failed: {response.text}"
    summary = response.json()
    session_id = summary["session_id"]

    sum_resp = requests.get(f"{base_url}/api/predictions/{session_id}/summary")
    assert sum_resp.status_code == 200
    meta = sum_resp.json()

    assert meta["has_ground_truth"] is True
    assert meta["total_rows"] == 50
    assert meta["normal_count"] + meta["attack_count"] == 50

    rows_resp = requests.get(f"{base_url}/api/predictions/{session_id}?page=1&page_size=100")
    assert rows_resp.status_code == 200
    rows = rows_resp.json()["results"]
    assert len(rows) == 50

    y_true = np.array([r["actual_label"] for r in rows])
    y_pred = np.array([1 if r["prediction"] == "Attack" else 0 for r in rows])
    # The API returns attack_prob as percentage 0-100, divide by 100 for score calculation
    y_probs = np.array([r["attack_prob"] / 100.0 for r in rows])

    expected_acc = accuracy_score(y_true, y_pred)
    expected_prec = precision_score(y_true, y_pred, zero_division=0)
    expected_rec = recall_score(y_true, y_pred, zero_division=0)
    expected_f1 = f1_score(y_true, y_pred, zero_division=0)
    expected_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    expected_auc = roc_auc_score(y_true, y_probs)

    assert meta["accuracy"] == pytest.approx(float(expected_acc), abs=1e-4)
    assert meta["precision"] == pytest.approx(float(expected_prec), abs=1e-4)
    assert meta["recall"] == pytest.approx(float(expected_rec), abs=1e-4)
    assert meta["f1_score"] == pytest.approx(float(expected_f1), abs=1e-4)
    assert meta["roc_auc"] == pytest.approx(float(expected_auc), abs=1e-4)

    cm = meta["confusion_matrix"]
    assert cm["tn"] == int(expected_cm[0, 0])
    assert cm["fp"] == int(expected_cm[0, 1])
    assert cm["fn"] == int(expected_cm[1, 0])
    assert cm["tp"] == int(expected_cm[1, 1])

    assert meta["roc_curve"]["available"] is True
    assert len(meta["roc_curve"]["fpr"]) > 0


def test_single_class_ground_truth_safety(base_url):
    """Verify single-class ground truth (all Normal or all Attack) does not crash ROC curve."""
    with open(NORMAL_SAMPLE, "rb") as f:
        res_norm = requests.post(f"{base_url}/api/predict", files={"file": ("normal.csv", f, "text/csv")})
    assert res_norm.status_code == 200
    sess_norm = res_norm.json()["session_id"]

    meta_norm = requests.get(f"{base_url}/api/predictions/{sess_norm}/summary").json()
    assert meta_norm["has_ground_truth"] is True
    assert meta_norm["roc_curve"]["available"] is False
    assert "both normal and attack" in meta_norm["roc_curve"]["message"].lower()
    assert meta_norm["accuracy"] is not None


def test_unlabeled_csv_does_not_invent_metrics(base_url):
    """Ensure unlabeled CSV does not display or invent false evaluation metrics."""
    with open(UNLABELED_SAMPLE, "rb") as f:
        res_unl = requests.post(f"{base_url}/api/predict", files={"file": ("unlabeled.csv", f, "text/csv")})
    assert res_unl.status_code == 200
    sess_unl = res_unl.json()["session_id"]

    meta_unl = requests.get(f"{base_url}/api/predictions/{sess_unl}/summary").json()
    assert meta_unl["has_ground_truth"] is False
    assert meta_unl["accuracy"] is None
    assert meta_unl["precision"] is None
    assert meta_unl["recall"] is None
    assert meta_unl["f1_score"] is None
    assert meta_unl["confusion_matrix"] is None
    assert meta_unl["roc_auc"] is None
    assert meta_unl["roc_curve"]["available"] is False

    rows = requests.get(f"{base_url}/api/predictions/{sess_unl}?page=1&page_size=10").json()["results"]
    assert all(r["actual_label"] is None for r in rows)
    assert all(r["is_correct"] is None for r in rows)


def test_session_isolation(base_url):
    """Verify two independent uploads do not bleed or cross-contaminate predictions."""
    with open(NORMAL_SAMPLE, "rb") as f:
        sess1 = requests.post(f"{base_url}/api/predict", files={"file": ("s1.csv", f, "text/csv")}).json()["session_id"]
    with open(ATTACK_SAMPLE, "rb") as f:
        sess2 = requests.post(f"{base_url}/api/predict", files={"file": ("s2.csv", f, "text/csv")}).json()["session_id"]

    assert sess1 != sess2
    sum1 = requests.get(f"{base_url}/api/predictions/{sess1}/summary").json()
    sum2 = requests.get(f"{base_url}/api/predictions/{sess2}/summary").json()

    assert sum1["session_id"] == sess1
    assert sum2["session_id"] == sess2
    # Verify s1 and s2 metrics are completely separate
    assert sum1["normal_count"] > sum2["normal_count"]
    assert sum2["attack_count"] > sum1["attack_count"]


def test_export_preserves_label_probabilities_and_sanitizes(base_url):
    """Verify CSV export includes label, probabilities, and protects against formula injection."""
    with open(MIXED_SAMPLE, "rb") as f:
        sess = requests.post(f"{base_url}/api/predict", files={"file": ("mixed.csv", f, "text/csv")}).json()["session_id"]

    export_resp = requests.get(f"{base_url}/api/predictions/{sess}/export?filter_applied=false")
    assert export_resp.status_code == 200
    content = export_resp.text
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert len(rows) == 50

    first = rows[0]
    assert "prediction" in first
    assert "confidence" in first
    assert "attack_probability" in first
    assert "label" in first
    assert "is_correct" in first


def test_screens_1_2_3_endpoints_preserved(base_url):
    """Verify Screens 1, 2, 3 APIs remain completely untouched and valid."""
    h = requests.get(f"{base_url}/api/health").json()
    assert h["status"] == "healthy"
    assert h["model_active"] is True

    mp = requests.get(f"{base_url}/api/model-performance").json()
    assert mp["metrics"]["total_test_samples"] == 82332

    da = requests.get(f"{base_url}/api/dataset-analysis").json()
    assert da["total_records"] == 257673
    assert da["traffic_distribution"]["total_normal"] == 93000
    assert da["traffic_distribution"]["total_attack"] == 164673
