import os
import re
import csv
import io
import time
import json
import sqlite3
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from ml.config import (
    MODELS_DIR, ACTIVE_MANIFEST_PATH, SESSIONS_DIR,
    FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    PREDICTION_CHUNK_SIZE, SESSION_EXPIRY_SECONDS
)
from ml.preprocessor import prepare_features, downcast_matrix_to_float32

class ModelBundle:
    def __init__(self, bundle_dir: Path, manifest: dict):
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.bundle_id = manifest.get("bundle_id", bundle_dir.name)
        self.use_float32 = manifest.get("hyperparameters", {}).get("float32_optimization", False)
        
        # Load artifacts
        self.model = joblib.load(bundle_dir / "model.joblib")
        self.preprocessor = joblib.load(bundle_dir / "preprocessor.joblib")
        
        # Load cached metrics & dataset analysis
        metrics_file = bundle_dir / "metrics.json"
        self.metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        
        analysis_file = bundle_dir / "dataset_analysis.json"
        self.dataset_analysis = json.loads(analysis_file.read_text(encoding="utf-8")) if analysis_file.exists() else {}

    def predict_dataframe(self, df: pd.DataFrame, return_probabilities: bool = False) -> Any:
        """
        Runs feature extraction, preprocessor transform, and model inference.
        Returns (predictions, confidences) or (predictions, confidences, attack_probs)
        """
        X_df, _ = prepare_features(df, is_training=False)
        X_trans = self.preprocessor.transform(X_df)
        if self.use_float32:
            X_trans = downcast_matrix_to_float32(X_trans)

        preds = self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)
        # Confidence is probability of the predicted class
        confidences = np.max(probs, axis=1)
        if return_probabilities:
            # Probability of attack (class 1)
            attack_col = 1 if (hasattr(self.model, "classes_") and len(self.model.classes_) > 1 and self.model.classes_[1] == 1) else 0
            attack_probs = probs[:, attack_col] if probs.shape[1] > 1 else probs[:, 0]
            return preds, confidences, attack_probs
        return preds, confidences

class ModelManager:
    """Thread-safe manager for loading and serving the active model bundle."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._current_bundle: Optional[ModelBundle] = None
                cls._instance._bundle_lock = threading.RLock()
        return cls._instance

    def get_bundle(self) -> Optional[ModelBundle]:
        with self._bundle_lock:
            if self._current_bundle is None:
                self.reload_active_bundle()
            return self._current_bundle

    def reload_active_bundle(self, manifest_path_override: Optional[Path] = None) -> bool:
        with self._bundle_lock:
            manifest_to_check = manifest_path_override if manifest_path_override is not None else ACTIVE_MANIFEST_PATH
            if not manifest_to_check.exists():
                return False

            try:
                active_meta = json.loads(manifest_to_check.read_text(encoding="utf-8"))
                bundle_rel_dir = active_meta.get("bundle_dir")
                if not bundle_rel_dir:
                    return False

                # Robust path resolution: handle absolute or relative (MODELS_DIR or BASE_DIR)
                raw_path = Path(bundle_rel_dir)
                if raw_path.is_absolute():
                    bundle_path = raw_path
                else:
                    bundle_path = MODELS_DIR / bundle_rel_dir
                    if not bundle_path.exists():
                        bundle_path = MODELS_DIR.parent / bundle_rel_dir

                manifest_path = bundle_path / "manifest.json"
                if not manifest_path.exists():
                    return False

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                
                # Verify checksums
                checksums = manifest.get("checksums", {})
                for fname, expected_hash in checksums.items():
                    fpath = bundle_path / fname
                    if not fpath.exists():
                        print(f"[ERROR] Missing artifact {fname} in {bundle_path.name}")
                        return False
                    h = hashlib.sha256(fpath.read_bytes()).hexdigest()
                    if h != expected_hash:
                        print(f"[ERROR] Checksum mismatch for {fname}")
                        return False

                # Load bundle
                new_bundle = ModelBundle(bundle_path, manifest)
                self._current_bundle = new_bundle
                print(f"[ModelManager] Activated bundle: {new_bundle.bundle_id}")
                return True
            except Exception as e:
                print(f"[ModelManager] Failed to load active bundle: {e}")
                return False

def init_session_db(db_path: Path):
    """Initializes a private SQLite session database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
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
            is_attack INTEGER,
            actual_label INTEGER,
            is_correct INTEGER,
            attack_prob REAL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(prediction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_proto ON predictions(proto)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            session_id TEXT PRIMARY KEY,
            total_rows INTEGER,
            attack_count INTEGER,
            normal_count INTEGER,
            attack_rate REAL,
            has_ground_truth INTEGER,
            evaluation_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def validate_csv_for_prediction(csv_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validates CSV existence, size, readable headers, and required feature columns.
    Returns (is_valid, error_message).
    """
    if not csv_path.exists():
        return False, "Uploaded file not found on server."
    if csv_path.stat().st_size == 0:
        return False, "Uploaded CSV file is empty (0 bytes)."

    try:
        header_df = pd.read_csv(csv_path, nrows=5)
        if len(header_df.columns) == 0:
            return False, "CSV contains no columns or is malformed."

        # Check required features (exclude id, attack_cat, label)
        cols = set(c.strip() for c in header_df.columns)
        missing = [col for col in FEATURE_COLUMNS if col not in cols]
        if missing:
            return False, f"Missing required feature columns: {', '.join(missing[:5])}" + (f" and {len(missing)-5} more" if len(missing) > 5 else "")
        return True, None
    except Exception as e:
        return False, f"Invalid or unparseable CSV file: {str(e)}"

def validate_session_id(session_id: str) -> bool:
    """Verifies that a session_id is a safe alphanumeric/UUID string to prevent path traversal."""
    import re
    if not session_id or not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        return False
    try:
        target = (SESSIONS_DIR / f"{session_id}.db").resolve()
        return target.parent == SESSIONS_DIR.resolve()
    except Exception:
        return False

def process_csv_into_session(csv_path: Path, session_id: str, model_bundle: ModelBundle) -> Dict[str, Any]:
    """
    Processes an uploaded CSV in chunks, performing model predictions and storing
    results in a private SQLite database. Enforces MAX_CSV_ROWS limit.
    Detects ground-truth 'label' column and evaluates authentic held-out metrics
    strictly from the uploaded records when valid actual labels are present.
    """
    if not validate_session_id(session_id):
        raise ValueError("Invalid session identifier.")

    valid, err_msg = validate_csv_for_prediction(csv_path)
    if not valid:
        raise ValueError(err_msg)

    # Detect if 'label' column is present in CSV header
    try:
        header_df = pd.read_csv(csv_path, nrows=5)
        raw_cols = [c.strip().lower() for c in header_df.columns]
        has_ground_truth = "label" in raw_cols
    except Exception:
        has_ground_truth = False

    db_path = SESSIONS_DIR / f"{session_id}.db"
    init_session_db(db_path)

    total_rows = 0
    attack_count = 0
    normal_count = 0
    preview_rows = []

    all_y_true = []
    all_y_pred = []
    all_attack_probs = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        chunk_idx = 0
        from ml.config import MAX_CSV_ROWS
        for chunk in pd.read_csv(csv_path, chunksize=PREDICTION_CHUNK_SIZE):
            if total_rows + len(chunk) > MAX_CSV_ROWS:
                raise ValueError(f"CSV row count exceeds maximum limit of {MAX_CSV_ROWS:,} rows.")

            preds, confs, attack_probs = model_bundle.predict_dataframe(chunk, return_probabilities=True)

            # Check if ground truth labels are valid in this chunk
            chunk_labels = None
            if has_ground_truth:
                # Find the label column case-insensitively
                lbl_col = next((c for c in chunk.columns if c.strip().lower() == "label"), None)
                if lbl_col:
                    numeric_labels = pd.to_numeric(chunk[lbl_col], errors="coerce")
                    if numeric_labels.isna().any():
                        has_ground_truth = False
                    else:
                        int_labels = numeric_labels.astype(int).values
                        if not set(int_labels).issubset({0, 1}):
                            has_ground_truth = False
                        else:
                            chunk_labels = int_labels
                else:
                    has_ground_truth = False

            records_to_insert = []
            for i in range(len(chunk)):
                row = chunk.iloc[i]
                pred_val = int(preds[i])
                is_attack = 1 if pred_val == 1 else 0
                pred_label = "Attack" if is_attack == 1 else "Normal"
                conf = float(confs[i])
                atk_prob = float(attack_probs[i])

                if is_attack == 1:
                    attack_count += 1
                else:
                    normal_count += 1

                row_id = total_rows + i + 1

                # Ground truth mapping
                actual_lbl = None
                is_corr = None
                if has_ground_truth and chunk_labels is not None:
                    actual_lbl = int(chunk_labels[i])
                    is_corr = 1 if (actual_lbl == is_attack) else 0
                    all_y_true.append(actual_lbl)

                all_y_pred.append(is_attack)
                all_attack_probs.append(atk_prob)

                # Extract fields with safe fallbacks
                src_ip = str(row.get("srcip", row.get("src_ip", f"Node-{row_id}")))
                dst_ip = str(row.get("dstip", row.get("dst_ip", f"Host-{row.get('service', 'srv')}")))
                proto = str(row.get("proto", "TCP")).upper()
                duration = float(row.get("dur", row.get("duration", 0.0)))
                src_bytes = float(row.get("sbytes", row.get("src_bytes", 0.0)))
                dst_bytes = float(row.get("dbytes", row.get("dst_bytes", 0.0)))
                
                # Count fields: use ct_src_ltm / ct_srv_dst or direct count/srv_count
                count_val = float(row.get("count", row.get("ct_src_ltm", 0.0)))
                srv_count_val = float(row.get("srv_count", row.get("ct_srv_dst", 0.0)))
                timestamp_val = str(row.get("timestamp", row.get("stime", f"Seq #{row_id}")))

                rec = (
                    row_id, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
                    count_val, srv_count_val, timestamp_val, pred_label, round(conf * 100, 1), is_attack,
                    actual_lbl, is_corr, atk_prob
                )
                records_to_insert.append(rec)

                if len(preview_rows) < 10:
                    preview_rows.append({
                        "id": row_id,
                        "src": src_ip,
                        "dst": dst_ip,
                        "proto": proto,
                        "duration": duration,
                        "src_bytes": src_bytes,
                        "dst_bytes": dst_bytes,
                        "count": count_val,
                        "srv_count": srv_count_val,
                        "timestamp": timestamp_val,
                        "prediction": pred_label,
                        "confidence": f"{round(conf * 100, 1)}%",
                        "attack_prob": round(atk_prob * 100, 1),
                        "actual_label": actual_lbl,
                        "actual_status": ("Attack" if actual_lbl == 1 else "Normal") if actual_lbl is not None else None,
                        "is_correct": bool(is_corr) if is_corr is not None else None
                    })

            cursor.executemany("""
                INSERT INTO predictions (
                    row_idx, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
                    count_val, srv_count_val, timestamp_val, prediction, confidence, is_attack,
                    actual_label, is_correct, attack_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_to_insert)
            conn.commit()
            total_rows += len(chunk)
            chunk_idx += 1

        # Calculate metrics if ground truth is verified for 100% of rows
        evaluation = None
        if has_ground_truth and len(all_y_true) == total_rows and total_rows > 0:
            y_true_arr = np.array(all_y_true, dtype=int)
            y_pred_arr = np.array(all_y_pred, dtype=int)
            y_scores_arr = np.array(all_attack_probs, dtype=float)

            tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
            fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
            fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
            tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

            acc = float((tp + tn) / total_rows)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            # ROC Curve & ROC-AUC (Requires both binary classes in ground truth)
            classes_present = np.unique(y_true_arr)
            if len(classes_present) == 2:
                from sklearn.metrics import roc_curve, roc_auc_score
                try:
                    auc_val = float(roc_auc_score(y_true_arr, y_scores_arr))
                    fpr_vals, tpr_vals, _ = roc_curve(y_true_arr, y_scores_arr)
                    if len(fpr_vals) > 50:
                        sampled_idxs = np.linspace(0, len(fpr_vals) - 1, 50, dtype=int)
                        fpr_out = [round(float(x), 4) for x in fpr_vals[sampled_idxs]]
                        tpr_out = [round(float(x), 4) for x in tpr_vals[sampled_idxs]]
                    else:
                        fpr_out = [round(float(x), 4) for x in fpr_vals]
                        tpr_out = [round(float(x), 4) for x in tpr_vals]
                    roc_info = {
                        "available": True,
                        "auc": round(auc_val, 4),
                        "fpr": fpr_out,
                        "tpr": tpr_out
                    }
                except Exception as roc_err:
                    roc_info = {
                        "available": False,
                        "reason": f"ROC computation error: {str(roc_err)}",
                        "message": f"ROC computation error: {str(roc_err)}",
                        "auc": None,
                        "fpr": [],
                        "tpr": []
                    }
            else:
                single_cls_name = "Normal (0)" if 0 in classes_present else "Attack (1)"
                reason_str = f"ROC curve requires both Normal and Attack classes in actual ground truth (uploaded CSV contains only {single_cls_name})."
                roc_info = {
                    "available": False,
                    "reason": reason_str,
                    "message": reason_str,
                    "auc": None,
                    "fpr": [],
                    "tpr": []
                }

            evaluation = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp
                },
                "roc": roc_info
            }
        else:
            has_ground_truth = False

        # Store session summary and evaluation in session_meta table
        attack_rate = round((attack_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        eval_json_str = json.dumps(evaluation) if evaluation is not None else None
        cursor.execute("""
            INSERT OR REPLACE INTO session_meta (
                session_id, total_rows, attack_count, normal_count, attack_rate, has_ground_truth, evaluation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, total_rows, attack_count, normal_count, attack_rate,
            1 if has_ground_truth else 0, eval_json_str
        ))
        conn.commit()

    except Exception:
        conn.close()
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass
        raise

    conn.close()

    return {
        "session_id": session_id,
        "total_rows": total_rows,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "attack_rate": round((attack_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "has_ground_truth": has_ground_truth,
        "evaluation": evaluation,
        "accuracy": evaluation.get("accuracy") if evaluation else None,
        "precision": evaluation.get("precision") if evaluation else None,
        "recall": evaluation.get("recall") if evaluation else None,
        "f1_score": evaluation.get("f1_score") if evaluation else None,
        "confusion_matrix": evaluation.get("confusion_matrix") if evaluation else None,
        "roc_auc": evaluation.get("roc", {}).get("auc") if evaluation else None,
        "roc_curve": evaluation.get("roc") if evaluation else {"available": False, "fpr": [], "tpr": []},
        "preview": preview_rows
    }

def query_session_predictions(
    session_id: str,
    page: int = 1,
    page_size: int = 50,
    search: str = "",
    proto: str = "ALL",
    status: str = "ALL"
) -> Optional[Dict[str, Any]]:
    """Queries paginated predictions from SQLite session. Returns None if session not found or invalid."""
    if not validate_session_id(session_id):
        return None

    db_path = SESSIONS_DIR / f"{session_id}.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    conditions = []
    params = []

    if proto and proto != "ALL":
        conditions.append("proto = ?")
        params.append(proto.upper())

    if status and status != "ALL":
        conditions.append("prediction = ?")
        params.append(status.capitalize())

    if search:
        search_like = f"%{search}%"
        conditions.append("(src_ip LIKE ? OR dst_ip LIKE ? OR proto LIKE ? OR prediction LIKE ?)")
        params.extend([search_like, search_like, search_like, search_like])

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    # Count total matching
    count_sql = f"SELECT COUNT(*) FROM predictions {where_clause}"
    cursor.execute(count_sql, params)
    total_matches = cursor.fetchone()[0]

    # Query page
    offset = max(0, (page - 1) * page_size)
    query_sql = f"""
        SELECT row_idx as id, src_ip as src, dst_ip as dst, proto,
               duration, src_bytes, dst_bytes, count_val as count,
               srv_count_val as srv_count, timestamp_val as timestamp,
               prediction, confidence, is_attack, actual_label, is_correct, attack_prob
        FROM predictions
        {where_clause}
        ORDER BY id ASC
        LIMIT ? OFFSET ?
    """
    cursor.execute(query_sql, params + [page_size, offset])
    rows = cursor.fetchall()
    
    results = []
    for r in rows:
        conf_val = float(r["confidence"]) if r["confidence"] is not None else 0.0
        act_lbl = int(r["actual_label"]) if r["actual_label"] is not None else None
        is_corr = bool(r["is_correct"]) if r["is_correct"] is not None else None
        atk_p = float(r["attack_prob"]) if r["attack_prob"] is not None else (1.0 if r["prediction"] == "Attack" else 0.0)

        results.append({
            "id": r["id"],
            "src": r["src"],
            "dst": r["dst"],
            "proto": r["proto"],
            "duration": r["duration"],
            "src_bytes": r["src_bytes"],
            "srcbytes": r["src_bytes"],
            "dst_bytes": r["dst_bytes"],
            "dstbytes": r["dst_bytes"],
            "count": r["count"],
            "srv_count": r["srv_count"],
            "srvcount": r["srv_count"],
            "timestamp": r["timestamp"],
            "prediction": r["prediction"],
            "confidence": f"{round(conf_val, 1)}%",
            "attack_prob": round(atk_p * 100, 1),
            "actual_label": act_lbl,
            "actual_status": ("Attack" if act_lbl == 1 else "Normal") if act_lbl is not None else None,
            "is_correct": is_corr
        })
    conn.close()

    total_pages = (total_matches + page_size - 1) // page_size if total_matches > 0 else 1

    return {
        "session_id": session_id,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "total_results": total_matches,
        "results": results
    }

def get_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves session summary metadata and evaluation metrics from SQLite."""
    if not validate_session_id(session_id):
        return None

    db_path = SESSIONS_DIR / f"{session_id}.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM session_meta WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None

        eval_data = json.loads(row["evaluation_json"]) if row["evaluation_json"] else None
        return {
            "session_id": row["session_id"],
            "total_rows": row["total_rows"],
            "attack_count": row["attack_count"],
            "normal_count": row["normal_count"],
            "attack_rate": row["attack_rate"],
            "has_ground_truth": bool(row["has_ground_truth"]),
            "evaluation": eval_data,
            "accuracy": eval_data.get("accuracy") if eval_data else None,
            "precision": eval_data.get("precision") if eval_data else None,
            "recall": eval_data.get("recall") if eval_data else None,
            "f1_score": eval_data.get("f1_score") if eval_data else None,
            "confusion_matrix": eval_data.get("confusion_matrix") if eval_data else None,
            "roc_auc": eval_data.get("roc", {}).get("auc") if eval_data else None,
            "roc_curve": eval_data.get("roc") if eval_data else {"available": False, "fpr": [], "tpr": []}
        }
    finally:
        conn.close()

def clean_expired_sessions():
    """Removes SQLite session files older than SESSION_EXPIRY_SECONDS."""
    now = time.time()
    try:
        for p in SESSIONS_DIR.glob("*.db"):
            if now - p.stat().st_mtime > SESSION_EXPIRY_SECONDS:
                p.unlink(missing_ok=True)
    except Exception as e:
        print(f"[SessionCleaner] Error: {e}")

# Strict numeric regex matching only standard signed/unsigned integer, decimal, and scientific literals
STRICT_NUMERIC_RE = re.compile(r"^[-+]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?$")

def sanitize_csv_cell(val: Any) -> Any:
    """
    Sanitizes values against CSV formula injection (OWASP CWE-1236)
    while preserving genuine numeric types and legitimate negative/positive numbers.
    """
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return val
    if val is None:
        return ""

    s = str(val)
    if not s:
        return s

    stripped = s.strip(" \t\r\n")
    if not stripped:
        return s

    has_leading_control = s[0] in ("\t", "\r", "\n")
    has_formula_trigger = stripped[0] in ("=", "+", "-", "@")

    if has_leading_control or has_formula_trigger:
        if STRICT_NUMERIC_RE.match(stripped):
            return val
        return "'" + s

    return s

def export_session_predictions_csv(
    session_id: str,
    filter_applied: bool = False,
    search: str = "",
    proto: str = "ALL",
    status: str = "ALL"
) -> Optional[str]:
    """
    Exports prediction records from the session SQLite database as sanitized CSV text.
    Preserves actual label and includes prediction probability.
    All non-numeric cells are protected against spreadsheet formula injection.
    """
    import csv
    import io

    if not validate_session_id(session_id):
        return None

    db_path = SESSIONS_DIR / f"{session_id}.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        has_gt = False
        try:
            cursor.execute("SELECT has_ground_truth FROM session_meta WHERE session_id = ?", (session_id,))
            meta_row = cursor.fetchone()
            has_gt = bool(meta_row[0]) if meta_row else False
        except sqlite3.OperationalError:
            has_gt = False

        # Inspect available columns in predictions table
        cursor.execute("PRAGMA table_info(predictions)")
        pred_cols = {col[1] for col in cursor.fetchall()}
        has_actual_label = has_gt and ("actual_label" in pred_cols)
        has_attack_prob = "attack_prob" in pred_cols

        conditions = []
        params = []

        if filter_applied:
            if proto and proto != "ALL":
                conditions.append("proto = ?")
                params.append(proto.upper())

            if status and status != "ALL":
                conditions.append("prediction = ?")
                params.append(status.capitalize())

            if search:
                search_like = f"%{search}%"
                conditions.append("(src_ip LIKE ? OR dst_ip LIKE ? OR proto LIKE ? OR prediction LIKE ?)")
                params.extend([search_like, search_like, search_like, search_like])

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        if has_actual_label:
            query_sql = f"""
                SELECT row_idx, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
                       count_val as count, srv_count_val as srv_count, timestamp_val as timestamp,
                       actual_label, prediction, is_correct, 
                       {'attack_prob' if has_attack_prob else '0.0 as attack_prob'}, confidence
                FROM predictions
                {where_clause}
                ORDER BY row_idx ASC
            """
            headers = [
                "row_id", "src_ip", "dst_ip", "proto", "duration",
                "src_bytes", "dst_bytes", "count", "srv_count", "timestamp",
                "label", "prediction", "is_correct", "attack_probability", "confidence"
            ]
        elif has_attack_prob:
            query_sql = f"""
                SELECT row_idx, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
                       count_val as count, srv_count_val as srv_count, timestamp_val as timestamp,
                       prediction, attack_prob, confidence
                FROM predictions
                {where_clause}
                ORDER BY row_idx ASC
            """
            headers = [
                "row_id", "src_ip", "dst_ip", "proto", "duration",
                "src_bytes", "dst_bytes", "count", "srv_count", "timestamp",
                "prediction", "attack_probability", "confidence"
            ]
        else:
            query_sql = f"""
                SELECT row_idx, src_ip, dst_ip, proto, duration, src_bytes, dst_bytes,
                       count_val as count, srv_count_val as srv_count, timestamp_val as timestamp,
                       prediction, confidence
                FROM predictions
                {where_clause}
                ORDER BY row_idx ASC
            """
            headers = [
                "row_id", "src_ip", "dst_ip", "proto", "duration",
                "src_bytes", "dst_bytes", "count", "srv_count", "timestamp",
                "prediction", "confidence"
            ]

        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(headers)

        for r in rows:
            conf_val = float(r["confidence"]) if r["confidence"] is not None else 0.0
            conf_str = f"{round(conf_val, 1)}%"

            if has_actual_label:
                atk_p = float(r["attack_prob"]) if ("attack_prob" in r.keys() and r["attack_prob"] is not None) else (1.0 if r["prediction"] == "Attack" else 0.0)
                atk_p_str = f"{round(atk_p * 100, 1)}%"
                act_lbl = int(r["actual_label"]) if r["actual_label"] is not None else ""
                is_corr = "Correct" if r["is_correct"] == 1 else "Incorrect" if r["is_correct"] == 0 else ""
                row_data = [
                    sanitize_csv_cell(r["row_idx"]),
                    sanitize_csv_cell(r["src_ip"]),
                    sanitize_csv_cell(r["dst_ip"]),
                    sanitize_csv_cell(r["proto"]),
                    sanitize_csv_cell(r["duration"]),
                    sanitize_csv_cell(r["src_bytes"]),
                    sanitize_csv_cell(r["dst_bytes"]),
                    sanitize_csv_cell(r["count"]),
                    sanitize_csv_cell(r["srv_count"]),
                    sanitize_csv_cell(r["timestamp"]),
                    sanitize_csv_cell(act_lbl),
                    sanitize_csv_cell(r["prediction"]),
                    sanitize_csv_cell(is_corr),
                    sanitize_csv_cell(atk_p_str),
                    sanitize_csv_cell(conf_str)
                ]
            elif has_attack_prob:
                atk_p = float(r["attack_prob"]) if ("attack_prob" in r.keys() and r["attack_prob"] is not None) else (1.0 if r["prediction"] == "Attack" else 0.0)
                atk_p_str = f"{round(atk_p * 100, 1)}%"
                row_data = [
                    sanitize_csv_cell(r["row_idx"]),
                    sanitize_csv_cell(r["src_ip"]),
                    sanitize_csv_cell(r["dst_ip"]),
                    sanitize_csv_cell(r["proto"]),
                    sanitize_csv_cell(r["duration"]),
                    sanitize_csv_cell(r["src_bytes"]),
                    sanitize_csv_cell(r["dst_bytes"]),
                    sanitize_csv_cell(r["count"]),
                    sanitize_csv_cell(r["srv_count"]),
                    sanitize_csv_cell(r["timestamp"]),
                    sanitize_csv_cell(r["prediction"]),
                    sanitize_csv_cell(atk_p_str),
                    sanitize_csv_cell(conf_str)
                ]
            else:
                row_data = [
                    sanitize_csv_cell(r["row_idx"]),
                    sanitize_csv_cell(r["src_ip"]),
                    sanitize_csv_cell(r["dst_ip"]),
                    sanitize_csv_cell(r["proto"]),
                    sanitize_csv_cell(r["duration"]),
                    sanitize_csv_cell(r["src_bytes"]),
                    sanitize_csv_cell(r["dst_bytes"]),
                    sanitize_csv_cell(r["count"]),
                    sanitize_csv_cell(r["srv_count"]),
                    sanitize_csv_cell(r["timestamp"]),
                    sanitize_csv_cell(r["prediction"]),
                    sanitize_csv_cell(conf_str)
                ]
            writer.writerow(row_data)

        return output.getvalue()
    finally:
        conn.close()
