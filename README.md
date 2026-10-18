# Machine Learning-Based Network Anomaly Detection System

An enterprise-grade, offline-ready network intrusion and anomaly detection dashboard powered by **FastAPI** and **Scikit-Learn**, evaluated on the **UNSW-NB15 benchmark dataset**.

The system provides real-time CSV traffic analysis, authentic machine learning metrics, confusion matrix breakdowns, and paginated detection logs with protocol/status filtering—packaged in a modern, responsive UI designed for security operations.

---

## 1. System Requirements & Prerequisites

### Client Laptop Requirements
- **Operating System:** Windows 10 or Windows 11 (64-bit recommended)
- **Processor:** Dual-Core 2.0 GHz or higher (x86_64)
- **Memory (RAM):** 4 GB minimum (8 GB recommended for processing large CSV files)
- **Disk Space:** 500 MB free space for application, model bundle, and temporary session databases
- **Browser:** Any modern Chromium-based browser (Google Chrome, Microsoft Edge, Brave) or Mozilla Firefox
- **Network Access:** **None required.** The system operates 100% offline with locally compiled stylesheets and bundled fonts.

### Software Prerequisites
- **Python:** Version **3.10** to **3.13** (64-bit)
- Make sure Python is added to your system `PATH`. You can verify by opening Command Prompt or PowerShell and running:
  ```bat
  python --version
  ```

---

## 2. Quick Start & Launch Instructions

### Option A: One-Click Launch (Recommended)
Double-click the `run.bat` file in the project root directory:
```
D:\Sharu\Network nomaly detection\run.bat
```
The launcher will:
1. Verify your Python runtime and PATH.
2. Check and auto-install any missing dependencies from `requirements.txt`.
3. Start the FastAPI Uvicorn server on `http://127.0.0.1:8000`.
4. Automatically open your default web browser to the dashboard.

### Option B: Manual Command-Line Launch
1. Open a terminal (PowerShell or Command Prompt) and navigate to the project root:
   ```bat
   cd "D:\Sharu\Network nomaly detection"
   ```
2. (Optional) Install dependencies:
   ```bat
   python -m pip install -r requirements.txt
   ```
3. Start the application:
   ```bat
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```
4. Open your browser and navigate to:
   - **Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Interactive API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **System Health:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 3. Dashboard Screens & Architecture

The single-page application features 5 dedicated screens accessible via sidebar navigation and hash routes:

| Route | Screen Name | Description |
| :--- | :--- | :--- |
| `#/overview` | **Overview** | High-level traffic summary, active model status, normal vs. attack ratio donut charts, and live system metrics. |
| `#/model-performance` | **Model Performance** | Authentic evaluation metrics on held-out UNSW-NB15 test set: **Accuracy (90.74%)**, **Precision (87.68%)**, **Recall (96.79%)**, **F1-Score (92.01%)**, **ROC-AUC (0.9845)**, full 2x2 Confusion Matrix, and detailed Classification Report. |
| `#/detection-results` | **Detection Results** | Paginated inspection table of analyzed network flows with source/destination IPs, protocol, anomaly classification, and probability confidence. Includes real-time search and protocol/status filtering. |
| `#/dataset-analysis` | **Dataset Analysis** | UNSW-NB15 benchmark breakdown across 257,673 records, showing training vs. test splits, protocol proportions (TCP, UDP, etc.), and attack category distribution (Generic, Exploits, Fuzzers, DoS, Reconnaissance, etc.). |
| `#/csv-upload-prediction` | **CSV Upload & Prediction** | Drag-and-drop CSV upload workflow with file-size limits, format validation, a 6-step preprocessing progress stepper, and instant navigation to prediction results. |

---

## 4. Input CSV Format Specifications

When uploading network capture files for anomaly prediction, ensure the CSV meets the following specifications:

- **Maximum File Size:** 10 MiB per upload.
- **Maximum Row Count:** 100,000 rows per batch.
- **File Encoding:** UTF-8 plain text.
- **Header Row:** Must include the canonical UNSW-NB15 feature columns (or subset of core network flow features):
  - `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`.
- Optional fields: `id` (row identifier), `label` (ground truth 0/1; ignored during inference).

### Sample Test CSV
You can test the upload immediately using rows from the benchmark test dataset located in:
```
data/raw/UNSW_NB15_testing-set.csv
```

---

## 5. REST API Endpoints

The backend provides well-structured REST endpoints documented via OpenAPI/Swagger:

- `GET /` — Serves the integrated single-page dashboard.
- `GET /api/health` — Checks server status, active bundle ID, model type, and uptime.
- `GET /api/model-performance` — Returns authentic evaluation metrics and confusion matrix from the active bundle.
- `GET /api/dataset-analysis` — Returns benchmark dataset distributions, protocol counts, and attack classes.
- `POST /api/predict` — Accepts a CSV upload (`multipart/form-data`), executes chunked inference, and creates an isolated SQLite session.
- `GET /api/predictions/{session_id}` — Returns paginated prediction records supporting `page`, `page_size`, `search`, `proto`, and `status` query parameters.

---

## 6. Offline Architecture & Local Bundling

This project is engineered to work in air-gapped or offline enterprise environments:
- **No External CDNs:** External CDN dependencies (`cdn.tailwindcss.com`, `cdnjs`, `unpkg`) have been completely eliminated.
- **Local Tailwind Stylesheet:** Tailwind CSS v3 is compiled into `static/css/app.css` (28.2 KB minified), including all custom color palettes and utility classes.
- **Bundled Fonts:** Inter web fonts (weights 400, 500, 600, 700, 800) are stored locally in `static/fonts/` and served via `@font-face`.

---

## 7. Running Verification & Test Suites

The project includes three comprehensive verification test suites:

```bat
:: 1. Verify Model Bundle Integrity & Inference
python -u tests/test_bundle_loading.py

:: 2. Verify FastAPI Backend Endpoints & Error Handling
python -u tests/test_backend_api.py

:: 3. Verify Offline Assets & Frontend Integration
python -u tests/test_frontend_integration.py

:: 4. Verify Final Improvements, Samples, Export & Security Sanitization
python -u tests/test_final_improvements.py
```

All suites execute against the local project files and return clear pass/fail output without modifying the underlying model or datasets.

---

## 8. Benchmark Samples & Prediction Export

### Three Usable Sample CSV Options
The application provides 3 verified benchmark sample files located in `data/samples/`:
- **Normal Traffic Sample (`normal_traffic_sample.csv`):** 50 genuine benign flows from UNSW-NB15 test set with 0 nulls and 0 duplicates.
- **Attack Traffic Sample (`attack_traffic_sample.csv`):** 50 genuine anomalous flows covering varied cyberattack categories (Generic, Exploits, Fuzzers, DoS, Reconnaissance).
- **Mixed Traffic Sample (`mixed_traffic_sample.csv`):** 50 genuine flows (25 Normal, 25 Attack).

Each sample contains exactly the 42 model input feature columns in canonical order. Target labels (`id`, `attack_cat`, `label`) are completely excluded so the files act as authentic unlabelled test vectors. Users can download samples directly or click "Load" to stage them into the upload card.

### Complete & Filtered CSV Export
- **Full Session Export (`GET /api/predictions/{session_id}/export`):** Exports 100% of prediction records for the current session, completely independent of table pagination.
- **Filtered Export (`GET /api/predictions/{session_id}/export?filter_applied=true`):** Exports all records matching active search, protocol, or status filters across the full session.

---

## 9. Limitations & Security Advisory

1. **Model Prediction Advisory:** Predictions are model estimates, not confirmation of a real cyberattack. Results may include false positives and false negatives. The application does not guarantee live network intrusion prevention or automatic packet blocking.
2. **Formula Injection Sanitization (OWASP CWE-1236):** Exported CSV text cells that begin with formula triggers (`=`, `+`, `-`, `@`)—including those preceded by spaces, tabs, carriage returns, or newlines—are sanitized with a leading apostrophe (`'`). Legitimate signed numeric values (such as negative numbers) are verified via strict numeric regex and preserved as genuine numbers without apostrophe prefixing.
   - *Limitation Note:* While apostrophe-prefixing aligns with OWASP CWE-1236, it cannot guarantee universal safety across every external spreadsheet application version or custom locale configuration, and plain-text editors will display the literal leading apostrophe.
3. **Session Storage:** CSV predictions are indexed in isolated temporary SQLite databases under `temp/sessions/` for fast pagination and searching. Sessions expire after 30 minutes.
4. **Upload Limits:** Uploads are capped at 10 MiB and 100,000 rows to ensure stability on client laptops.
