import os
from pathlib import Path

# Base Paths (strictly within current project folder)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
BUNDLES_DIR = MODELS_DIR / "bundles"
ACTIVE_MANIFEST_PATH = MODELS_DIR / "active_manifest.json"
TEMP_DIR = BASE_DIR / "temp"
SESSIONS_DIR = TEMP_DIR / "sessions"
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "data" / "samples"

# Ensure essential runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "fonts").mkdir(parents=True, exist_ok=True)

# Dataset URLs (Verified Hugging Face UNSW-NB15 small benchmark)
TRAIN_DATASET_URL = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv"
TEST_DATASET_URL = "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv"

TRAIN_DATASET_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_DATASET_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

# Expected Dataset Counts & Properties
EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TEST_ROWS = 82332
EXPECTED_TOTAL_COLUMNS = 45

# Canonical 45-column schema
CANONICAL_COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label"
]

# Excluded from model features (3 columns)
EXCLUDED_FEATURES = ["id", "attack_cat", "label"]

# Model feature columns (exactly 42 columns)
FEATURE_COLUMNS = [col for col in CANONICAL_COLUMNS if col not in EXCLUDED_FEATURES]

# Categorical vs Numeric Features
CATEGORICAL_FEATURES = ["proto", "service", "state"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

# Upload & Streaming Constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_CSV_ROWS = 100000
PREDICTION_CHUNK_SIZE = 10000
PAGE_SIZE_DEFAULT = 50
SESSION_EXPIRY_SECONDS = 1800  # 30 minutes
