import os
import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np

from ml.config import (
    TRAIN_DATASET_URL, TEST_DATASET_URL,
    TRAIN_DATASET_PATH, TEST_DATASET_PATH,
    EXPECTED_TRAIN_ROWS, EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_COLUMNS, CANONICAL_COLUMNS
)

def validate_dataset_partition(file_path: Path, expected_rows: int, partition_name: str) -> tuple[bool, str]:
    """
    Independently validate a dataset partition against the required 45-column schema,
    expected row count, column types, and binary target labels.
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if file_path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {file_path}"

    try:
        # Inspect headers first without loading whole file
        header_df = pd.read_csv(file_path, nrows=5)
        cols = list(header_df.columns)

        # Check total column count
        if len(cols) != EXPECTED_TOTAL_COLUMNS:
            return False, f"{partition_name}: Expected {EXPECTED_TOTAL_COLUMNS} columns, got {len(cols)}"

        # Check for duplicate columns
        if len(cols) != len(set(cols)):
            duplicates = [c for c in cols if cols.count(c) > 1]
            return False, f"{partition_name}: Duplicate columns detected: {set(duplicates)}"

        # Check for missing canonical columns
        missing_cols = [c for c in CANONICAL_COLUMNS if c not in cols]
        if missing_cols:
            return False, f"{partition_name}: Missing expected canonical columns: {missing_cols}"

        # Check for unexpected extra columns
        extra_cols = [c for c in cols if c not in CANONICAL_COLUMNS]
        if extra_cols:
            return False, f"{partition_name}: Unexpected extra columns: {extra_cols}"

        # Now load full partition to validate row count and labels
        df = pd.read_csv(file_path)

        if len(df) != expected_rows:
            return False, f"{partition_name}: Expected {expected_rows} rows, got {len(df)}"

        # Validate label column
        if "label" not in df.columns:
            return False, f"{partition_name}: 'label' column is missing."

        unique_labels = df["label"].dropna().unique()
        if not np.isin(unique_labels, [0, 1]).all():
            return False, f"{partition_name}: Target 'label' must contain only 0 and 1. Found: {unique_labels}"

        # Validate categorical columns exist and have non-empty values
        for cat_col in ["proto", "service", "state"]:
            if df[cat_col].isna().all():
                return False, f"{partition_name}: Column '{cat_col}' contains entirely null/empty values."

        return True, f"{partition_name}: Successfully verified ({len(df):,} rows, {len(cols)} columns, valid binary labels)."

    except Exception as e:
        return False, f"{partition_name}: Validation exception encountered: {str(e)}"

def download_file_stream(url: str, tmp_path: Path, timeout: int = 180):
    """
    Downloads a file in streaming chunks.
    If Python's bundled certifi encounters an enterprise/antivirus SSL proxy error,
    gracefully falls back to Windows native curl.exe which uses the OS certificate store.
    """
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 512  # 512 KB

    print(f"Downloading from {url} to {tmp_path.name}...")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb:.1f} MB / {tot_mb:.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)
        print("\nDownload complete.")
    except (requests.exceptions.SSLError, Exception) as req_err:
        print(f"\n[INFO] Standard HTTPS failed ({type(req_err).__name__}). Using Windows native curl with OS certificate store...")
        import subprocess
        # Use curl.exe with -L (follow redirects) and output to tmp_path
        cmd = ["curl.exe", "-L", "--fail", "--silent", "--show-error", "-o", str(tmp_path), url]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"curl download failed (exit code {res.returncode}): {res.stderr}")
        print("Download complete via Windows OS secure channel.")


def ensure_dataset(force_download: bool = False) -> bool:
    """
    Ensures both training and testing datasets are present, valid, and uncorrupted.
    Downloads to .tmp files and safely replaces only after 100% successful validation.
    """
    partitions = [
        ("Training Dataset", TRAIN_DATASET_URL, TRAIN_DATASET_PATH, EXPECTED_TRAIN_ROWS),
        ("Testing Dataset", TEST_DATASET_URL, TEST_DATASET_PATH, EXPECTED_TEST_ROWS)
    ]

    all_valid = True

    for name, url, path, expected_rows in partitions:
        print(f"\n--- Checking {name} ---")
        
        # 1. Check existing destination file
        if path.exists() and not force_download:
            valid, msg = validate_dataset_partition(path, expected_rows, name)
            if valid:
                print(f"[RETAIN] {msg}")
                continue
            else:
                print(f"[INVALID EXISTING] {msg}. Re-downloading required.")
        elif not path.exists():
            print(f"[MISSING] Destination file not found: {path.name}")

        # 2. Download to temporary staging file
        tmp_path = path.with_suffix(".csv.tmp")
        try:
            download_file_stream(url, tmp_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"\n[DOWNLOAD ERROR] Failed to download {name}: {e}")
            _print_recovery_instructions(name, url, path)
            all_valid = False
            continue

        # 3. Validate temporary staging file
        print(f"Validating {tmp_path.name}...")
        valid, msg = validate_dataset_partition(tmp_path, expected_rows, name)

        if valid:
            # 4. Safe atomic replace
            try:
                # Ensure canonical column order before final swap
                df_clean = pd.read_csv(tmp_path)
                df_clean = df_clean[CANONICAL_COLUMNS]
                df_clean.to_csv(tmp_path, index=False)
                
                # Atomic swap
                if path.exists():
                    path.unlink()  # On Windows, unlink first if destination exists to guarantee atomic replace
                os.replace(str(tmp_path), str(path))
                print(f"[SUCCESS] {name} validated and activated safely at {path.name}.")
            except Exception as e:
                print(f"[REPLACE ERROR] Could not activate {name}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                all_valid = False
        else:
            print(f"\n[VALIDATION FAILED] {msg}")
            if tmp_path.exists():
                tmp_path.unlink()
            _print_recovery_instructions(name, url, path)
            all_valid = False

    return all_valid

def _print_recovery_instructions(name: str, url: str, path: Path):
    print("=" * 70)
    print(f"MANUAL RECOVERY REQUIRED FOR: {name}")
    print(f"Please manually download the verified CSV from:")
    print(f"  {url}")
    print(f"and place it at:")
    print(f"  {path.resolve()}")
    print("Then re-run the application setup.")
    print("=" * 70)

if __name__ == "__main__":
    success = ensure_dataset()
    sys.exit(0 if success else 1)
