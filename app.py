import os
import sys
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ml.config import (
    MAX_UPLOAD_SIZE_BYTES, MAX_CSV_ROWS, SESSIONS_DIR, TEMP_DIR, STATIC_DIR, SAMPLES_DIR
)
from ml.predictor import (
    ModelManager, process_csv_into_session, query_session_predictions,
    export_session_predictions_csv, validate_session_id, clean_expired_sessions,
    get_session_summary
)

async def periodic_session_cleanup():
    """Background task to clean expired SQLite sessions periodically."""
    while True:
        try:
            await asyncio.sleep(600)  # Every 10 minutes
            clean_expired_sessions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[SessionCleaner] Unexpected error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify directories and load active model bundle
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    clean_expired_sessions()
    
    manager = ModelManager()
    bundle = manager.get_bundle()
    if bundle:
        print(f"[Server Startup] Active model bundle ready: {bundle.bundle_id}")
    else:
        print("[Server Startup WARNING] Active model bundle could not be loaded!")

    cleanup_task = asyncio.create_task(periodic_session_cleanup())
    yield
    cleanup_task.cancel()

app = FastAPI(
    title="Network Anomaly Detection System",
    description="Production-grade Machine Learning API for Network Anomaly Detection using UNSW-NB15",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for development and local network flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if it exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the main single-page application dashboard."""
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=404,
        content={"detail": "index.html not found in project root."}
    )

@app.get("/api/health")
async def get_health():
    """
    Health check endpoint reporting server status, active model bundle, and timestamp.
    """
    manager = ModelManager()
    bundle = manager.get_bundle()

    if bundle is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "model_active": False,
                "error": "No valid active model bundle is loaded.",
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
        )

    return {
        "status": "healthy",
        "model_active": True,
        "active_bundle_id": bundle.bundle_id,
        "model_type": bundle.manifest.get("model_type", "RandomForestClassifier"),
        "accuracy": bundle.metrics.get("accuracy"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/model-performance")
async def get_model_performance():
    """
    Returns authentic held-out evaluation metrics from the active model bundle.
    """
    manager = ModelManager()
    bundle = manager.get_bundle()

    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not active or unavailable."
        )

    return {
        "bundle_id": bundle.bundle_id,
        "metrics": bundle.metrics,
        "hyperparameters": bundle.manifest.get("hyperparameters", {}),
        "created_at_utc": bundle.manifest.get("created_at_utc")
    }

@app.get("/api/dataset-analysis")
async def get_dataset_analysis():
    """
    Returns verified UNSW-NB15 dataset distribution analytics from the active bundle.
    """
    manager = ModelManager()
    bundle = manager.get_bundle()

    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle not active or unavailable."
        )

    return bundle.dataset_analysis

@app.post("/api/predict")
async def predict_csv(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Accepts CSV upload via multipart form-data or raw request stream.
    Enforces 10 MiB size limit, verifies headers, performs chunked prediction,
    and stores results in an isolated SQLite session database.
    """
    manager = ModelManager()
    bundle = manager.get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model bundle is not loaded. Cannot process predictions."
        )

    session_id = uuid.uuid4().hex
    tmp_path = TEMP_DIR / f"upload_{session_id}.tmp.csv"

    try:
        # Stream file to disk in 64 KB chunks while strictly enforcing MAX_UPLOAD_SIZE_BYTES
        total_bytes = 0
        chunk_size = 64 * 1024  # 64 KB

        with open(tmp_path, "wb") as f_out:
            if file is not None:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MiB."
                        )
                    f_out.write(chunk)
            else:
                # Raw body stream fallback
                async for chunk in request.stream():
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MiB."
                        )
                    f_out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is empty (0 bytes)."
            )

        # Process the temporary CSV into SQLite session
        try:
            summary = process_csv_into_session(tmp_path, session_id, bundle)
            return summary
        except ValueError as val_err:
            msg = str(val_err)
            if "exceeds maximum limit" in msg or "row count" in msg:
                raise HTTPException(
                    status_code=413,
                    detail=msg
                )
            raise HTTPException(
                status_code=422,
                detail=msg
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process CSV records: {str(e)}"
            )

    finally:
        # Always remove staging temporary CSV file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

@app.get("/api/predictions/{session_id}")
async def get_session_predictions(
    session_id: str,
    page: int = Query(1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(50, ge=1, le=500, description="Records per page (1-500)"),
    search: str = Query("", description="Search term for IP, protocol, or prediction"),
    proto: str = Query("ALL", description="Protocol filter: ALL, TCP, UDP, etc."),
    status_filter: str = Query("ALL", alias="status", description="Status filter: ALL, Attack, Normal")
):
    """
    Retrieves paginated prediction rows from the disk-backed SQLite session database.
    """
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    results = query_session_predictions(
        session_id=session_id,
        page=page,
        page_size=page_size,
        search=search,
        proto=proto,
        status=status_filter
    )

    if results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    return results

@app.get("/api/predictions/{session_id}/summary")
async def get_session_summary_api(session_id: str):
    """
    Returns aggregated prediction summary and authentic evaluation metrics
    for the specified uploaded CSV session.
    """
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    summary = get_session_summary(session_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    return summary

@app.get("/api/samples/{sample_type}")
async def get_sample_csv(sample_type: str):
    """
    Serves verified benchmark sample CSVs (normal, attack, mixed, unlabeled)
    containing genuine UNSW-NB15 flow telemetry with the canonical 42 features.
    """
    valid_samples = {
        "normal": "normal_traffic_sample.csv",
        "attack": "attack_traffic_sample.csv",
        "mixed": "mixed_traffic_sample.csv",
        "unlabeled": "unlabeled_traffic_sample.csv"
    }
    sample_key = sample_type.lower().strip()
    if sample_key not in valid_samples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample type '{sample_type}' not found. Valid types: normal, attack, mixed, unlabeled."
        )

    file_path = SAMPLES_DIR / valid_samples[sample_key]
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample file '{valid_samples[sample_key]}' not found on server."
        )

    return FileResponse(
        path=str(file_path),
        media_type="text/csv",
        filename=valid_samples[sample_key],
        headers={"Content-Disposition": f'attachment; filename="{valid_samples[sample_key]}"'}
    )

@app.get("/api/predictions/{session_id}/export")
async def export_session_csv(
    session_id: str,
    filter_applied: bool = Query(False, description="Whether to apply active filters to export"),
    search: str = Query("", description="Search term filter"),
    proto: str = Query("ALL", description="Protocol filter"),
    status_filter: str = Query("ALL", alias="status", description="Status filter")
):
    """
    Exports prediction records as a sanitized CSV file.
    If filter_applied is False, exports 100% of session records regardless of pagination.
    If filter_applied is True, exports all matching records across the entire session.
    All non-numeric cells are protected against spreadsheet formula injection.
    """
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    csv_content = export_session_predictions_csv(
        session_id=session_id,
        filter_applied=filter_applied,
        search=search,
        proto=proto,
        status=status_filter
    )

    if csv_content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session expired or not found."
        )

    filename = f"prediction_results_{session_id[:8]}.csv" if not filter_applied else f"prediction_filtered_{session_id[:8]}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
