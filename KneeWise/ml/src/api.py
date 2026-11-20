"""
api.py - KneeWise High-Performance Standalone FastAPI Microservice
Provides REST endpoints for knee osteoarthritis grading and binary screening:
- GET  /health   : System diagnostics, checkpoint SHA-256 integrity, device status
- POST /predict  : Single knee radiograph classification & binary screening risk
"""

import os
import sys
import io
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from inference import KneeWiseInferenceEngine, PROVISIONAL_SCREENING_THRESHOLD

# Allowed image MIME types and extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Initializes the KneeWise inference engine exactly once on application startup.
    Ensures zero retraining, zero repeated loading, and verified SHA-256 checkpoint integrity.
    """
    print("[*] Starting KneeWise Medical AI API...")
    try:
        app.state.engine = KneeWiseInferenceEngine()
        print(f"[+] Model loaded successfully on device: {app.state.engine.device}")
        print(f"[+] Verified Checkpoint SHA-256: {app.state.engine.sha256}")
    except Exception as e:
        print(f"[!] Failed to initialize KneeWise inference engine: {e}")
        raise RuntimeError(f"Engine initialization failure: {e}")

    yield

    # Clean shutdown
    print("[*] Shutting down KneeWise API...")
    if hasattr(app.state, "engine"):
        del app.state.engine


# Instantiate FastAPI application
app = FastAPI(
    title="KneeWise Osteoarthritis AI Screening Microservice",
    description=(
        "Production inference service for Kellgren-Lawrence (KL) knee osteoarthritis grading "
        "and derived binary screening risk assessment using the frozen EfficientNet-B0 champion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    summary="Service Health & Diagnostic Status",
    response_description="Engine status, hardware allocation, and checkpoint integrity.",
)
async def health_check() -> Dict[str, Any]:
    """Returns real-time status of the model service, active device, and checkpoint verification."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized.",
        )

    engine: KneeWiseInferenceEngine = app.state.engine
    return {
        "status": "healthy",
        "service": "KneeWise-Screening-API",
        "model_name": "KneeWise-EfficientNet-B0-Champion",
        "experiment_id": "exp_d_efficientnet_b0",
        "checkpoint_sha256": engine.sha256,
        "device": str(engine.device),
        "total_parameters": 4013953,
        "input_resolution": "224x224x3 RGB",
        "default_screening_threshold": PROVISIONAL_SCREENING_THRESHOLD,
        "disclaimer": "AI screening-support service. Not an autonomous diagnostic system.",
    }


@app.post(
    "/predict",
    summary="Radiograph Severity Grading & Screening Inference",
    response_description="5-class KL probability distribution and derived binary screening classification.",
)
async def predict_radiograph(
    file: UploadFile = File(..., description="Knee radiograph image file (JPEG or PNG format)"),
    threshold: float = Query(
        default=PROVISIONAL_SCREENING_THRESHOLD,
        ge=0.05,
        le=0.95,
        description="Binary screening operating threshold for P(Definite OA) = P(KL2)+P(KL3)+P(KL4)",
    ),
) -> Dict[str, Any]:
    """
    Accepts an uploaded knee X-ray (PNG/JPEG) and returns:
    1. Five-class Kellgren-Lawrence severity probabilities (KL 0 to KL 4).
    2. Derived binary radiographic OA probability and referral classification.
    3. Latency benchmarks and cryptographic metadata.
    """
    if not hasattr(app.state, "engine") or app.state.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized.",
        )

    # 1. Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed image formats: {list(ALLOWED_EXTENSIONS)}",
        )

    # 2. Read file content safely
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    # 3. Execute inference
    try:
        engine: KneeWiseInferenceEngine = app.state.engine
        result = engine.predict(contents, screening_threshold=threshold)
        result["filename"] = file.filename
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {str(ve)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    print("[*] Launching KneeWise API server locally on http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
