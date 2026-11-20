"""
screening/services.py - FastAPI Inference Service Client
Provides robust, timeout-controlled HTTP communication between Django and
the standalone FastAPI ML screening service using httpx.
"""

import logging
from typing import Dict, Any, Union, BinaryIO
from pathlib import Path
import httpx
from django.conf import settings

logger = logging.getLogger("screening.services")


class FastAPIServiceError(Exception):
    """Base exception for FastAPI inference service failures."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class FastAPIConnectionError(FastAPIServiceError):
    """Raised when the FastAPI ML service is unreachable or offline."""
    def __init__(self, message: str = "The AI screening inference service is currently unavailable."):
        super().__init__(message, status_code=503)


class FastAPITimeoutError(FastAPIServiceError):
    """Raised when the ML inference request exceeds timeout limits."""
    def __init__(self, message: str = "The AI screening inference request timed out. Please retry."):
        super().__init__(message, status_code=504)


class FastAPISchemaValidationError(FastAPIServiceError):
    """Raised when the inference service returns an unexpected payload structure."""
    def __init__(self, message: str = "Inference service returned an invalid response structure."):
        super().__init__(message, status_code=502)


def send_xray_to_fastapi(
    image_file: Union[BinaryIO, bytes],
    filename: str = "xray.png",
    mime_type: str = "image/png",
    threshold: float = 0.30,
) -> Dict[str, Any]:
    """
    Transmits an X-ray image to the FastAPI /predict endpoint.
    Performs network timeout management, error classification, and schema validation.
    """
    base_url = settings.KNEEWISE_FASTAPI_URL.rstrip("/")
    predict_url = f"{base_url}/predict"
    timeout_sec = settings.KNEEWISE_FASTAPI_TIMEOUT

    # Prepare multipart files payload
    if isinstance(image_file, bytes):
        file_bytes = image_file
    elif hasattr(image_file, "read"):
        file_bytes = image_file.read()
    else:
        raise ValueError("image_file must be bytes or a file-like object with a read() method.")

    files = {
        "file": (filename, file_bytes, mime_type),
    }
    params = {
        "threshold": threshold,
    }

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            response = client.post(predict_url, files=files, params=params)

    except httpx.TimeoutException as exc:
        logger.error(f"[FastAPI Communication] Timeout after {timeout_sec}s calling {predict_url}: {exc}")
        raise FastAPITimeoutError() from exc

    except (httpx.ConnectError, httpx.NetworkError) as exc:
        logger.error(f"[FastAPI Communication] Connection failed to {predict_url}: {exc}")
        raise FastAPIConnectionError() from exc

    except Exception as exc:
        logger.error(f"[FastAPI Communication] Unexpected network error: {exc}")
        raise FastAPIServiceError(f"Network error communicating with AI screening service: {str(exc)}") from exc

    # Handle HTTP status codes
    if response.status_code == 400:
        logger.warning(f"[FastAPI Communication] 400 Bad Request: {response.text}")
        raise FastAPIServiceError("The uploaded radiograph could not be processed by the AI model.", status_code=400)

    if response.status_code >= 500:
        logger.error(f"[FastAPI Communication] Server Error {response.status_code}: {response.text}")
        raise FastAPIServiceError("The AI screening inference service encountered an internal error.", status_code=502)

    if response.status_code != 200:
        logger.error(f"[FastAPI Communication] Unexpected Status {response.status_code}: {response.text}")
        raise FastAPIServiceError(f"Inference service returned unexpected status {response.status_code}.", status_code=502)

    # Validate response schema
    try:
        payload = response.json()
    except Exception as exc:
        logger.error(f"[FastAPI Communication] Response is not valid JSON: {response.text}")
        raise FastAPISchemaValidationError("Inference service did not return valid JSON.") from exc

    required_keys = {"prediction", "binary_screening", "model_metadata"}
    missing = required_keys - set(payload.keys())
    if missing:
        logger.error(f"[FastAPI Communication] Missing schema keys in response: {missing}")
        raise FastAPISchemaValidationError(f"Inference response missing required fields: {missing}")

    pred = payload["prediction"]
    if "kellgren_lawrence_grade" not in pred or "probabilities" not in pred:
        logger.error("[FastAPI Communication] Prediction section missing grade or probabilities.")
        raise FastAPISchemaValidationError("Prediction payload missing grade or probabilities.")

    bin_screen = payload["binary_screening"]
    if "binary_oa_probability" not in bin_screen or "is_oa_likely" not in bin_screen:
        logger.error("[FastAPI Communication] Binary screening section missing risk probability.")
        raise FastAPISchemaValidationError("Binary screening payload missing risk probability.")

    return payload
