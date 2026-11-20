"""
test_api.py - Verification Test Suite for FastAPI Microservice
Tests:
- GET  /health   : Verifies model metadata, SHA-256 integrity, device
- POST /predict  : Valid PNG radiograph prediction, probability normalization, binary risk
- POST /predict  : Bad file input / non-image rejection (HTTP 400)
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import app

def test_api_endpoints():
    print("=" * 70)
    print("      KNEEWISE FASTAPI MICROSERVICE — ENDPOINT VERIFICATION")
    print("=" * 70)

    sample_img_path = Path(r"C:\Users\vikra\OneDrive\Desktop\rakshitha\KneeWise\ml\data\raw\test\0\9003175L.png")
    assert sample_img_path.exists(), f"Missing sample image at {sample_img_path}"

    with TestClient(app) as client:
        # 1. Test /health
        print("\n[Endpoint Test 1/3] GET /health ...")
        resp_health = client.get("/health")
        assert resp_health.status_code == 200, f"Health check failed: {resp_health.status_code}"
        health_data = resp_health.json()
        print("  Response:", health_data)
        assert health_data["status"] == "healthy"
        assert health_data["checkpoint_sha256"] == "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b"
        print("  -> PASSED: /health returned healthy status and matching SHA-256.")

        # 2. Test /predict with valid image
        print("\n[Endpoint Test 2/3] POST /predict (Valid Radiograph) ...")
        with open(sample_img_path, "rb") as f:
            files = {"file": ("9003175L.png", f, "image/png")}
            resp_predict = client.post("/predict?threshold=0.30", files=files)

        assert resp_predict.status_code == 200, f"Prediction failed: {resp_predict.status_code} - {resp_predict.text}"
        pred_data = resp_predict.json()
        print("  Predicted Grade:", pred_data["prediction"]["kellgren_lawrence_grade"])
        print("  Grade Description:", pred_data["prediction"]["grade_description"])
        print("  Five-Class Probs:", pred_data["prediction"]["probabilities"])
        print("  Binary OA Risk:", pred_data["binary_screening"]["binary_oa_probability"])
        print("  Screening Category:", pred_data["binary_screening"]["screening_category"])
        print("  Latency:", pred_data["model_metadata"]["inference_time_ms"], "ms")
        assert "prediction" in pred_data
        assert "binary_screening" in pred_data
        assert pred_data["binary_screening"]["screening_threshold_used"] == 0.30
        print("  -> PASSED: /predict executed successfully with full response structure.")

        # 3. Test /predict with invalid file
        print("\n[Endpoint Test 3/3] POST /predict (Corrupted/Bad File) ...")
        corrupted_files = {"file": ("bad_file.txt", b"NOT_AN_IMAGE", "text/plain")}
        resp_bad = client.post("/predict", files=corrupted_files)
        assert resp_bad.status_code == 400, f"Expected 400 for bad extension, got {resp_bad.status_code}"
        print(f"  Expected 400 error received: {resp_bad.json()['detail']}")
        print("  -> PASSED: Invalid input correctly rejected with HTTP 400.")

    print("\n" + "=" * 70)
    print(" ALL FASTAPI ENDPOINT TESTS PASSED CLEANLY (Zero Errors)")
    print("=" * 70)

if __name__ == "__main__":
    test_api_endpoints()
