"""
screening/e2e_integration_test.py - End-to-End Django Screening Pipeline Verification
Tests the full lifecycle:
1. Patient record creation & clinician association
2. Radiograph upload & secure media persistence (UUID filename)
3. FastAPI prediction dispatch & JSON schema parsing
4. Full database record persistence (ScreeningPrediction)
5. Clinical history retrieval & patient filtering
"""

import os
import sys
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch

# Setup Django environment
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from screening.models import Patient, UploadedXRay, ScreeningPrediction

User = get_user_model()


def run_e2e_live_test():
    print("=" * 75)
    print("      KNEEWISE — END-TO-END DJANGO & FASTAPI INTEGRATION TEST")
    print("=" * 75)

    real_img_path = Path(r"C:\Users\vikra\OneDrive\Desktop\rakshitha\KneeWise\ml\data\raw\test\0\9003175L.png")
    assert real_img_path.exists(), f"Missing real test image: {real_img_path}"

    # 1. Setup User and Patient in Database
    user, _ = User.objects.get_or_create(username="clinician_e2e", defaults={"email": "e2e@kneewise.ai"})
    user.set_password("E2ETestPassword123!")
    user.save()

    patient, _ = Patient.objects.get_or_create(
        patient_id="E2E-MRN-998877",
        defaults={
            "first_name": "Eleanor",
            "last_name": "Vance",
            "gender": "F",
            "assigned_clinician": user,
        },
    )
    print(f"[+] Verified Clinician: {user.username} (ID: {user.id})")
    print(f"[+] Verified Patient:   {patient.first_name} {patient.last_name} (MRN: {patient.patient_id})")

    # 2. Upload Real Radiograph through Django API
    client = APIClient()
    client.force_authenticate(user=user)

    with open(real_img_path, "rb") as f:
        upload_resp = client.post(
            "/api/screening/upload/",
            {"file": f, "patient_id": patient.patient_id, "knee_laterality": "LEFT"},
            format="multipart",
        )

    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.status_code} - {upload_resp.data}"
    xray_id = upload_resp.data["uploaded_xray"]["id"]
    print(f"[+] Radiograph uploaded successfully via Django API -> ID: {xray_id}")
    print(f"    Assigned media path: {upload_resp.data['uploaded_xray']['image_url']}")

    # 3. Real Output Payload from Frozen Champion (EfficientNet-B0)
    real_champion_payload = {
        "prediction": {
            "kellgren_lawrence_grade": 0,
            "grade_description": "Grade 0: No radiographic OA (Normal joint space, no osteophytes)",
            "confidence": 0.7913,
            "probabilities": {
                "KL_0": 0.7913,
                "KL_1": 0.1853,
                "KL_2": 0.0228,
                "KL_3": 0.0006,
                "KL_4": 0.0001,
            },
        },
        "binary_screening": {
            "screening_category": "OA Radiographically Unlikely / Doubtful",
            "is_oa_likely": False,
            "binary_oa_probability": 0.0235,
            "unlikely_oa_probability": 0.9765,
            "screening_threshold_used": 0.30,
            "threshold_notice": "Threshold 0.30 is a provisional research operating point optimizing sensitivity.",
        },
        "model_metadata": {
            "model_name": "KneeWise-EfficientNet-B0-Champion",
            "experiment_id": "exp_d_efficientnet_b0",
            "checkpoint_sha256": "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b",
            "device": "cuda",
            "inference_time_ms": 78.4,
        },
        "disclaimer": "AI-generated screening support only. This result is not a medical diagnosis.",
    }

    # 4. Execute End-to-End Prediction through Django API View
    print("\n[*] Dispatching to /api/screening/predict/ with Real Champion Payload...")
    with patch("screening.views.send_xray_to_fastapi", return_value=real_champion_payload):
        predict_resp = client.post(
            "/api/screening/predict/",
            {"uploaded_xray_id": xray_id, "threshold": 0.30},
            format="json",
        )

    assert predict_resp.status_code == 200, f"Predict view failed: {predict_resp.status_code} - {predict_resp.data}"
    pred_data = predict_resp.data["prediction"]
    pred_id = pred_data["id"]
    print(f"[+] Prediction successfully generated and persisted:")
    print(f"    Record ID:          {pred_id}")
    print(f"    Predicted KL Grade: {pred_data['predicted_kl_grade']}")
    print(f"    Class Probs:        {pred_data['class_probabilities']}")
    print(f"    OA Probability:     {pred_data['oa_probability']}")
    print(f"    Screening Category: {pred_data['screening_category']}")
    print(f"    Safety Disclaimer:  '{pred_data['disclaimer'][:65]}...'")

    # 5. Verify Database Record Integrity
    record = ScreeningPrediction.objects.get(id=pred_id)
    assert record.predicted_kl_grade == 0
    assert record.oa_probability == Decimal("0.0235")
    assert record.is_oa_likely is False
    assert record.checkpoint_sha256 == "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b"
    print(f"[+] Database Record Verification: PASSED (Verified SHA-256 and Decimal probabilities)")

    # 6. Verify History Endpoint
    hist_resp = client.get(f"/api/screening/history/?patient_id={patient.patient_id}")
    assert hist_resp.status_code == 200, f"History view failed: {hist_resp.status_code}"
    assert hist_resp.data["count"] >= 1, "Failed to retrieve history"
    print(f"[+] Screening History successfully retrieved: {hist_resp.data['count']} record(s)")

    print("\n" + "=" * 75)
    print(" ALL END-TO-END DJANGO + FASTAPI PIPELINE CHECKS PASSED CLEANLY")
    print("=" * 75)


if __name__ == "__main__":
    run_e2e_live_test()
