"""
screening/tests.py - Comprehensive Unit & Integration Tests for Screening App
Tests all requirements:
1. Unauthenticated request rejection (HTTP 401)
2. Successful X-ray upload (HTTP 201)
3. Invalid file extension rejection (HTTP 400)
4. Empty file rejection (HTTP 400)
5. Corrupted image rejection (HTTP 400)
6. Patient ownership & clinician authorization (HTTP 403)
7. Successful FastAPI communication & Database persistence (HTTP 200)
8. FastAPI timeout handling (HTTP 504)
9. FastAPI unavailable handling (HTTP 503)
10. Invalid prediction response handling (HTTP 502)
11. Screening history retrieval & patient filtering
12. Unauthorized prediction detail access (HTTP 403)
"""

import io
import uuid
from decimal import Decimal
from unittest.mock import patch
from PIL import Image

from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework import status

from .models import Patient, UploadedXRay, ScreeningPrediction
from .services import (
    FastAPIConnectionError,
    FastAPITimeoutError,
    FastAPISchemaValidationError,
    FastAPIServiceError,
)

User = get_user_model()


def create_dummy_image_file(name="sample.png", ext="PNG", size=(50, 50), color=(128, 128, 128)):
    """Generates a valid in-memory PNG or JPEG image file."""
    file_obj = io.BytesIO()
    image = Image.new("RGB", size=size, color=color)
    image.save(file_obj, format=ext)
    file_obj.seek(0)
    mime = "image/png" if ext.upper() == "PNG" else "image/jpeg"
    return SimpleUploadedFile(name, file_obj.read(), content_type=mime)


class ScreeningIntegrationTests(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Clinician A (Primary authorized user)
        self.user_a = User.objects.create_user(
            username="dr_sharma",
            password="SecurePassword123!",
            email="sharma@hospital.org",
        )

        # Clinician B (Unauthorized third-party user)
        self.user_b = User.objects.create_user(
            username="dr_jones",
            password="SecurePassword123!",
            email="jones@hospital.org",
        )

        # Patient assigned to Clinician A
        self.patient_a = Patient.objects.create(
            patient_id="MRN-100234",
            first_name="Jane",
            last_name="Doe",
            gender="F",
            assigned_clinician=self.user_a,
        )

        # Patient assigned to Clinician B
        self.patient_b = Patient.objects.create(
            patient_id="MRN-900567",
            first_name="John",
            last_name="Smith",
            gender="M",
            assigned_clinician=self.user_b,
        )

        # Mock valid FastAPI prediction response
        self.mock_fastapi_payload = {
            "prediction": {
                "kellgren_lawrence_grade": 2,
                "grade_description": "Grade 2: Mild radiographic OA (Definite osteophytes, possible joint space narrowing)",
                "confidence": 0.6918,
                "probabilities": {
                    "KL_0": 0.0228,
                    "KL_1": 0.1853,
                    "KL_2": 0.6918,
                    "KL_3": 0.0894,
                    "KL_4": 0.0107,
                },
            },
            "binary_screening": {
                "screening_category": "Definite Radiographic OA Likely (Refer for Clinical Evaluation)",
                "is_oa_likely": True,
                "binary_oa_probability": 0.7919,
                "unlikely_oa_probability": 0.2081,
                "screening_threshold_used": 0.30,
                "threshold_notice": "Provisional research threshold",
            },
            "model_metadata": {
                "model_name": "KneeWise-EfficientNet-B0-Champion",
                "experiment_id": "exp_d_efficientnet_b0",
                "checkpoint_sha256": "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b",
                "device": "cuda",
                "inference_time_ms": 74.5,
            },
            "disclaimer": "AI screening-support tool. Not an autonomous diagnostic system.",
        }

    # 1. Authentication Security
    def test_unauthenticated_request_rejection(self):
        """Unauthenticated requests must be rejected with HTTP 401."""
        url = reverse("screening:xray-upload")
        response = self.client.post(url, {})
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        url_predict = reverse("screening:screening-predict")
        resp_pred = self.client.post(url_predict, {})
        self.assertEqual(resp_pred.status_code, status.HTTP_401_UNAUTHORIZED)

        url_hist = reverse("screening:screening-history")
        resp_hist = self.client.get(url_hist)
        self.assertEqual(resp_hist.status_code, status.HTTP_401_UNAUTHORIZED)

    # 2. Successful Upload
    def test_successful_xray_upload(self):
        """Valid radiograph upload must succeed and persist with a secure UUID filename."""
        self.client.force_authenticate(user=self.user_a)
        img = create_dummy_image_file("patient_left_knee.png", ext="PNG")

        url = reverse("screening:xray-upload")
        data = {
            "file": img,
            "patient_id": self.patient_a.patient_id,
            "knee_laterality": "LEFT",
        }
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(response.data["success"])

        # Check DB persistence
        xray_id = response.data["uploaded_xray"]["id"]
        xray = UploadedXRay.objects.get(id=xray_id)
        self.assertEqual(xray.patient, self.patient_a)
        self.assertEqual(xray.uploaded_by, self.user_a)
        self.assertEqual(xray.knee_laterality, "LEFT")
        # Ensure random UUID filename path
        self.assertIn(f"xrays/patient_{self.patient_a.id}/", xray.image.name)

    # 3. Invalid File Type Rejection
    def test_invalid_file_extension_rejection(self):
        """Non-image files (e.g. .pdf, .txt) must be rejected with HTTP 400."""
        self.client.force_authenticate(user=self.user_a)
        bad_file = SimpleUploadedFile("report.pdf", b"%PDF-1.4 dummy pdf content", content_type="application/pdf")

        url = reverse("screening:xray-upload")
        data = {"file": bad_file, "patient_id": self.patient_a.patient_id}
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Unsupported image format", response.data["detail"])

    # 4. Empty File Rejection
    def test_empty_file_rejection(self):
        """Zero-byte files must be rejected with HTTP 400."""
        self.client.force_authenticate(user=self.user_a)
        empty_file = SimpleUploadedFile("empty.png", b"", content_type="image/png")

        url = reverse("screening:xray-upload")
        data = {"file": empty_file, "patient_id": self.patient_a.patient_id}
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("empty", response.data["detail"])

    # 5. Corrupted Image Rejection
    def test_corrupted_image_rejection(self):
        """Files with image extensions but corrupted/invalid binary data must be rejected."""
        self.client.force_authenticate(user=self.user_a)
        corrupted = SimpleUploadedFile("knee.png", b"CORRUPTED_NON_IMAGE_HEADER_12345", content_type="image/png")

        url = reverse("screening:xray-upload")
        data = {"file": corrupted, "patient_id": self.patient_a.patient_id}
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("corrupted or not a valid radiograph", response.data["detail"])

    # 6. Patient Ownership & Permission Security
    def test_patient_ownership_validation(self):
        """Clinician A cannot upload or run screening on Clinician B's patient."""
        self.client.force_authenticate(user=self.user_a)
        img = create_dummy_image_file("knee.png")

        url = reverse("screening:xray-upload")
        data = {"file": img, "patient_id": self.patient_b.patient_id}
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("permission", response.data["detail"])

    # 7. Successful FastAPI Communication & DB Persistence
    @patch("screening.views.send_xray_to_fastapi")
    def test_successful_fastapi_communication_and_db_persistence(self, mock_send):
        """Successful inference call must persist prediction and return formatted response."""
        mock_send.return_value = self.mock_fastapi_payload
        self.client.force_authenticate(user=self.user_a)

        # Upload an X-ray first
        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=len(img.read()),
            mime_type="image/png",
            knee_laterality="RIGHT",
        )

        url = reverse("screening:screening-predict")
        data = {
            "uploaded_xray_id": str(xray.id),
            "threshold": 0.30,
        }
        response = self.client.post(url, data, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["success"])

        pred_resp = response.data["prediction"]
        self.assertEqual(pred_resp["predicted_kl_grade"], 2)
        self.assertEqual(pred_resp["oa_probability"], 0.7919)
        self.assertTrue(pred_resp["is_oa_likely"])
        self.assertIn("AI-generated screening support only", pred_resp["disclaimer"])

        # Check Database persistence
        record = ScreeningPrediction.objects.get(id=pred_resp["id"])
        self.assertEqual(record.patient, self.patient_a)
        self.assertEqual(record.clinician, self.user_a)
        self.assertEqual(record.predicted_kl_grade, 2)
        self.assertEqual(record.oa_probability, Decimal("0.7919"))
        self.assertEqual(record.screening_threshold, Decimal("0.30"))
        self.assertEqual(record.model_name, "KneeWise-EfficientNet-B0-Champion")
        self.assertEqual(record.checkpoint_sha256, "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b")

    # 8. FastAPI Timeout Handling
    @patch("screening.views.send_xray_to_fastapi")
    def test_fastapi_timeout_handling(self, mock_send):
        """FastAPI timeouts must be caught and returned as HTTP 504 Gateway Timeout."""
        mock_send.side_effect = FastAPITimeoutError()
        self.client.force_authenticate(user=self.user_a)

        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=100,
            mime_type="image/png",
        )

        url = reverse("screening:screening-predict")
        response = self.client.post(url, {"uploaded_xray_id": str(xray.id)}, format="json")
        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)
        self.assertEqual(response.data["code"], "FASTAPI_REQUEST_TIMEOUT")

    # 9. FastAPI Unavailable Handling
    @patch("screening.views.send_xray_to_fastapi")
    def test_fastapi_unavailable_handling(self, mock_send):
        """FastAPI connection failures must return HTTP 503 Service Unavailable."""
        mock_send.side_effect = FastAPIConnectionError()
        self.client.force_authenticate(user=self.user_a)

        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=100,
            mime_type="image/png",
        )

        url = reverse("screening:screening-predict")
        response = self.client.post(url, {"uploaded_xray_id": str(xray.id)}, format="json")
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertEqual(response.data["code"], "FASTAPI_SERVICE_UNAVAILABLE")

    # 10. Invalid Prediction Schema Handling
    @patch("screening.views.send_xray_to_fastapi")
    def test_invalid_prediction_response_handling(self, mock_send):
        """Malformed FastAPI payloads must return HTTP 502 Bad Gateway."""
        mock_send.side_effect = FastAPISchemaValidationError("Corrupted response schema")
        self.client.force_authenticate(user=self.user_a)

        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=100,
            mime_type="image/png",
        )

        url = reverse("screening:screening-predict")
        response = self.client.post(url, {"uploaded_xray_id": str(xray.id)}, format="json")
        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)

    # 11. Screening History & Patient Filtering
    def test_screening_history_retrieval(self):
        """Clinician A can retrieve history for their patients, optionally filtered by patient."""
        self.client.force_authenticate(user=self.user_a)

        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=100,
        )

        # Create 2 predictions
        pred1 = ScreeningPrediction.objects.create(
            patient=self.patient_a,
            uploaded_xray=xray,
            clinician=self.user_a,
            predicted_kl_grade=0,
            kl0_probability=Decimal("0.8500"),
            kl1_probability=Decimal("0.1000"),
            kl2_probability=Decimal("0.0300"),
            kl3_probability=Decimal("0.0150"),
            kl4_probability=Decimal("0.0050"),
            oa_probability=Decimal("0.0500"),
            screening_threshold=Decimal("0.30"),
            is_oa_likely=False,
            screening_category="OA Radiographically Unlikely",
            model_name="KneeWise-EfficientNet-B0-Champion",
            experiment_id="exp_d_efficientnet_b0",
            inference_time=72.0,
        )
        pred2 = ScreeningPrediction.objects.create(
            patient=self.patient_a,
            uploaded_xray=xray,
            clinician=self.user_a,
            predicted_kl_grade=3,
            kl0_probability=Decimal("0.0100"),
            kl1_probability=Decimal("0.0400"),
            kl2_probability=Decimal("0.1500"),
            kl3_probability=Decimal("0.7000"),
            kl4_probability=Decimal("0.1000"),
            oa_probability=Decimal("0.9500"),
            screening_threshold=Decimal("0.30"),
            is_oa_likely=True,
            screening_category="Definite Radiographic OA",
            model_name="KneeWise-EfficientNet-B0-Champion",
            experiment_id="exp_d_efficientnet_b0",
            inference_time=73.0,
        )

        url = reverse("screening:screening-history")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)

        # Filter by patient MRN
        url_filtered = f"{url}?patient_id={self.patient_a.patient_id}"
        resp_filt = self.client.get(url_filtered)
        self.assertEqual(resp_filt.status_code, status.HTTP_200_OK)
        self.assertEqual(resp_filt.data["count"], 2)

    # 12. Unauthorized Prediction Detail Access
    def test_unauthorized_prediction_detail_access(self):
        """Clinician B cannot view Clinician A's patient screening record."""
        img = create_dummy_image_file("knee.png")
        xray = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=img,
            original_filename="knee.png",
            file_size_bytes=100,
        )
        pred = ScreeningPrediction.objects.create(
            patient=self.patient_a,
            uploaded_xray=xray,
            clinician=self.user_a,
            predicted_kl_grade=1,
            kl0_probability=Decimal("0.4000"),
            kl1_probability=Decimal("0.4500"),
            kl2_probability=Decimal("0.1000"),
            kl3_probability=Decimal("0.0300"),
            kl4_probability=Decimal("0.0200"),
            oa_probability=Decimal("0.1500"),
            screening_threshold=Decimal("0.30"),
            is_oa_likely=False,
            screening_category="OA Unlikely",
            model_name="KneeWise-EfficientNet-B0-Champion",
            experiment_id="exp_d_efficientnet_b0",
            inference_time=70.0,
        )

        # Authenticate as Clinician B (Unauthorized)
        self.client.force_authenticate(user=self.user_b)
        url = reverse("screening:screening-detail", kwargs={"pk": pred.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


class ProtectedXRayMediaSecurityTests(TestCase):
    """
    SEC-01 Security Regression Tests:
    Verifies that direct public media access is disabled and radiographs
    are strictly accessible through the authenticated, permission-checked endpoint:
    GET /api/screening/xrays/<uuid>/
    """
    def setUp(self):
        self.client = APIClient()

        # Clinician A (Primary authorized user)
        self.user_a = User.objects.create_user(
            username="dr_sharma_sec",
            password="SecurePassword123!",
            email="sharma_sec@hospital.org",
        )

        # Clinician B (Unauthorized user)
        self.user_b = User.objects.create_user(
            username="dr_jones_sec",
            password="SecurePassword123!",
            email="jones_sec@hospital.org",
        )

        # Admin user
        self.admin_user = User.objects.create_superuser(
            username="admin_sec",
            password="AdminPassword123!",
            email="admin_sec@hospital.org",
        )

        # Patient assigned to Clinician A
        self.patient_a = Patient.objects.create(
            patient_id="MRN-SEC-101",
            first_name="Alice",
            last_name="Walker",
            gender="F",
            assigned_clinician=self.user_a,
        )

        # Patient assigned to Clinician B
        self.patient_b = Patient.objects.create(
            patient_id="MRN-SEC-202",
            first_name="Bob",
            last_name="Miller",
            gender="M",
            assigned_clinician=self.user_b,
        )

        # Create dummy image on disk for Patient A
        self.dummy_img_a = create_dummy_image_file("patient_a_knee.png", ext="PNG", color=(100, 150, 200))
        self.xray_a = UploadedXRay.objects.create(
            patient=self.patient_a,
            uploaded_by=self.user_a,
            image=self.dummy_img_a,
            original_filename="patient_a_knee.png",
            file_size_bytes=1024,
            mime_type="image/png",
            knee_laterality="LEFT",
        )

        # Create dummy image on disk for Patient B
        self.dummy_img_b = create_dummy_image_file("patient_b_knee.png", ext="PNG", color=(200, 100, 50))
        self.xray_b = UploadedXRay.objects.create(
            patient=self.patient_b,
            uploaded_by=self.user_b,
            image=self.dummy_img_b,
            original_filename="patient_b_knee.png",
            file_size_bytes=1024,
            mime_type="image/png",
            knee_laterality="RIGHT",
        )

    def tearDown(self):
        # Clean up files created in test media storage safely on Windows
        for xray in [self.xray_a, self.xray_b]:
            if xray.image:
                try:
                    xray.image.close()
                except Exception:
                    pass
                try:
                    if xray.image.storage.exists(xray.image.name):
                        xray.image.storage.delete(xray.image.name)
                except (PermissionError, OSError):
                    pass

    # 1. Unauthenticated access returns 401
    def test_unauthenticated_xray_access_returns_401(self):
        url = reverse("screening:xray-media", kwargs={"pk": self.xray_a.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        response.close()

    # 2. Authorized clinician receives correct image with HTTP 200
    def test_authorized_clinician_access_returns_200(self):
        self.client.force_authenticate(user=self.user_a)
        url = reverse("screening:xray-media", kwargs={"pk": self.xray_a.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("Content-Type"), "image/png")
        self.assertIn("inline", response.headers.get("Content-Disposition", ""))
        self.assertIn(f"{self.xray_a.id}.png", response.headers.get("Content-Disposition", ""))
        # Verify content streamed
        content = b"".join(response.streaming_content)
        self.assertGreater(len(content), 0)
        response.close()

    # 3 & 4. Clinician A cannot access Clinician B's patient X-ray, returns 404 to avoid enumeration
    def test_unauthorized_clinician_access_returns_404(self):
        self.client.force_authenticate(user=self.user_a)
        # Attempt to access X-ray B belonging to Clinician B's patient
        url = reverse("screening:xray-media", kwargs={"pk": self.xray_b.id})
        response = self.client.get(url)
        # Must return 404 rather than 403 to prevent enumeration
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.data.get("detail"), "Radiograph not found.")
        response.close()

    # 5. Invalid UUID returns 404
    def test_nonexistent_or_invalid_uuid_returns_404(self):
        self.client.force_authenticate(user=self.user_a)
        non_existent_uuid = uuid.uuid4()
        url = reverse("screening:xray-media", kwargs={"pk": non_existent_uuid})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.data.get("detail"), "Radiograph not found.")
        response.close()

    # 6. Missing physical file returns safe 404
    def test_missing_physical_file_returns_safe_404(self):
        self.client.force_authenticate(user=self.user_a)
        # Delete file from storage while keeping database record
        try:
            self.xray_a.image.close()
            if self.xray_a.image.storage.exists(self.xray_a.image.name):
                self.xray_a.image.storage.delete(self.xray_a.image.name)
        except Exception:
            pass

        url = reverse("screening:xray-media", kwargs={"pk": self.xray_a.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(response.data.get("detail"), "Radiograph file not found on storage.")
        response.close()

    # 7. Arbitrary filename / path traversal is impossible
    def test_arbitrary_filename_or_path_traversal_is_impossible(self):
        self.client.force_authenticate(user=self.user_a)
        # URL pattern strictly enforces <uuid:pk>; non-UUID strings are rejected by Django routing
        url = "/api/screening/xrays/..%2F..%2Fetc%2Fpasswd/"
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        response.close()

        # Query param manipulation has zero effect
        url_with_query = f"{reverse('screening:xray-media', kwargs={'pk': self.xray_a.id})}?path=../../etc/passwd"
        response_query = self.client.get(url_with_query)
        self.assertEqual(response_query.status_code, status.HTTP_200_OK)
        # Content remains the legitimate radiograph
        self.assertEqual(response_query.headers.get("Content-Type"), "image/png")
        response_query.close()

    # 8. Direct public media URL access is disabled
    def test_direct_public_media_url_access_is_disabled(self):
        # Unauthenticated request to the direct media path must return 404
        direct_media_url = f"/media/{self.xray_a.image.name}"
        response = self.client.get(direct_media_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        response.close()

    # 9. Upload serializer returns protected media URL
    def test_upload_response_contains_protected_media_url(self):
        self.client.force_authenticate(user=self.user_a)
        img = create_dummy_image_file("new_knee.png")
        url = reverse("screening:xray-upload")
        data = {
            "file": img,
            "patient_id": self.patient_a.patient_id,
            "knee_laterality": "RIGHT",
        }
        response = self.client.post(url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        xray_id = response.data["uploaded_xray"]["id"]
        image_url = response.data["uploaded_xray"]["image_url"]

        # Ensure image_url points to protected endpoint, not direct /media/
        expected_endpoint = f"/api/screening/xrays/{xray_id}/"
        self.assertIn(expected_endpoint, image_url)
        self.assertNotIn("/media/xrays/", image_url)
        response.close()

    # 10. Admin access follows explicit authorization policy (superuser receives 200)
    def test_admin_access_allowed(self):
        self.client.force_authenticate(user=self.admin_user)
        # Superuser can access Patient A's X-ray
        url = reverse("screening:xray-media", kwargs={"pk": self.xray_a.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response.close()

        # Superuser can also access Patient B's X-ray
        url_b = reverse("screening:xray-media", kwargs={"pk": self.xray_b.id})
        response_b = self.client.get(url_b)
        self.assertEqual(response_b.status_code, status.HTTP_200_OK)
        response_b.close()


