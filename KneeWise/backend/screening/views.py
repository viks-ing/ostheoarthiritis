"""
screening/views.py - Django REST Framework API Views
Endpoints:
- POST /api/screening/upload/   : Validates and persists knee radiograph
- POST /api/screening/predict/  : Dispatches X-ray to FastAPI and saves AI screening result
- GET  /api/screening/history/  : Retrieves authenticated user's screening history
- GET  /api/screening/<id>/     : Retrieves single prediction details
- GET/POST /api/screening/patients/ : Lists and registers patients
"""

import logging
from pathlib import Path
from decimal import Decimal
from PIL import Image

from django.conf import settings
from django.shortcuts import get_object_or_404
from django.http import FileResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .models import Patient, UploadedXRay, ScreeningPrediction
from .serializers import (
    PatientSerializer,
    UploadedXRaySerializer,
    ScreeningPredictionSerializer,
    MEDICAL_AI_DISCLAIMER,
)
from .services import (
    send_xray_to_fastapi,
    FastAPIServiceError,
    FastAPIConnectionError,
    FastAPITimeoutError,
    FastAPISchemaValidationError,
)

logger = logging.getLogger("screening.views")

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def user_has_patient_access(user, patient) -> bool:
    """
    SEC-01: Centralized patient-access authorization verification.
    Staff and superusers have administrative access; clinicians can only access their assigned patients.
    """
    if not user or not user.is_authenticated:
        return False
    return bool(user.is_staff or user.is_superuser or (patient and patient.assigned_clinician_id == user.id))



class PatientListCreateView(APIView):
    """Lists and registers patients assigned to the authenticated clinician."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.is_staff or request.user.is_superuser:
            patients = Patient.objects.all()
        else:
            patients = Patient.objects.filter(assigned_clinician=request.user)
        serializer = PatientSerializer(patients, many=True, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = PatientSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            # Automatically assign to requesting clinician if not provided
            serializer.save(assigned_clinician=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class XRayUploadView(APIView):
    """
    POST /api/screening/upload/
    Accepts, validates, and securely stores knee radiographs.
    Associates file with authenticated clinician and patient.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        # 1. Validate file presence
        upload_file = request.FILES.get("file") or request.FILES.get("image")
        if not upload_file:
            return Response(
                {"detail": "No image file provided in request. Expecting 'file' or 'image' multipart field."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Validate empty file & file size
        if upload_file.size == 0:
            return Response(
                {"detail": "The uploaded file is empty (0 bytes)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if upload_file.size > MAX_UPLOAD_SIZE_BYTES:
            return Response(
                {"detail": f"File size exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB limit."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3. Validate file extension
        orig_filename = upload_file.name or "radiograph.png"
        ext = Path(orig_filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return Response(
                {"detail": f"Unsupported image format '{ext}'. Allowed formats: {list(ALLOWED_IMAGE_EXTENSIONS)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4. Validate image integrity with PIL
        try:
            upload_file.seek(0)
            with Image.open(upload_file) as img:
                img.verify()
            upload_file.seek(0)
        except Exception as e:
            return Response(
                {"detail": f"Uploaded file is corrupted or not a valid radiograph image: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 5. Validate patient ownership & clinician authorization
        patient_id = request.data.get("patient_id") or request.data.get("patient")
        if not patient_id:
            return Response(
                {"detail": "'patient_id' field is required to associate radiograph with a patient record."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Find patient by ID or MRN string
        patient_query = Patient.objects.filter(id=patient_id) if str(patient_id).isdigit() else Patient.objects.filter(patient_id=str(patient_id))
        patient = patient_query.first()

        if not patient:
            return Response(
                {"detail": f"Patient with identifier '{patient_id}' not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not (request.user.is_staff or request.user.is_superuser or patient.assigned_clinician == request.user):
            return Response(
                {"detail": "You do not have permission to upload radiographs for this patient."},
                status=status.HTTP_403_FORBIDDEN,
            )

        knee_laterality = (request.data.get("knee_laterality") or request.data.get("laterality", "BILATERAL")).upper()
        if knee_laterality not in ["LEFT", "RIGHT", "BILATERAL"]:
            knee_laterality = "BILATERAL"

        # 6. Save UploadedXRay instance (UUID filename is generated automatically by upload_to)
        xray = UploadedXRay.objects.create(
            patient=patient,
            uploaded_by=request.user,
            image=upload_file,
            original_filename=orig_filename,
            file_size_bytes=upload_file.size,
            mime_type=upload_file.content_type or ("image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"),
            knee_laterality=knee_laterality,
        )

        serializer = UploadedXRaySerializer(xray, context={"request": request})
        return Response(
            {
                "success": True,
                "message": "Knee radiograph uploaded and validated successfully.",
                "id": str(xray.id),
                "uploaded_xray": serializer.data,
            },
            status=status.HTTP_201_CREATED,
        )


class ScreeningPredictView(APIView):
    """
    POST /api/screening/predict/
    Dispatches X-ray to the standalone FastAPI ML microservice.
    Persists prediction record and returns formatted diagnostic screening output.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, pk=None):
        try:
            threshold_val = float(request.data.get("threshold") or request.query_params.get("threshold", settings.KNEEWISE_DEFAULT_THRESHOLD))
        except (ValueError, TypeError):
            threshold_val = settings.KNEEWISE_DEFAULT_THRESHOLD
        if not (0.05 <= threshold_val <= 0.95):
            threshold_val = settings.KNEEWISE_DEFAULT_THRESHOLD

        uploaded_xray = None
        image_bytes = None
        filename = "xray.png"
        mime_type = "image/png"

        # Option A: Reference previously uploaded X-ray by ID
        xray_id = pk or request.data.get("uploaded_xray_id") or request.data.get("xray_id") or request.data.get("id") or request.query_params.get("uploaded_xray_id")
        if xray_id:
            uploaded_xray = get_object_or_404(UploadedXRay, id=xray_id)
            patient = uploaded_xray.patient

            # Permission verification
            if not (request.user.is_staff or request.user.is_superuser or patient.assigned_clinician == request.user):
                return Response(
                    {"detail": "You do not have permission to analyze radiographs for this patient."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            filename = uploaded_xray.original_filename
            mime_type = uploaded_xray.mime_type
            try:
                with uploaded_xray.image.open("rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                logger.error(f"Failed to read stored image {uploaded_xray.image.name}: {e}")
                return Response(
                    {"detail": "Stored radiograph image file could not be read from media storage."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # Option B: Direct single-step image upload + predict
        elif "file" in request.FILES or "image" in request.FILES:
            upload_file = request.FILES.get("file") or request.FILES.get("image")
            patient_id = request.data.get("patient_id") or request.data.get("patient")
            if not patient_id:
                return Response(
                    {"detail": "'patient_id' is required when uploading a new radiograph."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            patient_query = Patient.objects.filter(id=patient_id) if str(patient_id).isdigit() else Patient.objects.filter(patient_id=str(patient_id))
            patient = patient_query.first()
            if not patient:
                return Response(
                    {"detail": f"Patient with identifier '{patient_id}' not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            if not (request.user.is_staff or request.user.is_superuser or patient.assigned_clinician == request.user):
                return Response(
                    {"detail": "You do not have permission to perform screening for this patient."},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Validate file
            orig_filename = upload_file.name or "radiograph.png"
            ext = Path(orig_filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                return Response(
                    {"detail": f"Unsupported image format '{ext}'. Allowed: {list(ALLOWED_IMAGE_EXTENSIONS)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                upload_file.seek(0)
                with Image.open(upload_file) as img:
                    img.verify()
                upload_file.seek(0)
            except Exception as e:
                return Response(
                    {"detail": f"Corrupted or invalid image file: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            filename = orig_filename
            mime_type = upload_file.content_type or "image/png"
            image_bytes = upload_file.read()
            upload_file.seek(0)

            knee_laterality = (request.data.get("knee_laterality") or request.data.get("laterality", "BILATERAL")).upper()
            if knee_laterality not in ["LEFT", "RIGHT", "BILATERAL"]:
                knee_laterality = "BILATERAL"

            # Persist UploadedXRay
            uploaded_xray = UploadedXRay.objects.create(
                patient=patient,
                uploaded_by=request.user,
                image=upload_file,
                original_filename=orig_filename,
                file_size_bytes=len(image_bytes),
                mime_type=mime_type,
                knee_laterality=knee_laterality,
            )
        else:
            return Response(
                {"detail": "Provide either 'uploaded_xray_id' or upload a 'file' with 'patient_id'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3. Call FastAPI ML Inference Service
        try:
            fastapi_resp = send_xray_to_fastapi(
                image_file=image_bytes,
                filename=filename,
                mime_type=mime_type,
                threshold=threshold_val,
            )
        except FastAPIConnectionError as ce:
            return Response(
                {
                    "success": False,
                    "detail": str(ce),
                    "code": "FASTAPI_SERVICE_UNAVAILABLE",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except FastAPITimeoutError as te:
            return Response(
                {
                    "success": False,
                    "detail": str(te),
                    "code": "FASTAPI_REQUEST_TIMEOUT",
                },
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except FastAPIServiceError as se:
            return Response(
                {
                    "success": False,
                    "detail": str(se),
                    "code": "FASTAPI_SERVICE_ERROR",
                },
                status=se.status_code,
            )
        except Exception as e:
            logger.error(f"Unexpected error calling FastAPI service: {e}")
            return Response(
                {
                    "success": False,
                    "detail": "Failed to communicate with AI screening service.",
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # 4. Extract and persist prediction record in Database
        pred_data = fastapi_resp["prediction"]
        bin_data = fastapi_resp["binary_screening"]
        meta_data = fastapi_resp.get("model_metadata", {})
        probs = pred_data.get("probabilities", {})

        prediction_record = ScreeningPrediction.objects.create(
            patient=patient,
            uploaded_xray=uploaded_xray,
            clinician=request.user,
            predicted_kl_grade=int(pred_data["kellgren_lawrence_grade"]),
            kl0_probability=Decimal(str(probs.get("KL_0", 0.0))),
            kl1_probability=Decimal(str(probs.get("KL_1", 0.0))),
            kl2_probability=Decimal(str(probs.get("KL_2", 0.0))),
            kl3_probability=Decimal(str(probs.get("KL_3", 0.0))),
            kl4_probability=Decimal(str(probs.get("KL_4", 0.0))),
            oa_probability=Decimal(str(bin_data.get("binary_oa_probability", 0.0))),
            screening_threshold=Decimal(str(bin_data.get("screening_threshold_used", threshold_val))),
            is_oa_likely=bool(bin_data.get("is_oa_likely", False)),
            screening_category=str(bin_data.get("screening_category", "Screening Evaluation Complete")),
            model_name=str(meta_data.get("model_name", "KneeWise-EfficientNet-B0-Champion")),
            experiment_id=str(meta_data.get("experiment_id", "exp_d_efficientnet_b0")),
            checkpoint_sha256=str(meta_data.get("checkpoint_sha256", "")),
            inference_time=float(meta_data.get("inference_time_ms", 0.0)),
        )

        # 5. Format structured response matching contract
        serializer = ScreeningPredictionSerializer(prediction_record, context={"request": request})
        resp_data = dict(serializer.data)
        probs_map = {f"KL{k[-1]}": v for k, v in (serializer.data.get("class_probabilities") or {}).items()}
        resp_data.update({
            "success": True,
            "prediction": serializer.data,
            "screening_result": serializer.data,
            "predicted_grade": serializer.data.get("predicted_kl_grade"),
            "binary_oa_risk": serializer.data.get("oa_probability"),
            "binary_referral_recommended": serializer.data.get("is_oa_likely"),
            "kl_probabilities": probs_map,
        })
        return Response(resp_data, status=status.HTTP_200_OK)


class ScreeningHistoryView(APIView):
    """
    GET /api/screening/history/
    Retrieves historical screening predictions for the authenticated clinician.
    Supports filtering by patient MRN/ID (?patient_id=<id>).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.is_staff or request.user.is_superuser:
            qs = ScreeningPrediction.objects.all()
        else:
            qs = ScreeningPrediction.objects.filter(patient__assigned_clinician=request.user)

        # Optional patient filter
        patient_filter = request.query_params.get("patient_id")
        if patient_filter:
            if str(patient_filter).isdigit():
                qs = qs.filter(patient__id=patient_filter)
            else:
                qs = qs.filter(patient__patient_id=patient_filter)

        serializer = ScreeningPredictionSerializer(qs, many=True, context={"request": request})
        return Response(
            {
                "count": qs.count(),
                "results": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class ScreeningDetailView(APIView):
    """
    GET /api/screening/<id>/
    Retrieves detailed results for a single screening prediction record.
    Enforces strict clinician/patient authorization checks.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        prediction = get_object_or_404(ScreeningPrediction, pk=pk)

        # Verify clinician permission
        if not (request.user.is_staff or request.user.is_superuser or prediction.patient.assigned_clinician == request.user):
            return Response(
                {"detail": "You do not have permission to view this screening record."},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = ScreeningPredictionSerializer(prediction, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ProtectedXRayMediaView(APIView):
    """
    GET /api/screening/xrays/<uuid:pk>/
    SEC-01: Protected Patient Radiograph Streaming Endpoint.
    Securely streams stored knee radiographs to authorized clinicians only.
    Enforces authentication, clinician-patient access rights, and returns
    HTTP 404 for unauthorized attempts to prevent patient record enumeration.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        # 1. Retrieve X-ray record using its UUID
        try:
            xray = UploadedXRay.objects.select_related("patient", "patient__assigned_clinician").get(pk=pk)
        except (UploadedXRay.DoesNotExist, ValueError):
            return Response(
                {"detail": "Radiograph not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # 2. Verify clinician permission (Return 404 to avoid leaking record existence)
        if not user_has_patient_access(request.user, xray.patient):
            logger.warning(
                f"[SEC-01 Protected Media] Unauthorized access attempt: User '{request.user.username}' (ID: {request.user.id}) "
                f"attempted to access radiograph {pk} belonging to Patient {xray.patient.patient_id}."
            )
            return Response(
                {"detail": "Radiograph not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # 3. Verify physical file presence on storage
        if not xray.image or not xray.image.storage.exists(xray.image.name):
            logger.error(
                f"[SEC-01 Protected Media] File '{xray.image.name if xray.image else 'None'}' missing on storage for X-ray {pk}."
            )
            return Response(
                {"detail": "Radiograph file not found on storage."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # 4. Open file stream safely
        try:
            file_handle = xray.image.open("rb")
        except Exception as e:
            logger.error(f"[SEC-01 Protected Media] Error reading radiograph file for X-ray {pk}: {e}")
            return Response(
                {"detail": "Unable to access radiograph file."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # 5. Determine validated image content type
        content_type = xray.mime_type or "image/png"
        if not content_type.startswith("image/"):
            ext = Path(xray.image.name).suffix.lower()
            content_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        # 6. Stream with safe headers
        ext = Path(xray.image.name).suffix.lower() or ".png"
        safe_filename = f"{xray.id}{ext}"

        response = FileResponse(file_handle, content_type=content_type)
        response["X-Content-Type-Options"] = "nosniff"
        response["Content-Disposition"] = f'inline; filename="{safe_filename}"'
        return response

