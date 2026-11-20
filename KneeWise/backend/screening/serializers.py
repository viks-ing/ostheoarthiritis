"""
screening/serializers.py - Django REST Framework Serializers
Handles validation, serialization, and presentation formatting for Patients,
X-ray uploads, and AI screening predictions with mandatory medical disclaimers.
"""

from django.urls import reverse
from rest_framework import serializers
from .models import Patient, UploadedXRay, ScreeningPrediction

MEDICAL_AI_DISCLAIMER = (
    "AI-generated screening support only. This result is not a medical diagnosis. "
    "Please consult a qualified radiologist or healthcare professional for clinical interpretation."
)


class PatientSerializer(serializers.ModelSerializer):
    """Serializes Patient demographic and identity information."""
    assigned_clinician_username = serializers.CharField(
        source="assigned_clinician.username",
        read_only=True,
    )

    class Meta:
        model = Patient
        fields = [
            "id",
            "patient_id",
            "first_name",
            "last_name",
            "date_of_birth",
            "gender",
            "assigned_clinician",
            "assigned_clinician_username",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class UploadedXRaySerializer(serializers.ModelSerializer):
    """Serializes uploaded X-ray metadata without exposing internal filesystem paths."""
    patient_mrn = serializers.CharField(source="patient.patient_id", read_only=True)
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedXRay
        fields = [
            "id",
            "patient",
            "patient_mrn",
            "original_filename",
            "file_size_bytes",
            "mime_type",
            "knee_laterality",
            "image_url",
            "uploaded_at",
        ]
        read_only_fields = ["id", "file_size_bytes", "mime_type", "uploaded_at"]

    def get_image_url(self, obj) -> str:
        """
        SEC-01: Generates the authenticated, permission-checked protected media URL.
        Prevents direct unauthenticated access to storage/filesystem paths.
        """
        if not obj or not getattr(obj, "id", None):
            return ""
        try:
            rel_url = reverse("screening:xray-media", kwargs={"pk": obj.id})
        except Exception:
            return ""
        request = self.context.get("request")
        if request:
            return request.build_absolute_uri(rel_url)
        return rel_url


class ScreeningPredictionSerializer(serializers.ModelSerializer):
    """
    Serializes KneeWise AI Screening Predictions.
    Formats five-class probabilities, derived binary OA risk, threshold, and medical safety notices.
    """
    patient_mrn = serializers.CharField(source="patient.patient_id", read_only=True)
    patient_name = serializers.SerializerMethodField()
    class_probabilities = serializers.SerializerMethodField()
    oa_probability = serializers.FloatField()
    screening_threshold = serializers.FloatField()
    model_metadata = serializers.SerializerMethodField()
    disclaimer = serializers.SerializerMethodField()

    class Meta:
        model = ScreeningPrediction
        fields = [
            "id",
            "patient",
            "patient_mrn",
            "patient_name",
            "uploaded_xray",
            "predicted_kl_grade",
            "class_probabilities",
            "oa_probability",
            "screening_threshold",
            "is_oa_likely",
            "screening_category",
            "model_metadata",
            "created_at",
            "disclaimer",
        ]
        read_only_fields = fields

    def get_patient_name(self, obj) -> str:
        return f"{obj.patient.last_name}, {obj.patient.first_name}"

    def get_class_probabilities(self, obj) -> dict:
        return {
            "KL0": float(obj.kl0_probability),
            "KL1": float(obj.kl1_probability),
            "KL2": float(obj.kl2_probability),
            "KL3": float(obj.kl3_probability),
            "KL4": float(obj.kl4_probability),
        }

    def get_model_metadata(self, obj) -> dict:
        return {
            "model_name": obj.model_name,
            "experiment_id": obj.experiment_id,
            "checkpoint_sha256": obj.checkpoint_sha256,
            "inference_time_ms": obj.inference_time,
        }

    def get_disclaimer(self, obj) -> str:
        return MEDICAL_AI_DISCLAIMER
