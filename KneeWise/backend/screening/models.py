"""
screening/models.py - KneeWise Clinical Screening Data Models
Implements relational entities for Patients, Secure Radiograph Uploads,
and Historical AI Screening Predictions with full model-provenance tracking.
"""

import os
import uuid
from pathlib import Path
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


def xray_upload_path(instance, filename: str) -> str:
    """
    Generates a secure, cryptographically random storage path for uploaded X-rays.
    Protects against path traversal attacks and prevents leaking internal filesystem paths.
    Format: xrays/patient_<id>/<uuid4>.<ext>
    """
    ext = Path(filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        ext = ".png"
    unique_id = uuid.uuid4().hex
    return f"xrays/patient_{instance.patient.id}/{unique_id}{ext}"


class Patient(models.Model):
    """
    Clinical Patient Record.
    Associates knee X-rays and predictions with an assigned healthcare provider.
    """
    patient_id = models.CharField(
        max_length=64,
        unique=True,
        help_text="Unique Medical Record Number (MRN) or Patient Identifier",
    )
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(
        max_length=20,
        choices=[("M", "Male"), ("F", "Female"), ("O", "Other"), ("U", "Unknown")],
        default="U",
    )
    assigned_clinician = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="patients",
        help_text="Clinician or radiologist authorized to view this patient",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Patient"
        verbose_name_plural = "Patients"

    def __str__(self) -> str:
        return f"{self.patient_id} - {self.last_name}, {self.first_name}"


class UploadedXRay(models.Model):
    """
    Uploaded Knee Radiograph Entity.
    Stores the raw image in media storage with metadata and ownership tracking.
    """
    LATERALITY_CHOICES = [
        ("LEFT", "Left Knee"),
        ("RIGHT", "Right Knee"),
        ("BILATERAL", "Bilateral / Unspecified"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="xrays",
    )
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="uploaded_xrays",
    )
    image = models.ImageField(upload_to=xray_upload_path)
    original_filename = models.CharField(max_length=255)
    file_size_bytes = models.PositiveIntegerField(help_text="Image size in bytes")
    mime_type = models.CharField(max_length=50, default="image/png")
    knee_laterality = models.CharField(
        max_length=15,
        choices=LATERALITY_CHOICES,
        default="BILATERAL",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name = "Uploaded Radiograph"
        verbose_name_plural = "Uploaded Radiographs"

    def __str__(self) -> str:
        return f"X-ray {self.id} (Patient {self.patient.patient_id}) - {self.knee_laterality}"


class ScreeningPrediction(models.Model):
    """
    KneeWise AI Screening Prediction Record.
    Persists historical 5-class Kellgren-Lawrence probabilities and derived binary screening status.
    Maintains full model version provenance for clinical auditability.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    uploaded_xray = models.ForeignKey(
        UploadedXRay,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    clinician = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="screening_predictions",
    )

    # 5-Class Severity Classification (Kellgren-Lawrence 0-4)
    predicted_kl_grade = models.PositiveSmallIntegerField(
        help_text="Predicted Kellgren-Lawrence Grade (0: Normal, 1: Doubtful, 2: Mild, 3: Moderate, 4: Severe)"
    )

    # Decimal probabilities stored from 0.0000 to 1.0000 (four decimal places)
    kl0_probability = models.DecimalField(max_digits=6, decimal_places=4, help_text="P(KL 0 - Normal)")
    kl1_probability = models.DecimalField(max_digits=6, decimal_places=4, help_text="P(KL 1 - Doubtful OA)")
    kl2_probability = models.DecimalField(max_digits=6, decimal_places=4, help_text="P(KL 2 - Mild OA)")
    kl3_probability = models.DecimalField(max_digits=6, decimal_places=4, help_text="P(KL 3 - Moderate OA)")
    kl4_probability = models.DecimalField(max_digits=6, decimal_places=4, help_text="P(KL 4 - Severe OA)")

    # Derived Binary OA Screening
    # P(Definite OA) = P(KL2) + P(KL3) + P(KL4)
    oa_probability = models.DecimalField(
        max_digits=6,
        decimal_places=4,
        help_text="P(Definite Radiographic OA) = P(KL2) + P(KL3) + P(KL4)",
    )
    screening_threshold = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        default=0.30,
        help_text="Operating threshold used for binary screening categorization",
    )
    is_oa_likely = models.BooleanField(
        help_text="True if oa_probability >= screening_threshold",
    )
    screening_category = models.CharField(
        max_length=120,
        help_text="Clinical screening recommendation category",
    )

    # Model & Pipeline Provenance Metadata
    model_name = models.CharField(max_length=100, default="KneeWise-EfficientNet-B0-Champion")
    experiment_id = models.CharField(max_length=100, default="exp_d_efficientnet_b0")
    checkpoint_sha256 = models.CharField(max_length=64, blank=True)
    inference_time = models.FloatField(help_text="Model inference latency in milliseconds")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Screening Prediction"
        verbose_name_plural = "Screening Predictions"

    def __str__(self) -> str:
        return f"Prediction {self.id}: Patient {self.patient.patient_id} -> KL Grade {self.predicted_kl_grade}"
