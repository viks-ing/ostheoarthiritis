"""
screening/admin.py - Django Admin Configuration for KneeWise Screening
"""

from django.contrib import admin
from .models import Patient, UploadedXRay, ScreeningPrediction


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("patient_id", "last_name", "first_name", "gender", "assigned_clinician", "created_at")
    search_fields = ("patient_id", "first_name", "last_name", "assigned_clinician__username")
    list_filter = ("gender", "created_at")


@admin.register(UploadedXRay)
class UploadedXRayAdmin(admin.ModelAdmin):
    list_display = ("id", "patient", "knee_laterality", "original_filename", "file_size_bytes", "uploaded_at")
    search_fields = ("id", "patient__patient_id", "original_filename")
    list_filter = ("knee_laterality", "uploaded_at")


@admin.register(ScreeningPrediction)
class ScreeningPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "patient",
        "predicted_kl_grade",
        "oa_probability",
        "screening_threshold",
        "is_oa_likely",
        "model_name",
        "created_at",
    )
    search_fields = ("id", "patient__patient_id", "model_name")
    list_filter = ("predicted_kl_grade", "is_oa_likely", "created_at")
    readonly_fields = [f.name for f in ScreeningPrediction._meta.fields]
