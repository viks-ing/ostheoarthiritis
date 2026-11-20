"""
rehab/serializers.py - DRF Serializers for Rehabilitation Sessions and Sets
"""

from rest_framework import serializers
from .models import RehabilitationSession, RehabilitationSet
from screening.models import Patient


class RehabilitationSetSerializer(serializers.ModelSerializer):
    class Meta:
        model = RehabilitationSet
        fields = [
            "id",
            "set_number",
            "reps_completed",
            "reps_rejected",
            "start_time",
            "end_time",
            "rest_duration_sec",
            "avg_excursion_deg",
            "max_extension_deg",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class RehabilitationSessionSerializer(serializers.ModelSerializer):
    sets = RehabilitationSetSerializer(many=True, read_only=True)
    patient_mrn = serializers.CharField(source="patient.patient_id", read_only=True)
    patient_name = serializers.SerializerMethodField()
    clinician_name = serializers.SerializerMethodField()

    class Meta:
        model = RehabilitationSession
        fields = [
            "id",
            "patient",
            "patient_mrn",
            "patient_name",
            "clinician",
            "clinician_name",
            "exercise_name",
            "selected_leg",
            "start_time",
            "end_time",
            "status",
            "total_reps",
            "rejected_reps",
            "set_count",
            "avg_excursion_deg",
            "best_excursion_deg",
            "tracking_availability_pct",
            "duration_sec",
            "notes",
            "sets",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "clinician",
            "clinician_name",
            "patient_mrn",
            "patient_name",
            "created_at",
            "updated_at",
        ]

    def get_patient_name(self, obj: RehabilitationSession) -> str:
        return f"{obj.patient.first_name} {obj.patient.last_name}"

    def get_clinician_name(self, obj: RehabilitationSession) -> str:
        return f"{obj.clinician.first_name} {obj.clinician.last_name}" or obj.clinician.username


class RehabilitationSessionCreateSerializer(serializers.ModelSerializer):
    patient = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all())

    class Meta:
        model = RehabilitationSession
        fields = [
            "id",
            "patient",
            "exercise_name",
            "selected_leg",
            "notes",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class SetInputItemSerializer(serializers.Serializer):
    set_number = serializers.IntegerField(min_value=1)
    reps_completed = serializers.IntegerField(min_value=0)
    reps_rejected = serializers.IntegerField(min_value=0, default=0)
    start_time = serializers.DateTimeField(required=False, allow_null=True)
    end_time = serializers.DateTimeField(required=False, allow_null=True)
    rest_duration_sec = serializers.FloatField(min_value=0.0, default=0.0)
    avg_excursion_deg = serializers.FloatField(min_value=0.0, default=0.0)
    max_extension_deg = serializers.FloatField(min_value=0.0, default=0.0)


class SessionCompleteInputSerializer(serializers.Serializer):
    total_reps = serializers.IntegerField(min_value=0)
    rejected_reps = serializers.IntegerField(min_value=0, default=0)
    set_count = serializers.IntegerField(min_value=0, default=1)
    avg_excursion_deg = serializers.FloatField(min_value=0.0, default=0.0)
    best_excursion_deg = serializers.FloatField(min_value=0.0, default=0.0)
    tracking_availability_pct = serializers.FloatField(min_value=0.0, max_value=100.0, default=0.0)
    duration_sec = serializers.FloatField(min_value=0.0, default=0.0)
    notes = serializers.CharField(required=False, allow_blank=True, default="")
    sets = SetInputItemSerializer(many=True, required=False, default=[])
