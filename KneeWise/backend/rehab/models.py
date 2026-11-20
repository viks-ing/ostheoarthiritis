"""
rehab/models.py - KneeWise Rehabilitation Data Models
Stores session metrics, repetition statistics, angular excursion history,
and set progressions linked to patients and assigned clinicians.

Safety & Regulatory Notice:
All kinematic and range-of-motion measurements stored in these records are
2D computer vision biomechanical estimates for exercise tracking.
They are not clinical diagnostic goniometry.
"""

import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()


class SessionStatus(models.TextChoices):
    IN_PROGRESS = "IN_PROGRESS", "In Progress"
    COMPLETED = "COMPLETED", "Completed"
    ABORTED = "ABORTED", "Aborted"


class LegChoice(models.TextChoices):
    LEFT = "left", "Left"
    RIGHT = "right", "Right"
    AUTO = "auto", "Auto"


class RehabilitationSession(models.Model):
    """
    Tracks a complete patient rehabilitation exercise session.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(
        "screening.Patient",
        on_delete=models.CASCADE,
        related_name="rehab_sessions",
        help_text="Patient undergoing rehabilitation exercise",
    )
    clinician = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="rehab_sessions",
        help_text="Clinician overseeing the rehabilitation session",
    )
    exercise_name = models.CharField(
        max_length=64,
        default="seated_knee_extension",
        help_text="Name of exercise protocol",
    )
    selected_leg = models.CharField(
        max_length=10,
        choices=LegChoice.choices,
        default=LegChoice.RIGHT,
        help_text="Target leg tracked during exercise",
    )
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=SessionStatus.choices,
        default=SessionStatus.IN_PROGRESS,
    )
    total_reps = models.PositiveIntegerField(
        default=0,
        help_text="Total validated repetitions completed",
    )
    rejected_reps = models.PositiveIntegerField(
        default=0,
        help_text="Count of incomplete or rejected attempts",
    )
    set_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of completed exercise sets",
    )
    avg_excursion_deg = models.FloatField(
        default=0.0,
        help_text="Average range of motion (excursion) in degrees",
    )
    best_excursion_deg = models.FloatField(
        default=0.0,
        help_text="Peak range of motion (excursion) achieved in degrees",
    )
    tracking_availability_pct = models.FloatField(
        default=0.0,
        help_text="Percentage of session duration with reliable landmark tracking",
    )
    duration_sec = models.FloatField(
        default=0.0,
        help_text="Total active session duration in seconds",
    )
    notes = models.TextField(
        blank=True,
        default="",
        help_text="Optional clinician observations or patient feedback",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Rehabilitation Session"
        verbose_name_plural = "Rehabilitation Sessions"

    def __str__(self) -> str:
        return f"RehabSession {self.id} - Patient {self.patient.patient_id} ({self.status})"


class RehabilitationSet(models.Model):
    """
    Sub-session set metrics and rest periods.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        RehabilitationSession,
        on_delete=models.CASCADE,
        related_name="sets",
        help_text="Associated rehabilitation session",
    )
    set_number = models.PositiveIntegerField(default=1)
    reps_completed = models.PositiveIntegerField(default=0)
    reps_rejected = models.PositiveIntegerField(default=0)
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    rest_duration_sec = models.FloatField(
        default=0.0,
        help_text="Rest interval elapsed before this set in seconds",
    )
    avg_excursion_deg = models.FloatField(
        default=0.0,
        help_text="Average ROM excursion during this set in degrees",
    )
    max_extension_deg = models.FloatField(
        default=0.0,
        help_text="Peak extension angle achieved during this set in degrees",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["set_number"]
        verbose_name = "Rehabilitation Set"
        verbose_name_plural = "Rehabilitation Sets"

    def __str__(self) -> str:
        return f"Set #{self.set_number} for Session {self.session_id} ({self.reps_completed} reps)"
