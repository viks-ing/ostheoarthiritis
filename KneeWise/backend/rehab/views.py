"""
rehab/views.py - REST Endpoints for Rehabilitation Sessions & Sets
Enforces strict clinician authentication and patient data isolation.
"""

from rest_framework import generics, status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied, NotFound
from django.utils import timezone
from django.shortcuts import get_object_or_404

import sys
import base64
import time
from pathlib import Path

# Ensure project root (workspace containing rehab_cv) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from rehab_cv.config import PoseConfig, ExerciseConfig
from rehab_cv.detection.pose_detector import PoseDetector
from rehab_cv.exercises.knee_extension import KneeExtensionExercise
from rehab_cv.feedback.feedback_engine import RehabilitationFeedbackEngine
from rehab_cv.feedback.session_analyzer import SessionAnalyzer

from .models import RehabilitationSession, RehabilitationSet, SessionStatus
from .serializers import (
    RehabilitationSessionSerializer,
    RehabilitationSessionCreateSerializer,
    RehabilitationSetSerializer,
    SessionCompleteInputSerializer,
)
from screening.models import Patient


# Active in-memory session CV pipelines
ACTIVE_SESSION_PIPELINES = {}


class SessionCVEngine:
    """Manages an active MediaPipe Pose + Biomechanics session pipeline."""

    def __init__(self, selected_leg="auto"):
        self.selected_leg = selected_leg.lower()
        self.detector = PoseDetector(PoseConfig(model_complexity=1))
        self.exercise = KneeExtensionExercise(ExerciseConfig(preferred_leg=self.selected_leg))
        self.feedback_engine = RehabilitationFeedbackEngine()
        self.analyzer = SessionAnalyzer(selected_leg=self.selected_leg)
        self.analyzer.start_session()
        self.last_active = time.time()

    def process(self, frame_bgr):
        self.last_active = time.time()
        detection = self.detector.process(frame_bgr)
        metrics = self.exercise.process(detection)
        feedback_event = self.feedback_engine.process(metrics)
        self.analyzer.record_frame(metrics)

        active_leg = detection.right_leg if metrics.selected_leg == "right" else detection.left_leg
        hip_pt = active_leg.hip
        knee_pt = active_leg.knee
        ankle_pt = active_leg.ankle

        landmarks_dict = {
            "hip": {"x": round(hip_pt.x, 4), "y": round(hip_pt.y, 4), "visibility": round(hip_pt.visibility, 3)} if hip_pt else None,
            "knee": {"x": round(knee_pt.x, 4), "y": round(knee_pt.y, 4), "visibility": round(knee_pt.visibility, 3)} if knee_pt else None,
            "ankle": {"x": round(ankle_pt.x, 4), "y": round(ankle_pt.y, 4), "visibility": round(ankle_pt.visibility, 3)} if ankle_pt else None,
        }

        tracking_status = "TRACKING" if metrics.is_tracking_reliable else ("NO_POSE" if not detection.detected else "LOW_CONFIDENCE")

        return {
            "tracking_status": tracking_status,
            "selected_leg": metrics.selected_leg.upper(),
            "knee_angle_deg": round(metrics.raw_angle, 1) if metrics.raw_angle is not None else None,
            "smoothed_angle_deg": round(metrics.smoothed_angle, 1) if metrics.smoothed_angle is not None else None,
            "repetitions": metrics.repetition_count,
            "exercise_state": metrics.exercise_state.value,
            "feedback": feedback_event.message if feedback_event else ("Tracking active" if metrics.is_tracking_reliable else "Position yourself sideways"),
            "feedback_severity": feedback_event.severity.value if feedback_event else "info",
            "timestamp": timezone.now().isoformat(),
            "is_tracking_reliable": metrics.is_tracking_reliable,
            "landmark_confidence": round(metrics.landmark_confidence, 2),
            "rep_completed_this_frame": metrics.rep_completed_this_frame,
            "landmarks": landmarks_dict,
            "metrics": {
                "max_extension_in_rep": round(metrics.max_extension_in_rep, 1),
                "current_excursion": round(metrics.current_excursion, 1),
            },
        }


class IsClinicianOrStaff(permissions.BasePermission):
    """
    Ensures user is authenticated and is the assigned clinician or staff.
    """
    def has_object_permission(self, request, view, obj):
        if request.user.is_staff or request.user.is_superuser:
            return True
        if isinstance(obj, RehabilitationSession):
            return obj.clinician == request.user or obj.patient.assigned_clinician == request.user
        if isinstance(obj, RehabilitationSet):
            return obj.session.clinician == request.user or obj.session.patient.assigned_clinician == request.user
        return False


class SessionListCreateView(generics.ListCreateAPIView):
    """
    GET: List rehabilitation sessions for clinician's authorized patients.
    POST: Create a new rehabilitation session for an authorized patient.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method == "POST":
            return RehabilitationSessionCreateSerializer
        return RehabilitationSessionSerializer

    def get_queryset(self):
        user = self.request.user
        qs = RehabilitationSession.objects.select_related("patient", "clinician").prefetch_related("sets")

        if not (user.is_staff or user.is_superuser):
            qs = qs.filter(patient__assigned_clinician=user)

        patient_param = self.request.query_params.get("patient")
        if patient_param:
            qs = qs.filter(patient__id=patient_param)

        return qs

    def perform_create(self, serializer):
        patient = serializer.validated_data.get("patient")
        user = self.request.user

        # Patient Isolation Check
        if not (user.is_staff or user.is_superuser):
            if patient.assigned_clinician != user:
                raise PermissionDenied("You are not authorized to initiate a session for this patient.")

        serializer.save(clinician=user, status=SessionStatus.IN_PROGRESS, start_time=timezone.now())


class SessionDetailView(generics.RetrieveAPIView):
    """
    GET: Retrieve detailed session record including sets and excursion statistics.
    """
    permission_classes = [permissions.IsAuthenticated, IsClinicianOrStaff]
    serializer_class = RehabilitationSessionSerializer
    queryset = RehabilitationSession.objects.select_related("patient", "clinician").prefetch_related("sets")


class SessionCompleteView(APIView):
    """
    POST: Mark session as COMPLETED and persist aggregate metrics and completed sets.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        session = get_object_or_404(
            RehabilitationSession.objects.select_related("patient", "clinician"),
            pk=pk,
        )

        # Check authorization
        if not (request.user.is_staff or request.user.is_superuser):
            if session.clinician != request.user and session.patient.assigned_clinician != request.user:
                raise PermissionDenied("You are not authorized to update this rehabilitation session.")

        input_serializer = SessionCompleteInputSerializer(data=request.data)
        input_serializer.is_valid(raise_exception=True)
        data = input_serializer.validated_data

        # Update Session from validated runtime measurements
        session.status = SessionStatus.COMPLETED
        session.end_time = timezone.now()

        sets_data = data.get("sets", [])

        # Derive duration from actual timestamps if not explicitly measured by client
        client_duration = data.get("duration_sec", 0.0)
        measured_duration = max(0.1, round((session.end_time - session.start_time).total_seconds(), 2))
        session.duration_sec = client_duration if client_duration > 0.0 else measured_duration

        # Derive set_count and rep metrics from actual submitted sets if available
        if sets_data:
            session.set_count = max(len(sets_data), data.get("set_count", 1))
            total_set_reps = sum(s["reps_completed"] for s in sets_data)
            session.total_reps = max(total_set_reps, data.get("total_reps", 0))

            total_set_rejected = sum(s.get("reps_rejected", 0) for s in sets_data)
            session.rejected_reps = max(total_set_rejected, data.get("rejected_reps", 0))

            # Excursion validations
            excursions = [s["avg_excursion_deg"] for s in sets_data if s.get("avg_excursion_deg", 0) > 0]
            if excursions and data.get("avg_excursion_deg", 0.0) == 0.0:
                session.avg_excursion_deg = round(sum(excursions) / len(excursions), 1)
            else:
                session.avg_excursion_deg = data["avg_excursion_deg"]

            max_exts = [s["max_extension_deg"] for s in sets_data if s.get("max_extension_deg", 0) > 0]
            if max_exts and data.get("best_excursion_deg", 0.0) == 0.0:
                session.best_excursion_deg = round(max(max_exts), 1)
            else:
                session.best_excursion_deg = data["best_excursion_deg"]
        else:
            session.set_count = data["set_count"]
            session.total_reps = data["total_reps"]
            session.rejected_reps = data["rejected_reps"]
            session.avg_excursion_deg = data["avg_excursion_deg"]
            session.best_excursion_deg = data["best_excursion_deg"]

        session.tracking_availability_pct = data["tracking_availability_pct"]
        if data.get("notes"):
            session.notes = data["notes"]
        session.save()
        for s in sets_data:
            RehabilitationSet.objects.create(
                session=session,
                set_number=s["set_number"],
                reps_completed=s["reps_completed"],
                reps_rejected=s.get("reps_rejected", 0),
                start_time=s.get("start_time"),
                end_time=s.get("end_time"),
                rest_duration_sec=s.get("rest_duration_sec", 0.0),
                avg_excursion_deg=s.get("avg_excursion_deg", 0.0),
                max_extension_deg=s.get("max_extension_deg", 0.0),
            )

        output_serializer = RehabilitationSessionSerializer(session)
        return Response(output_serializer.data, status=status.HTTP_200_OK)


class SetCreateView(generics.CreateAPIView):
    """
    POST: Append an individual completed set to an active rehabilitation session.
    """
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = RehabilitationSetSerializer

    def perform_create(self, serializer):
        session_id = self.kwargs.get("pk")
        session = get_object_or_404(RehabilitationSession, pk=session_id)

        user = self.request.user
        if not (user.is_staff or user.is_superuser):
            if session.clinician != user and session.patient.assigned_clinician != user:
                raise PermissionDenied("You are not authorized to add sets to this session.")

        serializer.save(session=session)


class RehabProcessFrameView(APIView):
    """
    POST: Processes a single real camera frame with the Python MediaPipe CV engine.
    Strictly authenticated and patient-isolated.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        session_id = request.data.get("session_id")
        if not session_id:
            return Response({"detail": "session_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        session = get_object_or_404(RehabilitationSession.objects.select_related("patient"), pk=session_id)

        # Check authorization
        if not (request.user.is_staff or request.user.is_superuser):
            if session.clinician != request.user and session.patient.assigned_clinician != request.user:
                raise PermissionDenied("You are not authorized to process rehabilitation frames for this patient.")

        # Decode image
        frame_bytes = None
        if "file" in request.FILES:
            frame_bytes = request.FILES["file"].read()
        elif "image" in request.data:
            b64_str = request.data["image"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                frame_bytes = base64.b64decode(b64_str)
            except Exception as e:
                return Response({"detail": f"Invalid base64 encoding: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        if not frame_bytes:
            return Response({"detail": "No image frame provided (expecting 'image' base64 or 'file' multipart)"}, status=status.HTTP_400_BAD_REQUEST)

        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return Response({"detail": "Failed to decode image frame"}, status=status.HTTP_400_BAD_REQUEST)

        # Get or create pipeline for this session
        selected_leg = request.data.get("selected_leg", session.selected_leg or "auto")
        pipeline = ACTIVE_SESSION_PIPELINES.get(str(session.id))
        if pipeline is None:
            pipeline = SessionCVEngine(selected_leg=selected_leg)
            ACTIVE_SESSION_PIPELINES[str(session.id)] = pipeline

        # Execute genuine CV
        result = pipeline.process(frame_bgr)
        return Response(result, status=status.HTTP_200_OK)
