"""
rehab/tests.py - Django REST Framework Integration Tests for Rehabilitation App
Tests:
- Authentication enforcement (unauthenticated requests rejected with 401)
- Patient isolation & authorization (cross-clinician access rejected with 403)
- Session lifecycle: creation, set tracking, and completion with aggregate metrics
- Patient filter queries
"""

import uuid
from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from screening.models import Patient
from rehab.models import RehabilitationSession, RehabilitationSet, SessionStatus

User = get_user_model()


class RehabilitationAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Clinician 1 & Patient 1
        self.clinician_1 = User.objects.create_user(
            username="clinician_one",
            email="dr.one@kneewise.ai",
            password="StrongPassword123!",
            first_name="Alice",
            last_name="Smith",
        )
        self.patient_1 = Patient.objects.create(
            patient_id="MRN-001",
            first_name="John",
            last_name="Doe",
            assigned_clinician=self.clinician_1,
        )

        # Clinician 2 & Patient 2
        self.clinician_2 = User.objects.create_user(
            username="clinician_two",
            email="dr.two@kneewise.ai",
            password="StrongPassword456!",
            first_name="Bob",
            last_name="Jones",
        )
        self.patient_2 = Patient.objects.create(
            patient_id="MRN-002",
            first_name="Jane",
            last_name="Miller",
            assigned_clinician=self.clinician_2,
        )

    def test_unauthenticated_requests_rejected(self):
        """Unauthenticated requests must return 401."""
        url = reverse("rehab:session_list_create")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)

        res_post = self.client.post(url, {"patient": self.patient_1.id})
        self.assertEqual(res_post.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_clinician_can_create_session_for_assigned_patient(self):
        """Clinician 1 can initiate a session for their assigned patient."""
        self.client.force_authenticate(user=self.clinician_1)
        url = reverse("rehab:session_list_create")
        payload = {
            "patient": self.patient_1.id,
            "exercise_name": "seated_knee_extension",
            "selected_leg": "right",
            "notes": "Post-op rehabilitation session 1",
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        self.assertIn("id", res.data)

        # Check DB state
        session = RehabilitationSession.objects.get(id=res.data["id"])
        self.assertEqual(session.clinician, self.clinician_1)
        self.assertEqual(session.patient, self.patient_1)
        self.assertEqual(session.status, SessionStatus.IN_PROGRESS)
        self.assertEqual(session.selected_leg, "right")

    def test_patient_isolation_clinician_cannot_create_session_for_unassigned_patient(self):
        """Clinician 2 must be rejected with 403 when trying to create session for Patient 1."""
        self.client.force_authenticate(user=self.clinician_2)
        url = reverse("rehab:session_list_create")
        payload = {
            "patient": self.patient_1.id,  # Owned by clinician 1
            "exercise_name": "seated_knee_extension",
            "selected_leg": "right",
        }
        res = self.client.post(url, payload)
        self.assertEqual(res.status_code, status.HTTP_403_FORBIDDEN)

    def test_patient_isolation_clinician_cannot_view_or_complete_other_clinician_session(self):
        """Clinician 2 cannot access or complete Clinician 1's session."""
        session = RehabilitationSession.objects.create(
            patient=self.patient_1,
            clinician=self.clinician_1,
            exercise_name="seated_knee_extension",
            selected_leg="right",
            status=SessionStatus.IN_PROGRESS,
        )

        self.client.force_authenticate(user=self.clinician_2)

        # GET detail
        detail_url = reverse("rehab:session_detail", kwargs={"pk": session.id})
        res_get = self.client.get(detail_url)
        self.assertEqual(res_get.status_code, status.HTTP_403_FORBIDDEN)

        # POST complete
        complete_url = reverse("rehab:session_complete", kwargs={"pk": session.id})
        complete_payload = {
            "total_reps": 10,
            "rejected_reps": 1,
            "set_count": 1,
            "avg_excursion_deg": 65.0,
            "best_excursion_deg": 70.0,
            "tracking_availability_pct": 95.0,
            "duration_sec": 45.0,
        }
        res_post = self.client.post(complete_url, complete_payload, format="json")
        self.assertEqual(res_post.status_code, status.HTTP_403_FORBIDDEN)

    def test_session_lifecycle_and_completion(self):
        """Clinician completes session with metrics and nested sets."""
        self.client.force_authenticate(user=self.clinician_1)

        # Create session
        session = RehabilitationSession.objects.create(
            patient=self.patient_1,
            clinician=self.clinician_1,
            exercise_name="seated_knee_extension",
            selected_leg="left",
            status=SessionStatus.IN_PROGRESS,
        )

        complete_url = reverse("rehab:session_complete", kwargs={"pk": session.id})
        complete_payload = {
            "total_reps": 12,
            "rejected_reps": 2,
            "set_count": 2,
            "avg_excursion_deg": 68.5,
            "best_excursion_deg": 75.0,
            "tracking_availability_pct": 98.2,
            "duration_sec": 120.5,
            "notes": "Patient demonstrated good extension without discomfort.",
            "sets": [
                {
                    "set_number": 1,
                    "reps_completed": 6,
                    "reps_rejected": 1,
                    "rest_duration_sec": 0.0,
                    "avg_excursion_deg": 67.0,
                    "max_extension_deg": 158.0,
                },
                {
                    "set_number": 2,
                    "reps_completed": 6,
                    "reps_rejected": 1,
                    "rest_duration_sec": 30.0,
                    "avg_excursion_deg": 70.0,
                    "max_extension_deg": 162.0,
                },
            ],
        }

        res = self.client.post(complete_url, complete_payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["status"], "COMPLETED")
        self.assertEqual(res.data["total_reps"], 12)
        self.assertEqual(res.data["rejected_reps"], 2)
        self.assertEqual(res.data["set_count"], 2)
        self.assertEqual(res.data["avg_excursion_deg"], 68.5)
        self.assertEqual(res.data["best_excursion_deg"], 75.0)
        self.assertEqual(len(res.data["sets"]), 2)

        # Verify DB records
        session.refresh_from_db()
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.sets.count(), 2)

    def test_clinician_session_list_filtering(self):
        """Session list returns only clinician's authorized sessions and filters by patient."""
        s1 = RehabilitationSession.objects.create(
            patient=self.patient_1,
            clinician=self.clinician_1,
            exercise_name="seated_knee_extension",
            selected_leg="right",
        )
        s2 = RehabilitationSession.objects.create(
            patient=self.patient_2,
            clinician=self.clinician_2,
            exercise_name="seated_knee_extension",
            selected_leg="right",
        )

        self.client.force_authenticate(user=self.clinician_1)
        url = reverse("rehab:session_list_create")
        res = self.client.get(url)
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        ids = [item["id"] for item in res.data["results"] if "results" in res.data] if isinstance(res.data, dict) and "results" in res.data else [item["id"] for item in res.data]
        self.assertIn(str(s1.id), ids)
        self.assertNotIn(str(s2.id), ids)

    def test_zero_metric_session_does_not_fabricate_metrics(self):
        """Verifies session completion calculates actual duration from timestamps and leaves 0-rep metrics as 0.0."""
        self.client.force_authenticate(user=self.clinician_1)
        session = RehabilitationSession.objects.create(
            patient=self.patient_1,
            clinician=self.clinician_1,
            exercise_name="seated_knee_extension",
            selected_leg="right",
            status=SessionStatus.IN_PROGRESS,
        )

        complete_url = reverse("rehab:session_complete", kwargs={"pk": session.id})
        payload = {
            "total_reps": 0,
            "rejected_reps": 0,
            "set_count": 1,
            "avg_excursion_deg": 0.0,
            "best_excursion_deg": 0.0,
            "tracking_availability_pct": 0.0,
            "duration_sec": 0.0,  # 0 passed -> backend must compute from actual timestamps
            "notes": "Session concluded with zero completed movements.",
        }
        res = self.client.post(complete_url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["total_reps"], 0)
        self.assertEqual(res.data["avg_excursion_deg"], 0.0)
        self.assertEqual(res.data["best_excursion_deg"], 0.0)
        self.assertGreater(res.data["duration_sec"], 0.0)  # Measured from DB start/end times!

    def test_set_level_aggregation_derives_session_totals(self):
        """Verifies that set excursions and reps are aggregated from actual set items."""
        self.client.force_authenticate(user=self.clinician_1)
        session = RehabilitationSession.objects.create(
            patient=self.patient_1,
            clinician=self.clinician_1,
            exercise_name="seated_knee_extension",
            selected_leg="right",
            status=SessionStatus.IN_PROGRESS,
        )

        complete_url = reverse("rehab:session_complete", kwargs={"pk": session.id})
        payload = {
            "total_reps": 0,  # Omitted -> backend derives from sets (4 + 6 = 10)
            "rejected_reps": 0,
            "set_count": 0,
            "avg_excursion_deg": 0.0,  # Omitted -> backend derives from sets ((50.0 + 60.0)/2 = 55.0)
            "best_excursion_deg": 0.0,  # Omitted -> backend derives max (165.0)
            "tracking_availability_pct": 92.5,
            "duration_sec": 45.0,
            "sets": [
                {
                    "set_number": 1,
                    "reps_completed": 4,
                    "reps_rejected": 0,
                    "rest_duration_sec": 0.0,
                    "avg_excursion_deg": 50.0,
                    "max_extension_deg": 155.0,
                },
                {
                    "set_number": 2,
                    "reps_completed": 6,
                    "reps_rejected": 1,
                    "rest_duration_sec": 30.0,
                    "avg_excursion_deg": 60.0,
                    "max_extension_deg": 165.0,
                },
            ],
        }
        res = self.client.post(complete_url, payload, format="json")
        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data["total_reps"], 10)
        self.assertEqual(res.data["set_count"], 2)
        self.assertEqual(res.data["avg_excursion_deg"], 55.0)
        self.assertEqual(res.data["best_excursion_deg"], 165.0)
