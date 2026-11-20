"""
rehab_cv/websocket_server.py - Real-Time Authenticated WebSocket CV Processor
Connects browser webcam frames directly to MediaPipe Pose, Angle Biomechanics,
Repetition Counter FSM, and Feedback Engine.
"""

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")
print("[*] Starting WebSocket Server Script...", flush=True)

import sys
import json
import base64
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import websockets

# Setup Django environment for JWT authentication & Patient isolation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "KneeWise" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
print("[*] Calling django.setup()...", flush=True)
django.setup()
print("[*] Django setup completed.", flush=True)

from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken
from asgiref.sync import sync_to_async
from screening.models import Patient
from rehab.models import RehabilitationSession
from rehab_cv.config import PoseConfig, ExerciseConfig
from rehab_cv.detection.pose_detector import PoseDetector
from rehab_cv.exercises.knee_extension import KneeExtensionExercise
from rehab_cv.feedback.feedback_engine import RehabilitationFeedbackEngine
from rehab_cv.feedback.session_analyzer import SessionAnalyzer

User = get_user_model()
PORT = int(os.environ.get("REHAB_WS_PORT", 8002))


@sync_to_async
def authenticate_token(token_str: str) -> Optional[Any]:
    """Validates JWT access token and returns User instance."""
    try:
        access_token = AccessToken(token_str)
        user_id = access_token.get("user_id")
        if str(user_id).isdigit():
            return User.objects.get(id=int(user_id))
        return User.objects.get(id=user_id)
    except Exception as e:
        print(f"[!] WebSocket auth failure: {e}", flush=True)
        return None


@sync_to_async
def verify_patient_authorization(user: Any, session_id: str) -> Optional[RehabilitationSession]:
    """Verifies that requesting user is authorized for the given rehabilitation session."""
    try:
        session = RehabilitationSession.objects.select_related("patient", "clinician").get(id=session_id)
        if user.is_staff or user.is_superuser:
            return session
        if session.clinician == user or session.patient.assigned_clinician == user:
            return session
        return None
    except Exception as e:
        print(f"[!] Session lookup error: {e}", flush=True)
        return None


class ClientCVSession:
    """Encapsulates a real-time CV session for a single connected client."""

    def __init__(self, selected_leg: str = "auto"):
        self.selected_leg = selected_leg.lower()
        self.detector = PoseDetector(PoseConfig(model_complexity=0))
        self.exercise = KneeExtensionExercise(ExerciseConfig(preferred_leg=self.selected_leg))
        self.feedback_engine = RehabilitationFeedbackEngine()
        self.analyzer = SessionAnalyzer(selected_leg=self.selected_leg)
        self.analyzer.start_session()

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs the full computer vision and biomechanics pipeline on a single frame."""
        h, w = frame_bgr.shape[:2]

        # 1. Pose detection
        detection = self.detector.process(frame_bgr)

        # 2. Knee extension biomechanics & FSM
        metrics = self.exercise.process(detection)

        # 3. Dynamic feedback evaluation
        feedback_event = self.feedback_engine.process(metrics)

        # 4. Session metric recording
        self.analyzer.record_frame(metrics)

        def _to_dict(leg):
            if not leg:
                return None
            h, k, a = leg.hip, leg.knee, leg.ankle
            return {
                "hip": {"x": round(h.x, 4), "y": round(h.y, 4), "visibility": round(h.visibility, 3)} if h else None,
                "knee": {"x": round(k.x, 4), "y": round(k.y, 4), "visibility": round(k.visibility, 3)} if k else None,
                "ankle": {"x": round(a.x, 4), "y": round(a.y, 4), "visibility": round(a.visibility, 3)} if a else None,
                "is_valid": leg.is_valid,
                "mean_visibility": round(leg.mean_visibility, 3)
            }

        left_dict = _to_dict(detection.left_leg)
        right_dict = _to_dict(detection.right_leg)
        landmarks_dict = right_dict if metrics.selected_leg == "right" else left_dict

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
            "timestamp": time.time(),
            "is_tracking_reliable": metrics.is_tracking_reliable,
            "landmark_confidence": round(metrics.landmark_confidence, 2),
            "rep_completed_this_frame": metrics.rep_completed_this_frame,
            "landmarks": landmarks_dict,
            "left_landmarks": left_dict,
            "right_landmarks": right_dict,
            "metrics": {
                "max_extension_in_rep": round(metrics.max_extension_in_rep, 1),
                "current_excursion": round(metrics.current_excursion, 1),
            },
        }


async def ws_handler(websocket):
    """Handles an individual client WebSocket connection."""
    client_addr = websocket.remote_address
    print(f"[+] WebSocket connection opened from {client_addr}")

    user = None
    session_record = None
    cv_session: Optional[ClientCVSession] = None

    try:
        # Step 1: Wait for authentication handshake message
        auth_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        auth_data = json.loads(auth_raw) if isinstance(auth_raw, str) else {}

        if auth_data.get("action") != "auth" or not auth_data.get("token") or not auth_data.get("session_id"):
            await websocket.send(json.dumps({"error": "Authentication required. Expected {'action': 'auth', 'token': '...', 'session_id': '...'}"}))
            await websocket.close(code=4001, reason="Unauthorized")
            return

        token = auth_data["token"]
        session_id = auth_data["session_id"]
        selected_leg = auth_data.get("selected_leg", "auto")

        user = await authenticate_token(token)
        if not user:
            await websocket.send(json.dumps({"error": "Invalid or expired JWT credentials."}))
            await websocket.close(code=4002, reason="Authentication failed")
            return

        session_record = await verify_patient_authorization(user, session_id)
        if not session_record:
            await websocket.send(json.dumps({"error": "Access denied: session not found or clinician not assigned to patient."}))
            await websocket.close(code=4003, reason="Permission denied")
            return

        # Instantiate dedicated CV pipeline for this session
        cv_session = ClientCVSession(selected_leg=selected_leg)
        await websocket.send(json.dumps({
            "status": "authenticated",
            "message": f"Connected as Dr. {user.username} for session {session_id}",
            "selected_leg": selected_leg,
        }))
        print(f"[+] Session {session_id} authorized for clinician {user.username}")

        # Step 2: Stream and process frames
        async for message in websocket:
            # Check for control messages
            if isinstance(message, str):
                try:
                    msg_obj = json.loads(message)
                    action = msg_obj.get("action")
                    if action == "change_leg":
                        new_leg = msg_obj.get("selected_leg", "auto").lower()
                        cv_session.exercise.leg_selector.preferred_leg = new_leg
                        if new_leg in ("left", "right"):
                            cv_session.exercise.leg_selector.active_side = new_leg
                        await websocket.send(json.dumps({"status": "leg_changed", "selected_leg": new_leg}))
                        continue
                    elif action == "reset":
                        cv_session.exercise.reset()
                        cv_session.feedback_engine.reset()
                        await websocket.send(json.dumps({"status": "reset_completed"}))
                        continue
                    elif action == "finish":
                        summary = cv_session.analyzer.finish_session()
                        await websocket.send(json.dumps({
                            "status": "session_finished",
                            "summary": {
                                "total_reps": summary.total_completed_reps,
                                "duration_sec": round(summary.duration_sec, 1),
                                "avg_excursion_deg": round(summary.avg_excursion_deg, 1),
                                "best_excursion_deg": round(summary.best_excursion_deg, 1),
                                "tracking_availability_pct": round(summary.tracking_availability_pct, 1),
                            }
                        }))
                        break
                    elif "image" in msg_obj:
                        # Base64 encoded frame
                        b64_str = msg_obj["image"]
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_str)
                    else:
                        continue
                except Exception as e:
                    await websocket.send(json.dumps({"error": f"Invalid JSON payload: {e}"}))
                    continue
            else:
                # Binary JPEG frame
                img_bytes = message

            # Decode frame
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send(json.dumps({"error": "Failed to decode image frame"}))
                continue

            # Run actual CV processing
            result = cv_session.process_frame(frame_bgr)
            await websocket.send(json.dumps(result))

    except asyncio.TimeoutError:
        print(f"[!] Auth timeout for {client_addr}")
        await websocket.close(code=4008, reason="Auth timeout")
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Connection closed for {client_addr}")
    except Exception as e:
        print(f"[!] Unexpected WebSocket error: {e}")
    finally:
        print(f"[-] Cleaned up session for {client_addr}")


async def main():
    print("=" * 70, flush=True)
    print(f"  KNEEWISE REHABILITATION CV WEBSOCKET SERVER — PORT {PORT}", flush=True)
    print("=" * 70, flush=True)
    print(f"[*] Binding WebSocket server to 127.0.0.1:{PORT}...", flush=True)
    async with websockets.serve(ws_handler, "127.0.0.1", PORT, max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=20):
        print(f"[+] WebSocket server running at ws://127.0.0.1:{PORT}", flush=True)
        await asyncio.Future()  # Keep running indefinitely


print("[*] Module loaded successfully, checking __name__...", flush=True)

if __name__ == "__main__":
    print("[*] Starting asyncio.run(main())...", flush=True)
    asyncio.run(main())
