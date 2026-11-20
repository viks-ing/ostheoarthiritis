"""
rehab_cv/visualization/overlay.py - Real-Time OpenCV HUD & Biomechanical Overlay
Renders high-visibility skeleton connections, joint highlights, dynamic angle arcs,
state banners, repetition counters, and tracking status indications.
"""

import time
import math
from typing import Optional, Tuple, Any
import cv2
import numpy as np

from ..config import OverlayConfig
from ..detection.pose_detector import PoseDetectionResult, LegLandmarks, LandmarkPoint
from ..exercises.knee_extension import ExerciseFrameMetrics, ExerciseState


class VisualOverlayRenderer:
    """
    Renders professional clinical HUD overlays on OpenCV frames.
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        self._rep_completion_flash_time: float = 0.0
        self._flash_duration_sec: float = 0.75

    def render(
        self,
        frame: np.ndarray,
        detection: PoseDetectionResult,
        metrics: ExerciseFrameMetrics,
        fps: float = 0.0,
        feedback_event: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Draws skeleton, knee angle, and HUD metrics onto a copy of the input frame.
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Check if rep was completed to trigger celebratory visual flash
        if metrics.rep_completed_this_frame:
            self._rep_completion_flash_time = time.time()

        is_flashing = (time.time() - self._rep_completion_flash_time) < self._flash_duration_sec

        # 1. Draw Skeleton and Limbs
        self._draw_limbs(output, detection, metrics.selected_leg)

        # 2. Draw Knee Angle Annotation at Knee Joint
        active_leg = detection.left_leg if metrics.selected_leg == "left" else detection.right_leg
        if active_leg.is_valid and active_leg.knee is not None:
            self._draw_knee_angle_display(output, active_leg, metrics)

        # 3. Draw Upper HUD Status Panel
        self._draw_top_hud(output, metrics, is_flashing)

        # 4. Draw Feedback Banner (if provided)
        if feedback_event is not None:
            self._draw_feedback_banner(output, feedback_event)

        # 5. Draw Lower Information / Key Control Bar
        self._draw_bottom_bar(output, metrics, fps)

        return output

    def _draw_feedback_banner(self, frame: np.ndarray, event: Any):
        """Draws prominent feedback guidance banner below the top HUD."""
        msg = getattr(event, "message", str(event))
        sev = getattr(event, "severity", None)
        sev_name = getattr(sev, "name", str(sev)) if sev else "INFO"

        color_map = {
            "INFO": (200, 200, 200),
            "GUIDANCE": (240, 200, 80),   # Cyan/Gold
            "WARNING": (50, 165, 255),    # Orange/Amber
            "SUCCESS": (80, 220, 80),     # Bright Green
            "ALERT": (60, 60, 240),       # Red
        }
        text_color = color_map.get(sev_name, (220, 220, 220))

        width = frame.shape[1]
        banner_h = 36
        y_start = 105

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, y_start), (width - 20, y_start + banner_h), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (20, y_start), (width - 20, y_start + banner_h), text_color, 1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(msg, self.config.font_face, 0.55, 1)
        cx = width // 2
        cv2.putText(
            frame,
            msg,
            (cx - (tw // 2), y_start + 24),
            self.config.font_face,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_limbs(
        self,
        frame: np.ndarray,
        detection: PoseDetectionResult,
        selected_side: str,
    ):
        """Draws hip-knee-ankle lines and circular joints for both legs."""
        for side, leg in [("left", detection.left_leg), ("right", detection.right_leg)]:
            if not leg.is_valid or leg.hip is None or leg.knee is None or leg.ankle is None:
                continue

            is_selected = (side == selected_side)

            # Colors
            if is_selected:
                limb_color = self.config.color_accent_cyan
                joint_color = self.config.color_success_green
                thickness = self.config.active_skeleton_thickness
                radius = self.config.active_joint_radius
            else:
                limb_color = self.config.color_unselected_leg
                joint_color = (130, 130, 130)
                thickness = self.config.skeleton_thickness
                radius = self.config.joint_radius

            hip_pt = leg.hip.as_pixel_tuple()
            knee_pt = leg.knee.as_pixel_tuple()
            ankle_pt = leg.ankle.as_pixel_tuple()

            # Thigh line: Hip to Knee
            cv2.line(frame, hip_pt, knee_pt, limb_color, thickness, cv2.LINE_AA)
            # Shank line: Knee to Ankle
            cv2.line(frame, knee_pt, ankle_pt, limb_color, thickness, cv2.LINE_AA)

            # Joints
            for pt in [hip_pt, knee_pt, ankle_pt]:
                cv2.circle(frame, pt, radius, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, radius + 2, (20, 20, 20), 1, cv2.LINE_AA)

            if is_selected:
                # Extra glowing concentric ring around the active knee
                cv2.circle(frame, knee_pt, radius + 6, limb_color, 2, cv2.LINE_AA)

    def _draw_knee_angle_display(
        self,
        frame: np.ndarray,
        leg: LegLandmarks,
        metrics: ExerciseFrameMetrics,
    ):
        """Draws the dynamic knee angle text badge and angular arc."""
        kx, ky = leg.knee.as_pixel_tuple()
        angle = metrics.smoothed_angle

        if angle is None:
            return

        # Determine color based on extension progression
        if angle >= 155.0:
            badge_color = self.config.color_success_green  # Terminal extension
        elif angle >= 120.0:
            badge_color = self.config.color_accent_cyan    # Extending upwards
        else:
            badge_color = self.config.color_warning_amber   # Seated flexion

        # Angle text label
        text = f"{int(round(angle))} deg"
        (t_w, t_h), _ = cv2.getTextSize(text, self.config.font_face, 0.7, 2)

        # Offset badge slightly outward from knee
        offset_x = 25 if leg.side == "left" else -t_w - 35
        bx = max(10, min(frame.shape[1] - t_w - 20, kx + offset_x))
        by = max(t_h + 10, min(frame.shape[0] - 20, ky))

        # Rounded background box
        cv2.rectangle(
            frame,
            (bx - 6, by - t_h - 6),
            (bx + t_w + 6, by + 6),
            (20, 20, 25),
            -1,
        )
        cv2.rectangle(
            frame,
            (bx - 6, by - t_h - 6),
            (bx + t_w + 6, by + 6),
            badge_color,
            2,
        )
        cv2.putText(
            frame,
            text,
            (bx, by),
            self.config.font_face,
            0.7,
            self.config.color_text_primary,
            2,
            cv2.LINE_AA,
        )

        # Raw angle subtitle if debugging
        if metrics.raw_angle is not None:
            raw_text = f"raw: {metrics.raw_angle:.1f}"
            cv2.putText(
                frame,
                raw_text,
                (bx, by + 18),
                self.config.font_face,
                0.4,
                self.config.color_text_muted,
                1,
                cv2.LINE_AA,
            )

    def _draw_top_hud(
        self,
        frame: np.ndarray,
        metrics: ExerciseFrameMetrics,
        is_flashing: bool,
    ):
        """Draws top semi-transparent HUD banner with reps, state, and leg selection."""
        width = frame.shape[1]
        hud_h = 100

        # Semi-transparent overlay panel
        overlay = frame.copy()
        panel_color = (30, 80, 30) if is_flashing else self.config.color_bg_panel
        cv2.rectangle(overlay, (0, 0), (width, hud_h), panel_color, -1)
        cv2.addWeighted(overlay, self.config.hud_alpha, frame, 1 - self.config.hud_alpha, 0, frame)

        # Title
        cv2.putText(
            frame,
            "KNEEWISE REHABILITATION",
            (20, 30),
            self.config.font_face,
            0.6,
            self.config.color_accent_cyan,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Seated Knee Extension",
            (20, 52),
            self.config.font_face,
            0.5,
            self.config.color_text_muted,
            1,
            cv2.LINE_AA,
        )

        # Selected Leg Badge
        leg_str = f"LEG: {metrics.selected_leg.upper()}"
        cv2.putText(
            frame,
            leg_str,
            (20, 80),
            self.config.font_face,
            0.55,
            self.config.color_text_primary,
            1,
            cv2.LINE_AA,
        )

        # Central Exercise State Pill
        state_str = metrics.exercise_state.value
        if metrics.exercise_state == ExerciseState.WAITING:
            state_color = self.config.color_warning_amber
        elif metrics.exercise_state == ExerciseState.EXTENDING:
            state_color = self.config.color_accent_cyan
        elif metrics.exercise_state == ExerciseState.EXTENDED:
            state_color = self.config.color_success_green
        else:  # RETURNING
            state_color = (220, 120, 200)  # Magenta/Purple

        (sw, sh), _ = cv2.getTextSize(state_str, self.config.font_face, 0.8, 2)
        center_x = width // 2
        pill_x1 = center_x - (sw // 2) - 16
        pill_x2 = center_x + (sw // 2) + 16
        cv2.rectangle(frame, (pill_x1, 25), (pill_x2, 75), state_color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            state_str,
            (center_x - (sw // 2), 58),
            self.config.font_face,
            0.8,
            state_color,
            2,
            cv2.LINE_AA,
        )

        # Tracking Status Pill (below state pill)
        if metrics.is_tracking_reliable:
            trk_txt = f"TRACKING OK ({int(metrics.landmark_confidence * 100)}%)"
            trk_color = self.config.color_success_green
        else:
            trk_txt = "POOR TRACKING / OCCLUDED"
            trk_color = self.config.color_danger_red

        cv2.putText(
            frame,
            trk_txt,
            (center_x - (sw // 2) - 10, 92),
            self.config.font_face,
            0.4,
            trk_color,
            1,
            cv2.LINE_AA,
        )

        # Right: Repetition Counter (Large numbers)
        rep_label = "REPS"
        rep_val = str(metrics.repetition_count)
        (rw, rh), _ = cv2.getTextSize(rep_val, self.config.font_face, 1.8, 3)
        rx = width - 110

        cv2.putText(
            frame,
            rep_label,
            (rx, 30),
            self.config.font_face,
            0.55,
            self.config.color_text_muted,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            rep_val,
            (rx, 80),
            self.config.font_face,
            1.6,
            self.config.color_success_green if is_flashing else self.config.color_text_primary,
            3,
            cv2.LINE_AA,
        )

    def _draw_bottom_bar(
        self,
        frame: np.ndarray,
        metrics: ExerciseFrameMetrics,
        fps: float,
    ):
        """Draws bottom bar with ROM excursion, peak angle, FPS, and hotkey hints."""
        height, width = frame.shape[:2]
        bar_h = 40
        y_top = height - bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (width, height), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Metrics: Peak Extension & Excursion
        metrics_str = f"Peak: {metrics.max_extension_in_rep:.0f} deg   ROM: {metrics.current_excursion:.0f} deg"
        cv2.putText(
            frame,
            metrics_str,
            (20, height - 14),
            self.config.font_face,
            0.5,
            self.config.color_accent_cyan,
            1,
            cv2.LINE_AA,
        )

        # Hotkeys in center
        keys_str = "[Q] Quit    [R] Reset Reps    [L] Switch Leg"
        (kw, _), _ = cv2.getTextSize(keys_str, self.config.font_face, 0.45, 1)
        cv2.putText(
            frame,
            keys_str,
            ((width - kw) // 2, height - 14),
            self.config.font_face,
            0.45,
            self.config.color_text_muted,
            1,
            cv2.LINE_AA,
        )

        # FPS on right
        fps_str = f"{fps:.1f} FPS"
        cv2.putText(
            frame,
            fps_str,
            (width - 80, height - 14),
            self.config.font_face,
            0.45,
            self.config.color_success_green if fps >= 20 else self.config.color_warning_amber,
            1,
            cv2.LINE_AA,
        )
