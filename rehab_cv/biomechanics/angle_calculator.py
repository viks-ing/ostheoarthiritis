"""
rehab_cv/biomechanics/angle_calculator.py - Robust 2D Knee-Angle Vector Biomechanics
Calculates knee flexion/extension angle using 2D hip-knee-ankle vector geometry.
Handles zero-length vectors, colinearity, NaN values, visibility gating,
and implements hysteresis-governed leg selection that prevents side-switching mid-repetition.

Important Biomechanical & Perspective Notice:
This module calculates 2D planar projection angles from camera pixel/normalized coordinates.
2D camera geometry is inherently sensitive to camera perspective, parallax, and foreshortening.
These calculations are designed for rehabilitation exercise feedback and rep counting;
they do NOT constitute clinically calibrated 3D goniometric joint angles.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..detection.pose_detector import LandmarkPoint, LegLandmarks


@dataclass
class AngleCalculationResult:
    """Encapsulates calculated knee angle, limb side, and reliability flags."""
    angle_deg: Optional[float]
    side: str  # "left" or "right"
    is_valid: bool
    confidence: float
    error_reason: Optional[str] = None


def calculate_knee_angle_2d(
    hip: Union[LandmarkPoint, Tuple[float, float]],
    knee: Union[LandmarkPoint, Tuple[float, float]],
    ankle: Union[LandmarkPoint, Tuple[float, float]],
    min_confidence: float = 0.50,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Computes 2D planar knee angle using vectors:
        v1 = Hip - Knee
        v2 = Ankle - Knee
    
    Formula:
        cos(theta) = (v1 . v2) / (|v1| * |v2|)
        theta = arccos(cos(theta)) in degrees [0°, 180°]

    Biomechanical convention:
        - Straight extended leg: ~180° (v1 and v2 point in opposite directions).
        - 90° seated flexion: ~90° (v1 and v2 are perpendicular).

    Returns:
        (angle_deg, error_reason)
        If valid: (float_in_range_0_180, None)
        If invalid: (None, "description of issue")
    """
    # 1. Extract coordinates and check confidence if LandmarkPoints are supplied
    if isinstance(hip, LandmarkPoint):
        if hip.visibility < min_confidence:
            return None, f"Hip landmark visibility ({hip.visibility:.2f}) below threshold ({min_confidence:.2f})"
        hx, hy = hip.x, hip.y
    else:
        hx, hy = float(hip[0]), float(hip[1])

    if isinstance(knee, LandmarkPoint):
        if knee.visibility < min_confidence:
            return None, f"Knee landmark visibility ({knee.visibility:.2f}) below threshold ({min_confidence:.2f})"
        kx, ky = knee.x, knee.y
    else:
        kx, ky = float(knee[0]), float(knee[1])

    if isinstance(ankle, LandmarkPoint):
        if ankle.visibility < min_confidence:
            return None, f"Ankle landmark visibility ({ankle.visibility:.2f}) below threshold ({min_confidence:.2f})"
        ax, ay = ankle.x, ankle.y
    else:
        ax, ay = float(ankle[0]), float(ankle[1])

    # 2. Check for NaN or Infinite values
    for val in (hx, hy, kx, ky, ax, ay):
        if math.isnan(val) or math.isinf(val):
            return None, "Coordinates contain NaN or infinite values"

    # 3. Vector calculations:
    # v1 = Hip - Knee
    # v2 = Ankle - Knee
    v1_x = hx - kx
    v1_y = hy - ky
    v2_x = ax - kx
    v2_y = ay - ky

    # 4. Magnitudes
    mag1 = math.hypot(v1_x, v1_y)
    mag2 = math.hypot(v2_x, v2_y)

    # 5. Guard against zero-length vectors (overlapping/identical joints)
    EPSILON = 1e-7
    if mag1 < EPSILON:
        return None, "Degenerate vector: Hip and Knee coordinates are coincident (distance ~ 0)"
    if mag2 < EPSILON:
        return None, "Degenerate vector: Knee and Ankle coordinates are coincident (distance ~ 0)"

    # 6. Dot product and cosine calculation
    dot_product = (v1_x * v2_x) + (v1_y * v2_y)
    cos_theta = dot_product / (mag1 * mag2)

    # 7. Numerical clipping to prevent domain errors in arccos [-1.0, 1.0]
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # 8. Compute angle in degrees
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return float(round(angle_deg, 2)), None


class LegSelectionManager:
    """
    Manages leg selection with strict hysteresis and mid-rep lock:
    - Supports "auto", "left", "right" preferences.
    - Evaluates visibility of left vs right limbs.
    - STRICT CONSTRAINT: Never silently switches active leg during an in-progress repetition.
      Side-switching is only permitted when exercise state is WAITING or when tracking is lost.
    """

    def __init__(self, preferred_leg: str = "auto"):
        self.preferred_leg = preferred_leg.lower()
        self.active_side: str = "left" if self.preferred_leg == "left" else "right" if self.preferred_leg == "right" else "unknown"
        self._consecutive_side_recommendations: int = 0
        self._pending_side: Optional[str] = None

    def select_leg(
        self,
        left_leg: LegLandmarks,
        right_leg: LegLandmarks,
        can_switch: bool = True,
    ) -> str:
        """
        Determines the active leg for tracking.
        Args:
            left_leg: LegLandmarks for left limb
            right_leg: LegLandmarks for right limb
            can_switch: True if exercise is in WAITING state or tracking lost;
                        False if mid-repetition (EXTENDING/EXTENDED/RETURNING)
        Returns:
            "left" or "right"
        """
        # If user explicitly configured "left" or "right", obey user preference
        if self.preferred_leg in ("left", "right"):
            self.active_side = self.preferred_leg
            return self.active_side

        # Evaluate leg visibility
        left_valid = left_leg.is_valid
        right_valid = right_leg.is_valid
        left_vis = left_leg.mean_visibility
        right_vis = right_leg.mean_visibility

        # Determine current recommended side based on visibility
        if left_valid and not right_valid:
            recommended = "left"
        elif right_valid and not left_valid:
            recommended = "right"
        elif left_valid and right_valid:
            # Significant visibility difference favors the clearer side (15% hysteresis margin)
            if left_vis > right_vis + 0.15:
                recommended = "left"
            elif right_vis > left_vis + 0.15:
                recommended = "right"
            else:
                # Maintain current side to avoid oscillating
                recommended = self.active_side if self.active_side in ("left", "right") else "left"
        else:
            # Neither is fully valid; pick whichever has higher mean visibility
            recommended = "left" if left_vis >= right_vis else "right"

        # If currently unknown (first frame), adopt immediately
        if self.active_side == "unknown":
            self.active_side = recommended
            return self.active_side

        # If mid-rep, forbid switching
        if not can_switch:
            return self.active_side

        # If switch is allowed, require 5 consecutive frames of recommendation before switching
        if recommended != self.active_side:
            if self._pending_side == recommended:
                self._consecutive_side_recommendations += 1
                if self._consecutive_side_recommendations >= 5:
                    self.active_side = recommended
                    self._pending_side = None
                    self._consecutive_side_recommendations = 0
            else:
                self._pending_side = recommended
                self._consecutive_side_recommendations = 1
        else:
            self._pending_side = None
            self._consecutive_side_recommendations = 0

        return self.active_side

    def force_switch(self) -> str:
        """Manually toggles active leg when user presses 'L' key."""
        self.active_side = "right" if self.active_side == "left" else "left"
        self._pending_side = None
        self._consecutive_side_recommendations = 0
        return self.active_side
