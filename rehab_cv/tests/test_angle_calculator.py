"""
rehab_cv/tests/test_angle_calculator.py - Unit Tests for 2D Knee Angle Biomechanics
Tests:
- Straight leg approximately 180°
- Right-angle bend approximately 90°
- Acute angle ~45° and obtuse angle ~135°
- Zero-length vectors (identical points)
- Invalid coordinates (NaN, Inf)
- Low-confidence visibility rejection
- LegSelectionManager mid-rep switching lock
"""

import math
import pytest
from rehab_cv.biomechanics.angle_calculator import (
    calculate_knee_angle_2d,
    LegSelectionManager,
)
from rehab_cv.detection.pose_detector import LandmarkPoint, LegLandmarks


def make_landmark(x: float, y: float, visibility: float = 0.95) -> LandmarkPoint:
    """Helper to generate dummy LandmarkPoint."""
    return LandmarkPoint(
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
        presence=1.0,
        pixel_x=int(x * 1000),
        pixel_y=int(y * 1000),
    )


def test_straight_leg_approaches_180_degrees():
    """
    Collinear horizontal arrangement:
    Hip (0.2, 0.5) ---- Knee (0.5, 0.5) ---- Ankle (0.8, 0.5)
    Vectors:
      v1 = Hip - Knee = (-0.3, 0.0)
      v2 = Ankle - Knee = (0.3, 0.0)
    Angle between them should be exactly 180.0°.
    """
    hip = (0.2, 0.5)
    knee = (0.5, 0.5)
    ankle = (0.8, 0.5)

    angle, err = calculate_knee_angle_2d(hip, knee, ankle)
    assert err is None
    assert angle is not None
    assert pytest.approx(angle, abs=0.1) == 180.0

    # Also test with LandmarkPoint objects
    lm_hip = make_landmark(0.2, 0.5)
    lm_knee = make_landmark(0.5, 0.5)
    lm_ankle = make_landmark(0.8, 0.5)
    angle_lm, err_lm = calculate_knee_angle_2d(lm_hip, lm_knee, lm_ankle)
    assert err_lm is None
    assert pytest.approx(angle_lm, abs=0.1) == 180.0


def test_right_angle_bend_approaches_90_degrees():
    """
    Right angle arrangement (seated position):
    Hip (0.2, 0.5) ---- Knee (0.5, 0.5)
                            |
                            |
                        Ankle (0.5, 0.8)
    v1 = Hip - Knee = (-0.3, 0.0)
    v2 = Ankle - Knee = (0.0, 0.3)
    Dot product = 0 -> angle = 90.0°
    """
    hip = (0.2, 0.5)
    knee = (0.5, 0.5)
    ankle = (0.5, 0.8)

    angle, err = calculate_knee_angle_2d(hip, knee, ankle)
    assert err is None
    assert angle is not None
    assert pytest.approx(angle, abs=0.1) == 90.0


def test_intermediate_angles_45_and_135():
    """Test 45-degree acute bend and 135-degree obtuse extension."""
    # 45 degrees:
    # v1 = (-1, 0)
    # v2 = (-1, -1) -> dot = 1, mag1 = 1, mag2 = sqrt(2) -> cos = 1/sqrt(2) -> 45 deg
    hip_45 = (0.0, 0.5)
    knee_45 = (1.0, 0.5)
    ankle_45 = (0.0, -0.5)  # v2 = (-1.0, -1.0)
    angle_45, err = calculate_knee_angle_2d(hip_45, knee_45, ankle_45)
    assert err is None
    assert pytest.approx(angle_45, abs=0.1) == 45.0

    # 135 degrees:
    # v1 = (-1, 0)
    # v2 = (1, 1) -> dot = -1, mag1 = 1, mag2 = sqrt(2) -> cos = -1/sqrt(2) -> 135 deg
    ankle_135 = (2.0, 1.5)  # v2 = (1.0, 1.0)
    angle_135, err = calculate_knee_angle_2d(hip_45, knee_45, ankle_135)
    assert err is None
    assert pytest.approx(angle_135, abs=0.1) == 135.0


def test_zero_length_vectors():
    """Zero-length vector (e.g. hip == knee or knee == ankle) returns None and error message."""
    # Hip and Knee coincident
    hip = (0.5, 0.5)
    knee = (0.5, 0.5)
    ankle = (0.8, 0.5)

    angle, err = calculate_knee_angle_2d(hip, knee, ankle)
    assert angle is None
    assert "Degenerate vector" in err

    # Knee and Ankle coincident
    angle2, err2 = calculate_knee_angle_2d((0.2, 0.5), (0.5, 0.5), (0.5, 0.5))
    assert angle2 is None
    assert "Degenerate vector" in err2


def test_nan_and_infinite_coordinates():
    """Coordinates containing NaN or Infinite values are safely caught and rejected."""
    angle_nan, err_nan = calculate_knee_angle_2d((float("nan"), 0.5), (0.5, 0.5), (0.8, 0.5))
    assert angle_nan is None
    assert "NaN" in err_nan

    angle_inf, err_inf = calculate_knee_angle_2d((0.2, 0.5), (float("inf"), 0.5), (0.8, 0.5))
    assert angle_inf is None
    assert "infinite" in err_inf


def test_low_confidence_rejection():
    """Landmarks with confidence below min_confidence are rejected."""
    hip = make_landmark(0.2, 0.5, visibility=0.9)
    knee = make_landmark(0.5, 0.5, visibility=0.25)  # Low visibility
    ankle = make_landmark(0.8, 0.5, visibility=0.9)

    angle, err = calculate_knee_angle_2d(hip, knee, ankle, min_confidence=0.50)
    assert angle is None
    assert "visibility" in err
    assert "0.25" in err


def test_leg_selection_manager_mid_rep_lock():
    """
    LegSelectionManager must NOT switch active tracking side mid-repetition
    when can_switch=False, even if the other leg suddenly becomes more visible.
    """
    manager = LegSelectionManager(preferred_leg="auto")

    # Initial state: Left leg is clear, Right leg is occluded
    left_leg_good = LegLandmarks("left", make_landmark(0.2, 0.5), make_landmark(0.5, 0.5), make_landmark(0.8, 0.5), True, 0.95)
    right_leg_poor = LegLandmarks("right", make_landmark(0.2, 0.5, 0.2), make_landmark(0.5, 0.5, 0.2), make_landmark(0.8, 0.5, 0.2), False, 0.20)

    selected = manager.select_leg(left_leg_good, right_leg_poor, can_switch=True)
    assert selected == "left"

    # Now simulate mid-repetition (can_switch=False):
    # Left leg temporarily becomes occluded and Right leg becomes visible
    left_leg_poor = LegLandmarks("left", make_landmark(0.2, 0.5, 0.2), make_landmark(0.5, 0.5, 0.2), make_landmark(0.8, 0.5, 0.2), False, 0.20)
    right_leg_good = LegLandmarks("right", make_landmark(0.2, 0.5), make_landmark(0.5, 0.5), make_landmark(0.8, 0.5), True, 0.95)

    # Must stay on "left" because can_switch is False!
    mid_rep_selected = manager.select_leg(left_leg_poor, right_leg_good, can_switch=False)
    assert mid_rep_selected == "left", "Leg selection switched mid-repetition, violating strict requirement!"

    # Once repetition completes (can_switch=True) and condition persists for 5 frames, it switches
    for _ in range(5):
        end_rep_selected = manager.select_leg(left_leg_poor, right_leg_good, can_switch=True)
    assert end_rep_selected == "right"
