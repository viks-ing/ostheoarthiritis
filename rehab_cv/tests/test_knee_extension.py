"""
rehab_cv/tests/test_knee_extension.py - Integration Tests for KneeExtensionExercise Engine
Tests:
- End-to-end processing of synthesized PoseDetectionResult frames
- Multi-repetition session metrics aggregation
- Leg switching and safe reset behavior
"""

import math
import pytest
from rehab_cv.exercises.knee_extension import KneeExtensionExercise, ExerciseState
from rehab_cv.detection.pose_detector import PoseDetectionResult, LegLandmarks, LandmarkPoint
from rehab_cv.config import ExerciseConfig, SmoothingConfig


def make_leg_result(side: str, angle_deg: float) -> PoseDetectionResult:
    """
    Constructs a PoseDetectionResult with synthetic left and right leg landmarks
    matching a specified knee angle.
    Knee is at (0.5, 0.5), Hip is at (0.2, 0.5).
    Ankle position is placed at distance 0.3 according to angle_deg.
    """
    # Deflection angle from horizontal
    deflection_rad = math.radians(180.0 - angle_deg)
    kx, ky = 0.5, 0.5
    hx, hy = 0.2, 0.5  # v1 = (-0.3, 0.0)
    ax = kx + 0.3 * math.cos(deflection_rad)
    ay = ky + 0.3 * math.sin(deflection_rad)

    hip = LandmarkPoint(hx, hy, 0.0, 0.95, 1.0, int(hx * 1000), int(hy * 1000))
    knee = LandmarkPoint(kx, ky, 0.0, 0.95, 1.0, int(kx * 1000), int(ky * 1000))
    ankle = LandmarkPoint(ax, ay, 0.0, 0.95, 1.0, int(ax * 1000), int(ay * 1000))

    active_leg = LegLandmarks(side, hip, knee, ankle, True, 0.95)
    inactive_leg = LegLandmarks("right" if side == "left" else "left", None, None, None, False, 0.0)

    return PoseDetectionResult(
        detected=True,
        raw_landmarks=None,
        left_leg=active_leg if side == "left" else inactive_leg,
        right_leg=inactive_leg if side == "left" else active_leg,
        frame_width=1280,
        frame_height=720,
    )


def test_knee_extension_end_to_end_single_rep():
    """Validates complete single-rep pipeline from detection results to session metrics."""
    ex_cfg = ExerciseConfig(
        flexion_threshold_deg=100.0,
        extension_threshold_deg=155.0,
        return_threshold_deg=115.0,
        min_excursion_deg=45.0,
        min_state_duration_sec=0.01,
        cooldown_sec=0.05,
        hysteresis_frames=1,
    )
    sm_cfg = SmoothingConfig(alpha=0.6)  # Fast alpha for synthetic test
    engine = KneeExtensionExercise(exercise_config=ex_cfg, smoothing_config=sm_cfg)

    # 1. Seated posture
    t = 1000.0
    dt = 0.05
    for _ in range(3):
        t += dt
        m = engine.process(make_leg_result("left", 90.0), current_time=t)
        assert m.selected_leg == "left"
        assert m.exercise_state == ExerciseState.WAITING

    # 2. Extending leg upwards to 165°
    for angle in [110.0, 130.0, 145.0, 160.0, 165.0, 165.0, 165.0]:
        t += dt
        m = engine.process(make_leg_result("left", angle), current_time=t)

    assert m.exercise_state == ExerciseState.EXTENDED
    assert m.max_extension_in_rep >= 160.0
    assert m.repetition_count == 0

    # 3. Returning back down to 90°
    for angle in [145.0, 125.0, 110.0, 95.0, 90.0]:
        t += dt
        m = engine.process(make_leg_result("left", angle), current_time=t)

    assert m.repetition_count == 1
    assert m.exercise_state == ExerciseState.WAITING

    session = engine.get_session_metrics()
    assert session.total_reps == 1
    assert session.average_excursion >= 65.0


def test_knee_extension_manual_leg_switch():
    """Manual leg switch resets current repetition state cleanly."""
    engine = KneeExtensionExercise()

    # Left leg in motion
    engine.process(make_leg_result("left", 140.0))
    assert engine.leg_selector.active_side == "left"

    # User presses 'L' to switch leg
    new_leg = engine.switch_leg()
    assert new_leg == "right"
    assert engine.counter.state == ExerciseState.WAITING
    assert engine.counter.repetition_count == 0
