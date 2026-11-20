"""
rehab_cv/tests/test_session_analyzer.py - Unit Tests for Session Analytics & Multi-Set Tracking
Tests:
- Frame tracking continuity and availability percentage
- Angular excursion aggregation (min, max, average excursion, best excursion)
- Multi-set lifecycle, rest intervals, and rep counting
- Summary serialization
"""

import pytest
from rehab_cv.feedback.session_analyzer import (
    SessionAnalyzer,
    SetMetrics,
    SessionSummaryMetrics,
)
from rehab_cv.exercises.knee_extension import ExerciseFrameMetrics, ExerciseState
from rehab_cv.exercises.repetition_counter import RepetitionData


def test_set_metrics_duration_and_dict():
    s = SetMetrics(
        set_number=1,
        reps_completed=10,
        reps_rejected=1,
        start_time=100.0,
        end_time=145.5,
        rest_duration_sec=30.0,
        avg_excursion_deg=62.5,
        max_extension_deg=160.0,
    )
    assert s.duration_sec() == 45.5
    d = s.to_dict()
    assert d["set_number"] == 1
    assert d["reps_completed"] == 10
    assert d["reps_rejected"] == 1
    assert d["duration_sec"] == 45.5
    assert d["rest_duration_sec"] == 30.0
    assert d["avg_excursion_deg"] == 62.5
    assert d["max_extension_deg"] == 160.0


def test_session_analyzer_frame_tracking_and_availability():
    analyzer = SessionAnalyzer(session_id="test-session-123", selected_leg="right")
    analyzer.start_session(now=0.0)

    # Record 8 tracked frames and 2 untracked frames
    for i in range(8):
        m = ExerciseFrameMetrics(
            selected_leg="right",
            raw_angle=90.0 + i * 5.0,
            smoothed_angle=90.0 + i * 5.0,
            repetition_count=0,
            exercise_state=ExerciseState.EXTENDING,
            max_extension_in_rep=90.0 + i * 5.0,
            current_excursion=i * 5.0,
            is_tracking_reliable=True,
            landmark_confidence=0.9,
            rep_completed_this_frame=False,
            last_completed_rep=None,
        )
        analyzer.record_frame(m, now=i * 0.1)

    for i in range(2):
        m_lost = ExerciseFrameMetrics(
            selected_leg="right",
            raw_angle=None,
            smoothed_angle=None,
            repetition_count=0,
            exercise_state=ExerciseState.EXTENDING,
            max_extension_in_rep=125.0,
            current_excursion=0.0,
            is_tracking_reliable=False,
            landmark_confidence=0.2,
            rep_completed_this_frame=False,
            last_completed_rep=None,
        )
        analyzer.record_frame(m_lost, now=0.8 + i * 0.1)

    summary = analyzer.finish_session(now=1.5)
    assert summary.total_frames == 10
    assert summary.tracked_frames == 8
    assert summary.tracking_availability_pct == 80.0
    assert summary.min_observed_angle == 90.0
    assert summary.max_observed_angle == 125.0


def test_session_analyzer_multi_set_and_reps():
    analyzer = SessionAnalyzer(session_id="multi-set-test")
    analyzer.start_session(now=100.0)

    # Set 1: 5 reps
    rep1 = RepetitionData(
        rep_number=1,
        side="right",
        start_angle=85.0,
        max_extension_angle=155.0,
        return_angle=90.0,
        excursion_deg=70.0,
        duration_sec=2.0,
        timestamp=103.0,
    )
    m_rep1 = ExerciseFrameMetrics(
        selected_leg="right",
        raw_angle=90.0,
        smoothed_angle=90.0,
        repetition_count=1,
        exercise_state=ExerciseState.WAITING,
        max_extension_in_rep=155.0,
        current_excursion=0.0,
        is_tracking_reliable=True,
        landmark_confidence=0.95,
        rep_completed_this_frame=True,
        last_completed_rep=rep1,
    )
    analyzer.record_frame(m_rep1, now=103.0)

    # Finish Set 1 at t=120
    analyzer.finish_set(reps_completed=5, reps_rejected=1, avg_excursion=68.0, max_extension=158.0, now=120.0)

    # Rest for 30 seconds (until t=150)
    analyzer.start_set(set_number=2, now=150.0)
    assert analyzer.current_set.rest_duration_sec == 30.0

    # Finish Set 2 at t=180
    analyzer.finish_set(reps_completed=6, reps_rejected=0, avg_excursion=72.0, max_extension=162.0, now=180.0)

    # Finish Session
    summary = analyzer.finish_session(now=185.0)

    assert summary.total_sets == 2
    assert summary.total_completed_reps == 11
    assert summary.total_rejected_reps == 1
    assert summary.duration_sec == 85.0
    assert summary.best_excursion_deg == 70.0
    assert len(summary.sets) == 2
    assert summary.sets[0].rest_duration_sec == 0.0
    assert summary.sets[1].rest_duration_sec == 30.0


def test_empty_session_never_fabricates_metrics():
    """Verifies that an unobserved or empty session yields explicit nulls/zeros, never placeholder metrics."""
    analyzer = SessionAnalyzer(session_id="empty-audit-session")
    analyzer.start_session(now=50.0)
    summary = analyzer.finish_session(now=52.5)

    assert summary.total_frames == 0
    assert summary.tracked_frames == 0
    assert summary.tracking_availability_pct == 0.0
    assert summary.total_completed_reps == 0
    assert summary.total_rejected_reps == 0
    assert summary.avg_excursion_deg == 0.0
    assert summary.best_excursion_deg == 0.0
    assert summary.min_observed_angle is None
    assert summary.max_observed_angle is None
    assert summary.duration_sec == pytest.approx(2.5, 0.01)


def test_excursions_calculated_dynamically_from_actual_measurements():
    """Verifies that average and best excursions are strictly derived from real observation records."""
    analyzer = SessionAnalyzer(session_id="dynamic-metrics-session")
    analyzer.start_session(now=10.0)

    # 3 distinct reps with varying excursions
    excursions = [52.4, 63.8, 71.2]
    for idx, exc in enumerate(excursions):
        rep = RepetitionData(
            rep_number=idx + 1,
            side="right",
            start_angle=90.0,
            max_extension_angle=90.0 + exc,
            return_angle=90.0,
            excursion_deg=exc,
            duration_sec=2.1,
            timestamp=12.0 + idx * 3.0,
        )
        m = ExerciseFrameMetrics(
            selected_leg="right",
            raw_angle=90.0 + exc,
            smoothed_angle=90.0 + exc,
            repetition_count=idx + 1,
            exercise_state=ExerciseState.WAITING,
            max_extension_in_rep=90.0 + exc,
            current_excursion=exc,
            is_tracking_reliable=True,
            landmark_confidence=0.92,
            rep_completed_this_frame=True,
            last_completed_rep=rep,
        )
        analyzer.record_frame(m, now=12.0 + idx * 3.0)

    summary = analyzer.finish_session(now=25.0)

    expected_avg = sum(excursions) / len(excursions)
    expected_best = max(excursions)

    assert summary.total_completed_reps == 3
    assert summary.avg_excursion_deg == pytest.approx(expected_avg, 0.1)
    assert summary.best_excursion_deg == pytest.approx(expected_best, 0.1)
    assert summary.duration_sec == pytest.approx(15.0, 0.01)
