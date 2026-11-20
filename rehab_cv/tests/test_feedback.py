"""
rehab_cv/tests/test_feedback.py - Unit Tests for CV-4 Feedback Engine & Rules
Tests:
- Feedback categories, severity levels, and structured event serialization
- Clinical rule evaluation across tracking states, movement states, and rep events
- Hysteresis, rate-limiting, and anti-spam message suppression in RehabilitationFeedbackEngine
- Priority escalation for critical events (ALERT and SUCCESS)
"""

import pytest
from rehab_cv.feedback.feedback_rules import (
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackEvent,
    FeedbackRuleSet,
)
from rehab_cv.feedback.feedback_engine import RehabilitationFeedbackEngine
from rehab_cv.exercises.knee_extension import ExerciseFrameMetrics, ExerciseState
from rehab_cv.exercises.repetition_counter import RepetitionData


def make_dummy_metrics(
    angle: float = 90.0,
    state: ExerciseState = ExerciseState.WAITING,
    tracking_reliable: bool = True,
    confidence: float = 0.95,
    rep_completed: bool = False,
    last_rep: RepetitionData = None,
) -> ExerciseFrameMetrics:
    return ExerciseFrameMetrics(
        selected_leg="right",
        raw_angle=angle,
        smoothed_angle=angle,
        repetition_count=1 if rep_completed else 0,
        exercise_state=state,
        max_extension_in_rep=angle if angle is not None else 0.0,
        current_excursion=0.0,
        is_tracking_reliable=tracking_reliable,
        landmark_confidence=confidence,
        rep_completed_this_frame=rep_completed,
        last_completed_rep=last_rep,
    )


def test_feedback_event_to_dict():
    event = FeedbackEvent(
        category=FeedbackCategory.MOVEMENT,
        message="Test message",
        severity=FeedbackSeverity.GUIDANCE,
        timestamp=100.0,
        measurement_deg=120.45,
        state="EXTENDING",
    )
    d = event.to_dict()
    assert d["category"] == "MOVEMENT"
    assert d["message"] == "Test message"
    assert d["severity"] == "GUIDANCE"
    assert d["timestamp"] == 100.0
    assert d["measurement_deg"] == 120.5
    assert d["state"] == "EXTENDING"


def test_tracking_loss_triggers_alert_and_warning():
    rules = FeedbackRuleSet(min_confidence=0.60)

    # Total loss (confidence 0.0)
    alert = rules.evaluate(
        is_tracking_reliable=False,
        confidence=0.0,
        exercise_state_str="WAITING",
        current_angle=None,
        current_excursion=0.0,
        rep_completed=False,
    )
    assert alert.category == FeedbackCategory.TRACKING
    assert alert.severity == FeedbackSeverity.ALERT
    assert "Position yourself sideways" in alert.message

    # Marginal loss / occlusion
    warning = rules.evaluate(
        is_tracking_reliable=False,
        confidence=0.45,
        exercise_state_str="WAITING",
        current_angle=90.0,
        current_excursion=0.0,
        rep_completed=False,
    )
    assert warning.category == FeedbackCategory.TRACKING
    assert warning.severity == FeedbackSeverity.WARNING
    assert "temporarily lost" in warning.message


def test_repetition_completion_triggers_success():
    rules = FeedbackRuleSet()
    event = rules.evaluate(
        is_tracking_reliable=True,
        confidence=0.90,
        exercise_state_str="WAITING",
        current_angle=95.0,
        current_excursion=65.0,
        rep_completed=True,
        last_rep_max_angle=160.0,
    )
    assert event.category == FeedbackCategory.REPETITION
    assert event.severity == FeedbackSeverity.SUCCESS
    assert "Repetition completed" in event.message
    assert event.measurement_deg == 160.0


def test_movement_state_guidance():
    rules = FeedbackRuleSet(
        flexion_target_deg=100.0,
        extension_target_deg=155.0,
        return_target_deg=115.0,
    )

    # 1. WAITING - angle good
    e_wait_ok = rules.evaluate(True, 0.9, "WAITING", 90.0, 0.0, False)
    assert "Ready. Begin extending" in e_wait_ok.message

    # 2. WAITING - angle too open
    e_wait_open = rules.evaluate(True, 0.9, "WAITING", 120.0, 0.0, False)
    assert "Bend your knee slightly more" in e_wait_open.message

    # 3. EXTENDING - mid range
    e_ext = rules.evaluate(True, 0.9, "EXTENDING", 130.0, 40.0, False)
    assert "Continue extending" in e_ext.message

    # 4. EXTENDING - near target
    e_ext_near = rules.evaluate(True, 0.9, "EXTENDING", 150.0, 60.0, False)
    assert "Almost there!" in e_ext_near.message

    # 5. EXTENDED - target reached
    e_held = rules.evaluate(True, 0.9, "EXTENDED", 158.0, 68.0, False)
    assert "Extension target reached!" in e_held.message
    assert e_held.severity == FeedbackSeverity.SUCCESS

    # 6. RETURNING
    e_ret = rules.evaluate(True, 0.9, "RETURNING", 140.0, 50.0, False)
    assert "Slowly return" in e_ret.message


def test_feedback_engine_duplicate_suppression_and_priority():
    engine = RehabilitationFeedbackEngine(min_message_hold_sec=1.0)

    # Time 10.0: WAITING state -> message "Ready. Begin extending your leg smoothly upwards."
    m1 = make_dummy_metrics(angle=90.0, state=ExerciseState.WAITING)
    ev1 = engine.process(m1, now=10.0)
    assert "Ready" in ev1.message
    first_msg = ev1.message

    # Time 10.2: Slightly different angle but same state/rule -> same message
    m2 = make_dummy_metrics(angle=92.0, state=ExerciseState.WAITING)
    ev2 = engine.process(m2, now=10.2)
    assert ev2.message == first_msg
    assert ev2.measurement_deg == 92.0

    # Time 10.4: State changes to EXTENDING (would normally trigger "Continue extending")
    # But hold window (1.0s) has NOT expired (only 0.4s elapsed) -> suppresses flicker
    m3 = make_dummy_metrics(angle=120.0, state=ExerciseState.EXTENDING)
    ev3 = engine.process(m3, now=10.4)
    assert ev3.message == first_msg  # Still holding previous message
    assert ev3.measurement_deg == 120.0

    # Time 10.5: Tracking lost (ALERT) -> BYPASSES debounce window immediately
    m_alert = make_dummy_metrics(angle=None, state=ExerciseState.EXTENDING, tracking_reliable=False, confidence=0.0)
    ev_alert = engine.process(m_alert, now=10.5)
    assert ev_alert.severity == FeedbackSeverity.ALERT
    assert ev_alert.category == FeedbackCategory.TRACKING

    # Time 12.0: Debounce window has passed, new message adopted
    m4 = make_dummy_metrics(angle=130.0, state=ExerciseState.EXTENDING)
    ev4 = engine.process(m4, now=12.0)
    assert "Continue extending" in ev4.message
