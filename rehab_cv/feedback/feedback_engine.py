"""
rehab_cv/feedback/feedback_engine.py - Rehabilitation Feedback Engine Coordinator
Manages real-time rule evaluation, duplicate feedback message suppression,
safety guardrails, and structured event streaming.
"""

import time
from typing import Optional
from .feedback_rules import (
    FeedbackCategory,
    FeedbackSeverity,
    FeedbackEvent,
    FeedbackRuleSet,
)
from ..exercises.knee_extension import ExerciseFrameMetrics


class RehabilitationFeedbackEngine:
    """
    Coordinates real-time exercise feedback generation with hysteresis and rate-limiting.
    Prevents visual flicker by suppressing duplicate messages within a configurable debounce window.
    """

    def __init__(
        self,
        rules: Optional[FeedbackRuleSet] = None,
        min_message_hold_sec: float = 0.80,
    ):
        self.rules = rules or FeedbackRuleSet()
        self.min_message_hold_sec = min_message_hold_sec

        self.last_event: Optional[FeedbackEvent] = None
        self.last_event_time: float = 0.0
        self.last_message_text: str = ""

    def process(
        self,
        metrics: ExerciseFrameMetrics,
        is_in_cooldown: bool = False,
        now: Optional[float] = None,
    ) -> FeedbackEvent:
        """
        Evaluates current frame metrics and returns a rate-limited, high-priority FeedbackEvent.
        """
        current_time = now if now is not None else time.time()

        last_rep_max = (
            metrics.last_completed_rep.max_extension_angle
            if metrics.last_completed_rep
            else metrics.max_extension_in_rep
        )

        # Generate candidate feedback event from rules
        candidate = self.rules.evaluate(
            is_tracking_reliable=metrics.is_tracking_reliable,
            confidence=metrics.landmark_confidence,
            exercise_state_str=metrics.exercise_state.value,
            current_angle=metrics.smoothed_angle,
            current_excursion=metrics.current_excursion,
            rep_completed=metrics.rep_completed_this_frame,
            last_rep_max_angle=last_rep_max,
            is_in_cooldown=is_in_cooldown,
            now=current_time,
        )

        # 1. Immediate Trigger Conditions:
        # Repetition completion (SUCCESS) or tracking failure (ALERT) bypass hold time
        if candidate.severity in (FeedbackSeverity.SUCCESS, FeedbackSeverity.ALERT):
            self.last_event = candidate
            self.last_event_time = current_time
            self.last_message_text = candidate.message
            return candidate

        # 2. Duplicate Message Check:
        # If identical message is produced within the hold window, return existing event
        time_since_last = current_time - self.last_event_time
        if candidate.message == self.last_message_text:
            if self.last_event is not None:
                # Update measurement and timestamp while preserving message
                self.last_event.measurement_deg = candidate.measurement_deg
                self.last_event.timestamp = current_time
                return self.last_event

        # 3. Message Change within Hold Window:
        # If hold window hasn't expired and candidate is of equal/lower severity, hold previous message
        if (
            self.last_event is not None
            and time_since_last < self.min_message_hold_sec
            and candidate.severity.value not in ("ALERT", "SUCCESS")
        ):
            self.last_event.measurement_deg = candidate.measurement_deg
            return self.last_event

        # 4. Adopt New Candidate Event
        self.last_event = candidate
        self.last_event_time = current_time
        self.last_message_text = candidate.message
        return candidate

    def reset(self):
        """Clears feedback history."""
        self.last_event = None
        self.last_event_time = 0.0
        self.last_message_text = ""
