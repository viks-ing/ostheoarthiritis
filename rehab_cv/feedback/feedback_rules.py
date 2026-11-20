"""
rehab_cv/feedback/feedback_rules.py - Structured Feedback Categories & Clinical Guidance Rules
Defines structured feedback event objects, severity tiers, anti-spam duplicate suppression,
and clinically safe, rule-based feedback heuristics for seated knee extensions.

Safety Notice:
Feedback messages are assistive movement guidance based on 2D camera geometry.
They do not diagnose injury or determine whether an exercise is medically safe.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


class FeedbackCategory(Enum):
    """Functional categories for exercise feedback events."""
    TRACKING = "TRACKING"         # Camera visibility, occlusion, lighting
    POSITIONING = "POSITIONING"   # Setup, posture, camera perspective
    MOVEMENT = "MOVEMENT"         # Dynamic joint motion during flexion/extension
    REPETITION = "REPETITION"     # Repetition completion, rejection, excursion validation
    SESSION = "SESSION"           # Set progression, rest intervals, session conclusion


class FeedbackSeverity(Enum):
    """Severity and urgency levels for user feedback."""
    INFO = "INFO"           # Standard informative status
    GUIDANCE = "GUIDANCE"   # Actionable clinical movement cue
    WARNING = "WARNING"     # Movement error or partial occlusion
    SUCCESS = "SUCCESS"     # Positive reinforcement (rep or set completed)
    ALERT = "ALERT"         # Critical failure (tracking lost, out of frame)


@dataclass
class FeedbackEvent:
    """Structured clinical feedback event passed to UI and logging."""
    category: FeedbackCategory
    message: str
    severity: FeedbackSeverity
    timestamp: float = field(default_factory=time.time)
    measurement_deg: Optional[float] = None
    state: str = "WAITING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": round(self.timestamp, 3),
            "measurement_deg": round(self.measurement_deg, 1) if self.measurement_deg is not None else None,
            "state": self.state,
        }


class FeedbackRuleSet:
    """
    Configurable clinical rule set evaluating real-time kinematic states
    and generating targeted, non-repetitive feedback events.
    """

    def __init__(
        self,
        flexion_target_deg: float = 100.0,
        extension_target_deg: float = 155.0,
        return_target_deg: float = 115.0,
        min_excursion_deg: float = 45.0,
        min_confidence: float = 0.60,
    ):
        self.flexion_target_deg = flexion_target_deg
        self.extension_target_deg = extension_target_deg
        self.return_target_deg = return_target_deg
        self.min_excursion_deg = min_excursion_deg
        self.min_confidence = min_confidence

    def evaluate(
        self,
        is_tracking_reliable: bool,
        confidence: float,
        exercise_state_str: str,
        current_angle: Optional[float],
        current_excursion: float,
        rep_completed: bool,
        last_rep_max_angle: Optional[float] = None,
        is_in_cooldown: bool = False,
        now: Optional[float] = None,
    ) -> FeedbackEvent:
        """
        Evaluates current kinematics against clinical rules and returns a structured FeedbackEvent.
        """
        timestamp = now if now is not None else time.time()

        # Rule 1: Tracking / Landmark Occlusion Safety (Highest Priority)
        if not is_tracking_reliable or confidence < self.min_confidence or current_angle is None:
            if confidence == 0.0:
                return FeedbackEvent(
                    category=FeedbackCategory.TRACKING,
                    message="Position yourself sideways to the camera.",
                    severity=FeedbackSeverity.ALERT,
                    timestamp=timestamp,
                    measurement_deg=None,
                    state=exercise_state_str,
                )
            return FeedbackEvent(
                category=FeedbackCategory.TRACKING,
                message="Knee tracking temporarily lost. Ensure your leg is visible.",
                severity=FeedbackSeverity.WARNING,
                timestamp=timestamp,
                measurement_deg=None,
                state=exercise_state_str,
            )

        # Rule 2: Repetition Just Completed (Celebratory Feedback)
        if rep_completed:
            max_ang_str = f" ({last_rep_max_angle:.0f}°)" if last_rep_max_angle else ""
            return FeedbackEvent(
                category=FeedbackCategory.REPETITION,
                message=f"Repetition completed! Good extension{max_ang_str}.",
                severity=FeedbackSeverity.SUCCESS,
                timestamp=timestamp,
                measurement_deg=last_rep_max_angle,
                state=exercise_state_str,
            )

        # Rule 3: Post-Repetition Cooldown
        if is_in_cooldown:
            return FeedbackEvent(
                category=FeedbackCategory.MOVEMENT,
                message="Pause briefly, then begin your next repetition.",
                severity=FeedbackSeverity.INFO,
                timestamp=timestamp,
                measurement_deg=current_angle,
                state=exercise_state_str,
            )

        # Rule 4: WAITING State (Seated resting posture)
        if exercise_state_str == "WAITING":
            if current_angle <= self.flexion_target_deg:
                return FeedbackEvent(
                    category=FeedbackCategory.MOVEMENT,
                    message="Ready. Begin extending your leg smoothly upwards.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )
            else:
                return FeedbackEvent(
                    category=FeedbackCategory.POSITIONING,
                    message="Bend your knee slightly more to reach starting position.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )

        # Rule 5: EXTENDING State (Leg ascending towards straight posture)
        if exercise_state_str == "EXTENDING":
            if current_angle >= (self.extension_target_deg - 10.0):
                return FeedbackEvent(
                    category=FeedbackCategory.MOVEMENT,
                    message="Almost there! Push gently to complete extension.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )
            else:
                return FeedbackEvent(
                    category=FeedbackCategory.MOVEMENT,
                    message="Continue extending your leg upwards.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )

        # Rule 6: EXTENDED State (Peak Terminal Extension Reached)
        if exercise_state_str == "EXTENDED":
            return FeedbackEvent(
                category=FeedbackCategory.MOVEMENT,
                message="Extension target reached! Hold momentarily, then lower slowly.",
                severity=FeedbackSeverity.SUCCESS,
                timestamp=timestamp,
                measurement_deg=current_angle,
                state=exercise_state_str,
            )

        # Rule 7: RETURNING State (Descending back to chair)
        if exercise_state_str == "RETURNING":
            if current_angle <= (self.return_target_deg + 10.0):
                return FeedbackEvent(
                    category=FeedbackCategory.MOVEMENT,
                    message="Almost down. Return fully to starting seated position.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )
            else:
                return FeedbackEvent(
                    category=FeedbackCategory.MOVEMENT,
                    message="Slowly return to the starting position.",
                    severity=FeedbackSeverity.GUIDANCE,
                    timestamp=timestamp,
                    measurement_deg=current_angle,
                    state=exercise_state_str,
                )

        # Default Neutral Fallback
        return FeedbackEvent(
            category=FeedbackCategory.POSITIONING,
            message="Maintain upright seated posture facing sideways.",
            severity=FeedbackSeverity.INFO,
            timestamp=timestamp,
            measurement_deg=current_angle,
            state=exercise_state_str,
        )
