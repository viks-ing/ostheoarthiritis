"""
rehab_cv/exercises/knee_extension.py - Seated Knee Extension Exercise Coordinator
Integrates Landmark Extraction, Vector Angle Computation, Temporal Smoothing,
and the Repetition Counter FSM into a cohesive exercise engine.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ..config import ExerciseConfig, SmoothingConfig
from ..detection.pose_detector import PoseDetectionResult, LegLandmarks
from ..biomechanics.angle_calculator import calculate_knee_angle_2d, LegSelectionManager
from ..biomechanics.smoothing import AngleSmoother
from .repetition_counter import RepetitionCounter, ExerciseState, RepetitionData


@dataclass
class ExerciseFrameMetrics:
    """Consolidated metrics produced per video frame."""
    selected_leg: str                       # "left" or "right"
    raw_angle: Optional[float]              # Unfiltered instantaneous angle
    smoothed_angle: Optional[float]         # Temporally filtered angle
    repetition_count: int                   # Total valid completed reps
    exercise_state: ExerciseState           # Current FSM state
    max_extension_in_rep: float             # Highest extension recorded in active rep
    current_excursion: float                # Angle excursion in current rep
    is_tracking_reliable: bool              # True if landmarks meet visibility thresholds
    landmark_confidence: float              # Mean visibility of active leg joints
    rep_completed_this_frame: bool          # Flag indicating rep completion on this frame
    last_completed_rep: Optional[RepetitionData] = None
    error_message: Optional[str] = None


@dataclass
class ExerciseSessionMetrics:
    """Cumulative metrics across a continuous exercise session."""
    total_reps: int = 0
    rejected_reps: int = 0
    selected_leg: str = "left"
    rep_history: List[RepetitionData] = field(default_factory=list)

    @property
    def average_excursion(self) -> float:
        if not self.rep_history:
            return 0.0
        return round(sum(r.excursion_deg for r in self.rep_history) / len(self.rep_history), 1)

    @property
    def average_duration(self) -> float:
        if not self.rep_history:
            return 0.0
        return round(sum(r.duration_sec for r in self.rep_history) / len(self.rep_history), 2)


class KneeExtensionExercise:
    """
    High-level engine that manages seated knee extension exercise tracking.
    """

    def __init__(
        self,
        exercise_config: Optional[ExerciseConfig] = None,
        smoothing_config: Optional[SmoothingConfig] = None,
    ):
        self.exercise_config = exercise_config or ExerciseConfig()
        self.smoothing_config = smoothing_config or SmoothingConfig()

        self.leg_selector = LegSelectionManager(preferred_leg=self.exercise_config.preferred_leg)
        self.smoother = AngleSmoother(self.smoothing_config)
        self.counter = RepetitionCounter(self.exercise_config)

    def process(
        self,
        detection: PoseDetectionResult,
        current_time: Optional[float] = None,
    ) -> ExerciseFrameMetrics:
        """
        Processes a single frame's pose detection result.
        Calculates angle, updates smoothing, and advances repetition state machine.
        """
        # 1. Determine whether side-switching is permissible
        # Mid-repetition lock: can only switch if WAITING or tracking completely lost
        can_switch = (self.counter.state == ExerciseState.WAITING) or (not detection.detected)

        active_side = self.leg_selector.select_leg(
            left_leg=detection.left_leg,
            right_leg=detection.right_leg,
            can_switch=can_switch,
        )

        active_leg_landmarks: LegLandmarks = (
            detection.left_leg if active_side == "left" else detection.right_leg
        )

        # 2. Compute 2D Vector Knee Angle
        raw_angle = None
        error_msg = None
        confidence = active_leg_landmarks.mean_visibility

        if active_leg_landmarks.is_valid:
            angle_val, err = calculate_knee_angle_2d(
                hip=active_leg_landmarks.hip,
                knee=active_leg_landmarks.knee,
                ankle=active_leg_landmarks.ankle,
                min_confidence=self.exercise_config.flexion_threshold_deg * 0.0,  # Pre-filtered
            )
            raw_angle = angle_val
            error_msg = err
        else:
            error_msg = f"{active_side.capitalize()} leg landmarks below visibility threshold ({confidence:.2f})"

        # 3. Apply Temporal Smoothing
        smoothed_angle = self.smoother.update(raw_angle)

        # 4. Advance Repetition State Machine
        # Use smoothed angle to prevent high-frequency noise from triggering false state transitions
        effective_angle = smoothed_angle if smoothed_angle is not None else raw_angle
        current_state, rep_count, rep_completed = self.counter.update(
            angle_deg=effective_angle,
            side=active_side,
            current_time=current_time,
        )

        last_rep = self.counter.completed_reps[-1] if self.counter.completed_reps else None

        return ExerciseFrameMetrics(
            selected_leg=active_side,
            raw_angle=raw_angle,
            smoothed_angle=smoothed_angle,
            repetition_count=rep_count,
            exercise_state=current_state,
            max_extension_in_rep=round(self.counter.current_max_extension, 1),
            current_excursion=round(self.counter.current_excursion, 1),
            is_tracking_reliable=active_leg_landmarks.is_valid,
            landmark_confidence=round(confidence, 2),
            rep_completed_this_frame=rep_completed,
            last_completed_rep=last_rep,
            error_message=error_msg,
        )

    def reset(self):
        """Resets repetition counter, smoothing filter, and session history."""
        self.smoother.reset()
        self.counter.reset()

    def switch_leg(self) -> str:
        """
        Manually switches active tracking leg.
        Resets current repetition state machine to prevent cross-limb contamination.
        """
        new_side = self.leg_selector.force_switch()
        self.smoother.reset()
        self.counter.reset()
        return new_side

    def get_session_metrics(self) -> ExerciseSessionMetrics:
        """Retrieves session summary statistics."""
        return ExerciseSessionMetrics(
            total_reps=self.counter.repetition_count,
            rejected_reps=self.counter.rejected_reps_count,
            selected_leg=self.leg_selector.active_side,
            rep_history=list(self.counter.completed_reps),
        )
