"""
rehab_cv/exercises/repetition_counter.py - 4-State Finite State Machine (FSM)
Tracks Seated Knee Extension Repetitions:
WAITING -> EXTENDING -> EXTENDED -> RETURNING -> WAITING (Count +1)

Features:
- Configurable angular thresholds (Flexion <= 100°, Extension >= 155°, Return <= 115°).
- Range of Motion / Excursion validation (Delta >= 45°).
- Debounce timers to reject rapid jitter / noisy threshold crossings.
- Cooldown timer after valid rep completion.
- Rejection of incomplete extensions and incomplete returns.
- Safe reset upon landmark tracking loss.
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple
from ..config import ExerciseConfig


class ExerciseState(Enum):
    """Finite State Machine States for Seated Knee Extension."""
    WAITING = "WAITING"       # Seated flexion resting position (approx 90°-100°)
    EXTENDING = "EXTENDING"   # Leg ascending / straightening towards extension threshold
    EXTENDED = "EXTENDED"     # Peak terminal extension reached (approx >= 155°)
    RETURNING = "RETURNING"   # Leg descending back towards seated position


@dataclass
class RepetitionData:
    """Record of a completed repetition."""
    rep_number: int
    side: str
    start_angle: float
    max_extension_angle: float
    return_angle: float
    excursion_deg: float
    duration_sec: float
    timestamp: float


class RepetitionCounter:
    """
    State machine that processes knee angles frame-by-frame and validates reps.
    """

    def __init__(self, config: Optional[ExerciseConfig] = None):
        self.config = config or ExerciseConfig()

        # FSM State
        self.state: ExerciseState = ExerciseState.WAITING
        self.repetition_count: int = 0
        self.rejected_reps_count: int = 0
        self.completed_reps: List[RepetitionData] = []

        # Current Repetition Tracking Variables
        self.rep_start_time: float = 0.0
        self.rep_start_angle: float = 90.0
        self.current_max_extension: float = 0.0
        self.current_min_flexion: float = 180.0
        self.current_excursion: float = 0.0

        # Timing and Debouncing
        self.state_enter_time: Optional[float] = None
        self.last_valid_frame_time: Optional[float] = None
        self.last_rep_complete_time: float = -9999.0

        # Threshold crossings confirmation buffer
        self._consecutive_threshold_frames: int = 0
        self._last_angle_observed: Optional[float] = None

    @property
    def is_in_cooldown(self) -> bool:
        """True if the system is currently in post-repetition cooldown."""
        return (time.time() - self.last_rep_complete_time) < self.config.cooldown_sec

    def reset(self):
        """Resets the state machine and repetition tallies."""
        self.state = ExerciseState.WAITING
        self.repetition_count = 0
        self.rejected_reps_count = 0
        self.completed_reps.clear()
        self.rep_start_time = 0.0
        self.current_max_extension = 0.0
        self.current_min_flexion = 180.0
        self.current_excursion = 0.0
        self.state_enter_time = None
        self.last_valid_frame_time = None
        self.last_rep_complete_time = -9999.0
        self._consecutive_threshold_frames = 0
        self._last_angle_observed = None

    def update(
        self,
        angle_deg: Optional[float],
        side: str = "left",
        current_time: Optional[float] = None,
    ) -> Tuple[ExerciseState, int, bool]:
        """
        Updates the FSM with a new knee angle measurement.
        Args:
            angle_deg: Smoothed knee angle in degrees [0, 180], or None if occluded.
            side: Limb side ("left" or "right").
            current_time: Optional timestamp (seconds), defaults to time.time().
        Returns:
            (current_state, repetition_count, rep_just_incremented)
        """
        now = current_time if current_time is not None else time.time()
        if self.state_enter_time is None:
            self.state_enter_time = now
        if self.last_valid_frame_time is None:
            self.last_valid_frame_time = now

        rep_just_incremented = False

        # 1. Handle Missing / Occluded Landmarks
        if angle_deg is None:
            # Check for tracking loss timeout
            if (now - self.last_valid_frame_time) > self.config.tracking_loss_reset_sec:
                if self.state != ExerciseState.WAITING:
                    self.rejected_reps_count += 1
                self.state = ExerciseState.WAITING
                self.current_max_extension = 0.0
                self.current_min_flexion = 180.0
                self.current_excursion = 0.0
                self._consecutive_threshold_frames = 0
            return (self.state, self.repetition_count, False)

        # Valid angle received
        self.last_valid_frame_time = now
        self._last_angle_observed = angle_deg

        # 2. Check Cooldown Period
        if (now - self.last_rep_complete_time) < self.config.cooldown_sec:
            # Stay in WAITING during cooldown to allow user to settle
            self.state = ExerciseState.WAITING
            return (self.state, self.repetition_count, False)

        # 3. Finite State Machine Transitions
        time_in_state = now - self.state_enter_time

        if self.state == ExerciseState.WAITING:
            # Look for leg seated flexion position
            if angle_deg <= self.config.flexion_threshold_deg:
                self.current_min_flexion = min(self.current_min_flexion, angle_deg)
                self.current_max_extension = angle_deg
                self.current_excursion = 0.0

            # Transition: WAITING -> EXTENDING
            # Initiated when angle starts rising above flexion threshold with upward velocity
            if angle_deg > (self.config.flexion_threshold_deg + 5.0):
                self._consecutive_threshold_frames += 1
                if self._consecutive_threshold_frames >= self.config.hysteresis_frames:
                    self.state = ExerciseState.EXTENDING
                    self.state_enter_time = now
                    self.rep_start_time = now
                    self.rep_start_angle = self.current_min_flexion if self.current_min_flexion < 180 else angle_deg
                    self.current_max_extension = angle_deg
                    self.current_excursion = max(0.0, angle_deg - self.rep_start_angle)
                    self._consecutive_threshold_frames = 0
            else:
                self._consecutive_threshold_frames = 0

        elif self.state == ExerciseState.EXTENDING:
            # Track peak extension continuously
            self.current_max_extension = max(self.current_max_extension, angle_deg)
            self.current_excursion = max(0.0, self.current_max_extension - self.rep_start_angle)

            # Check for Rejection: User gave up and returned to flexion without extending
            if angle_deg <= self.config.flexion_threshold_deg:
                self.rejected_reps_count += 1
                self.state = ExerciseState.WAITING
                self.state_enter_time = now
                self._consecutive_threshold_frames = 0
                return (self.state, self.repetition_count, False)

            # Transition: EXTENDING -> EXTENDED
            # Triggered when angle crosses extension threshold and meets minimum excursion
            if angle_deg >= self.config.extension_threshold_deg and self.current_excursion >= self.config.min_excursion_deg:
                self._consecutive_threshold_frames += 1
                if (
                    self._consecutive_threshold_frames >= self.config.hysteresis_frames
                    and time_in_state >= self.config.min_state_duration_sec
                ):
                    self.state = ExerciseState.EXTENDED
                    self.state_enter_time = now
                    self._consecutive_threshold_frames = 0
            else:
                self._consecutive_threshold_frames = 0

        elif self.state == ExerciseState.EXTENDED:
            # Update peak extension if leg straightens further
            self.current_max_extension = max(self.current_max_extension, angle_deg)
            self.current_excursion = self.current_max_extension - self.rep_start_angle

            # Transition: EXTENDED -> RETURNING
            # Triggered when knee starts flexing downward below extension threshold
            if angle_deg < (self.config.extension_threshold_deg - 5.0):
                self._consecutive_threshold_frames += 1
                if (
                    self._consecutive_threshold_frames >= self.config.hysteresis_frames
                    and time_in_state >= self.config.min_state_duration_sec
                ):
                    self.state = ExerciseState.RETURNING
                    self.state_enter_time = now
                    self._consecutive_threshold_frames = 0
            else:
                self._consecutive_threshold_frames = 0

        elif self.state == ExerciseState.RETURNING:
            # Leg is descending back to seated posture
            # If user stops and re-extends before completing return, maintain highest extension
            if angle_deg > self.current_max_extension:
                self.current_max_extension = angle_deg
                self.current_excursion = self.current_max_extension - self.rep_start_angle

            # Transition: RETURNING -> WAITING (Valid Repetition Completed!)
            # Must return below return threshold
            if angle_deg <= self.config.return_threshold_deg:
                self._consecutive_threshold_frames += 1
                if (
                    self._consecutive_threshold_frames >= self.config.hysteresis_frames
                    and time_in_state >= self.config.min_state_duration_sec
                ):
                    # Verify total excursion was sufficient
                    total_excursion = self.current_max_extension - self.rep_start_angle
                    if total_excursion >= self.config.min_excursion_deg:
                        # VALID REPETITION
                        self.repetition_count += 1
                        rep_just_incremented = True
                        rep_duration = now - self.rep_start_time

                        rep_data = RepetitionData(
                            rep_number=self.repetition_count,
                            side=side,
                            start_angle=round(self.rep_start_angle, 2),
                            max_extension_angle=round(self.current_max_extension, 2),
                            return_angle=round(angle_deg, 2),
                            excursion_deg=round(total_excursion, 2),
                            duration_sec=round(rep_duration, 2),
                            timestamp=now,
                        )
                        self.completed_reps.append(rep_data)
                        self.last_rep_complete_time = now
                    else:
                        self.rejected_reps_count += 1

                    # Reset to WAITING for next repetition
                    self.state = ExerciseState.WAITING
                    self.state_enter_time = now
                    self.current_min_flexion = angle_deg
                    self.current_max_extension = angle_deg
                    self.current_excursion = 0.0
                    self._consecutive_threshold_frames = 0
            else:
                self._consecutive_threshold_frames = 0

        return (self.state, self.repetition_count, rep_just_incremented)
