"""
rehab_cv/feedback/session_analyzer.py - Comprehensive Rehabilitation Session Analytics
Aggregates frame-by-frame kinematics, range-of-motion metrics, multi-set tracking,
rest intervals, and quality metrics into structured summary reports.

Safety Notice:
These metrics are 2D computer vision biomechanical estimates for patient adherence
and exercise quality tracking. They are not diagnostic clinical goniometry.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ..exercises.knee_extension import ExerciseFrameMetrics


@dataclass
class SetMetrics:
    """Performance metrics for an individual exercise set."""
    set_number: int
    reps_completed: int
    reps_rejected: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    rest_duration_sec: float = 0.0
    avg_excursion_deg: float = 0.0
    max_extension_deg: float = 0.0

    def duration_sec(self) -> float:
        if self.end_time is not None and self.end_time >= self.start_time:
            return round(self.end_time - self.start_time, 2)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "set_number": self.set_number,
            "reps_completed": self.reps_completed,
            "reps_rejected": self.reps_rejected,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3) if self.end_time is not None else None,
            "duration_sec": self.duration_sec(),
            "rest_duration_sec": round(self.rest_duration_sec, 2),
            "avg_excursion_deg": round(self.avg_excursion_deg, 1),
            "max_extension_deg": round(self.max_extension_deg, 1),
        }


@dataclass
class SessionSummaryMetrics:
    """Comprehensive performance and compliance summary for an entire exercise session."""
    session_id: Optional[str]
    exercise_name: str
    selected_leg: str
    start_time: float
    end_time: float
    duration_sec: float
    total_frames: int
    tracked_frames: int
    tracking_availability_pct: float
    total_completed_reps: int
    total_rejected_reps: int
    total_sets: int
    min_observed_angle: Optional[float]
    max_observed_angle: Optional[float]
    avg_excursion_deg: float
    best_excursion_deg: float
    sets: List[SetMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exercise_name": self.exercise_name,
            "selected_leg": self.selected_leg,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_sec": round(self.duration_sec, 2),
            "total_frames": self.total_frames,
            "tracked_frames": self.tracked_frames,
            "tracking_availability_pct": round(self.tracking_availability_pct, 1),
            "total_completed_reps": self.total_completed_reps,
            "total_rejected_reps": self.total_rejected_reps,
            "total_sets": self.total_sets,
            "min_observed_angle": round(self.min_observed_angle, 1) if self.min_observed_angle is not None else None,
            "max_observed_angle": round(self.max_observed_angle, 1) if self.max_observed_angle is not None else None,
            "avg_excursion_deg": round(self.avg_excursion_deg, 1),
            "best_excursion_deg": round(self.best_excursion_deg, 1),
            "sets": [s.to_dict() for s in self.sets],
        }


class SessionAnalyzer:
    """
    Session-level analytics aggregator.
    Tracks frame continuity, angular excursions, set progressions, and rest periods.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        exercise_name: str = "seated_knee_extension",
        selected_leg: str = "right",
    ):
        self.session_id = session_id
        self.exercise_name = exercise_name
        self.selected_leg = selected_leg

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.total_frames: int = 0
        self.tracked_frames: int = 0

        self.min_angle: Optional[float] = None
        self.max_angle: Optional[float] = None

        # Excursion history for completed reps
        self.rep_excursions: List[float] = []
        self.rep_max_angles: List[float] = []

        # Multi-set state
        self.sets: List[SetMetrics] = []
        self.current_set: Optional[SetMetrics] = None
        self.last_set_end_time: Optional[float] = None

        # Cumulative totals
        self.cumulative_completed_reps: int = 0
        self.cumulative_rejected_reps: int = 0

    def start_session(self, now: Optional[float] = None):
        """Initializes a new tracking session."""
        self.start_time = now if now is not None else time.time()
        self.total_frames = 0
        self.tracked_frames = 0
        self.min_angle = None
        self.max_angle = None
        self.rep_excursions.clear()
        self.rep_max_angles.clear()
        self.sets.clear()
        self.start_set(set_number=1, now=self.start_time)

    def start_set(self, set_number: int, now: Optional[float] = None):
        """Starts a new exercise set, recording preceding rest duration if applicable."""
        timestamp = now if now is not None else time.time()
        rest_duration = 0.0
        if self.last_set_end_time is not None:
            rest_duration = max(0.0, timestamp - self.last_set_end_time)

        self.current_set = SetMetrics(
            set_number=set_number,
            reps_completed=0,
            reps_rejected=0,
            start_time=timestamp,
            rest_duration_sec=rest_duration,
        )

    def finish_set(
        self,
        reps_completed: int,
        reps_rejected: int = 0,
        avg_excursion: float = 0.0,
        max_extension: float = 0.0,
        now: Optional[float] = None,
    ):
        """Finishes current set and appends it to completed sets list."""
        timestamp = now if now is not None else time.time()
        if self.current_set is not None:
            self.current_set.end_time = timestamp
            self.current_set.reps_completed = reps_completed
            self.current_set.reps_rejected = reps_rejected
            self.current_set.avg_excursion_deg = avg_excursion
            self.current_set.max_extension_deg = max_extension
            self.sets.append(self.current_set)
            self.cumulative_completed_reps += reps_completed
            self.cumulative_rejected_reps += reps_rejected
            self.last_set_end_time = timestamp
            self.current_set = None

    def record_frame(self, metrics: ExerciseFrameMetrics, now: Optional[float] = None):
        """Processes an incoming frame's metrics to update live excursion and visibility stats."""
        if self.start_time is None:
            self.start_session(now=now)

        self.total_frames += 1

        if metrics.is_tracking_reliable and metrics.smoothed_angle is not None:
            self.tracked_frames += 1
            ang = metrics.smoothed_angle
            if self.min_angle is None or ang < self.min_angle:
                self.min_angle = ang
            if self.max_angle is None or ang > self.max_angle:
                self.max_angle = ang

        if metrics.rep_completed_this_frame and metrics.last_completed_rep:
            rep = metrics.last_completed_rep
            self.rep_excursions.append(rep.excursion_deg)
            self.rep_max_angles.append(rep.max_extension_angle)
            if self.current_set is not None:
                self.current_set.reps_completed += 1

    def finish_session(self, now: Optional[float] = None) -> SessionSummaryMetrics:
        """Closes the session and generates final summary metrics."""
        self.end_time = now if now is not None else time.time()
        if self.start_time is None:
            self.start_time = self.end_time

        # If a set is still active, finish it
        if self.current_set is not None:
            self.finish_set(
                reps_completed=self.current_set.reps_completed,
                reps_rejected=self.current_set.reps_rejected,
                now=self.end_time,
            )

        duration = max(0.0, self.end_time - self.start_time)
        availability_pct = (
            (self.tracked_frames / self.total_frames * 100.0)
            if self.total_frames > 0
            else 0.0
        )

        avg_excursion = (
            sum(self.rep_excursions) / len(self.rep_excursions)
            if self.rep_excursions
            else 0.0
        )
        best_excursion = max(self.rep_excursions) if self.rep_excursions else 0.0

        return SessionSummaryMetrics(
            session_id=self.session_id,
            exercise_name=self.exercise_name,
            selected_leg=self.selected_leg,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_sec=duration,
            total_frames=self.total_frames,
            tracked_frames=self.tracked_frames,
            tracking_availability_pct=availability_pct,
            total_completed_reps=self.cumulative_completed_reps,
            total_rejected_reps=self.cumulative_rejected_reps,
            total_sets=len(self.sets),
            min_observed_angle=self.min_angle,
            max_observed_angle=self.max_angle,
            avg_excursion_deg=avg_excursion,
            best_excursion_deg=best_excursion,
            sets=list(self.sets),
        )
