"""
rehab_cv/tests/test_repetition_counter.py - Unit Tests for FSM Repetition Counting
Tests:
- Complete extension-and-return cycle counts exactly once
- Incomplete extension (< 155°) rejected
- Incomplete return (> 115°) rejected
- Noisy threshold crossings do not create duplicate reps
- Tracking loss does not create false repetitions
- Post-repetition cooldown blocks instant false triggering
- Full state sequence: WAITING -> EXTENDING -> EXTENDED -> RETURNING -> WAITING
"""

import time
from typing import Optional
import pytest
from rehab_cv.exercises.repetition_counter import RepetitionCounter, ExerciseState
from rehab_cv.config import ExerciseConfig


def create_counter() -> RepetitionCounter:
    """Helper to initialize a counter with predictable fast timing parameters for testing."""
    config = ExerciseConfig(
        flexion_threshold_deg=100.0,
        extension_threshold_deg=155.0,
        return_threshold_deg=115.0,
        min_excursion_deg=45.0,
        min_state_duration_sec=0.01,  # Short durations for deterministic unit tests
        cooldown_sec=0.10,
        tracking_loss_reset_sec=0.20,
        hysteresis_frames=2,
    )
    return RepetitionCounter(config)


class SimulatedSession:
    """Maintains a monotonically increasing simulation clock for feeding trajectories."""
    def __init__(self, counter: RepetitionCounter, start_time: float = 1000.0, dt: float = 0.03):
        self.counter = counter
        self.current_time = start_time
        self.dt = dt

    def feed(self, angles: list, dt: Optional[float] = None) -> list:
        step_dt = dt if dt is not None else self.dt
        results = []
        for angle in angles:
            self.current_time += step_dt
            state, reps, inc = self.counter.update(angle, current_time=self.current_time)
            results.append((state, reps, inc))
        return results


def test_complete_repetition_counts_exactly_once():
    """
    Simulates a perfect repetition:
    Seated at 90° -> Extends to 165° -> Returns back to 92°
    Must increment rep count from 0 to 1 exactly once upon returning.
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # 1. Seated flexion (WAITING)
    sim.feed([90.0, 92.0, 91.0])
    assert counter.state == ExerciseState.WAITING
    assert counter.repetition_count == 0

    # 2. Ascending extension (WAITING -> EXTENDING)
    sim.feed([110.0, 125.0, 140.0, 145.0])
    assert counter.state == ExerciseState.EXTENDING
    assert counter.repetition_count == 0

    # 3. Terminal extension reached (EXTENDING -> EXTENDED)
    sim.feed([156.0, 162.0, 165.0, 166.0])
    assert counter.state == ExerciseState.EXTENDED
    assert counter.repetition_count == 0

    # 4. Descending return (EXTENDED -> RETURNING)
    sim.feed([145.0, 130.0, 122.0])
    assert counter.state == ExerciseState.RETURNING
    assert counter.repetition_count == 0

    # 5. Return completed (RETURNING -> WAITING, Count = 1)
    sim.feed([110.0, 95.0, 92.0])
    assert counter.state == ExerciseState.WAITING
    assert counter.repetition_count == 1, f"Expected 1 rep, got {counter.repetition_count}"
    assert len(counter.completed_reps) == 1
    assert counter.completed_reps[0].max_extension_angle >= 165.0


def test_incomplete_extension_does_not_count():
    """
    User starts extending (90° -> 135°), but fails to reach the 155° threshold
    and drops back to seated position (90°).
    Must NOT count as a repetition.
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # Seated -> Extends partway to 135° -> Returns to 90°
    trajectory = [90.0, 95.0, 115.0, 125.0, 135.0, 130.0, 110.0, 95.0, 90.0, 90.0]
    sim.feed(trajectory)

    assert counter.repetition_count == 0, "Incomplete extension falsely counted as a rep!"
    assert counter.rejected_reps_count >= 1
    assert counter.state == ExerciseState.WAITING


def test_incomplete_return_does_not_count():
    """
    User extends to 165°, but only returns partway (e.g. 130°) before extending again.
    Repetition should NOT complete until the leg returns below return_threshold (115°).
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # Reach full extension
    sim.feed([90.0, 92.0, 120.0, 140.0, 160.0, 165.0])
    assert counter.state == ExerciseState.EXTENDED

    # Start return, but stop at 125° (above return threshold 115°)
    sim.feed([140.0, 130.0, 125.0, 128.0])
    assert counter.state == ExerciseState.RETURNING
    assert counter.repetition_count == 0, "Premature rep count before returning to flexion threshold!"

    # Now extend back up
    sim.feed([145.0, 160.0, 168.0])
    assert counter.repetition_count == 0


def test_noisy_threshold_crossings_do_not_produce_duplicate_reps():
    """
    Noise oscillating around the extension threshold (154° <-> 156°)
    must be debounced by hysteresis and state duration, avoiding duplicate counts.
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # Move from 90° to boundary 154°/156°
    noisy_extension = [90.0, 92.0, 120.0, 140.0, 154.0, 156.0, 154.0, 156.0, 154.0, 156.0, 160.0]
    sim.feed(noisy_extension)
    assert counter.state in (ExerciseState.EXTENDING, ExerciseState.EXTENDED)
    assert counter.repetition_count == 0

    # Return with noise around 114°/116°
    noisy_return = [140.0, 130.0, 116.0, 114.0, 116.0, 114.0, 105.0, 95.0]
    sim.feed(noisy_return)
    assert counter.repetition_count == 1, f"Jitter generated duplicate reps: {counter.repetition_count}"


def test_tracking_loss_does_not_create_false_repetitions():
    """
    If landmarks are lost (None) during extension, the counter must reset safely
    to WAITING without falsely incrementing repetition count.
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # Start extending
    sim.feed([90.0, 92.0, 120.0, 145.0])
    assert counter.state == ExerciseState.EXTENDING

    # Sudden tracking loss (camera covered / occlusion) for > tracking_loss_reset_sec (0.2s)
    sim.feed([None, None, None, None, None, None, None, None, None, None], dt=0.03)

    assert counter.state == ExerciseState.WAITING
    assert counter.repetition_count == 0, "Tracking loss falsely counted a repetition!"


def test_cooldown_blocks_immediate_duplicate_counting():
    """
    After completing a valid rep, counter enters cooldown (0.10s)
    and ignores instantaneous threshold re-crossings until settled.
    """
    counter = create_counter()
    sim = SimulatedSession(counter)

    # Complete rep 1
    full_rep = [90.0, 92.0, 120.0, 140.0, 160.0, 165.0, 140.0, 130.0, 110.0, 95.0, 90.0]
    sim.feed(full_rep, dt=0.02)
    assert counter.repetition_count == 1

    # Immediately inject high angle during cooldown window
    # dt=0.005s ensures time within 0.10s cooldown
    t = counter.last_rep_complete_time + 0.02
    state, reps, inc = counter.update(165.0, current_time=t)
    assert reps == 1, "Cooldown failed to prevent immediate rep trigger!"
    assert state == ExerciseState.WAITING

