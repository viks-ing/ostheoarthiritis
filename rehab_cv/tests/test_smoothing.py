"""
rehab_cv/tests/test_smoothing.py - Unit Tests for Temporal Angle Smoothing
Tests:
- Exponential Moving Average noise reduction
- Rolling Window average noise reduction
- Missing observations handling (does NOT insert fake 0.0 values)
- Graceful recovery after tracking resumption
- Filter reset behavior
"""

import pytest
from rehab_cv.biomechanics.smoothing import (
    ExponentialMovingAverage,
    RollingWindowAverage,
    AngleSmoother,
)
from rehab_cv.config import SmoothingConfig


def test_ema_noise_reduction():
    """EMA should dampen high-frequency oscillating noise around a true mean."""
    ema = ExponentialMovingAverage(alpha=0.30)

    # True angle is 150°, but sensor noise oscillates [140°, 160°]
    noisy_sequence = [150.0, 160.0, 140.0, 162.0, 138.0, 159.0, 141.0, 150.0]
    smoothed_values = [ema.update(val) for val in noisy_sequence]

    # Max variance of smoothed values should be strictly less than raw noisy amplitude
    raw_range = max(noisy_sequence) - min(noisy_sequence)  # 24°
    smoothed_tail = smoothed_values[2:]
    smoothed_range = max(smoothed_tail) - min(smoothed_tail)

    assert smoothed_range < raw_range * 0.75, f"Smoother failed to dampen noise: {smoothed_range} vs {raw_range}"


def test_missing_observations_do_not_insert_zero():
    """
    Critical requirement: When occlusion occurs (raw_angle is None),
    the smoother must NOT insert fake 0.0 values, which would ruin rep counting.
    """
    ema = ExponentialMovingAverage(alpha=0.40, max_missing_frames=4)

    # Initial valid angles
    ema.update(120.0)
    last_valid = ema.update(125.0)
    assert last_valid is not None and last_valid > 100.0

    # 3 frames of temporary occlusion (None)
    for _ in range(3):
        val = ema.update(None)
        assert val is not None, "Temporary occlusion dropped smoothed state prematurely"
        assert val != 0.0, "Smoother inserted fake 0.0 value!"
        assert pytest.approx(val, abs=1.0) == last_valid

    # Frame 4 of occlusion: still within tolerance
    val4 = ema.update(None)
    assert val4 is not None

    # Frame 5 of occlusion: exceeds max_missing_frames (4) -> resets to None
    val5 = ema.update(None)
    assert val5 is None


def test_recovery_after_tracking_resumes():
    """When tracking resumes after an extended gap, the filter initializes cleanly without lag spike."""
    ema = ExponentialMovingAverage(alpha=0.35, max_missing_frames=3)

    ema.update(90.0)
    ema.update(95.0)

    # Long occlusion gap
    for _ in range(6):
        ema.update(None)

    # Tracking resumes at 160.0°
    resumed = ema.update(160.0)
    # First measurement after reset should adopt 160.0 directly (not weighted with stale 95°)
    assert resumed == 160.0


def test_rolling_window_average():
    """Rolling window produces arithmetic mean of window observations."""
    window = RollingWindowAverage(window_size=3)

    assert window.update(90.0) == 90.0
    assert window.update(100.0) == 95.0   # (90 + 100) / 2
    assert window.update(110.0) == 100.0  # (90 + 100 + 110) / 3
    assert window.update(120.0) == 110.0  # (100 + 110 + 120) / 3


def test_unified_angle_smoother_interface():
    """AngleSmoother exposes both last_raw and last_smoothed for debugging."""
    config = SmoothingConfig(method="ema", alpha=0.5)
    smoother = AngleSmoother(config)

    s1 = smoother.update(100.0)
    assert smoother.last_raw == 100.0
    assert smoother.last_smoothed == 100.0

    s2 = smoother.update(120.0)
    assert smoother.last_raw == 120.0
    assert smoother.last_smoothed == 110.0  # 0.5 * 120 + 0.5 * 100

    smoother.reset()
    assert smoother.last_raw is None
    assert smoother.last_smoothed is None
