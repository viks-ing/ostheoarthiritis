"""
rehab_cv/biomechanics/smoothing.py - Angle Smoothing & Temporal Filtering
Implements Exponential Moving Average (EMA) and Rolling Window smoothing
with missing observation handling (zero-insertion avoidance) and low-latency response.
"""

from collections import deque
from typing import Optional, List
from ..config import SmoothingConfig


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) filter:
        S_0 = X_0
        S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    
    Attributes:
        alpha: Weight given to the newest observation [0.0, 1.0].
               Higher alpha = faster response, less smoothing.
               Lower alpha = smoother, slightly more latency.
               Default alpha=0.35 yields optimal compromise for rehabilitation motions (~15-30 deg/sec).
    """

    def __init__(self, alpha: float = 0.35, max_missing_frames: int = 6):
        self.alpha = float(min(1.0, max(0.01, alpha)))
        self.max_missing_frames = max_missing_frames
        self.current_smoothed: Optional[float] = None
        self.missing_count: int = 0

    def update(self, raw_value: Optional[float]) -> Optional[float]:
        """
        Updates the EMA with a new raw measurement.
        If raw_value is None (occlusion/low confidence):
            - Increments missing count.
            - If missing count <= max_missing_frames: returns last smoothed value (avoiding fake 0°).
            - If missing count > max_missing_frames: resets filter and returns None.
        """
        if raw_value is None:
            self.missing_count += 1
            if self.missing_count > self.max_missing_frames:
                self.current_smoothed = None
            return self.current_smoothed

        # Valid observation received
        self.missing_count = 0

        if self.current_smoothed is None:
            # First observation initializes filter immediately without ramp-up delay
            self.current_smoothed = float(raw_value)
        else:
            self.current_smoothed = (self.alpha * float(raw_value)) + ((1.0 - self.alpha) * self.current_smoothed)

        return round(self.current_smoothed, 2)

    def reset(self):
        """Clears smoothed state."""
        self.current_smoothed = None
        self.missing_count = 0


class RollingWindowAverage:
    """
    Moving Average over a sliding window of recent observations.
    """

    def __init__(self, window_size: int = 5, max_missing_frames: int = 6):
        self.window_size = max(1, window_size)
        self.max_missing_frames = max_missing_frames
        self.window: deque = deque(maxlen=self.window_size)
        self.missing_count: int = 0

    def update(self, raw_value: Optional[float]) -> Optional[float]:
        """Updates window and returns arithmetic mean of active observations."""
        if raw_value is None:
            self.missing_count += 1
            if self.missing_count > self.max_missing_frames:
                self.window.clear()
                return None
            return round(sum(self.window) / len(self.window), 2) if self.window else None

        self.missing_count = 0
        self.window.append(float(raw_value))
        return round(sum(self.window) / len(self.window), 2)

    def reset(self):
        self.window.clear()
        self.missing_count = 0


class AngleSmoother:
    """
    Unified Angle Smoother providing both raw and smoothed angle streams.
    Switches between EMA and Rolling Window based on SmoothingConfig.
    """

    def __init__(self, config: Optional[SmoothingConfig] = None):
        self.config = config or SmoothingConfig()
        if self.config.method.lower() == "window":
            self._filter = RollingWindowAverage(
                window_size=self.config.window_size,
                max_missing_frames=self.config.max_missing_frames,
            )
        else:
            self._filter = ExponentialMovingAverage(
                alpha=self.config.alpha,
                max_missing_frames=self.config.max_missing_frames,
            )
        self.last_raw: Optional[float] = None
        self.last_smoothed: Optional[float] = None

    def update(self, raw_angle: Optional[float]) -> Optional[float]:
        """Processes raw angle and returns smoothed output."""
        self.last_raw = raw_angle
        self.last_smoothed = self._filter.update(raw_angle)
        return self.last_smoothed

    def reset(self):
        """Resets smoother state."""
        self.last_raw = None
        self.last_smoothed = None
        self._filter.reset()
