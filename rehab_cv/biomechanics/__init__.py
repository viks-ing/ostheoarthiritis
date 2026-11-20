"""
biomechanics package - 2D Vector Angle Calculation & Smoothing
"""

from .angle_calculator import (
    calculate_knee_angle_2d,
    AngleCalculationResult,
    LegSelectionManager,
)
from .smoothing import AngleSmoother, ExponentialMovingAverage, RollingWindowAverage

__all__ = [
    "calculate_knee_angle_2d",
    "AngleCalculationResult",
    "LegSelectionManager",
    "AngleSmoother",
    "ExponentialMovingAverage",
    "RollingWindowAverage",
]
