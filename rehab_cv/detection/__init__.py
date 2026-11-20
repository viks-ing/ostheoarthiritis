"""
detection package - MediaPipe Pose Detection for KneeWise Rehabilitation
"""

from .pose_detector import PoseDetector, LandmarkPoint, LegLandmarks, PoseDetectionResult

__all__ = ["PoseDetector", "LandmarkPoint", "LegLandmarks", "PoseDetectionResult"]
