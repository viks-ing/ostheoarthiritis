"""
rehab_cv/detection/pose_detector.py - MediaPipe Pose Detection Wrapper
Extracts 33 body landmarks with visibility scoring, isolates left and right lower limb
triplets (Hip, Knee, Ankle), converts normalized coords to pixel space, and handles occlusion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np
import cv2
import mediapipe as mp

from ..config import PoseConfig


@dataclass
class LandmarkPoint:
    """Represents a single body landmark with normalized, pixel, and visibility attributes."""
    x: float          # Normalized X [0.0, 1.0]
    y: float          # Normalized Y [0.0, 1.0]
    z: float          # Normalized depth
    visibility: float # Confidence / visibility score [0.0, 1.0]
    presence: float   # Landmark presence probability [0.0, 1.0]
    pixel_x: int      # Absolute image coordinate in pixels
    pixel_y: int      # Absolute image coordinate in pixels

    def as_tuple_2d(self) -> Tuple[float, float]:
        """Returns 2D normalized coordinate pair."""
        return (self.x, self.y)

    def as_pixel_tuple(self) -> Tuple[int, int]:
        """Returns 2D integer pixel coordinate pair."""
        return (self.pixel_x, self.pixel_y)


@dataclass
class LegLandmarks:
    """Encapsulates the hip, knee, and ankle landmarks for one limb."""
    side: str  # "left" or "right"
    hip: Optional[LandmarkPoint]
    knee: Optional[LandmarkPoint]
    ankle: Optional[LandmarkPoint]
    is_valid: bool
    mean_visibility: float


@dataclass
class PoseDetectionResult:
    """High-level structured result for a single processed video frame."""
    detected: bool
    raw_landmarks: Any
    left_leg: LegLandmarks
    right_leg: LegLandmarks
    frame_width: int
    frame_height: int


class PoseDetector:
    """
    Encapsulates MediaPipe Pose estimator.
    Processes BGR frames, extracts normalized and pixel coordinates,
    and isolates lower extremity joint triplets.
    """

    # MediaPipe Landmark Index Mapping
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or PoseConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            enable_segmentation=self.config.enable_segmentation,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def _extract_landmark(
        self,
        raw_lm: Any,
        width: int,
        height: int,
    ) -> LandmarkPoint:
        """Converts MediaPipe landmark proto to strongly-typed LandmarkPoint with pixel mapping."""
        px = int(np.clip(raw_lm.x * width, 0, width - 1))
        py = int(np.clip(raw_lm.y * height, 0, height - 1))
        visibility = getattr(raw_lm, "visibility", 1.0)
        presence = getattr(raw_lm, "presence", 1.0)
        return LandmarkPoint(
            x=float(raw_lm.x),
            y=float(raw_lm.y),
            z=float(raw_lm.z),
            visibility=float(visibility),
            presence=float(presence),
            pixel_x=px,
            pixel_y=py,
        )

    def _build_leg_landmarks(
        self,
        side: str,
        hip_idx: int,
        knee_idx: int,
        ankle_idx: int,
        landmarks: Any,
        width: int,
        height: int,
    ) -> LegLandmarks:
        """Builds LegLandmarks struct and verifies minimum visibility thresholds."""
        if landmarks is None or len(landmarks.landmark) <= max(hip_idx, knee_idx, ankle_idx):
            return LegLandmarks(
                side=side,
                hip=None,
                knee=None,
                ankle=None,
                is_valid=False,
                mean_visibility=0.0,
            )

        raw_hip = landmarks.landmark[hip_idx]
        raw_knee = landmarks.landmark[knee_idx]
        raw_ankle = landmarks.landmark[ankle_idx]

        hip = self._extract_landmark(raw_hip, width, height)
        knee = self._extract_landmark(raw_knee, width, height)
        ankle = self._extract_landmark(raw_ankle, width, height)

        vis_thresh = self.config.min_landmark_visibility
        is_valid = (
            hip.visibility >= vis_thresh
            and knee.visibility >= vis_thresh
            and ankle.visibility >= vis_thresh
        )
        mean_vis = (hip.visibility + knee.visibility + ankle.visibility) / 3.0

        return LegLandmarks(
            side=side,
            hip=hip,
            knee=knee,
            ankle=ankle,
            is_valid=is_valid,
            mean_visibility=mean_vis,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseDetectionResult:
        """
        Executes MediaPipe pose detection on a single BGR OpenCV frame.
        Returns PoseDetectionResult containing parsed joints for left and right limbs.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseDetectionResult(
                detected=False,
                raw_landmarks=None,
                left_leg=LegLandmarks("left", None, None, None, False, 0.0),
                right_leg=LegLandmarks("right", None, None, None, False, 0.0),
                frame_width=0,
                frame_height=0,
            )

        height, width = frame_bgr.shape[:2]

        # MediaPipe expects RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)

        if not results or not results.pose_landmarks:
            return PoseDetectionResult(
                detected=False,
                raw_landmarks=None,
                left_leg=LegLandmarks("left", None, None, None, False, 0.0),
                right_leg=LegLandmarks("right", None, None, None, False, 0.0),
                frame_width=width,
                frame_height=height,
            )

        landmarks = results.pose_landmarks

        left_leg = self._build_leg_landmarks(
            side="left",
            hip_idx=self.LEFT_HIP,
            knee_idx=self.LEFT_KNEE,
            ankle_idx=self.LEFT_ANKLE,
            landmarks=landmarks,
            width=width,
            height=height,
        )

        right_leg = self._build_leg_landmarks(
            side="right",
            hip_idx=self.RIGHT_HIP,
            knee_idx=self.RIGHT_KNEE,
            ankle_idx=self.RIGHT_ANKLE,
            landmarks=landmarks,
            width=width,
            height=height,
        )

        return PoseDetectionResult(
            detected=True,
            raw_landmarks=landmarks,
            left_leg=left_leg,
            right_leg=right_leg,
            frame_width=width,
            frame_height=height,
        )

    def close(self):
        """Releases underlying MediaPipe graph resources."""
        if hasattr(self, "pose") and self.pose is not None:
            self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
