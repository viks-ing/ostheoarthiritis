"""
rehab_cv/utils/camera.py - Camera Capture Manager & Synthetic Simulator
Provides robust webcam capture with error handling, auto-mirroring,
and an integrated Synthetic Camera generator for test automation and headless execution.
"""

import time
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from ..config import CameraConfig


class SyntheticCamera:
    """
    Generates synthetic video frames simulating a human performing seated knee extension.
    Renders realistic humanoid geometric limb segments (torso, thigh, lower leg)
    undergoing a smooth periodic 90° -> 165° -> 90° knee extension motion.
    Useful for continuous integration, sandbox environments, and automated verification.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.cycle_period_sec = 4.0  # 4 seconds per complete repetition cycle
        self.is_opened = True

    def read(self) -> Tuple[bool, np.ndarray]:
        """Generates the next synthetic frame in the motion sequence."""
        if not self.is_opened:
            return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        frame = np.full((self.height, self.width, 3), (35, 30, 30), dtype=np.uint8)

        # Draw a simple room background with chair
        cv2.line(frame, (100, int(self.height * 0.8)), (self.width - 100, int(self.height * 0.8)), (60, 60, 60), 3)

        # Simulation Kinematics (Side profile: facing right)
        t = (self.frame_idx / self.fps) % self.cycle_period_sec
        # Phase 0.0 to 1.0 of the cycle
        phase = t / self.cycle_period_sec
        # Sinusoidal motion: 0 at start, 1 at peak (t=cycle/2), 0 at return
        # Angle ranges from 90° (seated flexion) to 165° (terminal extension)
        norm_extension = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        current_knee_angle = 90.0 + (75.0 * norm_extension)

        # Humanoid Dimensions
        hip_x = int(self.width * 0.40)
        hip_y = int(self.height * 0.45)
        thigh_len = int(self.height * 0.22)
        shank_len = int(self.height * 0.22)

        # Seated thigh is horizontal: Hip to Knee
        knee_x = hip_x + thigh_len
        knee_y = hip_y

        # Knee angle relative to thigh:
        # At 180° (straight leg), shank continues horizontal to the right (angle 0°).
        # At 90° (flexion), shank drops straight down (+90° in image coords).
        # Internal angle theta = 180° - deflection_angle => deflection = 180° - theta.
        deflection_deg = 180.0 - current_knee_angle
        deflection_rad = math.radians(deflection_deg)

        ankle_x = int(knee_x + shank_len * math.cos(deflection_rad))
        ankle_y = int(knee_y + shank_len * math.sin(deflection_rad))

        # Head and Torso
        shoulder_x = hip_x - 30
        shoulder_y = hip_y - int(self.height * 0.25)
        head_x = shoulder_x
        head_y = shoulder_y - 45

        # Render Humanoid Character (Skin & Clothing tones)
        # Head
        cv2.circle(frame, (head_x, head_y), 32, (180, 200, 220), -1)
        # Torso
        cv2.line(frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (140, 100, 70), 28)
        # Thigh (Leg)
        cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), (80, 80, 160), 22)
        # Shank (Lower leg)
        cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), (180, 200, 220), 18)
        # Foot
        cv2.line(frame, (ankle_x, ankle_y), (ankle_x + 35, ankle_y + 10), (50, 50, 50), 12)

        # Joints markers for visual feedback
        cv2.circle(frame, (hip_x, hip_y), 10, (0, 255, 255), -1)
        cv2.circle(frame, (knee_x, knee_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (ankle_x, ankle_y), 10, (255, 255, 0), -1)

        # Simulation watermark
        cv2.putText(
            frame,
            f"SYNTHETIC SIMULATOR - Target Angle: {current_knee_angle:.1f} deg",
            (30, self.height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )

        self.frame_idx += 1
        return True, frame

    def release(self):
        self.is_opened = False


class CameraManager:
    """
    Manages OpenCV VideoCapture lifecycle with resolution configuration,
    horizontal mirroring, and seamless fallback to SyntheticCamera if webcam is unavailable.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.synthetic: Optional[SyntheticCamera] = None
        self.is_synthetic: bool = False

        self._initialize_camera()

    def _initialize_camera(self):
        """Attempts to open physical webcam, falling back to synthetic generator if needed."""
        self.cap = cv2.VideoCapture(self.config.device_index)

        if self.cap.isOpened():
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            # Read test frame to verify actual image stream
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                self.is_synthetic = False
                return

        # Physical camera failed or unavailable
        if self.config.fallback_synthetic:
            print("[*] Physical camera unavailable. Initializing Synthetic Video Simulator...")
            if self.cap:
                self.cap.release()
            self.cap = None
            self.synthetic = SyntheticCamera(
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )
            self.is_synthetic = True
        else:
            raise RuntimeError(f"Unable to access camera device index {self.config.device_index}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves next frame from physical camera or synthetic stream.
        Applies horizontal mirroring if configured.
        """
        if self.is_synthetic and self.synthetic is not None:
            ret, frame = self.synthetic.read()
            return ret, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self):
        """Releases camera hardware or simulator resources cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.synthetic is not None:
            self.synthetic.release()
            self.synthetic = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
