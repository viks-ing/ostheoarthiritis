"""
rehab_cv/config.py - Centralized Configuration Module for KneeWise Rehabilitation CV Engine
Defines strongly-typed dataclasses for camera acquisition, MediaPipe pose detection,
knee-angle biomechanics calculation, repetition counting finite state machine, and overlay aesthetics.

Important Engineering Notice:
Thresholds and values configured here are initial engineering heuristics designed
for computer vision tracking stability. They do NOT constitute medical or physical
therapy prescriptions.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class CameraConfig:
    """Configuration for video acquisition."""
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True  # Horizontal flip for intuitive mirror-like user feedback
    fallback_synthetic: bool = True  # Use synthetic frame stream if physical webcam is unavailable


@dataclass
class PoseConfig:
    """Configuration for MediaPipe Pose detection and landmark filtering."""
    static_image_mode: bool = False
    model_complexity: int = 0  # 0=Lite (Fastest, ~10ms CPU inference), 1=Full, 2=Heavy
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    min_detection_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    min_landmark_visibility: float = 0.40  # Minimum confidence threshold for Hip, Knee, and Ankle joints


@dataclass
class SmoothingConfig:
    """Configuration for angle filtering and noise reduction."""
    method: str = "ema"  # Options: "ema" (Exponential Moving Average), "window" (Rolling Window)
    alpha: float = 0.65  # Snappy EMA factor: responsive real-time angle with noise suppression
    window_size: int = 3  # Number of frames for rolling window average
    max_missing_frames: int = 6  # Tolerated frames of occlusion before smoother resets


@dataclass
class ExerciseConfig:
    """
    Configuration for Seated Knee-Extension Exercise Tracking.
    FSM Sequence: WAITING -> EXTENDING -> EXTENDED -> RETURNING -> WAITING
    """
    name: str = "Seated Knee Extension"
    preferred_leg: str = "auto"  # Options: "auto", "left", "right"
    
    # Biomechanical Angle Thresholds (in degrees, 180° = fully straight leg, 90° = right-angle flexion)
    flexion_threshold_deg: float = 105.0   # Knee angle indicating rested/seated starting posture
    extension_threshold_deg: float = 150.0  # Knee angle indicating valid terminal knee extension (>= 150°)
    return_threshold_deg: float = 115.0     # Knee angle threshold to verify leg returned back to flexion (<= 115°)
    min_excursion_deg: float = 35.0         # Minimum range of motion (ROM) required between min flexion and peak extension
    
    # Temporal Debounce and Safety Settings
    min_state_duration_sec: float = 0.05    # Fast state confirmation without lag
    cooldown_sec: float = 0.25              # Quick cooldown before starting next repetition
    tracking_loss_reset_sec: float = 1.20   # Timeout after landmark occlusion before safely resetting active exercise state
    hysteresis_frames: int = 1              # Immediate threshold boundary response without 3-frame delay


@dataclass
class OverlayConfig:
    """Configuration for OpenCV real-time visual feedback and HUD overlay."""
    # BGR Color Palette
    color_bg_panel: Tuple[int, int, int] = (25, 25, 30)         # Deep slate dark background
    color_text_primary: Tuple[int, int, int] = (245, 245, 245)  # Bright off-white
    color_text_muted: Tuple[int, int, int] = (160, 160, 165)    # Slate gray
    color_accent_cyan: Tuple[int, int, int] = (235, 180, 50)    # Active tracking cyan (BGR)
    color_success_green: Tuple[int, int, int] = (80, 200, 80)   # Completed/Good green
    color_warning_amber: Tuple[int, int, int] = (40, 180, 240)  # Cautious/Waiting amber
    color_danger_red: Tuple[int, int, int] = (60, 60, 230)      # Tracking lost red
    color_unselected_leg: Tuple[int, int, int] = (100, 100, 100)# Dim gray for inactive limb

    # Line Thickness & Radii
    joint_radius: int = 7
    active_joint_radius: int = 11
    skeleton_thickness: int = 3
    active_skeleton_thickness: int = 5
    font_face: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    hud_alpha: float = 0.75  # Transparency of HUD panels


@dataclass
class RehabCVConfig:
    """Master configuration aggregate."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    exercise: ExerciseConfig = field(default_factory=ExerciseConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
