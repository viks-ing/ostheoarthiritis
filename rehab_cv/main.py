"""
rehab_cv/main.py - Main Entrypoint for KneeWise Rehabilitation CV Engine
Integrates Camera Acquisition, MediaPipe Pose Detection, 2D Angle Biomechanics,
Temporal Smoothing, Repetition Counting FSM, and OpenCV HUD Visualization.

Usage:
    # Run with default physical webcam
    python rehab_cv/main.py

    # Run with synthetic motion simulator (no webcam required)
    python rehab_cv/main.py --synthetic

    # Run headless verification mode (processes N frames without GUI window)
    python rehab_cv/main.py --synthetic --headless --max-frames 240

Keyboard Controls (during live GUI session):
    [Q] / [ESC] : Quit application cleanly
    [R]         : Reset repetition count and state machine
    [L]         : Switch active leg (safely resets active rep)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rehab_cv.config import (
    RehabCVConfig,
    CameraConfig,
    PoseConfig,
    SmoothingConfig,
    ExerciseConfig,
    OverlayConfig,
)
from rehab_cv.utils.camera import CameraManager
from rehab_cv.detection.pose_detector import PoseDetector
from rehab_cv.exercises.knee_extension import KneeExtensionExercise
from rehab_cv.visualization.overlay import VisualOverlayRenderer
from rehab_cv.feedback.feedback_engine import RehabilitationFeedbackEngine
from rehab_cv.feedback.session_analyzer import SessionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KneeWise Seated Knee-Extension Rehabilitation CV Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=1280, help="Video stream width")
    parser.add_argument("--height", type=int, default=720, help="Video stream height")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal selfie mirror flip")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic exercise motion generator")
    parser.add_argument("--leg", choices=["auto", "left", "right"], default="auto", help="Tracking leg preference")
    parser.add_argument("--complexity", type=int, choices=[0, 1, 2], default=1, help="MediaPipe Pose complexity (0=Lite, 1=Full, 2=Heavy)")
    parser.add_argument("--headless", action="store_true", help="Headless execution mode without cv2.imshow GUI")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop execution after processing N frames")
    return parser.parse_args()


def run_rehabilitation_engine(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()

    print("=" * 75)
    print("      KNEEWISE REHABILITATION CV ENGINE — SEATED KNEE EXTENSION")
    print("=" * 75)
    print(f"[*] Camera Device:     {'Synthetic Simulator' if args.synthetic else f'Index {args.camera}'}")
    print(f"[*] Target Stream:     {args.width}x{args.height} @ {args.fps} FPS")
    print(f"[*] Leg Selection:     {args.leg.upper()}")
    print(f"[*] Model Complexity:  {args.complexity} (MediaPipe Pose)")
    print(f"[*] GUI Mode:          {'Headless (No GUI)' if args.headless else 'Interactive OpenCV Window'}")
    print("-" * 75)

    # 1. Initialize Configuration
    cam_cfg = CameraConfig(
        device_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
        fallback_synthetic=True,
    )
    if args.synthetic:
        cam_cfg.fallback_synthetic = True

    pose_cfg = PoseConfig(
        model_complexity=args.complexity,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
        min_landmark_visibility=0.60,
    )

    exercise_cfg = ExerciseConfig(
        preferred_leg=args.leg,
        flexion_threshold_deg=100.0,
        extension_threshold_deg=155.0,
        return_threshold_deg=115.0,
        min_excursion_deg=45.0,
        min_state_duration_sec=0.12,
        cooldown_sec=0.60,
    )

    overlay_cfg = OverlayConfig()

    # 2. Instantiate Components
    camera = CameraManager(cam_cfg)
    if args.synthetic:
        # Force synthetic mode directly
        camera.is_synthetic = True
        from rehab_cv.utils.camera import SyntheticCamera
        camera.synthetic = SyntheticCamera(width=args.width, height=args.height, fps=args.fps)

    detector = PoseDetector(pose_cfg)
    exercise = KneeExtensionExercise(exercise_cfg)
    renderer = VisualOverlayRenderer(overlay_cfg)
    feedback_engine = RehabilitationFeedbackEngine()
    analyzer = SessionAnalyzer(selected_leg=args.leg)
    analyzer.start_session()

    # Performance Monitoring
    fps_estimate = 30.0
    frame_count = 0
    t_start = time.time()
    t_last_fps = time.time()
    fps_frame_count = 0

    window_name = "KneeWise Rehabilitation CV Engine"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    print("[+] Engine initialized. Streaming frames... Press 'Q' to quit.\n")

    try:
        while True:
            t_frame_start = time.time()

            # A. Capture Frame
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[!] End of video stream or camera read failure.")
                break

            frame_count += 1

            # B. Detect Body Pose
            detection = detector.process(frame)

            # C. Evaluate Knee Biomechanics & Rep FSM
            metrics = exercise.process(detection)
            feedback = feedback_engine.process(metrics)
            analyzer.record_frame(metrics)

            # Log rep completion in console
            if metrics.rep_completed_this_frame and metrics.last_completed_rep:
                rep = metrics.last_completed_rep
                print(
                    f"[+] REPETITION #{rep.rep_number:02d} COMPLETED! "
                    f"Limb: {rep.side.upper()} | Peak Extension: {rep.max_extension_angle:.1f}° | "
                    f"ROM: {rep.excursion_deg:.1f}° | Duration: {rep.duration_sec:.2f}s"
                )

            # D. Render Visual HUD Overlay
            annotated_frame = renderer.render(
                frame=frame,
                detection=detection,
                metrics=metrics,
                fps=fps_estimate,
                feedback_event=feedback,
            )

            # E. Calculate FPS
            fps_frame_count += 1
            t_now = time.time()
            if (t_now - t_last_fps) >= 1.0:
                fps_estimate = fps_frame_count / (t_now - t_last_fps)
                fps_frame_count = 0
                t_last_fps = t_now

            # F. Display Window (if not headless)
            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("\n[*] User requested application exit (Q/ESC).")
                    break
                elif key in (ord("r"), ord("R")):    # Reset
                    exercise.reset()
                    feedback_engine.reset()
                    print("[*] Repetition counter and state machine reset by user.")
                elif key in (ord("l"), ord("L")):    # Switch leg
                    new_side = exercise.switch_leg()
                    print(f"[*] Switched active tracking leg to: {new_side.upper()}")

            # G. Check Max Frames (for testing/benchmarks)
            if args.max_frames is not None and frame_count >= args.max_frames:
                print(f"[*] Reached target maximum frame count ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        # Cleanup
        camera.release()
        detector.close()
        if not args.headless:
            cv2.destroyAllWindows()

        # Session Summary Report
        summary = analyzer.finish_session()
        total_time = summary.duration_sec
        session = exercise.get_session_metrics()
        print("\n" + "=" * 75)
        print("                 REHABILITATION SESSION SUMMARY")
        print("=" * 75)
        print(f"Total Session Duration: {total_time:.2f} seconds")
        print(f"Total Frames Processed: {frame_count} frames ({frame_count / max(0.001, total_time):.1f} avg FPS)")
        print(f"Tracking Availability:  {summary.tracking_availability_pct:.1f}% ({summary.tracked_frames}/{summary.total_frames} frames)")
        print(f"Active Leg Tracked:     {session.selected_leg.upper()}")
        print(f"Valid Repetitions:      {session.total_reps}")
        print(f"Rejected Incomplete:    {session.rejected_reps}")
        print(f"Average ROM Excursion:  {summary.avg_excursion_deg:.1f}°")
        print(f"Best ROM Excursion:     {summary.best_excursion_deg:.1f}°")
        print(f"Average Rep Duration:   {session.average_duration:.2f} seconds")
        print("=" * 75)


if __name__ == "__main__":
    run_rehabilitation_engine()
