# KneeWise — Rehabilitation Computer Vision Engine
## Real-Time Seated Knee-Extension Tracking & Repetition Counting Prototype
### Milestone: CV-1 (Pose Detection) + CV-2 (Knee-Angle Biomechanics) + CV-3 (Repetition FSM)

---

## 1. Executive Summary & Objective

The **KneeWise Rehabilitation CV Engine** is an independent, low-latency computer vision module engineered to assist patients recovering from knee surgery or managing osteoarthritis through guided home exercise therapy.

This milestone prototype specifically tracks the **Seated Knee Extension** exercise—a foundational open-kinetic-chain quadriceps strengthening exercise. The system performs:
1. Real-time body pose estimation using **MediaPipe Pose** and **OpenCV**.
2. Robust 2D vector-based knee angle computation from hip, knee, and ankle joint coordinates.
3. Temporal angle smoothing via **Exponential Moving Average (EMA)** to suppress sensor noise without introducing phase delay.
4. A deterministic 4-state **Finite State Machine (FSM)** with debouncing and cooldown to accurately count valid repetitions, reject incomplete or truncated movements, and resist occlusion artifacts.

```
       +------------------+
       |   Webcam Frame   |
       +------------------+
                 |
                 v
       +------------------+
       |  MediaPipe Pose  | ----> Isolates Hip (23/24), Knee (25/26), Ankle (27/28)
       +------------------+
                 |
                 v
       +------------------+
       | Angle Calculator | ----> 2D Vector Geometry: cos(θ) = (v1 · v2) / (|v1||v2|)
       +------------------+
                 |
                 v
       +------------------+
       |  Angle Smoother  | ----> EMA Filtering: St = α·Xt + (1-α)·St-1 (Zero-Insertion Resilient)
       +------------------+
                 |
                 v
       +------------------+
       |  Repetition FSM  | ----> WAITING -> EXTENDING -> EXTENDED -> RETURNING -> WAITING
       +------------------+
                 |
                 v
       +------------------+
       | OpenCV HUD View  | ----> Real-Time Visual Feedback, Angle Badge, Rep Counter
       +------------------+
```

---

## 2. Hardware Architecture & System Requirements

Designed and benchmarked for consumer-grade hardware:
- **Operating System:** Windows 10 / Windows 11
- **CPU:** AMD Ryzen 5 / Intel Core i5 (Processes MediaPipe Pose on CPU at ~15–30 FPS)
- **RAM:** 8 GB
- **GPU (Optional):** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Webcam:** 720p / 1080p USB or integrated webcam

### Pinned Dependencies (`requirements.txt`)
- `opencv-contrib-python>=4.8.0` (Verified: `4.13.0`)
- `mediapipe==0.10.14` (Verified: `0.10.14` with bundled Pose graph)
- `numpy>=1.24.0` (Verified: `2.4.3`)
- `pytest>=8.0.0` (Verified: `9.1.1`)

---

## 3. Project Directory Structure

```text
rehab_cv/
├── main.py                     # Interactive CLI application & real-time OpenCV rendering loop
├── config.py                   # Centralized configuration dataclasses & engineering defaults
├── requirements.txt            # Pinned Python package dependencies
├── README.md                   # Complete architectural and clinical documentation
│
├── detection/
│   ├── __init__.py
│   └── pose_detector.py        # MediaPipe Pose abstraction, normalized/pixel joints, visibility
│
├── biomechanics/
│   ├── __init__.py
│   ├── angle_calculator.py     # 2D planar vector angle calculation & mid-rep leg selection lock
│   └── smoothing.py            # Exponential Moving Average & rolling window filters
│
├── exercises/
│   ├── __init__.py
│   ├── knee_extension.py       # High-level exercise session coordinator & metrics aggregator
│   └── repetition_counter.py   # 4-state Finite State Machine with debounce & excursion validation
│
├── visualization/
│   ├── __init__.py
│   └── overlay.py              # Medical HUD banner, highlighted skeleton, dynamic angle badges
│
├── utils/
│   ├── __init__.py
│   └── camera.py               # Robust VideoCapture wrapper & Synthetic Camera Simulator
│
└── tests/
    ├── __init__.py
    ├── test_angle_calculator.py    # Unit tests for 180°, 90°, collinear, zero-vector, NaN safety
    ├── test_smoothing.py           # Unit tests for EMA convergence, missing frames, recovery
    ├── test_repetition_counter.py  # Unit tests for FSM transitions, incomplete reps, cooldown
    └── test_knee_extension.py      # Integration tests for end-to-end sessions and leg switching
```

---

## 4. Biomechanical Angle Calculation Methodology

The 2D knee angle $\theta$ is calculated directly from the three key landmark coordinates of the active leg:
- **Hip:** $\vec{P}_{\text{hip}} = (x_h, y_h)$ (MediaPipe index 23 left, 24 right)
- **Knee:** $\vec{P}_{\text{knee}} = (x_k, y_k)$ (MediaPipe index 25 left, 26 right)
- **Ankle:** $\vec{P}_{\text{ankle}} = (x_a, y_a)$ (MediaPipe index 27 left, 28 right)

### Mathematical Formulation
Two 2D vectors are defined originating from the knee joint center:
$$\vec{v}_1 = \vec{P}_{\text{hip}} - \vec{P}_{\text{knee}} = (x_h - x_k, \, y_h - y_k)$$
$$\vec{v}_2 = \vec{P}_{\text{ankle}} - \vec{P}_{\text{knee}} = (x_a - x_k, \, y_a - y_k)$$

The internal angle $\theta$ subtended at the knee is given by the vector dot product:
$$\cos(\theta) = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|} = \frac{(x_h - x_k)(x_a - x_k) + (y_h - y_k)(y_a - y_k)}{\sqrt{(x_h - x_k)^2 + (y_h - y_k)^2} \cdot \sqrt{(x_a - x_k)^2 + (y_a - y_k)^2}}$$

$$\theta = \arccos\left(\text{clamp}\left(\cos(\theta), -1.0, 1.0\right)\right) \times \frac{180^\circ}{\pi}$$

### Angle Interpretation
- **Fully Straight Leg (Terminal Extension):** $\theta \approx 180^\circ$ ($\vec{v}_1$ and $\vec{v}_2$ point in opposite directions).
- **Seated Bent Leg (Resting Flexion):** $\theta \approx 90^\circ$ ($\vec{v}_1 \perp \vec{v}_2$).
- **Deep Flexion:** $\theta < 90^\circ$.

### Edge Cases Handled Safely
1. **Coincident Joints (Zero-Length Vectors):** If $\|\vec{v}_1\| < 10^{-7}$ or $\|\vec{v}_2\| < 10^{-7}$, returns `None` with an informative error string.
2. **Numerical Instability:** Dot product ratio is strictly clamped to $[-1.0, 1.0]$ prior to `acos()`.
3. **Visibility Gating:** Any joint with confidence $< 0.60$ causes the angle calculation to reject the observation.

---

## 5. Temporal Angle Smoothing

Raw landmark pixel predictions exhibit high-frequency jitter caused by camera sensor noise and lighting variations. An **Exponential Moving Average (EMA)** filter suppresses jitter while minimizing phase lag:
$$S_0 = X_0$$
$$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}$$

### Configuration Parameters
- **$\alpha = 0.35$:** Selected to provide smooth visual feedback without delaying the recognition of extension peaks.
- **Occlusion Handling (Zero-Insertion Resilient):** If a landmark is temporarily occluded ($X_t = \text{None}$), the filter **never** inserts a fake $0^\circ$ value (which would immediately corrupt rep tracking). Instead, it sustains the last valid smoothed value for up to `max_missing_frames = 6` frames (~200ms). If occlusion persists beyond this window, it resets safely.

---

## 6. Finite State Machine (FSM) Repetition Counting

Repetition counting is governed by a strict 4-state Finite State Machine:

```
          [Resting Seated Flexion <= 100°]
                    +----------+
                    | WAITING  | <------------------------------------+
                    +----------+                                      |
                         |                                            |
                         | Angle > 105°                               | Valid Return
                         v                                            | (Angle <= 115° and
                    +-----------+                                     |  ROM >= 45°)
     Incomplete     | EXTENDING |                                     |
     Return <= 100° +-----------+                                     |
        +--------->      |                                            |
        |                | Angle >= 155° and Excursion >= 45°         |
        |                v                                            |
        |           +----------+                                      |
        |           | EXTENDED | (Peak Extension Hold)                |
        |           +----------+                                      |
        |                |                                            |
        |                | Angle < 150°                               |
        |                v                                            |
        |           +-----------+                                     |
        +-----------| RETURNING | ------------------------------------+
                    +-----------+
```

### State Transitions & Rules
1. **`WAITING`:**
   - Patient is seated with knee bent ($\theta \le 100^\circ$).
   - Baselines the starting flexion angle $\theta_{\text{start}}$.
   - Transitions to `EXTENDING` when $\theta > 105^\circ$ for $\ge 3$ consecutive frames.
2. **`EXTENDING`:**
   - Leg is actively straightening. Tracks the maximum extension angle achieved ($\theta_{\text{peak}}$).
   - If user gives up and drops back to $\le 100^\circ$ without reaching $155^\circ$, the repetition is **rejected** as incomplete, and state reverts to `WAITING` without incrementing reps.
   - Transitions to `EXTENDED` when $\theta \ge 155^\circ$ AND excursion $\Delta \theta = (\theta_{\text{peak}} - \theta_{\text{start}}) \ge 45^\circ$.
3. **`EXTENDED`:**
   - Terminal extension achieved.
   - Transitions to `RETURNING` when knee drops below $150^\circ$.
4. **`RETURNING`:**
   - Leg descends back towards the chair.
   - When knee returns below $115^\circ$:
     - Validates range of motion $\Delta \theta \ge 45^\circ$.
     - Increments `repetition_count += 1`.
     - Logs repetition metadata (peak angle, ROM excursion, duration in seconds).
     - Activates a **$0.60\text{s}$ cooldown** timer to prevent duplicate triggers from settling jitter.
     - Transitions to `WAITING`.

### Side-Switching Safety Rule
In `auto` mode, the system monitors landmark visibility for both limbs. However, **the system will NEVER switch the active leg in the middle of a repetition** (during `EXTENDING`, `EXTENDED`, or `RETURNING`). Leg switching is locked until the state returns to `WAITING` or tracking is completely lost.

---

## 7. Camera Positioning Guidelines

For optimal 2D tracking accuracy:
1. **Camera Angle:** Position the camera at roughly knee height, approximately **$1.8\text{m}$ to $2.5\text{m}$ ($6–8\text{ ft}$)** away, perpendicular to the sagittal plane (direct side profile of the leg being exercised).
2. **Clothing:** Shorts or form-fitting clothing are strongly recommended. Loose blankets, skirts, or wide trousers can obscure the knee joint center and degrade landmark tracking.
3. **Lighting:** Ensure the room is well-lit from the front or side. Avoid strong backlighting (such as sitting directly in front of an unshaded window).
4. **Chair:** Use a stable, armless or low-arm chair where the thigh rests horizontally and the lower leg can hang freely at $\approx 90^\circ$ flexion.

---

## 8. Installation & Execution Guide

### Step 1: Install Dependencies
```powershell
pip install -r rehab_cv/requirements.txt
```

### Step 2: Run Automated Unit Tests
```powershell
python -m pytest rehab_cv/tests -v
```

### Step 3: Run Interactive Application
```powershell
# Run with default physical webcam
python rehab_cv/main.py

# Specify webcam device index and preferred leg
python rehab_cv/main.py --camera 0 --leg right

# Run with synthetic motion simulator (if webcam is unavailable)
python rehab_cv/main.py --synthetic

# Run headless verification mode (processes 180 frames without GUI)
python rehab_cv/main.py --synthetic --headless --max-frames 180
```

### Interactive Keyboard Controls
- **`[Q]` / `[ESC]`:** Exit the application cleanly and print the session summary report.
- **`[R]`:** Reset repetition counter, excursion history, and state machine to zero.
- **`[L]`:** Manually toggle the active tracking leg between `LEFT` and `RIGHT` (safely resets the active repetition).

---

## 9. Verification & Test Suite Results

The comprehensive test suite in `rehab_cv/tests/` verifies all components independently of webcam hardware:

```text
============================= test session starts =============================
rehab_cv/tests/test_angle_calculator.py::test_straight_leg_approaches_180_degrees PASSED [  5%]
rehab_cv/tests/test_angle_calculator.py::test_right_angle_bend_approaches_90_degrees PASSED [ 10%]
rehab_cv/tests/test_angle_calculator.py::test_intermediate_angles_45_and_135 PASSED [ 15%]
rehab_cv/tests/test_angle_calculator.py::test_zero_length_vectors PASSED [ 20%]
rehab_cv/tests/test_angle_calculator.py::test_nan_and_infinite_coordinates PASSED [ 25%]
rehab_cv/tests/test_angle_calculator.py::test_low_confidence_rejection PASSED [ 30%]
rehab_cv/tests/test_angle_calculator.py::test_leg_selection_manager_mid_rep_lock PASSED [ 35%]
rehab_cv/tests/test_knee_extension.py::test_knee_extension_end_to_end_single_rep PASSED [ 40%]
rehab_cv/tests/test_knee_extension.py::test_knee_extension_manual_leg_switch PASSED [ 45%]
rehab_cv/tests/test_repetition_counter.py::test_complete_repetition_counts_exactly_once PASSED [ 50%]
rehab_cv/tests/test_repetition_counter.py::test_incomplete_extension_does_not_count PASSED [ 55%]
rehab_cv/tests/test_repetition_counter.py::test_incomplete_return_does_not_count PASSED [ 60%]
rehab_cv/tests/test_repetition_counter.py::test_noisy_threshold_crossings_do_not_produce_duplicate_reps PASSED [ 65%]
rehab_cv/tests/test_repetition_counter.py::test_tracking_loss_does_not_create_false_repetitions PASSED [ 70%]
rehab_cv/tests/test_repetition_counter.py::test_cooldown_blocks_immediate_duplicate_counting PASSED [ 75%]
rehab_cv/tests/test_smoothing.py::test_ema_noise_reduction PASSED        [ 80%]
rehab_cv/tests/test_smoothing.py::test_missing_observations_do_not_insert_zero PASSED [ 85%]
rehab_cv/tests/test_smoothing.py::test_recovery_after_tracking_resumes PASSED [ 90%]
rehab_cv/tests/test_smoothing.py::test_rolling_window_average PASSED     [ 95%]
rehab_cv/tests/test_smoothing.py::test_unified_angle_smoother_interface PASSED [100%]
============================= 20 passed in 6.93s ==============================
```

---

## 10. Technical Limitations & Clinical Safety Notice

> [!WARNING]
> **2D Projection Geometry Limitation:**  
> The knee angles calculated by this module represent 2D planar projections onto the camera's image plane. If the camera is placed at an oblique angle rather than a true sagittal side profile, perspective foreshortening will alter the measured angle. **These angles must never be used as clinical diagnostic measurements or as substitutes for physical goniometric examination.**

> [!IMPORTANT]
> **Exercise Safety & Disclaimer:**  
> The angular thresholds ($100^\circ$ flexion, $155^\circ$ extension, $45^\circ$ excursion) are software engineering defaults for tracking stability. They do NOT constitute medical or rehabilitation advice. Patients should only perform knee extensions within the pain-free range of motion approved by their physical therapist or orthopedic physician.
