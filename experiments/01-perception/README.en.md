# Perception

中文： [README.md](./README.md)

## Core Question

How does a robot transform raw observations into state, structure, and semantic signals that later modules can actually use?

## Why This Direction Matters

Perception is the input layer for estimation, planning, and control. Without a runnable and interpretable perception baseline, the rest of the embodied loop has no trustworthy starting point.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Covers sensor simulation, noise modeling, state estimation, multi-sensor fusion, and basic vision; route entrance: [`level-1-python/README.en.md`](./level-1-python/README.en.md)
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 perception algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Richer fusion, multimodal perception, and open-vocabulary directions.

## Level 1 Experiments

### Experiment 1: Sensor Fundamentals & Noise Modeling

Build a sensor simulation pipeline from scratch: understand noise → simulate sensors → fuse multiple sensors.

| Module | Content |
|---|---|
| `noise.py` | Four noise models (Gaussian, bias, drift, outliers) |
| `sensors/` | GPS, LiDAR, IMU, and odometry sensor simulation |
| `fusion.py` | Multi-sensor weighted fusion |
| `scripts/noise.py` | Noise type comparison experiment |
| `scripts/sensors.py` | Multi-sensor comparison + CSV data output |
| `scripts/fusion.py` | Fusion result comparison |

### Experiment 2: Kalman Filter Family

Four filter implementations: KF → EKF → UKF → PF.

| Filter | When to Use |
|---|---|
| Linear Kalman Filter (KF) | Linear systems with Gaussian noise |
| Extended Kalman Filter (EKF) | Nonlinear observations (radar, LiDAR) |
| Unscented Kalman Filter (UKF) | Strong nonlinearity, no derivatives needed |
| Particle Filter (PF) | Arbitrary distributions, multimodal scenarios |

### Quick Start

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt

# Sensor fundamentals
python scripts/noise.py        # Noise comparison plot
python scripts/sensors.py      # Multi-sensor comparison + CSV
python scripts/fusion.py       # Fusion result

# Kalman filters
python scripts/kf.py           # Single filter
python scripts/all.py          # Four-filter comparison
python scripts/kf_tuning.py    # Parameter experiments
```

See `level-1-python/tutorials/` for full walkthroughs.

## Vision Track

Level 1 now covers three runnable tracks: sensor simulation, state estimation, and basic vision.

Current progress:

| Experiment | Core Question | Hardware | Status |
|-----------|--------------|----------|--------|
| Camera Calibration | How to get intrinsics and distortion coefficients from checkerboard images? | USB webcam | Landed |
| ArUco / AprilTag Detection | How to detect markers and estimate 6DoF pose in real time? | USB webcam | Landed |
| Classic OpenCV Vision | What can optical flow, feature matching, and background subtraction do? | USB webcam or offline sample | Landed |

These experiments run in parallel with the sensor/filter track without modifying the current structure.

Recommended hardware config: USB webcam, 640×480 @ 30fps.

## Representative Experiments

- `p01-sensor-fundamentals` — sensor simulation and noise modeling (landed)
- `p01-state-estimation-kalman` — four-filter implementation (landed)
- `p01-camera-calibration` — USB webcam checkerboard calibration (landed)
- `p01-aruco-pose` — ArUco / AprilTag detection and single-marker pose estimation (landed)
- `p01-classic-vision` — optical flow, feature matching, and background subtraction foundations (landed)
- `p02-vlm-grounding` — visual-language grounding (planned)
- future point-cloud and multi-sensor bridge experiments

## Research Extensions

Multi-sensor fusion, 3D perception, semantic reconstruction, open-vocabulary perception, and embodied scene understanding.
