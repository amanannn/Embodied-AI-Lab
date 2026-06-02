# Perception Level 1: Python Foundation Labs

中文: [README.md](./README.md)

This directory is the Level 1 entrance for the perception direction. Its goal is to build runnable undergraduate-level intuition in Python: first sensor errors, then state estimation, then basic camera vision.

The core principle is: **run first, observe first, then understand the formulas and code**.

## Learning Route

| Lab | Core Question | Entry Point | Tutorial |
|-----|---------------|-------------|----------|
| Lab 01: Sensor Noise | Why are sensor readings different from ground truth? | `python scripts/noise.py` | `tutorials/sensors.en.md` |
| Lab 02: Sensor Simulation | What do GPS, LiDAR, IMU, and odometry output? | `python scripts/sensors.py` | `tutorials/sensors.en.md` |
| Lab 03: Multi-Sensor Fusion | How can multiple imperfect measurements become a more reliable state? | `python scripts/fusion.py` | `tutorials/sensors.en.md` |
| Lab 04: Kalman Filter Family | How do filters turn noisy observations into state estimates? | `python scripts/kf.py`, `python scripts/all.py` | `tutorials/kalman.en.md` |
| Lab 05: Camera Calibration | How do we estimate USB camera intrinsics and distortion coefficients? | `python scripts/camera_calibration.py --generate-board` | `tutorials/camera_calibration.en.md` |
| Lab 06: ArUco / AprilTag Pose | How can visual markers be detected and used for pose estimation? | `python scripts/aruco_pose.py --generate-marker` | `tutorials/aruco_pose.en.md` |
| Lab 07: Classic OpenCV Vision | What can optical flow, feature matching, and background subtraction do? | `python scripts/classic_vision.py --generate-sample --mode optical-flow` | `tutorials/classic_vision.en.md` |

Recommended order:

```text
Lab 01 -> Lab 02 -> Lab 03 -> Lab 04 -> Lab 05 -> Lab 06 -> Lab 07
```

The first four labs build state-estimation foundations, the fifth introduces real USB-camera vision, the sixth completes marker-level 6DoF pose estimation, and the seventh adds basic intuition for image motion, local correspondence, and foreground detection.

## Code Modules

| Module | Responsibility |
|--------|----------------|
| `noise.py` | Gaussian noise, bias, drift, outliers, and basic noise models |
| `sensors/` | GPS, LiDAR, IMU, and odometry simulation |
| `fusion.py` | Time alignment and weighted fusion |
| `filters/` | KF, EKF, UKF, and PF implementations |
| `scripts/` | Runnable lab entry points |
| `tutorials/` | Bilingual tutorials |
| `utils/` | Trajectory generation, simulation helpers, and visualization |

The directory layout is organized by engineering responsibility, not learning order. Use the learning route first, then enter the corresponding modules when developing.

## Quick Start

```bash
pip install -r requirements.txt

python scripts/noise.py
python scripts/sensors.py
python scripts/fusion.py
python scripts/all.py
python scripts/camera_calibration.py --generate-board
python scripts/aruco_pose.py --generate-marker
python scripts/classic_vision.py --generate-sample --mode optical-flow
```

The ArUco lab needs `cv2.aruco`, so this level uses `opencv-contrib-python`. If your environment already has a manually installed `opencv-python`, uninstall the old package before installing these requirements to avoid mixing two OpenCV packages.

Generated plots, CSV files, JSON files, and animations are saved to `output/` by default. This directory is for experiment observation and is not tracked by git.

## What To Observe

- Noise lab: compare Gaussian noise, bias, drift, and outliers.
- Sensor lab: observe measurement dimensions, sampling rates, and error patterns for different sensors.
- Fusion lab: observe why weighted fusion is usually more stable than a single sensor.
- Filter lab: compare KF, EKF, UKF, and PF under different nonlinear and noise conditions.
- Camera calibration lab: observe checkerboard corners, reprojection error, camera matrix, and undistortion results.
- ArUco lab: observe marker IDs, corners, coordinate axes, and the physical meaning of `tvec_m`.
- Classic vision lab: observe optical-flow arrows, ORB match lines, and background-subtraction masks.

## Current Scope

- This level focuses on low-barrier, runnable, observable experiments, not industrial performance.
- C++ engineering reinforcement belongs in `../level-2-cpp-or-mixed/`.
- Research extensions belong in `../level-3-research/`.
- Current vision labs target ordinary USB cameras, not RGB-D, point clouds, or high-precision 3D reconstruction.
