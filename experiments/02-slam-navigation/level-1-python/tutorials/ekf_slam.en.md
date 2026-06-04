# s03-ekf-slam — Extended Kalman Filter SLAM

中文： [ekf_slam.md](./ekf_slam.md)

## What problem does this lab solve?

MCL answers "where am I when the map is known?" SLAM asks a harder question:

> Can the robot localize itself while estimating the map at the same time?

This lab implements a small EKF-SLAM demo. The robot observes 3 landmarks with range-bearing measurements, and the state vector contains both robot pose and landmark coordinates.

```text
state = [x, y, theta, lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y]
```

That is the key SLAM intuition: pose uncertainty affects map estimation, and map estimation feeds back into pose estimation.

## Quick Start

Run from `experiments/02-slam-navigation/level-1-python/`:

```bash
python scripts/ekf_slam.py
```

If you only want to verify the algorithm and JSON output:

```bash
python scripts/ekf_slam.py --no-plot
```

You can adjust horizon and noise:

```bash
python scripts/ekf_slam.py --steps 30 --range-sigma 0.18
```

Default output directory:

```text
output/ekf_slam/
```

Main outputs:

| File | Meaning |
|------|---------|
| `ekf_slam.png` | True trajectory, EKF-SLAM trajectory, true/estimated landmarks, and covariance trend |
| `ekf_slam_metrics.json` | Pose error, landmark RMSE, state size, and covariance trace |
| `ekf_slam_samples.json` | Per-step true pose, estimated pose, and error |
| `ekf_slam_landmarks.json` | True position, estimated position, and error for each landmark |
| `ekf_slam_covariance_diagonal.json` | Final joint covariance diagonal |

## What to inspect

The terminal prints:

```text
final_pose_error=...m
mean_pose_error=...m
landmark_rmse=...m
state_size=9
covariance_trace=...
```

Focus on three points:

- `state_size=9`: the state is no longer only the robot; it also contains 3 landmarks.
- `landmark_rmse`: whether the estimated map is close to the true landmarks.
- `covariance_trace`: how joint uncertainty changes with observations.

## Core loop

Each EKF-SLAM step contains:

1. `predict`: predict robot pose from control input and propagate joint covariance.
2. `observe`: generate range and bearing measurements for each landmark.
3. `linearize`: compute the observation Jacobian around the current estimate.
4. `update`: use the Kalman gain to correct both robot pose and landmark positions.

For teaching stability, this lab assumes known landmark association: each observation knows which landmark it came from. Real SLAM must also solve data association, which belongs in later research extensions.

## Parameters to try

```bash
python scripts/ekf_slam.py --steps 12 --no-plot
python scripts/ekf_slam.py --steps 36 --no-plot
python scripts/ekf_slam.py --range-sigma 0.25 --no-plot
python scripts/ekf_slam.py --bearing-sigma 0.08 --no-plot
```

Suggested observations:

- Does landmark RMSE decrease with more steps?
- Do pose error and covariance trace increase when observation noise gets larger?
- If you only look at `x, y, theta`, why can you not explain map error?

## ROS2 connection

Level 1 focuses on the intuition of joint EKF-SLAM state estimation. In Level 2, the state and covariance concepts can map to ROS2 `nav_msgs/Odometry`, `geometry_msgs/PoseWithCovariance`, TF frames, and landmark observation messages.
