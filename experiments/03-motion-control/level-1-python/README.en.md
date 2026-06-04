# Motion Control — Level 1: Core Python Lab

中文： [README.md](./README.md)

## Positioning

This directory is the **Level 1 entry** for the motion control direction. Level 1 is the current main product: pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment.

## Planned Experiments

| Experiment | Core Question | Status |
|-----------|---------------|--------|
| `c01-pid-control-playground` | How can interactive tuning build intuition for proportional, integral, and derivative feedback? | Complete |
| Trajectory Optimization | How to generate smooth, feasible motion trajectories? | Planned |

## Quick Start

Run the browser-free simulation first to verify the Python model and JSON outputs:

```bash
python scripts/pid_playground.py --simulate
```

Then start the local interactive page:

```bash
python scripts/pid_playground.py --serve
```

Open the local URL printed by the script, tune `Kp / Ki / Kd`, target position, and disturbance force, then observe trajectory, control effort, and metrics.

## Completed Labs

### `c01-pid-control-playground`

- Pure Python PID simulation core: one-dimensional mass-damper plant, short external disturbance, error/control logging.
- Standard-library HTTP service: no FastAPI, React, or Node.js toolchain required.
- Native HTML/CSS/JS frontend: sliders, Canvas plot, overshoot, steady-state error, settling time, and peak control.
- Tutorial: [pid_playground.en.md](./tutorials/pid_playground.en.md)

## Relationship to Level 2

Level 1 builds control algorithm intuition with Python. Level 2 (`../level-2-ros2-bridge/`) connects these algorithms into ROS2 / C++ / real robot software stacks.

## Related Documentation

- [Python-first, ROS2-ready Path](../../../docs/curriculum/python-first-ros2-ready.en.md)
- [Motion Control Direction Page](../README.en.md)
