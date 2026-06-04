# Motion Control

中文： [README.md](./README.md)

## Core Question

How does a robot track desired motion accurately, robustly, and with interpretable feedback behavior?

## Why This Direction Matters

Control is where models, sensors, and actuators become reliable physical motion instead of just desired trajectories.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Covers PID and trajectory optimization.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 control algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Robust control, MPC, force control, and safety constraints.

## Representative Experiments

- `c01-pid-control-playground` (complete): Python + browser interactive PID control lab
- `c02-trajectory-optimization`
- future realtime bridge work

## Suggested Entry Point

Start with `c01-pid-control-playground` under `level-1-python`: tune `Kp / Ki / Kd` with sliders, observe overshoot, oscillation, settling time, and disturbance recovery, then use trajectory optimization to connect feedback control with smoother motion generation.

## Research Extensions

Force control, impedance behavior, MPC, and disturbance-robust control in more realistic systems.
