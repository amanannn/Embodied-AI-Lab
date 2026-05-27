# Motion Control

## Core Question

How does a robot track desired motion accurately, robustly, and with interpretable feedback behavior?

## Why This Direction Matters

Control is where models, sensors, and actuators become reliable physical motion instead of just desired trajectories.

## Level Structure

- `level-1-python`: PID and trajectory optimization
- `level-2-cpp-or-mixed`: real-time control strengthening
- `level-3-research`: robust control, MPC, force control, and safety constraints

## Representative Experiments

- `c01-pid-control-lab`
- `c02-trajectory-optimization`
- future realtime bridge work

## Suggested Entry Point

Start with PID, then use trajectory optimization to connect feedback control with smoother motion generation.

## Research Extensions

Force control, impedance behavior, MPC, and disturbance-robust control in more realistic systems.
