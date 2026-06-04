# Simulation and Sim-to-Real

中文： [README.md](./README.md)

## Core Question

How do we build training environments in simulation and reduce the gap to real embodied systems?

## Why This Direction Matters

Simulation is where experimentation becomes scalable, but transfer quality determines whether that work matters outside the simulator.

Phase 1 gives this direction immediate repository relevance because `04-robot-sim` maps here and future reusable simulator pieces are expected to move into `shared/sim2d` instead of staying buried in one experiment.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable on Manjaro or any standard Python environment. Covers simulator foundations and domain-randomization concepts.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 simulation foundations into ROS2 / C++ / richer simulators, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Transfer, identification, and digital-twin extensions.

## Representative Experiments

- `r01-sim2d-foundation`
- future migration bridges from `04-robot-sim` into direction-local labs and `shared/sim2d`

## Suggested Entry Point

Use the simplified simulator foundation to understand world stepping and observation loops before transfer-focused work. In Phase 1, this page clarifies that the repository does not yet contain migrated experiments here even though the direction already owns the sim-to-real path.

## Research Extensions

Domain randomization, system identification, simulator fidelity tradeoffs, and transfer diagnostics.
