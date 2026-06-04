# Manipulation

中文： [README.md](./README.md)

## Core Question

How does a robot move its body and end effector to interact with objects purposefully?

## Why This Direction Matters

Manipulation is where kinematics, dynamics, perception, and control meet object interaction.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Covers kinematics, manipulator dynamics, and grasping.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 manipulation algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Dexterous, force-guided, and bimanual manipulation.

## Representative Experiments

- `m01-robot-kinematics`
- `m02-manipulator-dynamics-control`
- `m03-grasping-and-placing`

## Suggested Entry Point

Start with kinematics, then move into manipulator dynamics before tackling grasping pipelines.

## Research Extensions

Force-guided manipulation, dexterous hands, bimanual coordination, and task-conditioned manipulation.
