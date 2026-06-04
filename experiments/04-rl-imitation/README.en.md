# RL and Imitation

中文： [README.md](./README.md)

## Core Question

How does a robot learn behavior from reward signals, demonstrations, or iterative policy improvement?

## Why This Direction Matters

Learning-based behavior becomes important when hand-designed control logic is too brittle or too expensive to specify directly.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Covers Q-learning, deep RL, and imitation learning.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 policy learning algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Sim-to-real RL, offline RL, and transfer extensions.

## Representative Experiments

- `l01-q-learning`
- `l02-deep-rl`
- `l03-imitation-learning`

## Suggested Entry Point

Use tabular Q-learning to build intuition before moving into deep policies or imitation-based behavior.

## Research Extensions

Offline RL, policy transfer, safer exploration, and demonstration-efficient learning.
