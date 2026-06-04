# World Models

中文： [README.md](./README.md)

## Core Question

Can a robot learn predictive dynamics or latent structure that helps it plan before acting?

## Why This Direction Matters

Predictive modeling helps robots reason about consequences instead of reacting only at the current timestep.

In this repository, Phase 1 establishes the direction landing page first and does not yet contain migrated experiments beyond the placeholder identity work needed to anchor future world-model labs.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable on Manjaro or any standard Python environment. Covers dynamics prediction and short-horizon rollout.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 predictive models into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Latent world models and model-based RL.

## Representative Experiments

- `w01-dynamics-prediction-world-model`
- future bridges from control, RL, and sim-to-real once predictive assets are mature enough to split out

## Suggested Entry Point

Start with a simple predictive model in a toy environment before touching latent or multimodal world models; in Phase 1, that means using this page as the curriculum contract even though the repository does not yet contain migrated experiments for this direction.

## Research Extensions

Latent rollouts, planning over learned models, multimodal prediction, and model-based embodied agents.
