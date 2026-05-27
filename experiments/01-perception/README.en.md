# Perception

中文： [README.md](./README.md)

## Core Question

How does a robot transform raw observations into state, structure, and semantic signals that later modules can actually use?

## Why This Direction Matters

Perception is the input layer for estimation, planning, and control. Without a runnable and interpretable perception baseline, the rest of the embodied loop has no trustworthy starting point.

## Level Structure

- `level-1-python`: state estimation, grounding, and lightweight perception experiments
- `level-2-cpp-or-mixed`: C++ Kalman and point cloud strengthening
- `level-3-research`: richer fusion, multimodal perception, and open-vocabulary directions

## Representative Experiments

- `p01-state-estimation-kalman`
- `p02-vlm-grounding`
- future point-cloud and multi-sensor bridge experiments

## Suggested Entry Point

Start with Kalman-based state estimation to build intuition for noise, observations, and uncertainty before moving into richer perception tasks.

## Research Extensions

Multi-sensor fusion, 3D perception, semantic reconstruction, open-vocabulary perception, and embodied scene understanding.
