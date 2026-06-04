# Vision-Language Navigation

中文： [README.md](./README.md)

## Core Question

How does a robot interpret natural-language goals and move toward semantically relevant targets?

## Why This Direction Matters

This direction connects language understanding, perception, memory, and navigation in a single embodied loop.

In the current repository, this direction is intentionally ahead of the code migration: Phase 1 documents the landing page first so language grounding and semantic navigation work can be attached to a stable direction home later.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable on Manjaro or any standard Python environment. Covers language goal parsing and semantic goal search.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 language navigation algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Long-horizon VLN and benchmark-oriented extensions.

## Representative Experiments

- `v01-language-goal-navigation`
- future language grounding bridges from perception and vertical application work

## Suggested Entry Point

Begin with a toy semantic search task before attempting richer navigation environments. During Phase 1, this README is the discoverability anchor because the repository does not yet contain migrated experiments under this direction.

## Research Extensions

Long-horizon instruction following, semantic maps, multimodal memory, and benchmark-grade VLN systems.
