# Python-first, ROS2-ready Path

中文： [python-first-ros2-ready.md](./python-first-ros2-ready.md)

## Positioning

This project follows a **Python-first, ROS2-ready** path:

- **Level 1** builds algorithmic intuition with runnable Python labs
- **Level 2** bridges those outputs into ROS2/C++/mixed robotics systems
- **Level 3** frames research extensions

This is not a simplified "write Python first, then C++" story. It is an intentional learning path: run algorithms with the lowest barrier first, engineer them into real systems next, and explore open questions with a research lens last.

## Why This Design

### Level 1: Core Python Lab

Level 1 is the current main product.

- Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies
- Runnable on Manjaro or any standard Python environment
- Goal: let undergraduates and early learners run their first experiment within 30 minutes
- Emphasizes from-scratch implementation, visualization, fast feedback, and intuition

Level 1 is not a "lite version." It is the algorithmic intuition layer. Many core concepts (Kalman filters, path planning, reinforcement learning) are actually clearer in Python.

### Level 2: ROS2 / C++ / Mixed Bridge

Level 2 is the engineering bridge layer.

- Connects Level 1 algorithmic intuition into ROS2 / C++ / real robot software stacks
- Targets Ubuntu development environments with ROS2 and C++ toolchains
- Emphasizes performance, ROS2 messages/nodes, geometric constraints, and real-time behavior
- Goal: help learners understand how algorithm code becomes part of a robot system

Level 2 is not an "advanced version." It is an engineering bridge. Its value lies in connecting Python prototypes to real systems, not in simply rewriting everything in C++.

### Level 3: Research Extension

Level 3 is the research layer.

- Connects course versions to theses, papers, and deeper implementation paths
- Does not promise vague "complete implementations" — instead, clarifies research questions and extension boundaries
- Goal: provide entry points for learners interested in research

## Device Strategy

| Layer | Environment | Dependencies |
|-------|------------|--------------|
| Level 1 | Manjaro or any Python environment | Python 3.10+, with each lab's requirements.txt as the source of truth |
| Level 2 | Ubuntu | ROS2, C++ build toolchains, CMake |
| Level 3 | Depends on research direction | Depends on specific topic |

## How This Appears in the Repository

- Each direction's `level-1-python/` directory contains runnable Python experiments
- Each direction's `level-2-ros2-bridge/` directory contains ROS2 / C++ bridge documentation
- Each direction's `level-3-research/` directory contains research extension directions
- `shared/ros2_interfaces/` defines Level 1 → Level 2 ROS2 message and interface contracts

## Relationship to Legacy Structure

Legacy experiment directories (e.g., `02-particle-filter-mcl`, `07-path-planning`) are gradually migrating to the direction-first structure. Until migration is complete, legacy directories remain searchable and understandable. See `docs/curriculum/legacy-to-direction-map.md` for mapping details.
