# SLAM & Navigation — Level 1: Core Python Lab

中文： [README.md](./README.md)

This directory is the Level 1 Python entry point for SLAM and navigation. The goal is to build runnable, visible intuition for localization, mapping, and path planning before introducing ROS2.

## Current Entry

| Lab | Core Question | Prerequisite | Status |
|-----|---------------|-------------|--------|
| `s01-grid-search` | How to find the shortest path in a known map? | None | Complete |
| `s02-mcl-localization` | Where is the robot in a known map? | Particle filter | Complete |
| `s03-ekf-slam` | How to localize and map simultaneously in an unknown environment? | Extended Kalman filter | Complete |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/grid_search.py --map maze --algorithm astar
python scripts/grid_search.py --map maze --algorithm dijkstra
python scripts/grid_search.py --map maze --compare
python scripts/mcl_localization.py
python scripts/ekf_slam.py
```

Outputs are written under `output/`:

- `output/grid_search/`: inspect `grid_search.png`, `grid_search_astar.png`, `grid_search_dijkstra.png`, and JSON metrics.
- `output/mcl_localization/`: inspect the true trajectory, odometry trajectory, MCL estimate, particle cloud, and error curve.
- `output/ekf_slam/`: inspect the true trajectory, EKF-SLAM estimate, true/estimated landmarks, and covariance trend.

## Learning Route

1. Run `s01-grid-search` first and compare A* against Dijkstra on a known occupancy grid.
2. Move to `s02-mcl-localization` next to answer "where am I?" before "where should I go?"
3. Use `s03-ekf-slam` to understand why localization and mapping become coupled in unknown environments.

## Lab Guide

- [Grid Search path-planning tutorial](tutorials/grid_search.en.md)
- [MCL localization tutorial](tutorials/mcl_localization.en.md)
- [EKF-SLAM tutorial](tutorials/ekf_slam.en.md)

## Directory Structure

```text
level-1-python/
├── localization/   # Localization algorithms such as MCL
├── planners/       # Path-planning algorithms
├── slam/           # Mapping algorithms such as EKF-SLAM
├── scripts/        # Runnable Python scripts
├── tutorials/      # Introductory tutorials
├── output/         # Runtime output (not tracked)
└── requirements.txt
```
