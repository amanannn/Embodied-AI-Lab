# s01-grid-search — Known-Map Path Planning

中文： [grid_search.md](./grid_search.md)

## Core Question

When a robot already has a 2D occupancy grid, how does it find a path from a start cell to a goal cell?

`s01-grid-search` implements A* and Dijkstra in pure Python. It does not use ROS2 or real robot hardware. The goal is to make known-map path search runnable and inspectable first.

## Run

```bash
pip install -r requirements.txt
python scripts/grid_search.py --map maze --algorithm astar
python scripts/grid_search.py --map maze --algorithm dijkstra
python scripts/grid_search.py --map maze --compare
python scripts/grid_search.py --map rooms --algorithm astar
```

The default learning experience is matplotlib-first. After running the script, inspect:

```text
output/grid_search/grid_search.png
output/grid_search/grid_search_astar.png
output/grid_search/grid_search_dijkstra.png
```

For minimal environments without plotting dependencies, skip PNG rendering and keep JSON metrics only:

```bash
python scripts/grid_search.py --algorithm astar --no-plot
```

If `matplotlib` is not installed, the script automatically skips PNG rendering and keeps JSON metrics. This does not affect the path-planning algorithm.

## Map Convention

The script provides three reproducible built-in map scenarios:

| Map | Purpose |
|-----|---------|
| `maze` | Default map. A 21x31 procedural maze with branches, bottlenecks, and dead ends for comparing A* and Dijkstra. |
| `rooms` | A room-and-corridor layout closer to an indoor floor plan. |
| `simple` | A small legacy map for quick debugging. |

These maps are inputs for known-map path planning. They are not SLAM mapping outputs. Later MCL and EKF-SLAM labs continue into localization and mapping.

- `0` means free space
- `1` means obstacle
- coordinates use `(row, col)` order
- movement is 4-connected: up, right, down, left

## What to Observe

The default output directory is `output/grid_search/`. Inspect:

- `grid_search.png`: map, expanded search area, and final path for a single algorithm
- `grid_search_astar.png`: A* visualization in comparison mode
- `grid_search_dijkstra.png`: Dijkstra visualization in comparison mode
- `grid_search_metrics.json`: metrics for one algorithm
- `grid_search_comparison.json`: A* versus Dijkstra metrics

Core metrics:

- `path_cost`: path cost. In a 4-connected grid, each step costs 1.
- `path_length`: number of path nodes, including start and goal.
- `expanded_nodes`: number of nodes expanded by the algorithm.
- `runtime_ms`: algorithm runtime measured by the script.

## A* vs Dijkstra

Dijkstra only considers the cost already paid from the start to the current node. It is reliable, but tends to expand outward broadly.

A* adds a heuristic on top of Dijkstra. This lab uses Manhattan distance:

```text
heuristic = abs(row - goal_row) + abs(col - goal_col)
```

That makes A* prefer cells closer to the goal. As long as the heuristic does not overestimate the true remaining cost, A* still finds a shortest path while usually expanding fewer nodes.

## Extensions

This lab is the entry point for later navigation work:

- MCL answers "where is the robot in the map?"
- EKF-SLAM answers "how do localization and mapping interact when the map is unknown?"
- Level 2 bridges paths, maps, and poses into ROS2 `nav_msgs`, RViz2, and Nav2 concepts.
