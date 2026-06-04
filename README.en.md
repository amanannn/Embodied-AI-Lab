# Embodied AI Lab

中文： [README.md](./README.md)

![Embodied AI Lab banner](./figure/embodied-ai-lab-banner.png)

### A direction-first, experiment-driven curriculum for embodied AI

Embodied AI Lab is a public course repository for learners and builders who want to understand embodied intelligence through executable experiments rather than theory-only summaries.

This project follows a **Python-first, ROS2-ready** path: Level 1 builds algorithmic intuition with runnable Python labs, Level 2 bridges those outputs into ROS2/C++/mixed robotics systems, and Level 3 frames research extensions.

The repository is currently organized around research directions first and experiments second: directions provide the top-level map, while experiments remain the real learning and implementation units.

## What This Repository Is

Embodied AI emphasizes agents that perceive, decide, and act through a physical body interacting with the world. That makes it an inherently interdisciplinary field spanning robotics, control, computer vision, machine learning, geometry, and systems engineering.

This repository is not an awesome-list style index and not a theory-only note collection. Its purpose is to break embodied AI into buildable course units that can be run, inspected, and extended.

## Why This Course Exists

Most embodied AI materials do one of two things:

- summarize what the field contains
- assume you already know how to connect perception, planning, control, and systems code

This repository takes a different path:

- experiments are the primary learning unit
- running code comes before abstraction
- visualization and observed behavior are part of the explanation
- hard directions may begin as high-quality partial implementations instead of hollow end-to-end demos

## Course Panorama

The course is organized around 10 mainstream research directions. The panorama below answers three questions: what each direction studies, where students usually enter, and how that work later strengthens into more serious engineering.

| Direction | Core Question | Typical Level 1 Entry | Typical Level 2 Bridge |
|---|---|---|---|
| Perception | How does a robot interpret the world? | Kalman-based state estimation | ROS2 perception topics, TF, odometry bridge |
| SLAM and Navigation | How does a robot know where it is and how to move? | Grid Search path planning, MCL, EKF-SLAM | ROS2 map/path topics, RViz2, Nav2 concept bridge |
| Motion Control | How does a robot move accurately and robustly? | PID and trajectory optimization | ROS2 control loop and trajectory messages |
| RL and Imitation | How does a robot learn behavior from reward or demonstration? | Q-learning, deep RL, imitation learning | policy deployment bridge and action interface |
| World Models | Can a robot predict future dynamics and use them for decisions? | dynamics prediction and rollout | prediction service / planner bridge |
| Vision-Language Navigation | Can a robot follow language to find semantic goals? | toy semantic search | language goal to ROS2 navigation goal bridge |
| Manipulation | How does a robot interact with objects? | kinematics, dynamics, grasping | MoveIt2 / trajectory / gripper interface bridge |
| LLM and Robotics | How do large models help robots plan and act? | task decomposition and tool use | task planning to ROS2 services/actions bridge |
| Simulation and Sim-to-Real | How do we train in simulation and transfer to the real world? | lightweight simulation foundations | Gazebo / Isaac / ROS2 simulation bridge |
| Vertical Applications | How do these capabilities land in real scenarios? | scenario-scoped exercises | multi-module ROS2 integration bridge |

## Learning Architecture

Each direction advances through three levels:

- **Level 1: Core Python Lab**  
  The current main product. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Built for undergraduates and early learners, emphasizing from-scratch implementation, visualization, fast feedback, and intuition.
- **Level 2: ROS2 / C++ / Mixed Bridge**  
  The engineering bridge layer. Connects Level 1 algorithmic intuition into ROS2 / C++ / real robot software stacks, targeting Ubuntu development environments. Emphasizes performance, ROS2 messages/nodes, geometric constraints, and real-time behavior.
- **Level 3: Research Extension**  
  The research layer. Connects course versions to theses, papers, and deeper implementation paths.

Python builds intuition quickly.  
ROS2 / C++ bridges into real robotic systems.  
Research extensions explore boundaries and open questions.

## Research Directions

This section emphasizes the repository state of each direction: where it sits today and where a learner should enter.

| Direction | Role in This Repository | Current Entry |
|---|---|---|
| 01. Perception | Provides the first complete runnable Level 1 baseline. | Start at `experiments/01-perception/level-1-python`, which contains sensor, filtering, and basic vision labs. |
| 02. SLAM and Navigation | Extends estimation into localization, mapping, and planning loops. | Currently represented by candidate experiment pages; start with Grid Search path planning, then expand into MCL and EKF-SLAM. |
| 03. Motion Control | Opens the path from sensing loops to action loops. | Currently represented by direction pages and structure for PID and trajectory work. |
| 04. RL and Imitation | Adds policy learning once simulation anchors are mature. | Currently represented by direction pages and scaffold only. |
| 05. World Models | Focuses on predictive modeling and stronger decision-making. | Currently a direction entry with future anchor experiments. |
| 06. Vision-Language Navigation | Connects language, semantics, and navigation. | Currently a direction entry with future semantic search and VLN work. |
| 07. Manipulation | Shifts from navigation to object interaction and manipulation. | Currently a direction entry for future kinematics, dynamics, and grasping work. |
| 08. LLM and Robotics | Explores large models as planning and tool-use layers. | Currently a direction entry with future task planning agent work. |
| 09. Simulation and Sim-to-Real | Provides training and transfer foundations. | Currently the direction most closely tied to future migration from `archive/legacy-experiments/04-robot-sim`. |
| 10. Vertical Applications | Packages cross-direction skills into scenario-driven systems. | Currently a direction entry and capstone planning surface. |

## Recommended Learning Paths

### Path A: General undergraduate entry

1. Perception
2. SLAM and Navigation
3. Motion Control
4. RL and Imitation
5. Manipulation or Simulation and Sim-to-Real

### Path B: Control-oriented learners

1. Motion Control
2. Perception
3. SLAM and Navigation
4. Manipulation
5. Simulation and Sim-to-Real

### Path C: Computer vision / AI-oriented learners

1. Perception
2. RL and Imitation
3. Vision-Language Navigation
4. LLM and Robotics
5. World Models

### Path D: Research preparation

1. Complete at least three Level 1 directions
2. Bridge one direction into Level 2
3. Choose one Level 3 extension as a research launch point

## Current Progress

The repository is transitioning from a flat experiment list to a direction-first structure. At the moment:

- Perception Level 1 now covers three complete experiment groups: sensor simulation, state estimation, and basic vision
- most later directions still sit at the direction-page and scaffold stage
- the direction-first structure exists, and code migration is gradually expanding

| Area | Status |
|---|---|
| Direction-first information architecture | In progress |
| Root homepage rewrite | Complete |
| Direction landing pages | First batch complete |
| Perception Level 1 vision track | Complete |
| Shared module extraction | Planned |
| World Models / VLN / LLM+Robot anchor experiments | Planned |

## Example Outputs

The figures below are representative Level 1 outputs that have already been generated and promoted into `figure/` as public showcase assets. Runtime outputs still live under each lab's `output/` directory; the homepage only references curated public assets.

| Perception: Vision Loop | Navigation: A* Maze Path Planning | Navigation: Dijkstra Search Expansion |
|---|---|---|
| ![Perception vision loop showcase](./figure/showcase/perception-vision-panel.jpg) | ![A* maze path planning showcase](./figure/showcase/navigation-astar-maze.png) | ![Dijkstra maze path planning showcase](./figure/showcase/navigation-dijkstra-maze.png) |
| Combines marker pose estimation, optical flow, and foreground segmentation into one runnable vision panel. | Shows heuristic-search expansion and the final shortest path on a known occupancy grid. | Uses the same map and shortest path as A*, but expands a wider region, making the value of heuristic search visible. |

## Repository Structure

```text
README.md
README.en.md
shared/
experiments/
docs/
tests/
archive/
```

- `shared/` contains reusable cross-direction infrastructure and public documentation
- `experiments/` contains direction pages and experiment content
- `docs/` contains public curriculum-facing documentation
- `tests/` contains regression checks that matter to the public repository
- `archive/` stores legacy experiments and is not the current learning entry point

## Quick Start

### Python path

The perception direction contains three complete experiment groups: sensor fundamentals, Kalman filters, and basic vision.

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt

# Experiment 1: Sensor Fundamentals & Noise Modeling
python scripts/noise.py        # Noise type comparison
python scripts/sensors.py      # Multi-sensor comparison
python scripts/fusion.py       # Fusion result

# Experiment 2: Kalman Filter Family
python scripts/kf.py           # Linear KF
python scripts/ekf.py          # Extended Kalman Filter
python scripts/ukf.py          # Unscented Kalman Filter
python scripts/pf.py           # Particle Filter
python scripts/all.py          # Four-filter comparison
python scripts/kf_tuning.py    # Parameter experiments

# Experiment 3: Basic Vision
python scripts/camera_calibration.py --generate-board   # Generate checkerboard
python scripts/aruco_pose.py --generate-marker          # Generate ArUco marker
python scripts/classic_vision.py --generate-sample --mode optical-flow  # Optical flow sample
```

The SLAM and navigation direction now provides its first Level 1 path-planning lab.

```bash
cd experiments/02-slam-navigation/level-1-python
pip install -r requirements.txt

# Experiment 1: Known-map path planning
python scripts/grid_search.py --map maze --algorithm astar
python scripts/grid_search.py --map maze --compare
```

### ROS2 / C++ bridge path

Level 2 bridges Level 1 Python algorithmic intuition into ROS2 / C++ / real robot software stacks. The current recommendation is to complete Level 1 first, then enter Level 2 engineering bridge experiments through the direction pages.

> Device strategy: Level 1 targets a Python 3.10+ runtime environment, uses each lab requirements.txt as the dependency source of truth, and can be managed with venv or Conda; Level 2 targets Ubuntu with ROS2 and C++ toolchains.

## Roadmap

### Phase 1

- establish the direction-first structure
- establish the bilingual public entry points
- complete the first batch of direction and shared documentation
- define the Python-first, ROS2-ready three-level positioning

### Phase 2

- gradually move legacy experiment docs into the direction structure
- extract reusable code into `shared/`
- establish Level 1 → Level 2 ROS2 / C++ bridge interfaces

### Phase 3

- add anchor experiments for World Models, Vision-Language Navigation, and LLM and Robotics
- expand more bilingual direction pages and experiment tutorials
- add polished visuals and showcase outputs
