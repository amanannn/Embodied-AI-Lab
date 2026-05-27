# LLM and Robotics

## Core Question

How can language models decompose tasks, invoke tools, and support robot decision-making?

## Why This Direction Matters

Large models are increasingly used as planning and interaction layers, but they must be grounded in executable robot-facing behavior.

This repository's Phase 1 state matters here: the landing page comes first so future tool-using and ROS2-connected agent work has a clear home before any legacy experiments are migrated.

## Level Structure

- `level-1-python`: task decomposition and toy-world tool use
- `level-2-cpp-or-mixed`: grounded execution interfaces
- `level-3-research`: long-horizon, multimodal, and agentic robotics systems

## Representative Experiments

- `a01-llm-task-planning-agent`
- future tool-using middleware bridges that connect language planning to ROS2 execution

## Suggested Entry Point

Begin with tool-using planning in a constrained environment before attempting real middleware integration. In this repository, Phase 1 means the direction is defined before migrated experiments land, so the page sets scope for later grounded agent work.

## Research Extensions

Grounded planning, multimodal agents, execution monitoring, and long-horizon robotic task orchestration.
