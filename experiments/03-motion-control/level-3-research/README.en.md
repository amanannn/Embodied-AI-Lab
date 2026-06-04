# Motion Control — Level 3: Research Extension

中文： [README.md](./README.md)

## Positioning

This directory is the **research extension layer** for the motion control direction. Level 3 connects course versions to research questions, papers, and deeper implementation paths — it does not promise vague "complete implementations" but clarifies research boundaries.

## Research Directions

| Direction | Research Question | Prerequisites |
|-----------|------------------|---------------|
| Robust Control | How to maintain stable control under model uncertainty and external disturbances? | Level 1 PID + control theory foundations |
| Model Predictive Control (MPC) | How to optimize multi-step control sequences under constraints? | Level 1 trajectory optimization + optimization foundations |
| Force and Impedance Control | How to achieve safe interaction with the environment? | Level 2 real-time control interfaces |

## Relationship to Level 1/2

- Level 1 builds PID and trajectory optimization algorithm intuition
- Level 2 connects control algorithms into ROS2 / C++ real-time systems
- Level 3 explores more open research questions and extension boundaries

## Related Documentation

- [Python-first, ROS2-ready Path](../../../docs/curriculum/python-first-ros2-ready.en.md)
- [Motion Control Direction Page](../README.en.md)
