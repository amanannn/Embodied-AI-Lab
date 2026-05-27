# Perception

English: [README.en.md](./README.en.md)

## Core Question

机器人如何把原始观测转化为可用于定位、规划或理解的状态、结构与语义信息？

## Why This Direction Matters

感知是后续状态估计、规划与控制的输入层。没有可解释、可运行的感知基线，后面的闭环都缺少可信起点。

## Level Structure

- `level-1-python`：状态估计、grounding 与轻量感知实验
- `level-2-cpp-or-mixed`：C++ Kalman 与点云处理强化
- `level-3-research`：更丰富的融合、多模态感知与开放词汇方向

## Representative Experiments

- `p01-state-estimation-kalman`
- `p02-vlm-grounding`
- 后续点云与多传感器融合桥接实验

## Suggested Entry Point

先从卡尔曼状态估计开始，建立对噪声、观测和不确定性的直觉，再进入更强的感知任务。

## Research Extensions

多传感器融合、3D 感知、语义重建、开放词汇感知与具身场景理解。
