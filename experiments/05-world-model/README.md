# World Models

English: [README.en.md](./README.en.md)

## Core Question

机器人能否学习预测性动力学或潜在结构，以在行动前进行规划？

## Why This Direction Matters

预测建模帮助机器人推理后果，而不是仅在当前时间步做出反应。

在本仓库中，Phase 1 先建立方向着陆页，尚未包含迁移实验，仅保留定位所需的占位工作。

## Level Structure

- `level-1-python`：动力学预测与短期展开
- `level-2-cpp-or-mixed`：规划器耦合的预测环境
- `level-3-research`：潜在世界模型与基于模型的强化学习

## Representative Experiments

- `w01-dynamics-prediction-world-model`
- 后续从控制、RL 和 sim-to-real 迁移的桥接实验

## Suggested Entry Point

在接触潜在或多模态世界模型之前，先在玩具环境中建立简单预测模型。在 Phase 1，此页面作为课程契约，即使仓库尚未包含该方向的迁移实验。

## Research Extensions

潜在展开、在学习模型上规划、多模态预测、基于模型的具身智能体。
