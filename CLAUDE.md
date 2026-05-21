# CLAUDE.md — Embodied AI Lab 项目指引

## 项目定位

**具身智能实验课** — 实验驱动的中文具身智能入门课程。每个实验是一个独立、可运行、有可视化的 Python 项目。核心理念：**先跑代码，再理解理论。**

## 目录结构

```
experiments/
├── 01-kalman-filter/      ✅ KF/EKF/UKF/PF + 动画可视化
├── 02-particle-filter-mcl/ 🚧 粒子滤波定位
├── 03-ekf-slam/           🚧 EKF-SLAM
├── 04-robot-sim/          🚧 2D 仿真世界（共享基础设施）
├── 05-pid-control/        🚧 PID 控制实验室
├── 06-robot-kinematics/   🚧 机械臂运动学
├── 07-path-planning/      🚧 搜索与采样规划
├── 08-trajectory-optimization/ 🚧 轨迹优化
├── 09-q-learning/         🚧 表格 Q-Learning
├── 10-deep-rl/            🚧 深度强化学习
├── 11-imitation-learning/ 🚧 模仿学习
├── 12-vla-grounding/      🚧 视觉-语言 Grounding
└── S1-fullstack-nav/      🚧 全栈自主导航
```

## 实验设计规范

每个实验必须包含：
- `tutorial.md` — 教程正文，按"问题→直觉→代码→公式→实验"结构组织
- `requirements.txt` — 尽量只用 numpy + matplotlib，降低安装门槛
- `src/` — 核心实现模块，可直接 import
- `demos/` — 至少一个可运行的 demo 脚本
- `utils/` — 可视化和辅助工具
- `output/` — 生成的可视化输出（图片/GIF）

**设计原则：**
- 每个实验产物是**可运行的 Python 代码 + 可视化输出**，不是知识总结
- 实验间通过共享基础设施（04 仿真器）建立依赖，后续实验 import 前序模块
- 教学循环：动手构建 → 观察行为 → 理解原理 → 调参实验
- 面向零基础本科生和转行者，中文写作，纯 Python 实现

**代码风格：**
- 中文注释解释"为什么"，英文变量名
- 深色主题 Matplotlib 可视化（沿用 01 的风格）
- 终端输出包含可读的指标汇总（RMSE、降幅百分比等）

## 当前状态

- 01 卡尔曼滤波已完成（~1500 行 Python + 完整 tutorial.md）
- 其余 11 个实验待建设
- 建设优先级见 README.md

## 禁止事项

- 不要引入需要 GPU 的依赖（除非实验明确标注需要，如 VLA/DRL）
- 不要做纯理论教程——每个实验必须有可运行的代码
- 不要假设学习者有 ROS2/Isaac/真实机器人
- 不要在单个实验里覆盖太多主题——做深一个，不做宽十个

## 参考项目

- rlabbe/Kalman-and-Bayesian-Filters-in-Python — 经典 KF 英文教程
- PythonRobotics — 纯 Python 机器人算法集
