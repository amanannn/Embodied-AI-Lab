# Awesome Embodied AI

> 从零开始，走进具身智能 —— 面向本科生和零基础入门者的学习资源大全。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

具身智能（Embodied AI）是人工智能、机器人学、计算机视觉和控制理论的交叉领域，研究智能体如何在物理世界或仿真环境中感知、推理和行动。

本仓库以 **awesome 列表** 的形式收录优质学习资源，同时提供 **自建中文教程**，适合本科低年级或零基础转行者系统入门。

---

## 目录

- [自建教程](#-自建教程)
- [数学基础](#-数学基础)
- [感知与状态估计](#-感知与状态估计)
- [仿真环境](#-仿真环境)
- [机器人操作系统](#-机器人操作系统)
- [强化学习与模仿学习](#-强化学习与模仿学习)
- [课程与书籍](#-课程与书籍)
- [工具与框架](#-工具与框架)
- [学习路线建议](#-学习路线建议)
- [贡献指南](#-贡献指南)

---

## 📖 自建教程

> 面向零基础，中文写作，代码可运行。

- [**卡尔曼滤波：从零到精通**](./tutorials/kalman-filter/tutorial.md) —— 从直觉理解到数学推导，覆盖 KF / EKF / UKF / 粒子滤波四种滤波器的完整实现。附带动画可视化、参数调优实验和滤波器对比。
- [更多教程等你贡献……](#-贡献指南)

---

## 🧮 数学基础

> 具身智能的三大数学支柱：线性代数、概率论、最优化。

- [3Blue1Brown — 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) —— 动画直观理解向量、矩阵、特征值，零基础首选。
- [MIT 18.06 — Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) —— 经典线性代数课，Strang 老爷子的讲法无人能敌。
- [MIT 6.041 — Probabilistic Systems Analysis](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) —— 概率论入门，覆盖高斯分布、贝叶斯、马尔可夫链。
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) —— 交互式线性代数教材，可拖拽操作矩阵。
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) —— 矩阵求导速查手册，推公式时的救星。

---

## 👁️ 感知与状态估计

> "我在哪？周围有什么？"—— 机器人感知的核心问题。

### 卡尔曼滤波与状态估计

- [**本仓库教程：卡尔曼滤波从零到精通**](./tutorials/kalman-filter/tutorial.md) —— KF / EKF / UKF / PF 四种滤波器，带完整代码和可视化。
- [rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) —— 交互式 Jupyter Notebook，中文社区公认最好的卡尔曼滤波教程。
- [Probabilistic Robotics (Thrun, Burgard, Fox)](https://www.probabilistic-robotics.org/) —— 机器人状态估计圣经，涵盖 KF、EKF、UKF、粒子滤波、图优化 SLAM。

### SLAM

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) —— 视觉 SLAM 标杆开源项目，支持单目/双目/RGB-D。
- [Cartographer](https://github.com/cartographer-project/cartographer) —— Google 开源的 2D/3D 激光 SLAM，ROS 生态标配。
- [g2o — General Graph Optimization](https://github.com/RainerKuemmerle/g2o) —— 图优化框架，SLAM 后端的数学引擎。
- [SLAMBook (中文)](https://github.com/gaoxiang12/slambook2) —— 高翔《视觉SLAM十四讲》代码，中文 SLAM 必读。

### 计算机视觉

- [CS231A — Computer Vision: From 3D Reconstruction to Recognition](https://web.stanford.edu/class/cs231a/) —— Stanford 3D 视觉课程，含多视图几何、深度估计。
- [OpenCV 官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html) —— 计算机视觉瑞士军刀。

---

## 🏗️ 仿真环境

> 没有机器人？先在仿真里跑。

- [Habitat-Sim / Habitat-Lab](https://github.com/facebookresearch/habitat-sim) —— Meta 开源的 3D 室内仿真平台，支持具身导航任务（PointNav、ObjectNav）。
- [Isaac Sim (NVIDIA)](https://developer.nvidia.com/isaac-sim) —— 基于 Omniverse 的高保真机器人仿真，支持 ROS2 桥接。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) —— Isaac Sim 之上的 RL 训练框架，统一强化学习和模仿学习接口。
- [MuJoCo](https://github.com/google-deepmind/mujoco) —— DeepMind 开源的物理引擎，轻量、精确，RL 研究标配。
- [PyBullet](https://pybullet.org/) —— Python 友好的物理仿真，入门快。
- [AI2-THOR](https://ai2thor.allenai.org/) —— 交互式室内场景，支持物体操作任务。
- [GibsonEnv](https://github.com/StanfordVL/GibsonEnv) —— Stanford 的 3D 环境，侧重真实感渲染。

---

## 🤖 机器人操作系统

> ROS2 是具身智能的神经中枢。

- [ROS 2 官方文档](https://docs.ros.org/en/rolling/) —— ROS2 Humble/Iron 入门必读。
- [ROS2 中文教程 (鱼香ROS)](http://fishros.com/) —— 中文社区最活跃的 ROS2 教程，含一键安装工具。
- [Micro-ROS](https://micro.ros.org/) —— 把 ROS2 搬到 MCU 上，嵌入式机器人必备。
- [Foxglove](https://foxglove.dev/) —— 机器人数据可视化利器，替代 RViz。

---

## 🧠 强化学习与模仿学习

> 让机器人自己学会做事。

- [OpenAI Spinning Up](https://spinningup.openai.com/) —— RL 入门最佳实践，从 VPG 到 SAC 都有清晰实现。
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) —— PyTorch RL 算法库，开箱即用的 PPO / SAC / TD3。
- [Sutton & Barto — Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) —— RL 领域圣经，免费在线阅读。
- [李宏毅 — 强化学习 (中文)](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_) —— 中文 RL 课程，讲解风趣易懂。
- [Robomimic](https://github.com/ARISE-Initiative/robomimic) —— 机器人模仿学习框架，统一 BC / IRIS / Diffusion Policy 等方法。
- [LeRobot](https://github.com/huggingface/lerobot) —— HuggingFace 的机器人学习库，PyTorch 原生，社区活跃。

---

## 📚 课程与书籍

> 系统学习的最佳路径。

### 在线课程

- [CS223A — Introduction to Robotics (Stanford, Oussama Khatib)](https://see.stanford.edu/Course/CS223A) —— 机器人学入门，正运动学、逆运动学、雅可比。
- [6.4210 — Robotic Manipulation (MIT, Russ Tedrake)](https://manipulation.csail.mit.edu/) —— 机器人操作，从基础到高级规划。
- [Self-Driving Cars (U. Tübingen)](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/) —— 自动驾驶感知系统全景课。

### 必读书单

| 书名 | 适合阶段 | 说明 |
|------|:--------:|------|
| [Probabilistic Robotics](https://www.probabilistic-robotics.org/) | 大一～大二 | 状态估计圣经，卡尔曼滤波、AMCL、SLAM |
| [Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) | 大二～大三 | 多视图几何，视觉 SLAM 数学地基 |
| [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) | 大二～大三 | RL 圣经，免费在线阅读 |
| [Modern Robotics](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) | 大一～大二 | 现代机器人学，附 Coursera 配套课程 |
| [视觉SLAM十四讲 (高翔)](https://github.com/gaoxiang12/slambook2) | 大一～大三 | 中文 SLAM 必读，从理论到代码 |

---

## 🔧 工具与框架

> 日常开发会用到的轮子。

- [PyTorch](https://pytorch.org/) —— 深度学习框架，科研首选。
- [NumPy](https://numpy.org/) —— 数值计算基石。
- [OpenCV](https://opencv.org/) —— 计算机视觉库。
- [GTSAM](https://github.com/borglab/gtsam) —— 因子图优化库，SLAM / 状态估计的高级工具。
- [PyRoboPlan](https://github.com/njxue/PyRoboPlan) —— Python 机器人规划算法合集。
- [Rerun](https://rerun.io/) —— 机器人/计算机视觉日志可视化，调试神器。

---

## 🎯 学习路线建议

> 以下路径面向 **本科低年级 / 零基础转行**，按优先级排序。

```
第 0 步：打好地基
├── Python 基础（NumPy + Matplotlib 能跑起来就行）
├── 线性代数（矩阵乘法、特征值 —— 3Blue1Brown + 前 5 讲 MIT 18.06）
└── 概率论（高斯分布、条件概率 —— MIT 6.041 前半）

第 1 步：感知入门（你在这）
└── 🎯 卡尔曼滤波：从零到精通（本仓库教程）
    ├── 线性 KF → 跑起来，看动画，调参数
    ├── EKF → 明白"雅可比 = 在曲线上画切线"
    ├── UKF → 理解"sigma 点 = 派侦察兵探路"
    └── PF → 体会"粒子群 = 用一群蚂蚁表示概率"

第 2 步：仿真环境
├── 安装 Habitat-Sim 或 PyBullet
├── 跑通随机智能体基线
└── 理解仿真环境与真实机器人的关系

第 3 步：SLAM
├── 读 Probabilistic Robotics 前 10 章
├── 跑通 ORB-SLAM3 示例
└── 理解因子图优化（g2o / GTSAM）

第 4 步：学习如何让机器人"学会"
├── OpenAI Spinning Up
├── 在 Isaac Lab / MuJoCo 上练 RL
└── 看 Robomimic 模仿学习的工作流

第 5 步：走向真实世界
├── ROS2 + 真实机器人
└── 从仿真迁移到现实（Sim2Real）
```

> **核心原则**：先动手跑代码，再回头补理论。你在 `demo_kf.py` 调一次参数获得的直觉，比读十页公式更深刻。

---

## 🤝 贡献指南

欢迎贡献！这个仓库的目标是成为中文最好的具身智能入门资源。

**贡献方式**：
- **新增资源** —— Fork → 添加链接 → 提 PR（请确保资源对零基础者友好，并附一句话中文简介）
- **撰写教程** —— 在 `tutorials/` 下新建目录，参考 [kalman-filter](./tutorials/kalman-filter/) 的结构
- **改进现有内容** —— 发现教程错误或过时链接？提 Issue 或直接 PR

资源收录标准：
- ✅ 面向初学者，有清晰的中文/英文文档
- ✅ 至少提供可运行的代码或交互式内容
- ❌ 纯学术论文且无配套实现（初学者需要先建立直觉，再看论文）

---

## 📄 License

[MIT](./LICENSE)
