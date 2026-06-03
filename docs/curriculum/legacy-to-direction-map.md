# 旧实验到新方向结构的映射

English: [legacy-to-direction-map.en.md](./legacy-to-direction-map.en.md)

## 迁移规则

- 第一阶段不会把代码直接从旧实验目录搬走，当前首先建立的是新的方向首页和导航结构。
- 这些映射的作用是帮助读者在迁移期间保持可发现性，而不是宣称代码已经全部重排完毕。
- 只要对应方向下还没有真实迁移后的实验资产，旧实验名称就应该继续保持可搜索、可理解。

## Planned Identity 的含义

`Planned identity` 表示迁移完成后，这个旧实验在方向优先结构中应当对应的实验命名目标。它是未来课程结构的命名锚点，不代表该重命名实验今天已经完整存在。

## 映射表

| 旧实验 | 新方向归属 | 目标实验命名 |
|---|---|---|
| ~~`01-kalman-filter`~~ | `experiments/01-perception/level-1-python/` | `p01-state-estimation-kalman` (已迁移) |
| `02-particle-filter-mcl` | `experiments/02-slam-navigation/` | `s02-mcl-localization` |
| `03-ekf-slam` | `experiments/02-slam-navigation/` | `s03-ekf-slam` |
| `04-robot-sim` | `experiments/09-sim-to-real/` | `r01-sim2d-foundation` |
| `05-pid-control` | `experiments/03-motion-control/` | `c01-pid-control-lab` |
| `06-robot-kinematics` | `experiments/07-manipulation/` | `m01-robot-kinematics` |
| `07-path-planning` | `experiments/02-slam-navigation/` | `s01-grid-search` |
| `08-trajectory-optimization` | `experiments/03-motion-control/` | `c02-trajectory-optimization` |
| `09-q-learning` | `experiments/04-rl-imitation/` | `l01-q-learning` |
| `10-deep-rl` | `experiments/04-rl-imitation/` | `l02-deep-rl` |
| `11-imitation-learning` | `experiments/04-rl-imitation/` | `l03-imitation-learning` |
| `12-vla-grounding` | `experiments/01-perception/` | `p02-vlm-grounding` |
| `g1-grasping` | `experiments/07-manipulation/` | `m03-grasping-and-placing` |
| `g2-dynamics` | `experiments/07-manipulation/` | `m02-manipulator-dynamics-control` |
| `S1-fullstack-nav` | `experiments/10-vertical-apps/` | `x01-autonomous-inspection-capstone` |
