# 旧实验归档到新方向结构的映射

English: [legacy-to-direction-map.en.md](./legacy-to-direction-map.en.md)

## 迁移规则

- 旧实验已统一归档到 `archive/legacy-experiments/`，避免和 10 个新方向在 `experiments/` 顶层混杂。
- 这些映射的作用是帮助读者在迁移期间找到历史来源和新方向归属，而不是宣称代码已经全部重排完毕。
- 当前学习入口统一从 `experiments/` 开始；归档目录主要用于历史参考和后续迁移。

## Planned Identity 的含义

`Planned identity` 表示迁移完成后，这个旧实验在方向优先结构中应当对应的实验命名目标。它是未来课程结构的命名锚点，不代表该重命名实验今天已经完整存在。

## 映射表

| 旧实验归档位置 | 新方向归属 | 目标实验命名 |
|---|---|---|
| ~~`01-kalman-filter`~~ | `experiments/01-perception/level-1-python/` | `p01-state-estimation-kalman` (已迁移) |
| `archive/legacy-experiments/02-particle-filter-mcl` | `experiments/02-slam-navigation/` | `s02-mcl-localization` |
| `archive/legacy-experiments/03-ekf-slam` | `experiments/02-slam-navigation/` | `s03-ekf-slam` |
| `archive/legacy-experiments/04-robot-sim` | `experiments/09-sim-to-real/` | `r01-sim2d-foundation` |
| `archive/legacy-experiments/05-pid-control` | `experiments/03-motion-control/` | `c01-pid-control-lab` |
| `archive/legacy-experiments/06-robot-kinematics` | `experiments/07-manipulation/` | `m01-robot-kinematics` |
| `archive/legacy-experiments/07-path-planning` | `experiments/02-slam-navigation/level-1-python/` | `s01-grid-search` (已迁移) |
| `archive/legacy-experiments/08-trajectory-optimization` | `experiments/03-motion-control/` | `c02-trajectory-optimization` |
| `archive/legacy-experiments/09-q-learning` | `experiments/04-rl-imitation/` | `l01-q-learning` |
| `archive/legacy-experiments/10-deep-rl` | `experiments/04-rl-imitation/` | `l02-deep-rl` |
| `archive/legacy-experiments/11-imitation-learning` | `experiments/04-rl-imitation/` | `l03-imitation-learning` |
| `archive/legacy-experiments/12-vla-grounding` | `experiments/01-perception/` | `p02-vlm-grounding` |
| `archive/legacy-experiments/g1-grasping` | `experiments/07-manipulation/` | `m03-grasping-and-placing` |
| `archive/legacy-experiments/g2-dynamics` | `experiments/07-manipulation/` | `m02-manipulator-dynamics-control` |
| 历史记录缺失：`S1-fullstack-nav` | `experiments/10-vertical-apps/` | `x01-autonomous-inspection-capstone` |
