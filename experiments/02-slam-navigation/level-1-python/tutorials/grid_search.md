# s01-grid-search — 已知地图路径规划

English: [grid_search.en.md](./grid_search.en.md)

## 核心问题

机器人已经有一张二维栅格地图时，如何从起点走到终点？

`s01-grid-search` 用纯 Python 实现 A* 和 Dijkstra。这个实验不接 ROS2，也不使用真实机器人，目的是先把“已知地图中的路径搜索”讲清楚。

## 运行方式

```bash
pip install -r requirements.txt
python scripts/grid_search.py --map maze --algorithm astar
python scripts/grid_search.py --map maze --algorithm dijkstra
python scripts/grid_search.py --map maze --compare
python scripts/grid_search.py --map rooms --algorithm astar
```

默认体验以 `matplotlib` PNG 可视化为主。运行后优先打开：

```text
output/grid_search/grid_search.png
output/grid_search/grid_search_astar.png
output/grid_search/grid_search_dijkstra.png
```

如果只想在没有图形依赖的最小环境中验证算法和 JSON 指标，可以跳过 PNG：

```bash
python scripts/grid_search.py --algorithm astar --no-plot
```

如果没有安装 `matplotlib`，脚本会自动跳过 PNG 渲染，只保留 JSON 指标；这不影响路径规划算法本身。

## 地图约定

脚本内置三种可复现实验地图：

| 地图 | 用途 |
|------|------|
| `maze` | 默认地图。21x31 的程序化迷宫，包含分支、瓶颈和死胡同，适合观察 A* 与 Dijkstra 的搜索差异。 |
| `rooms` | 房间与走廊结构，更接近室内平面图。 |
| `simple` | 小型旧地图，适合快速调试。 |

这些地图是“已知地图路径规划”的输入，不是 SLAM 建图结果。后续 MCL 和 EKF-SLAM 会继续回答定位与建图问题。

- `0` 表示可通行区域
- `1` 表示障碍物
- 坐标格式为 `(row, col)`
- 默认只允许上下左右 4 邻接移动

## 观察指标

输出目录默认为 `output/grid_search/`。重点看：

- `grid_search.png`：单个算法的地图、搜索展开区域和最终路径
- `grid_search_astar.png`：对比模式下 A* 的可视化结果
- `grid_search_dijkstra.png`：对比模式下 Dijkstra 的可视化结果
- `grid_search_metrics.json`：单个算法的指标
- `grid_search_comparison.json`：A* 与 Dijkstra 的对比指标

核心指标：

- `path_cost`：路径代价。4 邻接网格中，每走一步代价为 1。
- `path_length`：路径节点数量，包含起点和终点。
- `expanded_nodes`：算法实际展开过的节点数量。
- `runtime_ms`：脚本中算法搜索耗时。

## A* 与 Dijkstra 的差别

Dijkstra 只关心“从起点走到当前节点已经花了多少代价”。它很稳，但会向四周扩散搜索。

A* 在 Dijkstra 的基础上加入启发函数。这里使用曼哈顿距离：

```text
heuristic = abs(row - goal_row) + abs(col - goal_col)
```

所以 A* 会更偏向终点方向。只要启发函数不高估真实代价，A* 仍然能找到最短路径，同时通常扩展更少节点。

## 延伸思考

这个实验是后续导航方向的入口：

- MCL 会回答“机器人在地图中的哪个位置？”
- EKF-SLAM 会回答“地图未知时，定位和建图如何一起做？”
- Level 2 会把路径、地图和位姿桥接到 ROS2 的 `nav_msgs`、RViz2 和 Nav2 概念中。
