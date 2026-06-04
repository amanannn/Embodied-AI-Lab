# c01-pid-control-playground：互动 PID 控制实验

English: [pid_playground.en.md](./pid_playground.en.md)

## 学习目标

这个实验用一维质量-阻尼系统建立 PID 直觉。学习者不需要 ROS2、Gazebo 或真实机器人，只需要 Python 3.10+ 和浏览器。

完成后应能回答：

- `Kp` 为什么会提高响应速度，也可能带来超调和振荡？
- `Ki` 为什么能消除稳态误差，也可能让系统恢复变慢？
- `Kd` 为什么能抑制快速变化，但不能单独完成控制？
- 外力扰动出现后，反馈控制如何把系统拉回目标？

## 先跑无浏览器版本

```bash
python scripts/pid_playground.py --simulate
```

脚本会生成：

- `output/pid_playground/pid_playground_metrics.json`
- `output/pid_playground/pid_playground_samples.json`

也可以显式调整参数：

```bash
python scripts/pid_playground.py --simulate --kp 8 --ki 1.2 --kd 2.6
```

## 启动互动页面

```bash
python scripts/pid_playground.py --serve
```

打开终端中打印的本地地址，例如：

```text
http://127.0.0.1:8003
```

如果端口被占用，可以换一个端口：

```bash
python scripts/pid_playground.py --serve --port 8004
```

实验结束后在终端按 `Ctrl+C` 关闭本地服务，避免下次启动时端口仍被占用。

页面包含五个滑块：

- `Kp`：比例增益
- `Ki`：积分增益
- `Kd`：微分增益
- `Target`：目标位置
- `Disturbance`：短时外力扰动

每次修改参数后，前端会请求 Python 后端的 `/api/simulate`，再用 Canvas 重绘轨迹。

## 观察指标

- `Steady Error`：最终位置和目标位置的差距。
- `Overshoot`：位置超过目标的百分比。
- `Settling Time`：轨迹进入目标附近并保持稳定的大致时间。
- `Peak Control`：控制力的最大绝对值。

## 推荐实验步骤

1. 只保留 `Kp`，把 `Ki` 和 `Kd` 调到 0，观察响应速度和超调。
2. 慢慢加入 `Kd`，观察振荡是否被抑制。
3. 加入少量 `Ki`，观察稳态误差是否变小。
4. 增大扰动力，观察系统能否回到目标。
5. 记录一组“响应快但超调大”和一组“响应慢但稳定”的参数。

## 与真实机器人的关系

真实机器人中的 PID 往往控制轮速、关节角度、位置或姿态。本实验只保留一维系统，是为了先看清反馈控制的核心闭环：测量当前位置、计算误差、输出控制量、系统状态改变，再重新测量。
