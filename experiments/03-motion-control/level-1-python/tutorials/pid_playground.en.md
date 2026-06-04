# c01-pid-control-playground: Interactive PID Control Lab

中文： [pid_playground.md](./pid_playground.md)

## Learning Goals

This lab builds PID intuition with a one-dimensional mass-damper system. Learners only need Python 3.10+ and a browser; ROS2, Gazebo, and real robot hardware are not required.

After completing the lab, learners should be able to explain:

- Why `Kp` improves response speed but can increase overshoot and oscillation.
- Why `Ki` can reduce steady-state error but may slow recovery.
- Why `Kd` damps fast changes but cannot control the plant alone.
- How feedback pulls the system back after an external disturbance.

## Run the Browser-Free Version First

```bash
python scripts/pid_playground.py --simulate
```

The script writes:

- `output/pid_playground/pid_playground_metrics.json`
- `output/pid_playground/pid_playground_samples.json`

You can also tune parameters explicitly:

```bash
python scripts/pid_playground.py --simulate --kp 8 --ki 1.2 --kd 2.6
```

## Start the Interactive Page

```bash
python scripts/pid_playground.py --serve
```

Open the local URL printed by the terminal, for example:

```text
http://127.0.0.1:8003
```

If the port is already in use, choose another one:

```bash
python scripts/pid_playground.py --serve --port 8004
```

When the lab is finished, press `Ctrl+C` in the terminal to stop the local server and avoid leaving the port occupied.

The page exposes five sliders:

- `Kp`: proportional gain
- `Ki`: integral gain
- `Kd`: derivative gain
- `Target`: target position
- `Disturbance`: short external force

Whenever parameters change, the frontend calls the Python backend at `/api/simulate` and redraws the response with Canvas.

## Metrics to Observe

- `Steady Error`: final position error relative to the target.
- `Overshoot`: how far the position exceeds the target in percent.
- `Settling Time`: approximate time until the trajectory enters and stays near the target.
- `Peak Control`: maximum absolute control effort.

## Suggested Procedure

1. Keep only `Kp` by setting `Ki` and `Kd` to 0, then observe speed and overshoot.
2. Add `Kd` gradually and observe whether oscillation decreases.
3. Add a small amount of `Ki` and observe whether steady-state error decreases.
4. Increase the disturbance and check whether the system returns to the target.
5. Record one parameter set that is fast but overshoots, and one that is slower but stable.

## Connection to Real Robots

Real robots often use PID loops for wheel speed, joint angle, position, or attitude. This lab keeps only a one-dimensional plant so the feedback loop is visible: measure position, compute error, output control, update the system state, then measure again.
