"""PID control simulation for a one-dimensional mass-damper system.

The lab intentionally starts with a small plant model: one point mass moves
along a line, the PID controller pushes it toward a target position, and a
single external disturbance can be injected mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationResult:
    samples: list[dict[str, float]]
    metrics: dict[str, float]


def simulate_pid(
    *,
    setpoint: float = 1.0,
    kp: float = 8.0,
    ki: float = 1.2,
    kd: float = 2.6,
    mass: float = 1.0,
    damping: float = 1.1,
    duration: float = 6.0,
    dt: float = 0.02,
    disturbance_time: float | None = None,
    disturbance_force: float = 0.0,
    initial_position: float = 0.0,
    initial_velocity: float = 0.0,
) -> SimulationResult:
    """Run a deterministic PID closed-loop simulation.

    Returns a list of time samples and a compact metrics dictionary suitable
    for JSON output and browser visualization.
    """

    _validate_parameters(kp, ki, kd, mass, damping, duration, dt)

    position = float(initial_position)
    velocity = float(initial_velocity)
    integral = 0.0
    previous_error = setpoint - position
    max_position = position
    min_position = position
    samples: list[dict[str, float]] = []

    steps = int(duration / dt)
    for step in range(steps + 1):
        time_s = step * dt
        error = setpoint - position
        integral += error * dt
        derivative = (error - previous_error) / dt if step > 0 else 0.0
        control = kp * error + ki * integral + kd * derivative

        disturbance = 0.0
        if disturbance_time is not None and time_s >= disturbance_time:
            if time_s < disturbance_time + 0.18:
                disturbance = disturbance_force

        acceleration = (control + disturbance - damping * velocity) / mass
        velocity += acceleration * dt
        position += velocity * dt
        max_position = max(max_position, position)
        min_position = min(min_position, position)

        samples.append(
            {
                "time": round(time_s, 4),
                "setpoint": round(setpoint, 6),
                "position": round(position, 6),
                "velocity": round(velocity, 6),
                "error": round(error, 6),
                "control": round(control, 6),
                "disturbance": round(disturbance, 6),
            }
        )
        previous_error = error

    return SimulationResult(
        samples=samples,
        metrics=_compute_metrics(samples, setpoint, max_position, min_position, duration),
    )


def _validate_parameters(
    kp: float,
    ki: float,
    kd: float,
    mass: float,
    damping: float,
    duration: float,
    dt: float,
) -> None:
    gains = {"kp": kp, "ki": ki, "kd": kd}
    for name, value in gains.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if mass <= 0:
        raise ValueError("mass must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if dt >= duration:
        raise ValueError("dt must be smaller than duration")


def _compute_metrics(
    samples: list[dict[str, float]],
    setpoint: float,
    max_position: float,
    min_position: float,
    duration: float,
) -> dict[str, float]:
    final_position = samples[-1]["position"]
    steady_state_error = abs(setpoint - final_position)

    if abs(setpoint) < 1e-9:
        overshoot_percent = 0.0
    elif setpoint > 0:
        overshoot_percent = max(0.0, (max_position - setpoint) / abs(setpoint) * 100.0)
    else:
        overshoot_percent = max(0.0, (setpoint - min_position) / abs(setpoint) * 100.0)

    tolerance = max(0.02 * abs(setpoint), 0.02)
    settling_time = duration
    for index, sample in enumerate(samples):
        remaining = samples[index:]
        if all(abs(item["position"] - setpoint) <= tolerance for item in remaining):
            settling_time = sample["time"]
            break

    peak_control = max(abs(sample["control"]) for sample in samples)

    return {
        "final_position": round(final_position, 6),
        "steady_state_error": round(steady_state_error, 6),
        "overshoot_percent": round(overshoot_percent, 3),
        "settling_time": round(settling_time, 4),
        "peak_control": round(peak_control, 6),
        "sample_count": float(len(samples)),
    }
