"""Monte Carlo Localization on a small known beacon map.

The lab keeps the math deliberately compact: a robot moves in a 2D room, sees
distances to fixed beacons, and uses a particle filter to recover its pose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


Pose = tuple[float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class MclScenario:
    name: str
    width: float
    height: float
    beacons: tuple[Point, ...]
    start: Pose
    controls: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    theta: float
    weight: float


@dataclass(frozen=True)
class MclResult:
    metrics: dict[str, float | int | str]
    samples: list[dict[str, float | int]]
    beacons: list[dict[str, float | int]]
    particles: list[dict[str, float]]


def build_indoor_beacon_scenario(steps: int = 24) -> MclScenario:
    """Build a deterministic room trajectory with four range beacons."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    pattern = [
        (0.30, 0.08),
        (0.30, 0.06),
        (0.28, -0.04),
        (0.26, -0.10),
        (0.24, -0.16),
        (0.27, 0.12),
    ]
    controls = tuple(pattern[index % len(pattern)] for index in range(steps))
    return MclScenario(
        name="indoor-beacons",
        width=8.0,
        height=6.0,
        beacons=((0.8, 0.8), (7.2, 0.9), (7.0, 5.2), (0.9, 5.1)),
        start=(1.4, 1.2, 0.12),
        controls=controls,
    )


def run_mcl_demo(
    *,
    seed: int = 11,
    particle_count: int = 600,
    steps: int = 24,
    motion_sigma: float = 0.035,
    turn_sigma: float = 0.025,
    sensor_sigma: float = 0.22,
    odometry_drift: float = 0.045,
) -> MclResult:
    """Run a deterministic teaching demo and return JSON-friendly results."""

    _validate_parameters(
        particle_count=particle_count,
        steps=steps,
        motion_sigma=motion_sigma,
        turn_sigma=turn_sigma,
        sensor_sigma=sensor_sigma,
        odometry_drift=odometry_drift,
    )
    rng = random.Random(seed)
    scenario = build_indoor_beacon_scenario(steps)
    particles = _initialize_particles(rng, scenario, particle_count)
    true_pose = scenario.start
    odometry_pose = scenario.start

    samples: list[dict[str, float | int]] = []
    final_effective_count = float(particle_count)

    for step, control in enumerate(scenario.controls, start=1):
        true_pose = _move_pose(true_pose, control, scenario)
        noisy_control = (
            control[0] + rng.gauss(0.0, motion_sigma * 2.0) + odometry_drift,
            control[1] + rng.gauss(0.0, turn_sigma * 1.5),
        )
        odometry_pose = _move_pose(odometry_pose, noisy_control, scenario)
        observation = _observe_distances(true_pose, scenario.beacons, rng, sensor_sigma)

        predicted = [
            _jitter_particle(
                _move_pose((p.x, p.y, p.theta), control, scenario),
                rng,
                motion_sigma,
                turn_sigma,
                scenario,
            )
            for p in particles
        ]
        weighted = _weight_particles(predicted, observation, scenario.beacons, sensor_sigma)
        final_effective_count = _effective_particle_count(weighted)
        particles = _systematic_resample(weighted, rng)
        estimate = _estimate_pose(particles)

        estimate_error = _distance((estimate[0], estimate[1]), (true_pose[0], true_pose[1]))
        odometry_error = _distance((odometry_pose[0], odometry_pose[1]), (true_pose[0], true_pose[1]))
        samples.append(
            {
                "step": step,
                "true_x": round(true_pose[0], 4),
                "true_y": round(true_pose[1], 4),
                "true_theta": round(true_pose[2], 4),
                "estimate_x": round(estimate[0], 4),
                "estimate_y": round(estimate[1], 4),
                "estimate_theta": round(estimate[2], 4),
                "odometry_x": round(odometry_pose[0], 4),
                "odometry_y": round(odometry_pose[1], 4),
                "error": round(estimate_error, 4),
                "odometry_error": round(odometry_error, 4),
                "effective_particle_count": round(final_effective_count, 3),
            }
        )

    errors = [float(sample["error"]) for sample in samples]
    odometry_errors = [float(sample["odometry_error"]) for sample in samples]
    metrics: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "particle_count": particle_count,
        "steps": steps,
        "seed": seed,
        "final_error": round(errors[-1], 4),
        "mean_error": round(sum(errors) / len(errors), 4),
        "max_error": round(max(errors), 4),
        "final_odometry_error": round(odometry_errors[-1], 4),
        "mean_odometry_error": round(sum(odometry_errors) / len(odometry_errors), 4),
        "effective_particle_count": round(final_effective_count, 3),
        "sensor_sigma": sensor_sigma,
        "motion_sigma": motion_sigma,
    }

    return MclResult(
        metrics=metrics,
        samples=samples,
        beacons=[
            {"id": index, "x": round(x, 4), "y": round(y, 4)}
            for index, (x, y) in enumerate(scenario.beacons)
        ],
        particles=[
            {"x": round(p.x, 4), "y": round(p.y, 4), "theta": round(p.theta, 4)}
            for p in particles
        ],
    )


def _validate_parameters(
    *,
    particle_count: int,
    steps: int,
    motion_sigma: float,
    turn_sigma: float,
    sensor_sigma: float,
    odometry_drift: float,
) -> None:
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if motion_sigma < 0:
        raise ValueError("motion_sigma must be non-negative")
    if turn_sigma < 0:
        raise ValueError("turn_sigma must be non-negative")
    if sensor_sigma <= 0:
        raise ValueError("sensor_sigma must be positive")
    if odometry_drift < 0:
        raise ValueError("odometry_drift must be non-negative")


def _initialize_particles(
    rng: random.Random,
    scenario: MclScenario,
    particle_count: int,
) -> list[Particle]:
    weight = 1.0 / particle_count
    start_x, start_y, start_theta = scenario.start
    return [
        Particle(
            x=_clamp(rng.gauss(start_x, 0.55), 0.3, scenario.width - 0.3),
            y=_clamp(rng.gauss(start_y, 0.55), 0.3, scenario.height - 0.3),
            theta=_wrap_angle(rng.gauss(start_theta, 0.45)),
            weight=weight,
        )
        for _ in range(particle_count)
    ]


def _move_pose(pose: Pose, control: tuple[float, float], scenario: MclScenario) -> Pose:
    distance, turn = control
    theta = _wrap_angle(pose[2] + turn)
    x = _clamp(pose[0] + distance * math.cos(theta), 0.05, scenario.width - 0.05)
    y = _clamp(pose[1] + distance * math.sin(theta), 0.05, scenario.height - 0.05)
    return x, y, theta


def _jitter_particle(
    pose: Pose,
    rng: random.Random,
    motion_sigma: float,
    turn_sigma: float,
    scenario: MclScenario,
) -> Particle:
    x = _clamp(pose[0] + rng.gauss(0.0, motion_sigma), 0.05, scenario.width - 0.05)
    y = _clamp(pose[1] + rng.gauss(0.0, motion_sigma), 0.05, scenario.height - 0.05)
    theta = _wrap_angle(pose[2] + rng.gauss(0.0, turn_sigma))
    return Particle(x=x, y=y, theta=theta, weight=0.0)


def _observe_distances(
    pose: Pose,
    beacons: tuple[Point, ...],
    rng: random.Random,
    sensor_sigma: float,
) -> list[float]:
    return [
        _distance((pose[0], pose[1]), beacon) + rng.gauss(0.0, sensor_sigma)
        for beacon in beacons
    ]


def _weight_particles(
    particles: list[Particle],
    observation: list[float],
    beacons: tuple[Point, ...],
    sensor_sigma: float,
) -> list[Particle]:
    weights: list[float] = []
    for particle in particles:
        error_sq = 0.0
        for observed, beacon in zip(observation, beacons):
            expected = _distance((particle.x, particle.y), beacon)
            error_sq += (observed - expected) ** 2
        weights.append(math.exp(-0.5 * error_sq / (sensor_sigma**2)))

    total = sum(weights)
    if total <= 0.0:
        normalized = 1.0 / len(particles)
        return [
            Particle(x=p.x, y=p.y, theta=p.theta, weight=normalized)
            for p in particles
        ]

    return [
        Particle(x=p.x, y=p.y, theta=p.theta, weight=w / total)
        for p, w in zip(particles, weights)
    ]


def _effective_particle_count(particles: list[Particle]) -> float:
    total_sq = sum(p.weight**2 for p in particles)
    if total_sq == 0.0:
        return 0.0
    return 1.0 / total_sq


def _systematic_resample(particles: list[Particle], rng: random.Random) -> list[Particle]:
    count = len(particles)
    positions = [(rng.random() + index) / count for index in range(count)]
    cumulative: list[float] = []
    total = 0.0
    for particle in particles:
        total += particle.weight
        cumulative.append(total)

    result: list[Particle] = []
    index = 0
    uniform_weight = 1.0 / count
    for position in positions:
        while index < count - 1 and position > cumulative[index]:
            index += 1
        selected = particles[index]
        result.append(
            Particle(
                x=selected.x,
                y=selected.y,
                theta=selected.theta,
                weight=uniform_weight,
            )
        )
    return result


def _estimate_pose(particles: list[Particle]) -> Pose:
    x = sum(p.x for p in particles) / len(particles)
    y = sum(p.y for p in particles) / len(particles)
    sin_sum = sum(math.sin(p.theta) for p in particles)
    cos_sum = sum(math.cos(p.theta) for p in particles)
    theta = math.atan2(sin_sum, cos_sum)
    return x, y, theta


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
