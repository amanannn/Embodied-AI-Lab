"""Extended Kalman Filter SLAM with a tiny range-bearing landmark map.

This module is intentionally small and dependency-free. It demonstrates the
core EKF-SLAM loop: predict robot pose, update pose and landmark estimates with
range-bearing observations, and track uncertainty in one joint covariance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


Pose = tuple[float, float, float]
Point = tuple[float, float]
Vector = list[float]
Matrix = list[list[float]]


@dataclass(frozen=True)
class EkfSlamScenario:
    name: str
    start: Pose
    landmarks: tuple[Point, ...]
    controls: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EkfSlamResult:
    metrics: dict[str, float | int | str]
    samples: list[dict[str, float | int]]
    landmarks: list[dict[str, float | int]]
    covariance_diagonal: list[float]


def build_range_bearing_scenario(steps: int = 24) -> EkfSlamScenario:
    """Build a small landmark scenario suitable for a first EKF-SLAM lab."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    pattern = [
        (0.34, 0.08),
        (0.36, 0.03),
        (0.32, -0.05),
        (0.30, -0.10),
        (0.34, 0.10),
        (0.30, 0.05),
    ]
    controls = tuple(pattern[index % len(pattern)] for index in range(steps))
    return EkfSlamScenario(
        name="range-bearing-landmarks",
        start=(1.0, 1.0, 0.16),
        landmarks=((2.0, 4.8), (5.8, 4.1), (6.6, 1.2)),
        controls=controls,
    )


def run_ekf_slam_demo(
    *,
    seed: int = 7,
    steps: int = 24,
    range_sigma: float = 0.12,
    bearing_sigma: float = 0.045,
    motion_sigma: float = 0.035,
    turn_sigma: float = 0.025,
) -> EkfSlamResult:
    """Run a deterministic EKF-SLAM teaching demo."""

    _validate_parameters(
        steps=steps,
        range_sigma=range_sigma,
        bearing_sigma=bearing_sigma,
        motion_sigma=motion_sigma,
        turn_sigma=turn_sigma,
    )
    rng = random.Random(seed)
    scenario = build_range_bearing_scenario(steps)
    landmark_count = len(scenario.landmarks)
    state = _initial_state(scenario, rng)
    covariance = _initial_covariance(landmark_count)
    true_pose = scenario.start
    samples: list[dict[str, float | int]] = []

    for step, control in enumerate(scenario.controls, start=1):
        true_pose = _move_pose(
            true_pose,
            (
                control[0] + rng.gauss(0.0, motion_sigma * 0.4),
                control[1] + rng.gauss(0.0, turn_sigma * 0.4),
            ),
        )
        noisy_control = (
            control[0] + rng.gauss(0.0, motion_sigma),
            control[1] + rng.gauss(0.0, turn_sigma),
        )
        state, covariance = _predict(state, covariance, noisy_control, motion_sigma, turn_sigma)

        for landmark_index, landmark in enumerate(scenario.landmarks):
            measurement = _observe_landmark(
                true_pose,
                landmark,
                rng=rng,
                range_sigma=range_sigma,
                bearing_sigma=bearing_sigma,
            )
            state, covariance = _update_landmark(
                state,
                covariance,
                landmark_index,
                measurement,
                range_sigma,
                bearing_sigma,
            )

        pose_error = _distance((state[0], state[1]), (true_pose[0], true_pose[1]))
        samples.append(
            {
                "step": step,
                "true_x": round(true_pose[0], 4),
                "true_y": round(true_pose[1], 4),
                "true_theta": round(true_pose[2], 4),
                "estimate_x": round(state[0], 4),
                "estimate_y": round(state[1], 4),
                "estimate_theta": round(state[2], 4),
                "pose_error": round(pose_error, 4),
                "covariance_trace": round(_trace(covariance), 4),
            }
        )

    landmark_estimates = _landmark_estimates(state, scenario.landmarks)
    errors = [float(sample["pose_error"]) for sample in samples]
    metrics: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "steps": steps,
        "seed": seed,
        "landmark_count": landmark_count,
        "state_size": len(state),
        "final_pose_error": round(errors[-1], 4),
        "mean_pose_error": round(sum(errors) / len(errors), 4),
        "landmark_rmse": round(_landmark_rmse(landmark_estimates), 4),
        "final_covariance_trace": round(_trace(covariance), 4),
        "range_sigma": range_sigma,
        "bearing_sigma": bearing_sigma,
    }

    return EkfSlamResult(
        metrics=metrics,
        samples=samples,
        landmarks=[
            {
                "id": item["id"],
                "true_x": round(float(item["true_x"]), 4),
                "true_y": round(float(item["true_y"]), 4),
                "estimate_x": round(float(item["estimate_x"]), 4),
                "estimate_y": round(float(item["estimate_y"]), 4),
                "error": round(float(item["error"]), 4),
            }
            for item in landmark_estimates
        ],
        covariance_diagonal=[round(covariance[index][index], 6) for index in range(len(state))],
    )


def _validate_parameters(
    *,
    steps: int,
    range_sigma: float,
    bearing_sigma: float,
    motion_sigma: float,
    turn_sigma: float,
) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if range_sigma <= 0:
        raise ValueError("range_sigma must be positive")
    if bearing_sigma <= 0:
        raise ValueError("bearing_sigma must be positive")
    if motion_sigma < 0:
        raise ValueError("motion_sigma must be non-negative")
    if turn_sigma < 0:
        raise ValueError("turn_sigma must be non-negative")


def _initial_state(scenario: EkfSlamScenario, rng: random.Random) -> Vector:
    state = [
        scenario.start[0] + rng.gauss(0.0, 0.08),
        scenario.start[1] + rng.gauss(0.0, 0.08),
        _wrap_angle(scenario.start[2] + rng.gauss(0.0, 0.03)),
    ]
    for landmark_x, landmark_y in scenario.landmarks:
        state.extend(
            [
                landmark_x + rng.gauss(0.0, 0.65),
                landmark_y + rng.gauss(0.0, 0.65),
            ]
        )
    return state


def _initial_covariance(landmark_count: int) -> Matrix:
    size = 3 + 2 * landmark_count
    covariance = _zeros(size, size)
    diagonal = [0.14, 0.14, 0.05] + [1.2, 1.2] * landmark_count
    for index, value in enumerate(diagonal):
        covariance[index][index] = value
    return covariance


def _move_pose(pose: Pose, control: tuple[float, float]) -> Pose:
    distance, turn = control
    theta = _wrap_angle(pose[2] + turn)
    x = pose[0] + distance * math.cos(theta)
    y = pose[1] + distance * math.sin(theta)
    return x, y, theta


def _predict(
    state: Vector,
    covariance: Matrix,
    control: tuple[float, float],
    motion_sigma: float,
    turn_sigma: float,
) -> tuple[Vector, Matrix]:
    distance, turn = control
    theta = _wrap_angle(state[2] + turn)
    predicted = state[:]
    predicted[0] = state[0] + distance * math.cos(theta)
    predicted[1] = state[1] + distance * math.sin(theta)
    predicted[2] = theta

    size = len(state)
    g = _identity(size)
    g[0][2] = -distance * math.sin(theta)
    g[1][2] = distance * math.cos(theta)

    process_noise = _zeros(size, size)
    process_noise[0][0] = motion_sigma**2
    process_noise[1][1] = motion_sigma**2
    process_noise[2][2] = turn_sigma**2

    predicted_covariance = _add(_matmul(_matmul(g, covariance), _transpose(g)), process_noise)
    return predicted, predicted_covariance


def _observe_landmark(
    pose: Pose,
    landmark: Point,
    *,
    rng: random.Random,
    range_sigma: float,
    bearing_sigma: float,
) -> tuple[float, float]:
    dx = landmark[0] - pose[0]
    dy = landmark[1] - pose[1]
    distance = math.hypot(dx, dy)
    bearing = _wrap_angle(math.atan2(dy, dx) - pose[2])
    return (
        distance + rng.gauss(0.0, range_sigma),
        _wrap_angle(bearing + rng.gauss(0.0, bearing_sigma)),
    )


def _update_landmark(
    state: Vector,
    covariance: Matrix,
    landmark_index: int,
    measurement: tuple[float, float],
    range_sigma: float,
    bearing_sigma: float,
) -> tuple[Vector, Matrix]:
    lx_index = 3 + 2 * landmark_index
    ly_index = lx_index + 1
    dx = state[lx_index] - state[0]
    dy = state[ly_index] - state[1]
    q = max(dx * dx + dy * dy, 1e-9)
    predicted_range = math.sqrt(q)
    predicted_bearing = _wrap_angle(math.atan2(dy, dx) - state[2])
    innovation = [
        measurement[0] - predicted_range,
        _wrap_angle(measurement[1] - predicted_bearing),
    ]

    size = len(state)
    h = _zeros(2, size)
    h[0][0] = -dx / predicted_range
    h[0][1] = -dy / predicted_range
    h[0][lx_index] = dx / predicted_range
    h[0][ly_index] = dy / predicted_range
    h[1][0] = dy / q
    h[1][1] = -dx / q
    h[1][2] = -1.0
    h[1][lx_index] = -dy / q
    h[1][ly_index] = dx / q

    measurement_noise = [
        [range_sigma**2, 0.0],
        [0.0, bearing_sigma**2],
    ]
    s = _add(_matmul(_matmul(h, covariance), _transpose(h)), measurement_noise)
    k = _matmul(_matmul(covariance, _transpose(h)), _inverse_2x2(s))
    updated_state = _add_vector(state, _matvec(k, innovation))
    updated_state[2] = _wrap_angle(updated_state[2])

    identity = _identity(size)
    updated_covariance = _matmul(_subtract(identity, _matmul(k, h)), covariance)
    updated_covariance = _symmetrize(updated_covariance)
    return updated_state, updated_covariance


def _landmark_estimates(state: Vector, truth: tuple[Point, ...]) -> list[dict[str, float | int]]:
    estimates: list[dict[str, float | int]] = []
    for index, (true_x, true_y) in enumerate(truth):
        lx_index = 3 + 2 * index
        estimate_x = state[lx_index]
        estimate_y = state[lx_index + 1]
        estimates.append(
            {
                "id": index,
                "true_x": true_x,
                "true_y": true_y,
                "estimate_x": estimate_x,
                "estimate_y": estimate_y,
                "error": _distance((estimate_x, estimate_y), (true_x, true_y)),
            }
        )
    return estimates


def _landmark_rmse(landmarks: list[dict[str, float | int]]) -> float:
    if not landmarks:
        return 0.0
    mse = sum(float(item["error"]) ** 2 for item in landmarks) / len(landmarks)
    return math.sqrt(mse)


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _identity(size: int) -> Matrix:
    matrix = _zeros(size, size)
    for index in range(size):
        matrix[index][index] = 1.0
    return matrix


def _transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix)]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    inner = len(right)
    cols = len(right[0])
    result = _zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            result[row][col] = sum(left[row][k] * right[k][col] for k in range(inner))
    return result


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(value * vector[col] for col, value in enumerate(row)) for row in matrix]


def _add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][col] + right[row][col] for col in range(len(left[0]))]
        for row in range(len(left))
    ]


def _subtract(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][col] - right[row][col] for col in range(len(left[0]))]
        for row in range(len(left))
    ]


def _add_vector(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right)]


def _inverse_2x2(matrix: Matrix) -> Matrix:
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise ValueError("innovation covariance is singular")
    inv_det = 1.0 / determinant
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det],
    ]


def _symmetrize(matrix: Matrix) -> Matrix:
    size = len(matrix)
    result = _zeros(size, size)
    for row in range(size):
        for col in range(size):
            result[row][col] = 0.5 * (matrix[row][col] + matrix[col][row])
    return result


def _trace(matrix: Matrix) -> float:
    return sum(matrix[index][index] for index in range(len(matrix)))


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
