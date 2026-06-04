"""Grid-search path planning algorithms for known occupancy maps.

Coordinates use ``(row, col)`` order. Grid cells with value ``0`` are free,
and cells with value ``1`` are obstacles.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import NamedTuple


Coordinate = tuple[int, int]
Grid = list[list[int]] | tuple[tuple[int, ...], ...]


class SearchResult(NamedTuple):
    algorithm: str
    found: bool
    path: list[Coordinate]
    path_cost: float
    expanded_nodes: int
    visited: set[Coordinate]
    runtime_ms: float


def astar(grid: Grid, start: Coordinate, goal: Coordinate) -> SearchResult:
    """Find a shortest path with A* and Manhattan-distance guidance."""

    return _run_grid_search(grid, start, goal, algorithm="astar")


def dijkstra(grid: Grid, start: Coordinate, goal: Coordinate) -> SearchResult:
    """Find a shortest path with uniform-cost search."""

    return _run_grid_search(grid, start, goal, algorithm="dijkstra")


def render_ascii_grid(
    grid: Grid,
    path: list[Coordinate] | None,
    start: Coordinate,
    goal: Coordinate,
    visited: set[Coordinate] | None = None,
) -> str:
    """Render a compact ASCII map for terminal-first teaching."""

    normalized = _normalize_grid(grid)
    path_cells = set(path or [])
    visited_cells = visited or set()
    lines: list[str] = []
    for row_idx, row in enumerate(normalized):
        chars: list[str] = []
        for col_idx, cell in enumerate(row):
            coord = (row_idx, col_idx)
            if coord == start:
                chars.append("S")
            elif coord == goal:
                chars.append("G")
            elif cell == 1:
                chars.append("#")
            elif coord in path_cells:
                chars.append("*")
            elif coord in visited_cells:
                chars.append("+")
            else:
                chars.append(".")
        lines.append(" ".join(chars))
    return "\n".join(lines)


def result_to_metrics(result: SearchResult) -> dict[str, int | float | bool | str]:
    """Convert a search result into JSON-friendly scalar metrics."""

    return {
        "algorithm": result.algorithm,
        "found": result.found,
        "path_length": len(result.path),
        "path_cost": result.path_cost,
        "expanded_nodes": result.expanded_nodes,
        "visited_nodes": len(result.visited),
        "runtime_ms": round(result.runtime_ms, 3),
    }


def _run_grid_search(
    grid: Grid,
    start: Coordinate,
    goal: Coordinate,
    algorithm: str,
) -> SearchResult:
    normalized = _normalize_grid(grid)
    _validate_query(normalized, start, goal)

    start_time = time.perf_counter()
    frontier: list[tuple[float, int, Coordinate]] = []
    heapq.heappush(frontier, (0.0, 0, start))
    came_from: dict[Coordinate, Coordinate | None] = {start: None}
    cost_so_far: dict[Coordinate, float] = {start: 0.0}
    visited: set[Coordinate] = set()
    expanded_nodes = 0
    tie_breaker = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expanded_nodes += 1

        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return SearchResult(
                algorithm=algorithm,
                found=True,
                path=path,
                path_cost=cost_so_far[goal],
                expanded_nodes=expanded_nodes,
                visited=visited,
                runtime_ms=(time.perf_counter() - start_time) * 1000,
            )

        for neighbor in _neighbors(normalized, current):
            new_cost = cost_so_far[current] + 1.0
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                tie_breaker += 1
                priority = new_cost
                if algorithm == "astar":
                    priority += _manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, tie_breaker, neighbor))

    return SearchResult(
        algorithm=algorithm,
        found=False,
        path=[],
        path_cost=math.inf,
        expanded_nodes=expanded_nodes,
        visited=visited,
        runtime_ms=(time.perf_counter() - start_time) * 1000,
    )


def _normalize_grid(grid: Grid) -> tuple[tuple[int, ...], ...]:
    if not grid:
        raise ValueError("grid must not be empty")

    normalized = tuple(tuple(int(cell) for cell in row) for row in grid)
    width = len(normalized[0])
    if width == 0:
        raise ValueError("grid rows must not be empty")

    for row in normalized:
        if len(row) != width:
            raise ValueError("grid must be rectangular")
        for cell in row:
            if cell not in (0, 1):
                raise ValueError("grid cells must be 0 for free or 1 for obstacle")
    return normalized


def _validate_query(
    grid: tuple[tuple[int, ...], ...],
    start: Coordinate,
    goal: Coordinate,
) -> None:
    for name, coord in [("start", start), ("goal", goal)]:
        row, col = coord
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
            raise ValueError(f"{name} coordinate is outside the grid: {coord}")
        if grid[row][col] == 1:
            raise ValueError(f"{name} coordinate is blocked by an obstacle: {coord}")


def _neighbors(
    grid: tuple[tuple[int, ...], ...],
    coord: Coordinate,
) -> list[Coordinate]:
    row, col = coord
    candidates = [
        (row - 1, col),
        (row, col + 1),
        (row + 1, col),
        (row, col - 1),
    ]
    valid: list[Coordinate] = []
    for next_row, next_col in candidates:
        if (
            0 <= next_row < len(grid)
            and 0 <= next_col < len(grid[0])
            and grid[next_row][next_col] == 0
        ):
            valid.append((next_row, next_col))
    return valid


def _reconstruct_path(
    came_from: dict[Coordinate, Coordinate | None],
    goal: Coordinate,
) -> list[Coordinate]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
