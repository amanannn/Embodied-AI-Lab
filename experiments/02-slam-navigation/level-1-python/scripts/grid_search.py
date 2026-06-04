"""Run the s01-grid-search path-planning lab.

Usage:
    python scripts/grid_search.py --algorithm astar
    python scripts/grid_search.py --algorithm dijkstra
    python scripts/grid_search.py --compare
    python scripts/grid_search.py --map rooms --algorithm astar
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import NamedTuple


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from planners.grid_search import (  # noqa: E402
    SearchResult,
    astar,
    dijkstra,
    result_to_metrics,
)


class MapScenario(NamedTuple):
    name: str
    grid: tuple[tuple[int, ...], ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    description: str


def freeze_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def build_simple_grid() -> tuple[tuple[int, ...], ...]:
    return (
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0),
        (1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
        (0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0),
    )


def build_seeded_maze(rows: int = 21, cols: int = 31, seed: int = 4) -> tuple[tuple[int, ...], ...]:
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("maze dimensions must be odd")

    rng = random.Random(seed)
    grid = [[1 for _ in range(cols)] for _ in range(rows)]
    stack = [(1, 1)]
    grid[1][1] = 0
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

    while stack:
        row, col = stack[-1]
        candidates = []
        for d_row, d_col in directions:
            next_row = row + d_row
            next_col = col + d_col
            if (
                1 <= next_row < rows - 1
                and 1 <= next_col < cols - 1
                and grid[next_row][next_col] == 1
            ):
                candidates.append((next_row, next_col, d_row, d_col))

        if not candidates:
            stack.pop()
            continue

        next_row, next_col, d_row, d_col = rng.choice(candidates)
        grid[row + d_row // 2][col + d_col // 2] = 0
        grid[next_row][next_col] = 0
        stack.append((next_row, next_col))

    return freeze_grid(grid)


def build_rooms_grid() -> tuple[tuple[int, ...], ...]:
    rows, cols = 15, 23
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        grid[row][0] = 1
        grid[row][cols - 1] = 1
    for col in range(cols):
        grid[0][col] = 1
        grid[rows - 1][col] = 1

    for col in (6, 14):
        for row in range(1, rows - 1):
            grid[row][col] = 1
        for door_row in (3, 10):
            grid[door_row][col] = 0

    for row in (4, 8, 12):
        for col in range(1, cols - 1):
            grid[row][col] = 1
        for door_col in (3, 10, 18):
            grid[row][door_col] = 0

    return freeze_grid(grid)


MAP_SCENARIOS = {
    "simple": MapScenario(
        name="simple",
        grid=build_simple_grid(),
        start=(0, 0),
        goal=(9, 14),
        description="small legacy grid for quick debugging",
    ),
    "maze": MapScenario(
        name="maze",
        grid=build_seeded_maze(),
        start=(1, 1),
        goal=(19, 29),
        description="seeded 21x31 maze with branches, bottlenecks, and dead ends",
    ),
    "rooms": MapScenario(
        name="rooms",
        grid=build_rooms_grid(),
        start=(1, 1),
        goal=(13, 20),
        description="room-and-corridor map that resembles an indoor floor plan",
    ),
}

DEFAULT_MAP_NAME = "maze"
DEFAULT_GRID = MAP_SCENARIOS[DEFAULT_MAP_NAME].grid
DEFAULT_START = MAP_SCENARIOS[DEFAULT_MAP_NAME].start
DEFAULT_GOAL = MAP_SCENARIOS[DEFAULT_MAP_NAME].goal
DEFAULT_OUTPUT_DIR = BASE_DIR / "output/grid_search"


def parse_coord(raw: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("coordinate must use row,col format")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coordinate values must be integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="s01-grid-search: A* and Dijkstra on a known occupancy grid",
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        choices=sorted(MAP_SCENARIOS),
        default=DEFAULT_MAP_NAME,
        help="built-in map scenario to run",
    )
    parser.add_argument(
        "--algorithm",
        choices=["astar", "dijkstra"],
        default="astar",
        help="single algorithm to run when --compare is not set",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run both A* and Dijkstra and write comparison metrics",
    )
    parser.add_argument(
        "--start",
        type=parse_coord,
        default=None,
        help="optional start coordinate in row,col format; defaults to the map start",
    )
    parser.add_argument(
        "--goal",
        type=parse_coord,
        default=None,
        help="optional goal coordinate in row,col format; defaults to the map goal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for JSON metrics and optional PNG visualization outputs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip PNG rendering; useful in minimal environments",
    )
    return parser.parse_args()


def get_scenario(map_name: str) -> MapScenario:
    try:
        return MAP_SCENARIOS[map_name]
    except KeyError as exc:
        raise ValueError(f"unknown map scenario: {map_name}") from exc


def get_grid(map_name: str) -> tuple[tuple[int, ...], ...]:
    return get_scenario(map_name).grid


def run_algorithm(
    algorithm: str,
    map_name: str,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> SearchResult:
    grid = get_grid(map_name)
    if algorithm == "astar":
        return astar(grid, start=start, goal=goal)
    return dijkstra(grid, start=start, goal=goal)


def write_single_outputs(
    result: SearchResult,
    map_name: str,
    grid: tuple[tuple[int, ...], ...],
    start: tuple[int, int],
    goal: tuple[int, int],
    output_dir: Path,
    no_plot: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_to_metrics(result)
    metrics["map_name"] = map_name
    metrics["visualization"] = "disabled" if no_plot else "png"
    (output_dir / "grid_search_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not no_plot:
        prepare_matplotlib_cache(output_dir)
        maybe_write_plot(
            result,
            map_name=map_name,
            grid=grid,
            start=start,
            goal=goal,
            output_path=output_dir / "grid_search.png",
        )


def write_comparison_outputs(
    results: list[SearchResult],
    map_name: str,
    grid: tuple[tuple[int, ...], ...],
    start: tuple[int, int],
    goal: tuple[int, int],
    output_dir: Path,
    no_plot: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for result in results:
        result_metrics = result_to_metrics(result)
        result_metrics["map_name"] = map_name
        result_metrics["visualization"] = "disabled" if no_plot else "png"
        metrics.append(result_metrics)
    (output_dir / "grid_search_comparison.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for result in results:
        if not no_plot:
            prepare_matplotlib_cache(output_dir)
            maybe_write_plot(
                result,
                map_name=map_name,
                grid=grid,
                start=start,
                goal=goal,
                output_path=output_dir / f"grid_search_{result.algorithm}.png",
            )


def prepare_matplotlib_cache(output_dir: Path) -> Path:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    return cache_dir


def maybe_write_plot(
    result: SearchResult,
    map_name: str,
    grid: tuple[tuple[int, ...], ...],
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed; skip PNG rendering")
        return

    rows = len(grid)
    cols = len(grid[0])
    cell_colors = []
    path_cells = set(result.path)
    for row in range(rows):
        rendered_row = []
        for col in range(cols):
            coord = (row, col)
            if grid[row][col] == 1:
                rendered_row.append(0)
            elif coord in path_cells:
                rendered_row.append(3)
            elif coord in result.visited:
                rendered_row.append(2)
            else:
                rendered_row.append(1)
        cell_colors.append(rendered_row)

    cmap = matplotlib.colors.ListedColormap(["#1f2933", "#f7efe1", "#9bd6ff", "#f97316"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cell_colors, cmap=cmap, interpolation="nearest")
    ax.plot(
        [col for _, col in result.path],
        [row for row, _ in result.path],
        color="#c2410c",
        linewidth=3.0,
        alpha=0.88,
        label="shortest path",
    )
    ax.scatter(start[1], start[0], marker="o", s=180, c="#16a34a", edgecolors="white", label="start")
    ax.scatter(goal[1], goal[0], marker="*", s=260, c="#dc2626", edgecolors="white", label="goal")
    ax.set_xticks(range(cols), minor=True)
    ax.set_yticks(range(rows), minor=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(which="minor", color="#ffffff", linewidth=0.35, alpha=0.28)
    ax.set_title(
        f"{map_name} / {result.algorithm.upper()}: cost={result.path_cost:.0f}, "
        f"expanded={result.expanded_nodes}"
    )
    ax.set_facecolor("#111827")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def print_result(result: SearchResult) -> None:
    print(
        f"  {result.algorithm:<8} found={result.found} "
        f"cost={result.path_cost:.0f} "
        f"path_len={len(result.path)} "
        f"expanded={result.expanded_nodes} "
        f"runtime={result.runtime_ms:.3f}ms"
    )


def main() -> None:
    args = parse_args()
    scenario = get_scenario(args.map_name)
    start = args.start if args.start is not None else scenario.start
    goal = args.goal if args.goal is not None else scenario.goal

    print("=" * 64)
    print("  s01-grid-search: known-map path planning")
    print("=" * 64)
    print(f"  map={scenario.name} ({scenario.description})")
    print(f"  start={start}, goal={goal}")

    if args.compare:
        results = [
            run_algorithm("astar", map_name=scenario.name, start=start, goal=goal),
            run_algorithm("dijkstra", map_name=scenario.name, start=start, goal=goal),
        ]
        print("\nMetrics:")
        for result in results:
            print_result(result)
        write_comparison_outputs(
            results,
            scenario.name,
            scenario.grid,
            start,
            goal,
            args.output_dir,
            args.no_plot,
        )
        print("  visualization=disabled" if args.no_plot else "  visualization=png")
        print(f"\n[OK] outputs written to {args.output_dir}")
        return

    result = run_algorithm(args.algorithm, map_name=scenario.name, start=start, goal=goal)
    print("\nMetrics:")
    print_result(result)
    write_single_outputs(
        result,
        scenario.name,
        scenario.grid,
        start,
        goal,
        args.output_dir,
        args.no_plot,
    )
    print("  visualization=disabled" if args.no_plot else "  visualization=png")
    print(f"\n[OK] outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
