import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "experiments/02-slam-navigation/level-1-python"
MODULE_PATH = LEVEL1 / "planners/grid_search.py"
SCRIPT_PATH = LEVEL1 / "scripts/grid_search.py"
README = LEVEL1 / "README.md"
README_EN = LEVEL1 / "README.en.md"
TUTORIAL = LEVEL1 / "tutorials/grid_search.md"
TUTORIAL_EN = LEVEL1 / "tutorials/grid_search.en.md"
DIRECTION_README = ROOT / "experiments/02-slam-navigation/README.md"
DIRECTION_README_EN = ROOT / "experiments/02-slam-navigation/README.en.md"
MAP_README = ROOT / "docs/curriculum/legacy-to-direction-map.md"
MAP_README_EN = ROOT / "docs/curriculum/legacy-to-direction-map.en.md"


def load_grid_search_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing grid search module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("slam_grid_search", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_grid_search_script_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing grid search script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("slam_grid_search_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SlamNavigationLevel1GridSearchTest(unittest.TestCase):
    def test_astar_and_dijkstra_find_same_shortest_path(self) -> None:
        grid_search = load_grid_search_module()
        grid = [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        astar = grid_search.astar(grid, start=(0, 0), goal=(4, 4))
        dijkstra = grid_search.dijkstra(grid, start=(0, 0), goal=(4, 4))

        self.assertEqual(astar.path[0], (0, 0))
        self.assertEqual(astar.path[-1], (4, 4))
        self.assertEqual(astar.path_cost, 8)
        self.assertEqual(dijkstra.path_cost, 8)
        self.assertLessEqual(astar.expanded_nodes, dijkstra.expanded_nodes)
        self.assertEqual(astar.algorithm, "astar")
        self.assertEqual(dijkstra.algorithm, "dijkstra")

    def test_grid_search_rejects_blocked_or_invalid_queries(self) -> None:
        grid_search = load_grid_search_module()
        grid = [
            [0, 1],
            [0, 0],
        ]

        with self.assertRaises(ValueError):
            grid_search.astar(grid, start=(0, 1), goal=(1, 1))
        with self.assertRaises(ValueError):
            grid_search.dijkstra(grid, start=(-1, 0), goal=(1, 1))

    def test_ascii_renderer_marks_start_goal_obstacles_and_path(self) -> None:
        grid_search = load_grid_search_module()
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

        rendered = grid_search.render_ascii_grid(
            grid,
            path=path,
            start=(0, 0),
            goal=(2, 2),
        )

        self.assertIn("S", rendered)
        self.assertIn("G", rendered)
        self.assertIn("*", rendered)
        self.assertIn("#", rendered)

    def test_grid_search_cli_writes_metrics_and_skips_ascii_by_default(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing script: {SCRIPT_PATH}")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--algorithm",
                    "astar",
                    "--output-dir",
                    tmpdir,
                    "--no-plot",
                ],
                cwd=LEVEL1,
                text=True,
                capture_output=True,
                check=True,
            )

            metrics_path = pathlib.Path(tmpdir) / "grid_search_metrics.json"
            ascii_path = pathlib.Path(tmpdir) / "grid_search_map.txt"
            self.assertTrue(metrics_path.is_file())
            self.assertFalse(ascii_path.exists())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["algorithm"], "astar")
            self.assertGreater(metrics["path_cost"], 0)
            self.assertGreater(metrics["expanded_nodes"], 0)
            self.assertIn("s01-grid-search", result.stdout)
            self.assertIn("visualization=disabled", result.stdout)

    def test_default_map_is_a_teaching_maze_not_a_short_corridor(self) -> None:
        script = load_grid_search_script_module()

        astar = script.run_algorithm(
            "astar",
            map_name=script.DEFAULT_MAP_NAME,
            start=script.DEFAULT_START,
            goal=script.DEFAULT_GOAL,
        )
        dijkstra = script.run_algorithm(
            "dijkstra",
            map_name=script.DEFAULT_MAP_NAME,
            start=script.DEFAULT_START,
            goal=script.DEFAULT_GOAL,
        )
        grid = script.get_grid(script.DEFAULT_MAP_NAME)

        self.assertEqual(script.DEFAULT_MAP_NAME, "maze")
        self.assertEqual(len(grid), 21)
        self.assertEqual(len(grid[0]), 31)
        self.assertGreaterEqual(astar.path_cost, 70)
        self.assertGreaterEqual(astar.expanded_nodes, 80)
        self.assertLess(astar.expanded_nodes, dijkstra.expanded_nodes)
        self.assertGreaterEqual(dijkstra.expanded_nodes - astar.expanded_nodes, 40)

    def test_cli_supports_named_maps(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing script: {SCRIPT_PATH}")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--map",
                    "rooms",
                    "--algorithm",
                    "astar",
                    "--output-dir",
                    tmpdir,
                    "--no-plot",
                ],
                cwd=LEVEL1,
                text=True,
                capture_output=True,
                check=True,
            )

            metrics = json.loads(
                (pathlib.Path(tmpdir) / "grid_search_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["map_name"], "rooms")
            self.assertIn("map=rooms", result.stdout)

    def test_plot_cache_uses_writable_output_directory(self) -> None:
        script = load_grid_search_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_value = script.os.environ.pop("MPLCONFIGDIR", None)
            try:
                cache_dir = script.prepare_matplotlib_cache(pathlib.Path(tmpdir))
                self.assertTrue(cache_dir.is_dir())
                self.assertEqual(script.os.environ["MPLCONFIGDIR"], str(cache_dir))
                self.assertTrue(cache_dir.is_relative_to(pathlib.Path(tmpdir)))
            finally:
                if old_value is not None:
                    script.os.environ["MPLCONFIGDIR"] = old_value
                else:
                    script.os.environ.pop("MPLCONFIGDIR", None)

    def test_grid_search_docs_are_bilingual_and_runnable(self) -> None:
        for path in [README, README_EN, TUTORIAL, TUTORIAL_EN]:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Missing doc: {path}")

        zh = README.read_text(encoding="utf-8")
        en = README_EN.read_text(encoding="utf-8")
        tutorial = TUTORIAL.read_text(encoding="utf-8")
        tutorial_en = TUTORIAL_EN.read_text(encoding="utf-8")

        self.assertIn("s01-grid-search", zh)
        self.assertIn("s01-grid-search", en)
        self.assertIn("已完成", zh)
        self.assertIn("Complete", en)
        self.assertIn("python scripts/grid_search.py", tutorial)
        self.assertIn("python scripts/grid_search.py", tutorial_en)
        self.assertIn("--map maze", tutorial)
        self.assertIn("--map rooms", tutorial_en)
        self.assertIn("matplotlib", tutorial)
        self.assertIn("PNG", tutorial_en)
        self.assertNotIn("ASCII 输出", tutorial)
        self.assertNotIn("ASCII output", tutorial_en)
        self.assertIn("A*", tutorial)
        self.assertIn("Dijkstra", tutorial_en)

    def test_direction_pages_mark_grid_search_as_complete(self) -> None:
        direction = DIRECTION_README.read_text(encoding="utf-8")
        direction_en = DIRECTION_README_EN.read_text(encoding="utf-8")
        legacy_map = MAP_README.read_text(encoding="utf-8")
        legacy_map_en = MAP_README_EN.read_text(encoding="utf-8")

        self.assertIn("s01-grid-search", direction)
        self.assertIn("已完成", direction)
        self.assertNotIn("s01-grid-search` — A* 与 Dijkstra 路径规划（候选）", direction)
        self.assertIn("s01-grid-search", direction_en)
        self.assertIn("complete", direction_en)
        self.assertNotIn("s01-grid-search` — A* and Dijkstra path planning (candidate)", direction_en)
        self.assertIn("`s01-grid-search` (已迁移)", legacy_map)
        self.assertIn("`s01-grid-search` (migrated)", legacy_map_en)


if __name__ == "__main__":
    unittest.main()
