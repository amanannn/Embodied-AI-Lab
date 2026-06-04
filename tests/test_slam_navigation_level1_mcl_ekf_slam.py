import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "experiments/02-slam-navigation/level-1-python"
MCL_MODULE = LEVEL1 / "localization/mcl.py"
MCL_SCRIPT = LEVEL1 / "scripts/mcl_localization.py"
EKF_MODULE = LEVEL1 / "slam/ekf_slam.py"
EKF_SCRIPT = LEVEL1 / "scripts/ekf_slam.py"
README = LEVEL1 / "README.md"
README_EN = LEVEL1 / "README.en.md"
DIRECTION_README = ROOT / "experiments/02-slam-navigation/README.md"
DIRECTION_README_EN = ROOT / "experiments/02-slam-navigation/README.en.md"
MAP_README = ROOT / "docs/curriculum/legacy-to-direction-map.md"
MAP_README_EN = ROOT / "docs/curriculum/legacy-to-direction-map.en.md"
MCL_TUTORIAL = LEVEL1 / "tutorials/mcl_localization.md"
MCL_TUTORIAL_EN = LEVEL1 / "tutorials/mcl_localization.en.md"
EKF_TUTORIAL = LEVEL1 / "tutorials/ekf_slam.md"
EKF_TUTORIAL_EN = LEVEL1 / "tutorials/ekf_slam.en.md"


def load_module(path: pathlib.Path, name: str):
    if not path.is_file():
        raise AssertionError(f"Missing module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SlamNavigationLevel1MclEkfSlamTest(unittest.TestCase):
    def test_mcl_demo_converges_better_than_dead_reckoning(self) -> None:
        mcl = load_module(MCL_MODULE, "slam_mcl")

        result = mcl.run_mcl_demo(seed=11, particle_count=450, steps=24)

        self.assertEqual(result.metrics["scenario"], "indoor-beacons")
        self.assertEqual(result.metrics["particle_count"], 450)
        self.assertGreaterEqual(len(result.samples), 24)
        self.assertLess(result.metrics["final_error"], 0.9)
        self.assertLess(result.metrics["mean_error"], 0.9)
        self.assertLess(result.metrics["final_error"], result.metrics["final_odometry_error"])
        self.assertGreater(result.metrics["effective_particle_count"], 40)
        self.assertTrue(all(math.isfinite(sample["estimate_x"]) for sample in result.samples))
        self.assertTrue(all(math.isfinite(sample["estimate_y"]) for sample in result.samples))

    def test_mcl_rejects_invalid_teaching_parameters(self) -> None:
        mcl = load_module(MCL_MODULE, "slam_mcl_invalid")

        with self.assertRaises(ValueError):
            mcl.run_mcl_demo(particle_count=0)
        with self.assertRaises(ValueError):
            mcl.run_mcl_demo(sensor_sigma=0.0)
        with self.assertRaises(ValueError):
            mcl.run_mcl_demo(steps=0)

    def test_mcl_default_parameters_are_good_for_teaching(self) -> None:
        mcl = load_module(MCL_MODULE, "slam_mcl_defaults")

        result = mcl.run_mcl_demo()

        self.assertLess(result.metrics["final_error"], 0.35)
        self.assertLess(result.metrics["final_error"], result.metrics["final_odometry_error"])
        self.assertGreater(result.metrics["effective_particle_count"], 80)

    def test_mcl_cli_writes_json_outputs_without_plot(self) -> None:
        self.assertTrue(MCL_SCRIPT.is_file(), f"Missing script: {MCL_SCRIPT}")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(MCL_SCRIPT),
                    "--particles",
                    "300",
                    "--steps",
                    "18",
                    "--seed",
                    "5",
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
                (pathlib.Path(tmpdir) / "mcl_localization_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            samples = json.loads(
                (pathlib.Path(tmpdir) / "mcl_localization_samples.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metrics["particle_count"], 300)
            self.assertLess(metrics["final_error"], metrics["final_odometry_error"])
            self.assertGreaterEqual(len(samples), 18)
            self.assertIn("s02-mcl-localization", result.stdout)
            self.assertIn("visualization=disabled", result.stdout)

    def test_mcl_script_marks_visualization_unavailable_when_plot_is_skipped(self) -> None:
        script = load_module(MCL_SCRIPT, "slam_mcl_script_plot_status")
        mcl = load_module(MCL_MODULE, "slam_mcl_plot_status")
        result = mcl.run_mcl_demo(steps=3, particle_count=120)
        script.maybe_write_plot = lambda *_args, **_kwargs: False

        with tempfile.TemporaryDirectory() as tmpdir:
            visualization = script.write_outputs(result, pathlib.Path(tmpdir), no_plot=False)
            metrics = json.loads(
                (pathlib.Path(tmpdir) / "mcl_localization_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(visualization, "unavailable")
        self.assertEqual(metrics["visualization"], "unavailable")

    def test_ekf_slam_demo_estimates_pose_and_landmarks(self) -> None:
        ekf = load_module(EKF_MODULE, "slam_ekf")

        result = ekf.run_ekf_slam_demo(seed=7, steps=22)

        self.assertEqual(result.metrics["scenario"], "range-bearing-landmarks")
        self.assertEqual(result.metrics["landmark_count"], 3)
        self.assertEqual(result.metrics["state_size"], 9)
        self.assertGreaterEqual(len(result.samples), 22)
        self.assertLess(result.metrics["final_pose_error"], 0.5)
        self.assertLess(result.metrics["mean_pose_error"], 0.45)
        self.assertLess(result.metrics["landmark_rmse"], 0.7)
        self.assertGreater(result.metrics["final_covariance_trace"], 0.0)
        self.assertTrue(all(math.isfinite(sample["estimate_x"]) for sample in result.samples))

    def test_ekf_slam_rejects_invalid_teaching_parameters(self) -> None:
        ekf = load_module(EKF_MODULE, "slam_ekf_invalid")

        with self.assertRaises(ValueError):
            ekf.run_ekf_slam_demo(steps=0)
        with self.assertRaises(ValueError):
            ekf.run_ekf_slam_demo(range_sigma=0.0)
        with self.assertRaises(ValueError):
            ekf.run_ekf_slam_demo(bearing_sigma=0.0)

    def test_ekf_slam_default_parameters_are_good_for_teaching(self) -> None:
        ekf = load_module(EKF_MODULE, "slam_ekf_defaults")

        result = ekf.run_ekf_slam_demo()

        self.assertLess(result.metrics["final_pose_error"], 0.5)
        self.assertLess(result.metrics["mean_pose_error"], 0.35)
        self.assertLess(result.metrics["landmark_rmse"], 0.5)

    def test_ekf_slam_cli_writes_json_outputs_without_plot(self) -> None:
        self.assertTrue(EKF_SCRIPT.is_file(), f"Missing script: {EKF_SCRIPT}")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(EKF_SCRIPT),
                    "--steps",
                    "18",
                    "--seed",
                    "9",
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
                (pathlib.Path(tmpdir) / "ekf_slam_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            landmarks = json.loads(
                (pathlib.Path(tmpdir) / "ekf_slam_landmarks.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(metrics["landmark_count"], 3)
            self.assertLess(metrics["final_pose_error"], 0.6)
            self.assertEqual(len(landmarks), 3)
            self.assertIn("s03-ekf-slam", result.stdout)
            self.assertIn("visualization=disabled", result.stdout)

    def test_ekf_slam_script_marks_visualization_unavailable_when_plot_is_skipped(self) -> None:
        script = load_module(EKF_SCRIPT, "slam_ekf_script_plot_status")
        ekf = load_module(EKF_MODULE, "slam_ekf_plot_status")
        result = ekf.run_ekf_slam_demo(steps=3)
        script.maybe_write_plot = lambda *_args, **_kwargs: False

        with tempfile.TemporaryDirectory() as tmpdir:
            visualization = script.write_outputs(result, pathlib.Path(tmpdir), no_plot=False)
            metrics = json.loads(
                (pathlib.Path(tmpdir) / "ekf_slam_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(visualization, "unavailable")
        self.assertEqual(metrics["visualization"], "unavailable")

    def test_slam_navigation_docs_mark_mcl_and_ekf_slam_complete(self) -> None:
        docs = [README, README_EN, DIRECTION_README, DIRECTION_README_EN]
        tutorials = [MCL_TUTORIAL, MCL_TUTORIAL_EN, EKF_TUTORIAL, EKF_TUTORIAL_EN]
        for path in docs + tutorials:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Missing doc: {path}")

        zh = README.read_text(encoding="utf-8")
        en = README_EN.read_text(encoding="utf-8")
        direction = DIRECTION_README.read_text(encoding="utf-8")
        direction_en = DIRECTION_README_EN.read_text(encoding="utf-8")
        mcl_tutorial = MCL_TUTORIAL.read_text(encoding="utf-8")
        ekf_tutorial = EKF_TUTORIAL.read_text(encoding="utf-8")
        mcl_tutorial_en = MCL_TUTORIAL_EN.read_text(encoding="utf-8")
        ekf_tutorial_en = EKF_TUTORIAL_EN.read_text(encoding="utf-8")

        for text in [zh, direction]:
            self.assertIn("s02-mcl-localization", text)
            self.assertIn("s03-ekf-slam", text)
            self.assertIn("已完成", text)
            self.assertNotIn("蒙特卡洛定位（候选）", text)
            self.assertNotIn("扩展卡尔曼 SLAM（候选）", text)

        for text in [en, direction_en]:
            self.assertIn("s02-mcl-localization", text)
            self.assertIn("s03-ekf-slam", text)
            self.assertIn("complete", text.lower())
            self.assertNotIn("Monte Carlo Localization (candidate)", text)
            self.assertNotIn("Extended Kalman Filter SLAM (candidate)", text)

        self.assertIn("python scripts/mcl_localization.py", mcl_tutorial)
        self.assertIn("python scripts/ekf_slam.py", ekf_tutorial)
        self.assertIn("--no-plot", mcl_tutorial_en)
        self.assertIn("--no-plot", ekf_tutorial_en)

    def test_legacy_map_marks_mcl_and_ekf_slam_as_migrated(self) -> None:
        legacy_map = MAP_README.read_text(encoding="utf-8")
        legacy_map_en = MAP_README_EN.read_text(encoding="utf-8")

        self.assertIn("`s02-mcl-localization` (已迁移)", legacy_map)
        self.assertIn("`s03-ekf-slam` (已迁移)", legacy_map)
        self.assertIn("`s02-mcl-localization` (migrated)", legacy_map_en)
        self.assertIn("`s03-ekf-slam` (migrated)", legacy_map_en)


if __name__ == "__main__":
    unittest.main()
