import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "experiments/03-motion-control/level-1-python"
CONTROLLERS = LEVEL1 / "controllers/pid.py"
SCRIPT = LEVEL1 / "scripts/pid_playground.py"
STATIC_INDEX = LEVEL1 / "web/index.html"
STATIC_APP = LEVEL1 / "web/app.js"
STATIC_STYLE = LEVEL1 / "web/styles.css"
README = LEVEL1 / "README.md"
README_EN = LEVEL1 / "README.en.md"
TUTORIAL = LEVEL1 / "tutorials/pid_playground.md"
TUTORIAL_EN = LEVEL1 / "tutorials/pid_playground.en.md"
DIRECTION_README = ROOT / "experiments/03-motion-control/README.md"
DIRECTION_README_EN = ROOT / "experiments/03-motion-control/README.en.md"


def load_module(path: pathlib.Path, module_name: str):
    if not path.is_file():
        raise AssertionError(f"Missing module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MotionControlLevel1PidPlaygroundTest(unittest.TestCase):
    def test_pid_simulation_converges_and_reports_teaching_metrics(self) -> None:
        pid = load_module(CONTROLLERS, "motion_pid")

        result = pid.simulate_pid(
            setpoint=1.0,
            kp=8.0,
            ki=1.2,
            kd=2.6,
            duration=6.0,
            dt=0.02,
            disturbance_time=3.0,
            disturbance_force=-1.5,
        )

        self.assertLess(abs(result.samples[-1]["position"] - 1.0), 0.08)
        self.assertLess(result.metrics["steady_state_error"], 0.08)
        self.assertGreater(result.metrics["overshoot_percent"], 0.0)
        self.assertGreater(result.metrics["settling_time"], 0.0)
        self.assertIn("control", result.samples[0])
        self.assertIn("error", result.samples[0])

    def test_pid_rejects_unstable_or_invalid_parameters(self) -> None:
        pid = load_module(CONTROLLERS, "motion_pid")

        with self.assertRaises(ValueError):
            pid.simulate_pid(kp=-1.0)
        with self.assertRaises(ValueError):
            pid.simulate_pid(duration=0.0)
        with self.assertRaises(ValueError):
            pid.simulate_pid(dt=1.0, duration=0.5)

    def test_pid_reports_directional_overshoot_for_negative_targets(self) -> None:
        pid = load_module(CONTROLLERS, "motion_pid")

        result = pid.simulate_pid(
            setpoint=-1.0,
            kp=8.0,
            ki=1.2,
            kd=2.6,
            duration=6.0,
            dt=0.02,
        )

        self.assertLess(abs(result.samples[-1]["position"] + 1.0), 0.08)
        self.assertLess(result.metrics["overshoot_percent"], 30.0)

    def test_cli_writes_metrics_without_requiring_a_browser(self) -> None:
        self.assertTrue(SCRIPT.is_file(), f"Missing script: {SCRIPT}")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--simulate",
                    "--kp",
                    "8.0",
                    "--ki",
                    "1.2",
                    "--kd",
                    "2.6",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=LEVEL1,
                text=True,
                capture_output=True,
                check=True,
            )

            metrics_path = pathlib.Path(tmpdir) / "pid_playground_metrics.json"
            samples_path = pathlib.Path(tmpdir) / "pid_playground_samples.json"
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(samples_path.is_file())

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertLess(metrics["steady_state_error"], 0.1)
            self.assertIn("c01-pid-control-playground", result.stdout)
            self.assertIn("steady_state_error", result.stdout)

    def test_static_frontend_exposes_interactive_pid_controls(self) -> None:
        for path in [STATIC_INDEX, STATIC_APP, STATIC_STYLE]:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Missing frontend asset: {path}")

        index = STATIC_INDEX.read_text(encoding="utf-8")
        app = STATIC_APP.read_text(encoding="utf-8")
        style = STATIC_STYLE.read_text(encoding="utf-8")

        self.assertIn("PID Control Playground", index)
        self.assertIn('id="kp"', index)
        self.assertIn('id="ki"', index)
        self.assertIn('id="kd"', index)
        self.assertIn("<canvas", index)
        self.assertIn('id="statusMessage"', index)
        self.assertIn('id="initialPayload"', index)
        self.assertIn("__PID_INITIAL_PAYLOAD__", index)
        self.assertIn("/api/simulate", app)
        self.assertIn("drawTrajectory", app)
        self.assertIn("readInitialPayload", app)
        self.assertIn("drawLoadingState", app)
        self.assertIn("setStatus", app)
        self.assertIn("catch", app)
        self.assertIn("--accent", style)
        self.assertIn("@media", style)

    def test_docs_mark_pid_playground_as_complete_and_runnable(self) -> None:
        for path in [README, README_EN, TUTORIAL, TUTORIAL_EN]:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Missing doc: {path}")

        zh = README.read_text(encoding="utf-8")
        en = README_EN.read_text(encoding="utf-8")
        tutorial = TUTORIAL.read_text(encoding="utf-8")
        tutorial_en = TUTORIAL_EN.read_text(encoding="utf-8")
        direction = DIRECTION_README.read_text(encoding="utf-8")
        direction_en = DIRECTION_README_EN.read_text(encoding="utf-8")

        self.assertIn("c01-pid-control-playground", zh)
        self.assertIn("已完成", zh)
        self.assertIn("c01-pid-control-playground", en)
        self.assertIn("Complete", en)
        self.assertIn("python scripts/pid_playground.py --serve", tutorial)
        self.assertIn("python scripts/pid_playground.py --simulate", tutorial)
        self.assertIn("--port 8004", tutorial)
        self.assertIn("Ctrl+C", tutorial)
        self.assertIn("--port 8004", tutorial_en)
        self.assertIn("Ctrl+C", tutorial_en)
        self.assertIn("python scripts/pid_playground.py --serve", tutorial_en)
        self.assertIn("PID", direction)
        self.assertIn("c01-pid-control-playground", direction)
        self.assertIn("c01-pid-control-playground", direction_en)


if __name__ == "__main__":
    unittest.main()
