import ast
import csv
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KF_PATH = ROOT / "experiments/01-perception/level-1-python/scripts/kf.py"


def load_csv_rows_function():
    source = KF_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(KF_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_csv_rows":
            selected.append(node)
            break
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"csv": csv}
    exec(compile(module, str(KF_PATH), "exec"), namespace)
    helper = types.SimpleNamespace(load_csv_rows=namespace["load_csv_rows"])
    return helper


class KfExternalDataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kf = load_csv_rows_function()

    def write_csv(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        Path(path).unlink(missing_ok=True)
        csv_path = Path(path)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "sensor_id", "x", "y", "cov_xx", "cov_xy", "cov_yy"]
            )
            writer.writerows(rows)
        return csv_path

    def test_rejects_mixed_sensor_stream(self) -> None:
        csv_path = self.write_csv(
            [
                [0.0, "gps", 1.0, 2.0, 0.25, 0.0, 0.25],
                [0.0, "lidar", 1.1, 2.1, 0.09, 0.0, 0.09],
            ]
        )
        with self.assertRaisesRegex(ValueError, "只接受单一位置观测流"):
            self.kf.load_csv_rows(str(csv_path))

    def test_sorts_and_deduplicates_single_stream(self) -> None:
        csv_path = self.write_csv(
            [
                [0.2, "fused", 2.0, 3.0, 0.1, 0.0, 0.1],
                [0.1, "fused", 1.0, 2.0, 0.1, 0.0, 0.1],
                [0.2, "fused", 9.0, 9.0, 0.1, 0.0, 0.1],
            ]
        )
        timestamps, observations, covariances, sensor_id = self.kf.load_csv_rows(
            str(csv_path)
        )
        self.assertEqual(sensor_id, "fused")
        self.assertEqual(timestamps, [0.1, 0.2])
        self.assertEqual(observations, [[1.0, 2.0], [2.0, 3.0]])
        self.assertEqual(len(covariances), 2)

    def test_external_mode_uses_no_ground_truth_visualization_path(self) -> None:
        content = KF_PATH.read_text(encoding="utf-8")
        self.assertIn("draw_tracking_figure(", content)
        self.assertIn("show_error_panel=False", content)
        self.assertIn('reference_label="Observation Stream"', content)
