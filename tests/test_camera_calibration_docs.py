from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/01-perception/level-1-python/scripts/camera_calibration.py"
TUTORIAL_ZH = ROOT / "experiments/01-perception/level-1-python/tutorials/camera_calibration.md"
TUTORIAL_EN = ROOT / "experiments/01-perception/level-1-python/tutorials/camera_calibration.en.md"
SCRIPTS_README = ROOT / "experiments/01-perception/level-1-python/scripts/README.md"


class CameraCalibrationDocsTest(unittest.TestCase):
    def test_camera_calibration_files_exist(self) -> None:
        required = [SCRIPT, TUTORIAL_ZH, TUTORIAL_EN]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        self.assertEqual(missing, [], f"Missing camera calibration files: {missing}")

    def test_camera_calibration_tutorials_have_language_links(self) -> None:
        zh = TUTORIAL_ZH.read_text(encoding="utf-8")
        en = TUTORIAL_EN.read_text(encoding="utf-8")
        self.assertIn("English: [camera_calibration.en.md](./camera_calibration.en.md)", zh)
        self.assertIn("中文： [camera_calibration.md](./camera_calibration.md)", en)

    def test_scripts_readme_mentions_camera_calibration(self) -> None:
        content = SCRIPTS_README.read_text(encoding="utf-8")
        self.assertIn("camera_calibration.py", content)
        self.assertIn("output/camera_calibration.json", content)

    def test_script_exposes_core_calibration_workflow(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")
        for expected in [
            "--generate-board",
            "--input-dir",
            "--capture",
            "cv2.findChessboardCorners",
            "cv2.calibrateCamera",
            "output/camera_calibration.json",
        ]:
            self.assertIn(expected, content)

    def test_script_normalizes_relative_image_paths_before_json_output(self) -> None:
        content = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("def display_path", content)
        self.assertIn("resolved = path.resolve()", content)
        self.assertIn("resolved.relative_to(BASE_DIR)", content)
        self.assertNotIn("path.relative_to(BASE_DIR)", content)


if __name__ == "__main__":
    unittest.main()
