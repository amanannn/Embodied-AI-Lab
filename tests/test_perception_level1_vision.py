from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "experiments/01-perception/level-1-python"
SCRIPTS = LEVEL1 / "scripts"
TUTORIALS = LEVEL1 / "tutorials"
ASSETS = LEVEL1 / "assets"

ARUCO_SCRIPT = SCRIPTS / "aruco_pose.py"
CLASSIC_SCRIPT = SCRIPTS / "classic_vision.py"
ARUCO_ZH = TUTORIALS / "aruco_pose.md"
ARUCO_EN = TUTORIALS / "aruco_pose.en.md"
CLASSIC_ZH = TUTORIALS / "classic_vision.md"
CLASSIC_EN = TUTORIALS / "classic_vision.en.md"


class PerceptionLevel1VisionTest(unittest.TestCase):
    def test_vision_lab_files_exist(self) -> None:
        required = [
            ARUCO_SCRIPT,
            CLASSIC_SCRIPT,
            ARUCO_ZH,
            ARUCO_EN,
            CLASSIC_ZH,
            CLASSIC_EN,
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        self.assertEqual(missing, [], f"Missing vision lab files: {missing}")

    def test_vision_tutorials_have_language_links(self) -> None:
        aruco_zh = ARUCO_ZH.read_text(encoding="utf-8")
        aruco_en = ARUCO_EN.read_text(encoding="utf-8")
        classic_zh = CLASSIC_ZH.read_text(encoding="utf-8")
        classic_en = CLASSIC_EN.read_text(encoding="utf-8")

        self.assertIn("English: [aruco_pose.en.md](./aruco_pose.en.md)", aruco_zh)
        self.assertIn("中文： [aruco_pose.md](./aruco_pose.md)", aruco_en)
        self.assertIn("English: [classic_vision.en.md](./classic_vision.en.md)", classic_zh)
        self.assertIn("中文： [classic_vision.md](./classic_vision.md)", classic_en)

    def test_aruco_script_exposes_marker_and_pose_workflow(self) -> None:
        content = ARUCO_SCRIPT.read_text(encoding="utf-8")

        for expected in [
            "--generate-marker",
            "--input-image",
            "--calibration",
            "--marker-size",
            "cv2.aruco",
            "ArucoDetector",
            "estimatePoseSingleMarkers",
            "drawFrameAxes",
            "output/aruco_pose",
        ]:
            with self.subTest(expected=expected):
                self.assertIn(expected, content)

    def test_classic_vision_script_exposes_three_foundation_modes(self) -> None:
        content = CLASSIC_SCRIPT.read_text(encoding="utf-8")

        for expected in [
            "--mode",
            "--input-video",
            "--camera-index",
            "optical-flow",
            "feature-matching",
            "background-subtraction",
            "cv2.calcOpticalFlowPyrLK",
            "cv2.ORB_create",
            "cv2.createBackgroundSubtractorMOG2",
            "output/classic_vision",
        ]:
            with self.subTest(expected=expected):
                self.assertIn(expected, content)

    def test_readmes_promote_completed_vision_labs(self) -> None:
        level1 = (LEVEL1 / "README.md").read_text(encoding="utf-8")
        scripts = (SCRIPTS / "README.md").read_text(encoding="utf-8")

        for expected in [
            "aruco_pose.py",
            "classic_vision.py",
            "tutorials/aruco_pose.md",
            "tutorials/classic_vision.md",
        ]:
            with self.subTest(expected=expected):
                self.assertIn(expected, level1)

        self.assertIn("aruco_pose.py", scripts)
        self.assertIn("classic_vision.py", scripts)


if __name__ == "__main__":
    unittest.main()
