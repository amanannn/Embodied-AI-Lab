from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
KALMAN_TUTORIAL = (
    ROOT / "experiments/01-perception/level-1-python/tutorials/kalman.md"
)


class PerceptionLevel1DocsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.content = KALMAN_TUTORIAL.read_text(encoding="utf-8")

    def test_kalman_tutorial_uses_current_layout(self) -> None:
        self.assertIn("cd experiments/01-perception/level-1-python", self.content)
        self.assertIn("python scripts/all.py", self.content)
        self.assertIn("python scripts/kf.py", self.content)

    def test_kalman_tutorial_does_not_reference_legacy_demo_layout(self) -> None:
        forbidden_fragments = [
            "demos/",
            "demo_kf.py",
            "demo_all.py",
            "kalman-filter-lab/",
            "tutorial.md",
        ]
        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.content)


if __name__ == "__main__":
    unittest.main()
