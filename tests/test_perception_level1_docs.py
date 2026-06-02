from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "experiments/01-perception/level-1-python"
KALMAN_TUTORIAL = (
    LEVEL1 / "tutorials/kalman.md"
)


class PerceptionLevel1DocsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.level1 = LEVEL1
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

    def test_level1_readmes_exist_and_link_languages(self) -> None:
        zh = self.level1 / "README.md"
        en = self.level1 / "README.en.md"

        self.assertTrue(zh.exists(), "Level 1 中文 README 缺失")
        self.assertTrue(en.exists(), "Level 1 英文 README 缺失")
        self.assertIn("English: [README.en.md](./README.en.md)", zh.read_text(encoding="utf-8"))
        self.assertIn("中文: [README.md](./README.md)", en.read_text(encoding="utf-8"))

    def test_level1_readme_explains_learning_route(self) -> None:
        text = (self.level1 / "README.md").read_text(encoding="utf-8")

        required_terms = [
            "Lab 01",
            "Lab 02",
            "Lab 03",
            "Lab 04",
            "Lab 05",
            "噪声",
            "传感器",
            "多传感器融合",
            "Kalman",
            "相机标定",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, text)

    def test_level1_readme_separates_route_from_modules(self) -> None:
        text = (self.level1 / "README.md").read_text(encoding="utf-8")

        required_terms = [
            "学习路线",
            "代码模块",
            "scripts/",
            "sensors/",
            "filters/",
            "tutorials/",
        ]
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
