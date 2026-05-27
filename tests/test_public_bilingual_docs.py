import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    "README.md",
    "README.en.md",
    "experiments/01-perception/README.md",
    "experiments/01-perception/README.en.md",
    "experiments/02-slam-navigation/README.md",
    "experiments/02-slam-navigation/README.en.md",
    "shared/README.md",
    "shared/README.en.md",
    "docs/curriculum/legacy-to-direction-map.md",
    "docs/curriculum/legacy-to-direction-map.en.md",
]

PRIVATE_FILES = [
    "AGENTS.md",
    "tests/test_guidance_docs.py",
]

PRIVATE_DIRS = [
    "docs/superpowers",
]

ROOT_README = ROOT / "README.md"
ROOT_README_EN = ROOT / "README.en.md"
PERCEPTION_README = ROOT / "experiments/01-perception/README.md"
PERCEPTION_README_EN = ROOT / "experiments/01-perception/README.en.md"
SLAM_README = ROOT / "experiments/02-slam-navigation/README.md"
SLAM_README_EN = ROOT / "experiments/02-slam-navigation/README.en.md"
SHARED_README = ROOT / "shared/README.md"
SHARED_README_EN = ROOT / "shared/README.en.md"
MAP_README = ROOT / "docs/curriculum/legacy-to-direction-map.md"
MAP_README_EN = ROOT / "docs/curriculum/legacy-to-direction-map.en.md"


class PublicBilingualDocsTest(unittest.TestCase):
    def test_public_bilingual_files_exist(self) -> None:
        missing = [path for path in PUBLIC_DOCS if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], f"Missing public bilingual files: {missing}")

    def test_private_agent_files_absent(self) -> None:
        present = [path for path in PRIVATE_FILES if (ROOT / path).exists()]
        self.assertEqual(present, [], f"Private files should be absent: {present}")

    def test_private_agent_directories_absent(self) -> None:
        present = [path for path in PRIVATE_DIRS if (ROOT / path).exists()]
        self.assertEqual(present, [], f"Private directories should be absent: {present}")

    def test_root_homepage_has_language_links(self) -> None:
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)
        self.assertIn("## 这是什么项目", zh)
        self.assertIn("## What This Repository Is", en)

    def test_direction_pages_have_language_links(self) -> None:
        zh = PERCEPTION_README.read_text(encoding="utf-8")
        en = PERCEPTION_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)

        zh = SLAM_README.read_text(encoding="utf-8")
        en = SLAM_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)

    def test_shared_and_map_have_language_links(self) -> None:
        zh = SHARED_README.read_text(encoding="utf-8")
        en = SHARED_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)

        zh = MAP_README.read_text(encoding="utf-8")
        en = MAP_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [legacy-to-direction-map.en.md](./legacy-to-direction-map.en.md)", zh)
        self.assertIn("中文： [legacy-to-direction-map.md](./legacy-to-direction-map.md)", en)


if __name__ == "__main__":
    unittest.main()
