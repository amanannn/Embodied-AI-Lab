import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    "README.md",
    "README.en.md",
    "experiments/01-perception/README.md",
    "experiments/01-perception/README.en.md",
    "experiments/02-slam-navigation/README.md",
    "experiments/02-slam-navigation/README.en.md",
    "experiments/03-motion-control/README.md",
    "experiments/03-motion-control/README.en.md",
    "experiments/04-rl-imitation/README.md",
    "experiments/04-rl-imitation/README.en.md",
    "experiments/05-world-model/README.md",
    "experiments/05-world-model/README.en.md",
    "experiments/06-vision-language-navigation/README.md",
    "experiments/06-vision-language-navigation/README.en.md",
    "experiments/07-manipulation/README.md",
    "experiments/07-manipulation/README.en.md",
    "experiments/08-llm-robot/README.md",
    "experiments/08-llm-robot/README.en.md",
    "experiments/09-sim-to-real/README.md",
    "experiments/09-sim-to-real/README.en.md",
    "experiments/10-vertical-apps/README.md",
    "experiments/10-vertical-apps/README.en.md",
    "experiments/01-perception/level-2-cpp-or-mixed/README.md",
    "experiments/01-perception/level-2-cpp-or-mixed/README.en.md",
    "experiments/02-slam-navigation/level-1-python/README.md",
    "experiments/02-slam-navigation/level-1-python/README.en.md",
    "experiments/02-slam-navigation/level-2-cpp-or-mixed/README.md",
    "experiments/02-slam-navigation/level-2-cpp-or-mixed/README.en.md",
    "experiments/02-slam-navigation/level-3-research/README.md",
    "experiments/02-slam-navigation/level-3-research/README.en.md",
    "experiments/03-motion-control/level-1-python/README.md",
    "experiments/03-motion-control/level-1-python/README.en.md",
    "experiments/03-motion-control/level-2-cpp-or-mixed/README.md",
    "experiments/03-motion-control/level-2-cpp-or-mixed/README.en.md",
    "experiments/03-motion-control/level-3-research/README.md",
    "experiments/03-motion-control/level-3-research/README.en.md",
    "experiments/04-rl-imitation/level-1-python/README.md",
    "experiments/04-rl-imitation/level-1-python/README.en.md",
    "experiments/04-rl-imitation/level-2-cpp-or-mixed/README.md",
    "experiments/04-rl-imitation/level-2-cpp-or-mixed/README.en.md",
    "experiments/04-rl-imitation/level-3-research/README.md",
    "experiments/04-rl-imitation/level-3-research/README.en.md",
    "experiments/05-world-model/level-1-python/README.md",
    "experiments/05-world-model/level-1-python/README.en.md",
    "experiments/05-world-model/level-2-cpp-or-mixed/README.md",
    "experiments/05-world-model/level-2-cpp-or-mixed/README.en.md",
    "experiments/05-world-model/level-3-research/README.md",
    "experiments/05-world-model/level-3-research/README.en.md",
    "experiments/06-vision-language-navigation/level-1-python/README.md",
    "experiments/06-vision-language-navigation/level-1-python/README.en.md",
    "experiments/06-vision-language-navigation/level-2-cpp-or-mixed/README.md",
    "experiments/06-vision-language-navigation/level-2-cpp-or-mixed/README.en.md",
    "experiments/06-vision-language-navigation/level-3-research/README.md",
    "experiments/06-vision-language-navigation/level-3-research/README.en.md",
    "experiments/07-manipulation/level-1-python/README.md",
    "experiments/07-manipulation/level-1-python/README.en.md",
    "experiments/07-manipulation/level-2-cpp-or-mixed/README.md",
    "experiments/07-manipulation/level-2-cpp-or-mixed/README.en.md",
    "experiments/07-manipulation/level-3-research/README.md",
    "experiments/07-manipulation/level-3-research/README.en.md",
    "experiments/08-llm-robot/level-1-python/README.md",
    "experiments/08-llm-robot/level-1-python/README.en.md",
    "experiments/08-llm-robot/level-2-cpp-or-mixed/README.md",
    "experiments/08-llm-robot/level-2-cpp-or-mixed/README.en.md",
    "experiments/08-llm-robot/level-3-research/README.md",
    "experiments/08-llm-robot/level-3-research/README.en.md",
    "experiments/09-sim-to-real/level-1-python/README.md",
    "experiments/09-sim-to-real/level-1-python/README.en.md",
    "experiments/09-sim-to-real/level-2-cpp-or-mixed/README.md",
    "experiments/09-sim-to-real/level-2-cpp-or-mixed/README.en.md",
    "experiments/09-sim-to-real/level-3-research/README.md",
    "experiments/09-sim-to-real/level-3-research/README.en.md",
    "experiments/10-vertical-apps/level-1-python/README.md",
    "experiments/10-vertical-apps/level-1-python/README.en.md",
    "experiments/10-vertical-apps/level-2-cpp-or-mixed/README.md",
    "experiments/10-vertical-apps/level-2-cpp-or-mixed/README.en.md",
    "experiments/10-vertical-apps/level-3-research/README.md",
    "experiments/10-vertical-apps/level-3-research/README.en.md",
    "shared/README.md",
    "shared/README.en.md",
    "shared/sim2d/README.md",
    "shared/sim2d/README.en.md",
    "shared/viz/README.md",
    "shared/viz/README.en.md",
    "shared/math/README.md",
    "shared/math/README.en.md",
    "shared/ros2_interfaces/README.md",
    "shared/ros2_interfaces/README.en.md",
    "shared/datasets/README.md",
    "shared/datasets/README.en.md",
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
BANNER_IMAGE = ROOT / "figure/embodied-ai-lab-banner.png"
PERCEPTION_README = ROOT / "experiments/01-perception/README.md"
PERCEPTION_README_EN = ROOT / "experiments/01-perception/README.en.md"
SLAM_README = ROOT / "experiments/02-slam-navigation/README.md"
SLAM_README_EN = ROOT / "experiments/02-slam-navigation/README.en.md"
SHARED_README = ROOT / "shared/README.md"
SHARED_README_EN = ROOT / "shared/README.en.md"
MAP_README = ROOT / "docs/curriculum/legacy-to-direction-map.md"
MAP_README_EN = ROOT / "docs/curriculum/legacy-to-direction-map.en.md"


class PublicBilingualDocsTest(unittest.TestCase):
    def _is_tracked(self, path: str) -> bool:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def test_public_bilingual_files_exist(self) -> None:
        missing = [path for path in PUBLIC_DOCS if not (ROOT / path).is_file()]
        self.assertEqual(missing, [], f"Missing public bilingual files: {missing}")

    def test_private_agent_files_not_tracked(self) -> None:
        tracked = [path for path in PRIVATE_FILES if self._is_tracked(path)]
        self.assertEqual(tracked, [], f"Private files should not be tracked: {tracked}")

    def test_private_agent_directories_not_tracked(self) -> None:
        tracked = [path for path in PRIVATE_DIRS if self._is_tracked(path)]
        self.assertEqual(
            tracked, [], f"Private directories should not be tracked: {tracked}"
        )

    def test_root_homepage_has_language_links(self) -> None:
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)
        self.assertIn("## 这是什么项目", zh)
        self.assertIn("## What This Repository Is", en)

    def test_root_homepage_uses_public_banner_asset(self) -> None:
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")
        banner = "![Embodied AI Lab banner](./figure/embodied-ai-lab-banner.png)"
        self.assertTrue(BANNER_IMAGE.is_file(), "Missing root README banner image")
        self.assertIn(banner, zh)
        self.assertIn(banner, en)

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

    def test_batch2_direction_pages_have_language_links(self) -> None:
        """第二批方向页（03-10）必须有双向语言链接。"""
        batch2_dirs = [
            "03-motion-control",
            "04-rl-imitation",
            "05-world-model",
            "06-vision-language-navigation",
            "07-manipulation",
            "08-llm-robot",
            "09-sim-to-real",
            "10-vertical-apps",
        ]
        for d in batch2_dirs:
            with self.subTest(direction=d):
                zh = (ROOT / f"experiments/{d}/README.md").read_text(encoding="utf-8")
                en = (ROOT / f"experiments/{d}/README.en.md").read_text(encoding="utf-8")
                self.assertIn("English: [README.en.md](./README.en.md)", zh)
                self.assertIn("中文： [README.md](./README.md)", en)

    def test_shared_submodule_pages_have_language_links(self) -> None:
        """shared/ 子模块页必须有双向语言链接。"""
        submodules = ["sim2d", "viz", "math", "ros2_interfaces", "datasets"]
        for mod in submodules:
            with self.subTest(module=mod):
                zh = (ROOT / f"shared/{mod}/README.md").read_text(encoding="utf-8")
                en = (ROOT / f"shared/{mod}/README.en.md").read_text(encoding="utf-8")
                self.assertIn("English: [README.en.md](./README.en.md)", zh)
                self.assertIn("中文： [README.md](./README.md)", en)

    def test_level_scaffold_pages_have_language_links(self) -> None:
        """新增 level 骨架页必须有双向语言链接。"""
        scaffold_paths = [
            "experiments/01-perception/level-2-cpp-or-mixed",
            "experiments/02-slam-navigation/level-1-python",
            "experiments/02-slam-navigation/level-2-cpp-or-mixed",
            "experiments/02-slam-navigation/level-3-research",
            "experiments/03-motion-control/level-1-python",
            "experiments/03-motion-control/level-2-cpp-or-mixed",
            "experiments/03-motion-control/level-3-research",
            "experiments/04-rl-imitation/level-1-python",
            "experiments/04-rl-imitation/level-2-cpp-or-mixed",
            "experiments/04-rl-imitation/level-3-research",
            "experiments/05-world-model/level-1-python",
            "experiments/05-world-model/level-2-cpp-or-mixed",
            "experiments/05-world-model/level-3-research",
            "experiments/06-vision-language-navigation/level-1-python",
            "experiments/06-vision-language-navigation/level-2-cpp-or-mixed",
            "experiments/06-vision-language-navigation/level-3-research",
            "experiments/07-manipulation/level-1-python",
            "experiments/07-manipulation/level-2-cpp-or-mixed",
            "experiments/07-manipulation/level-3-research",
            "experiments/08-llm-robot/level-1-python",
            "experiments/08-llm-robot/level-2-cpp-or-mixed",
            "experiments/08-llm-robot/level-3-research",
            "experiments/09-sim-to-real/level-1-python",
            "experiments/09-sim-to-real/level-2-cpp-or-mixed",
            "experiments/09-sim-to-real/level-3-research",
            "experiments/10-vertical-apps/level-1-python",
            "experiments/10-vertical-apps/level-2-cpp-or-mixed",
            "experiments/10-vertical-apps/level-3-research",
        ]
        for rel in scaffold_paths:
            with self.subTest(path=rel):
                zh = (ROOT / rel / "README.md").read_text(encoding="utf-8")
                en = (ROOT / rel / "README.en.md").read_text(encoding="utf-8")
                self.assertIn("English: [README.en.md](./README.en.md)", zh)
                self.assertIn("中文： [README.md](./README.md)", en)


if __name__ == "__main__":
    unittest.main()
