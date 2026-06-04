import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    "README.md",
    "README.en.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING.en.md",
    "archive/README.md",
    "archive/README.en.md",
    "archive/legacy-experiments/README.md",
    "archive/legacy-experiments/README.en.md",
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
    "experiments/01-perception/level-2-ros2-bridge/README.md",
    "experiments/01-perception/level-2-ros2-bridge/README.en.md",
    "experiments/02-slam-navigation/level-1-python/README.md",
    "experiments/02-slam-navigation/level-1-python/README.en.md",
    "experiments/02-slam-navigation/level-1-python/tutorials/grid_search.md",
    "experiments/02-slam-navigation/level-1-python/tutorials/grid_search.en.md",
    "experiments/02-slam-navigation/level-2-ros2-bridge/README.md",
    "experiments/02-slam-navigation/level-2-ros2-bridge/README.en.md",
    "experiments/02-slam-navigation/level-3-research/README.md",
    "experiments/02-slam-navigation/level-3-research/README.en.md",
    "experiments/03-motion-control/level-1-python/README.md",
    "experiments/03-motion-control/level-1-python/README.en.md",
    "experiments/03-motion-control/level-2-ros2-bridge/README.md",
    "experiments/03-motion-control/level-2-ros2-bridge/README.en.md",
    "experiments/03-motion-control/level-3-research/README.md",
    "experiments/03-motion-control/level-3-research/README.en.md",
    "experiments/04-rl-imitation/level-1-python/README.md",
    "experiments/04-rl-imitation/level-1-python/README.en.md",
    "experiments/04-rl-imitation/level-2-ros2-bridge/README.md",
    "experiments/04-rl-imitation/level-2-ros2-bridge/README.en.md",
    "experiments/04-rl-imitation/level-3-research/README.md",
    "experiments/04-rl-imitation/level-3-research/README.en.md",
    "experiments/05-world-model/level-1-python/README.md",
    "experiments/05-world-model/level-1-python/README.en.md",
    "experiments/05-world-model/level-2-ros2-bridge/README.md",
    "experiments/05-world-model/level-2-ros2-bridge/README.en.md",
    "experiments/05-world-model/level-3-research/README.md",
    "experiments/05-world-model/level-3-research/README.en.md",
    "experiments/06-vision-language-navigation/level-1-python/README.md",
    "experiments/06-vision-language-navigation/level-1-python/README.en.md",
    "experiments/06-vision-language-navigation/level-2-ros2-bridge/README.md",
    "experiments/06-vision-language-navigation/level-2-ros2-bridge/README.en.md",
    "experiments/06-vision-language-navigation/level-3-research/README.md",
    "experiments/06-vision-language-navigation/level-3-research/README.en.md",
    "experiments/07-manipulation/level-1-python/README.md",
    "experiments/07-manipulation/level-1-python/README.en.md",
    "experiments/07-manipulation/level-2-ros2-bridge/README.md",
    "experiments/07-manipulation/level-2-ros2-bridge/README.en.md",
    "experiments/07-manipulation/level-3-research/README.md",
    "experiments/07-manipulation/level-3-research/README.en.md",
    "experiments/08-llm-robot/level-1-python/README.md",
    "experiments/08-llm-robot/level-1-python/README.en.md",
    "experiments/08-llm-robot/level-2-ros2-bridge/README.md",
    "experiments/08-llm-robot/level-2-ros2-bridge/README.en.md",
    "experiments/08-llm-robot/level-3-research/README.md",
    "experiments/08-llm-robot/level-3-research/README.en.md",
    "experiments/09-sim-to-real/level-1-python/README.md",
    "experiments/09-sim-to-real/level-1-python/README.en.md",
    "experiments/09-sim-to-real/level-2-ros2-bridge/README.md",
    "experiments/09-sim-to-real/level-2-ros2-bridge/README.en.md",
    "experiments/09-sim-to-real/level-3-research/README.md",
    "experiments/09-sim-to-real/level-3-research/README.en.md",
    "experiments/10-vertical-apps/level-1-python/README.md",
    "experiments/10-vertical-apps/level-1-python/README.en.md",
    "experiments/10-vertical-apps/level-2-ros2-bridge/README.md",
    "experiments/10-vertical-apps/level-2-ros2-bridge/README.en.md",
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
    "docs/curriculum/python-first-ros2-ready.md",
    "docs/curriculum/python-first-ros2-ready.en.md",
    "docs/curriculum/README.md",
    "docs/curriculum/README.en.md",
    "experiments/01-perception/level-1-python/README.md",
    "experiments/01-perception/level-1-python/README.en.md",
    "experiments/01-perception/level-3-research/README.md",
    "experiments/01-perception/level-3-research/README.en.md",
]

PYTHON_FIRST_ROS2_READY = ROOT / "docs/curriculum/python-first-ros2-ready.md"
PYTHON_FIRST_ROS2_READY_EN = ROOT / "docs/curriculum/python-first-ros2-ready.en.md"
CURRICULUM_README = ROOT / "docs/curriculum/README.md"
CURRICULUM_README_EN = ROOT / "docs/curriculum/README.en.md"
EXPERIMENTS_DIR = ROOT / "experiments"
LEGACY_ARCHIVE_DIR = ROOT / "archive/legacy-experiments"

PRIVATE_FILES = [
    "AGENTS.md",
    "tests/test_guidance_docs.py",
]

PRIVATE_DIRS = [
    "docs/superpowers",
]

ROOT_README = ROOT / "README.md"
ROOT_README_EN = ROOT / "README.en.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
CONTRIBUTING_EN = ROOT / "CONTRIBUTING.en.md"
BANNER_IMAGE = ROOT / "figure/embodied-ai-lab-banner.png"
SHOWCASE_ASSETS = [
    "figure/showcase/perception-vision-panel.jpg",
    "figure/showcase/navigation-astar-maze.png",
    "figure/showcase/navigation-dijkstra-maze.png",
]
PERCEPTION_README = ROOT / "experiments/01-perception/README.md"
PERCEPTION_README_EN = ROOT / "experiments/01-perception/README.en.md"
SLAM_README = ROOT / "experiments/02-slam-navigation/README.md"
SLAM_README_EN = ROOT / "experiments/02-slam-navigation/README.en.md"
SHARED_README = ROOT / "shared/README.md"
SHARED_README_EN = ROOT / "shared/README.en.md"
MAP_README = ROOT / "docs/curriculum/legacy-to-direction-map.md"
MAP_README_EN = ROOT / "docs/curriculum/legacy-to-direction-map.en.md"
SLAM_LEVEL1_README = ROOT / "experiments/02-slam-navigation/level-1-python/README.md"
SLAM_LEVEL1_README_EN = ROOT / "experiments/02-slam-navigation/level-1-python/README.en.md"


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

    def test_root_homepage_uses_public_showcase_assets(self) -> None:
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")

        for asset in SHOWCASE_ASSETS:
            with self.subTest(asset=asset):
                self.assertTrue((ROOT / asset).is_file(), f"Missing showcase asset: {asset}")
                self.assertIn(f"./{asset}", zh)
                self.assertIn(f"./{asset}", en)

        self.assertNotIn("level-1-python/output/", zh)
        self.assertNotIn("level-1-python/output/", en)
        self.assertNotIn("perception-aruco-pose.jpg", zh)
        self.assertNotIn("perception-aruco-pose.jpg", en)

    def test_root_homepage_orders_showcase_outputs_by_learning_flow(self) -> None:
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")

        self.assertLess(zh.index("感知：视觉闭环"), zh.index("导航：A* 迷宫路径规划"))
        self.assertLess(zh.index("导航：A* 迷宫路径规划"), zh.index("导航：Dijkstra 搜索展开"))
        self.assertLess(
            zh.index("./figure/showcase/perception-vision-panel.jpg"),
            zh.index("./figure/showcase/navigation-astar-maze.png"),
        )
        self.assertLess(
            zh.index("./figure/showcase/navigation-astar-maze.png"),
            zh.index("./figure/showcase/navigation-dijkstra-maze.png"),
        )

        self.assertLess(en.index("Perception: Vision Loop"), en.index("Navigation: A* Maze Path Planning"))
        self.assertLess(
            en.index("Navigation: A* Maze Path Planning"),
            en.index("Navigation: Dijkstra Search Expansion"),
        )
        self.assertLess(
            en.index("./figure/showcase/perception-vision-panel.jpg"),
            en.index("./figure/showcase/navigation-astar-maze.png"),
        )
        self.assertLess(
            en.index("./figure/showcase/navigation-astar-maze.png"),
            en.index("./figure/showcase/navigation-dijkstra-maze.png"),
        )

    def test_direction_pages_have_language_links(self) -> None:
        zh = PERCEPTION_README.read_text(encoding="utf-8")
        en = PERCEPTION_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)

        zh = SLAM_README.read_text(encoding="utf-8")
        en = SLAM_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md](./README.en.md)", zh)
        self.assertIn("中文： [README.md](./README.md)", en)

    def test_slam_navigation_candidate_ids_are_consistent(self) -> None:
        zh = SLAM_README.read_text(encoding="utf-8")
        en = SLAM_README_EN.read_text(encoding="utf-8")
        level1_zh = SLAM_LEVEL1_README.read_text(encoding="utf-8")
        level1_en = SLAM_LEVEL1_README_EN.read_text(encoding="utf-8")
        map_zh = MAP_README.read_text(encoding="utf-8")
        map_en = MAP_README_EN.read_text(encoding="utf-8")

        expected = [
            "s01-grid-search",
            "s02-mcl-localization",
            "s03-ekf-slam",
        ]
        for doc in [zh, en, level1_zh, level1_en]:
            for lab_id in expected:
                with self.subTest(lab_id=lab_id):
                    self.assertIn(lab_id, doc)

        self.assertEqual(zh.count("s01-"), 1)
        self.assertEqual(en.count("s01-"), 1)
        self.assertIn("`archive/legacy-experiments/07-path-planning` | `experiments/02-slam-navigation/level-1-python/` | `s01-grid-search` (已迁移)", map_zh)
        self.assertIn("`archive/legacy-experiments/02-particle-filter-mcl` | `experiments/02-slam-navigation/` | `s02-mcl-localization`", map_zh)
        self.assertIn("`archive/legacy-experiments/03-ekf-slam` | `experiments/02-slam-navigation/` | `s03-ekf-slam`", map_zh)
        self.assertIn("`archive/legacy-experiments/07-path-planning` | `experiments/02-slam-navigation/level-1-python/` | `s01-grid-search` (migrated)", map_en)
        self.assertIn("`archive/legacy-experiments/02-particle-filter-mcl` | `experiments/02-slam-navigation/` | `s02-mcl-localization`", map_en)
        self.assertIn("`archive/legacy-experiments/03-ekf-slam` | `experiments/02-slam-navigation/` | `s03-ekf-slam`", map_en)

    def test_legacy_archive_map_paths_exist(self) -> None:
        """映射表中声明的归档路径必须真实存在，不能指向不存在的旧目录。"""
        for map_path in [MAP_README, MAP_README_EN]:
            with self.subTest(map_path=map_path.name):
                content = map_path.read_text(encoding="utf-8")
                missing = []
                for line in content.splitlines():
                    if not line.startswith("| `archive/legacy-experiments/"):
                        continue
                    archive_path = line.split("`", maxsplit=2)[1]
                    if not (ROOT / archive_path).exists():
                        missing.append(archive_path)
                self.assertEqual(missing, [])

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
            "experiments/01-perception/level-2-ros2-bridge",
            "experiments/02-slam-navigation/level-1-python",
            "experiments/02-slam-navigation/level-2-ros2-bridge",
            "experiments/02-slam-navigation/level-3-research",
            "experiments/03-motion-control/level-1-python",
            "experiments/03-motion-control/level-2-ros2-bridge",
            "experiments/03-motion-control/level-3-research",
            "experiments/04-rl-imitation/level-1-python",
            "experiments/04-rl-imitation/level-2-ros2-bridge",
            "experiments/04-rl-imitation/level-3-research",
            "experiments/05-world-model/level-1-python",
            "experiments/05-world-model/level-2-ros2-bridge",
            "experiments/05-world-model/level-3-research",
            "experiments/06-vision-language-navigation/level-1-python",
            "experiments/06-vision-language-navigation/level-2-ros2-bridge",
            "experiments/06-vision-language-navigation/level-3-research",
            "experiments/07-manipulation/level-1-python",
            "experiments/07-manipulation/level-2-ros2-bridge",
            "experiments/07-manipulation/level-3-research",
            "experiments/08-llm-robot/level-1-python",
            "experiments/08-llm-robot/level-2-ros2-bridge",
            "experiments/08-llm-robot/level-3-research",
            "experiments/09-sim-to-real/level-1-python",
            "experiments/09-sim-to-real/level-2-ros2-bridge",
            "experiments/09-sim-to-real/level-3-research",
            "experiments/10-vertical-apps/level-1-python",
            "experiments/10-vertical-apps/level-2-ros2-bridge",
            "experiments/10-vertical-apps/level-3-research",
        ]
        for rel in scaffold_paths:
            with self.subTest(path=rel):
                zh = (ROOT / rel / "README.md").read_text(encoding="utf-8")
                en = (ROOT / rel / "README.en.md").read_text(encoding="utf-8")
                self.assertIn("English: [README.en.md](./README.en.md)", zh)
                self.assertIn("中文： [README.md](./README.md)", en)


    def test_root_readme_has_python_first_positioning(self) -> None:
        """根 README 必须包含 Python-first, ROS2-ready 定位。"""
        zh = ROOT_README.read_text(encoding="utf-8")
        en = ROOT_README_EN.read_text(encoding="utf-8")
        self.assertIn("Python-first, ROS2-ready", zh)
        self.assertIn("Python-first, ROS2-ready", en)
        self.assertIn("Core Python Lab", zh)
        self.assertIn("Core Python Lab", en)

    def test_python_first_ros2_ready_docs_have_language_links(self) -> None:
        """Python-first, ROS2-ready 路线文档必须有双向语言链接。"""
        zh = PYTHON_FIRST_ROS2_READY.read_text(encoding="utf-8")
        en = PYTHON_FIRST_ROS2_READY_EN.read_text(encoding="utf-8")
        self.assertIn("English: [python-first-ros2-ready.en.md]", zh)
        self.assertIn("中文： [python-first-ros2-ready.md]", en)

    def test_curriculum_readme_has_language_links(self) -> None:
        """docs/curriculum/README 必须有双向语言链接。"""
        zh = CURRICULUM_README.read_text(encoding="utf-8")
        en = CURRICULUM_README_EN.read_text(encoding="utf-8")
        self.assertIn("English: [README.en.md]", zh)
        self.assertIn("中文： [README.md]", en)

    def test_archive_pages_have_language_links(self) -> None:
        """archive/ 归档页必须有双向语言链接。"""
        archive_pairs = [
            ROOT / "archive/README.md",
            ROOT / "archive/legacy-experiments/README.md",
        ]
        for zh_path in archive_pairs:
            en_path = zh_path.with_name("README.en.md")
            with self.subTest(path=zh_path):
                zh = zh_path.read_text(encoding="utf-8")
                en = en_path.read_text(encoding="utf-8")
                self.assertIn("English: [README.en.md](./README.en.md)", zh)
                self.assertIn("中文： [README.md](./README.md)", en)

    def test_level2_readmes_are_engineering_bridge(self) -> None:
        """Level 2 README 必须包含 Engineering Bridge 语义。"""
        directions = [
            "01-perception",
            "02-slam-navigation",
            "03-motion-control",
            "04-rl-imitation",
            "05-world-model",
            "06-vision-language-navigation",
            "07-manipulation",
            "08-llm-robot",
            "09-sim-to-real",
            "10-vertical-apps",
        ]
        for d in directions:
            with self.subTest(direction=d):
                zh = (ROOT / f"experiments/{d}/level-2-ros2-bridge/README.md").read_text(encoding="utf-8")
                en = (ROOT / f"experiments/{d}/level-2-ros2-bridge/README.en.md").read_text(encoding="utf-8")
                self.assertIn("工程桥接层", zh, f"{d} Level 2 zh missing '工程桥接层'")
                self.assertIn("engineering bridge layer", en.lower(), f"{d} Level 2 en missing 'engineering bridge layer'")

    def test_contributing_guide_uses_ros2_bridge_level2_label(self) -> None:
        """贡献指南目录树必须使用 ROS2 Bridge 语义描述 Level 2。"""
        zh = CONTRIBUTING.read_text(encoding="utf-8")
        en = CONTRIBUTING_EN.read_text(encoding="utf-8")

        self.assertIn("level-2-ros2-bridge/", zh)
        self.assertIn("level-2-ros2-bridge/", en)
        self.assertIn("Level 2: ROS2 Bridge", zh)
        self.assertIn("Level 2: ROS2 bridge", en)
        self.assertNotIn("Level 2: C++ 进阶", zh)
        self.assertNotIn("Level 2: C++ advanced", en)

    def test_level2_readmes_use_candidate_bridge_language(self) -> None:
        """Level 2 仍处于规划阶段时，桥接条目必须使用候选语气。"""
        directions = [
            "01-perception",
            "02-slam-navigation",
            "03-motion-control",
            "04-rl-imitation",
            "05-world-model",
            "06-vision-language-navigation",
            "07-manipulation",
            "08-llm-robot",
            "09-sim-to-real",
            "10-vertical-apps",
        ]
        for d in directions:
            with self.subTest(direction=d):
                zh = (ROOT / f"experiments/{d}/level-2-ros2-bridge/README.md").read_text(encoding="utf-8")
                en = (ROOT / f"experiments/{d}/level-2-ros2-bridge/README.en.md").read_text(encoding="utf-8")
                self.assertIn("候选桥接主题包括", zh)
                self.assertIn("Candidate bridge themes", en)

    def test_direction_readmes_have_level_semantics(self) -> None:
        """方向 README 必须包含 Level 语义说明。"""
        directions = [
            "01-perception",
            "02-slam-navigation",
            "03-motion-control",
            "04-rl-imitation",
            "05-world-model",
            "06-vision-language-navigation",
            "07-manipulation",
            "08-llm-robot",
            "09-sim-to-real",
            "10-vertical-apps",
        ]
        for d in directions:
            with self.subTest(direction=d):
                zh = (ROOT / f"experiments/{d}/README.md").read_text(encoding="utf-8")
                en = (ROOT / f"experiments/{d}/README.en.md").read_text(encoding="utf-8")
                self.assertIn("当前主产品", zh, f"{d} zh missing '当前主产品'")
                self.assertIn("Current main product", en, f"{d} en missing 'Current main product'")

    def test_experiments_top_level_contains_only_direction_directories(self) -> None:
        """experiments/ 顶层只保留 10 个新方向，旧实验必须归档。"""
        expected = [
            "01-perception",
            "02-slam-navigation",
            "03-motion-control",
            "04-rl-imitation",
            "05-world-model",
            "06-vision-language-navigation",
            "07-manipulation",
            "08-llm-robot",
            "09-sim-to-real",
            "10-vertical-apps",
        ]
        actual = sorted(path.name for path in EXPERIMENTS_DIR.iterdir() if path.is_dir())
        self.assertEqual(actual, expected)
        self.assertTrue(LEGACY_ARCHIVE_DIR.is_dir())

    def test_no_public_doc_references_old_level2_path(self) -> None:
        """公开文档中不应再出现旧的 level-2-cpp-or-mixed 路径。"""
        forbidden = "level-2-cpp-or-mixed"
        checked_files = [
            ROOT_README, ROOT_README_EN,
            PYTHON_FIRST_ROS2_READY, PYTHON_FIRST_ROS2_READY_EN,
            CURRICULUM_README, CURRICULUM_README_EN,
        ]
        for f in checked_files:
            with self.subTest(file=f.name):
                text = f.read_text(encoding="utf-8")
                self.assertNotIn(forbidden, text, f"{f.name} still contains '{forbidden}'")

        directions = [
            "01-perception", "02-slam-navigation", "03-motion-control",
            "04-rl-imitation", "05-world-model", "06-vision-language-navigation",
            "07-manipulation", "08-llm-robot", "09-sim-to-real", "10-vertical-apps",
        ]
        for d in directions:
            for level in ["level-1-python", "level-2-ros2-bridge", "level-3-research"]:
                for suffix in ["README.md", "README.en.md"]:
                    fpath = ROOT / f"experiments/{d}/{level}/{suffix}"
                    if fpath.is_file():
                        with self.subTest(file=f"{d}/{level}/{suffix}"):
                            text = fpath.read_text(encoding="utf-8")
                            self.assertNotIn(forbidden, text, f"{d}/{level}/{suffix} still contains '{forbidden}'")

    def test_level2_readmes_have_no_problematic_version_strings(self) -> None:
        """Level 2 README 中不应出现不严谨的版本绑定表述。"""
        forbidden_patterns = [
            "Ubuntu 22.04+",
            "Humble or newer",
            "Humble 或更新版本",
        ]
        directions = [
            "01-perception", "02-slam-navigation", "03-motion-control",
            "04-rl-imitation", "05-world-model", "06-vision-language-navigation",
            "07-manipulation", "08-llm-robot", "09-sim-to-real", "10-vertical-apps",
        ]
        for d in directions:
            for suffix in ["README.md", "README.en.md"]:
                fpath = ROOT / f"experiments/{d}/level-2-ros2-bridge/{suffix}"
                with self.subTest(file=f"{d}/level-2-ros2-bridge/{suffix}"):
                    text = fpath.read_text(encoding="utf-8")
                    for pat in forbidden_patterns:
                        self.assertNotIn(pat, text, f"{d}/level-2-ros2-bridge/{suffix} contains '{pat}'")

    def test_no_public_doc_references_python_312_plus(self) -> None:
        """公开文档中不应出现 Python 3.12+ 的硬编码版本门槛。"""
        forbidden = "Python 3.12+"
        docs_to_check = [
            PYTHON_FIRST_ROS2_READY, PYTHON_FIRST_ROS2_READY_EN,
        ]
        for f in docs_to_check:
            with self.subTest(file=f.name):
                text = f.read_text(encoding="utf-8")
                self.assertNotIn(forbidden, text, f"{f.name} still contains '{forbidden}'")

        directions = [
            "01-perception", "02-slam-navigation", "03-motion-control",
            "04-rl-imitation", "05-world-model", "06-vision-language-navigation",
            "07-manipulation", "08-llm-robot", "09-sim-to-real", "10-vertical-apps",
        ]
        for d in directions:
            fpath = ROOT / f"experiments/{d}/level-2-ros2-bridge/README.md"
            with self.subTest(file=f"{d}/level-2-ros2-bridge/README.md"):
                text = fpath.read_text(encoding="utf-8")
                self.assertNotIn(forbidden, text, f"{d}/level-2-ros2-bridge/README.md contains '{forbidden}'")

    def test_public_docs_do_not_expose_local_distro_or_conflate_python_and_conda(self) -> None:
        """公开文档不应暴露作者本地发行版，也不应把 Python 与 Conda 并列成同类概念。"""
        forbidden_patterns = [
            "Manjaro",
            "Python / Conda",
            "standard Python / Conda",
            "标准 Python / Conda",
        ]
        docs_to_check = [ROOT / path for path in PUBLIC_DOCS]
        docs_to_check.extend([PYTHON_FIRST_ROS2_READY, PYTHON_FIRST_ROS2_READY_EN])

        for fpath in sorted(set(docs_to_check)):
            with self.subTest(file=fpath.relative_to(ROOT).as_posix()):
                text = fpath.read_text(encoding="utf-8")
                for pattern in forbidden_patterns:
                    self.assertNotIn(pattern, text)


if __name__ == "__main__":
    unittest.main()
