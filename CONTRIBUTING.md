# 贡献指南

English: [CONTRIBUTING.en.md](./CONTRIBUTING.en.md)

感谢你对 Embodied AI Lab 的关注。本指南说明如何参与贡献。

## 贡献类型

| 类型 | 说明 |
|------|------|
| 新实验 | 为某个研究方向添加新的可运行实验 |
| 教程 | 为已有实验补充入门教程或进阶说明 |
| Bug 修复 | 修复代码错误、路径问题、文档不一致 |
| 双语镜像 | 为中文文档补充英文翻译 |
| 文档改进 | 改善说明、补充示例、修正错误 |

## 目录结构

```
experiments/
├── 01-perception/          # 研究方向
│   ├── level-1-python/     # Level 1: Python 入门
│   │   ├── scripts/        # 可运行脚本
│   │   ├── tutorials/      # 教程
│   │   ├── filters/        # 滤波器实现
│   │   └── output/         # 运行输出（不跟踪）
│   ├── level-2-ros2-bridge/  # Level 2: ROS2 Bridge
│   └── level-3-research/   # Level 3: 研究出口
├── 02-slam-navigation/
│   ...
shared/                     # 共享库（可视化、数学工具等）
```

## 提交流程

1. Fork 本仓库
2. 创建分支：`git checkout -b feature/your-feature`
3. 提交改动：`git commit -m "描述你的改动"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 双语规则

- 中文文档为主，英文文档为镜像
- 每个 `.md` 文件应有对应的 `.en.md` 文件
- 文件顶部必须有语言切换链接
- 不要求逐句直译，但要求信息等价

## 代码规范

- Python 脚本必须可独立运行：`python scripts/xxx.py`
- 使用 `argparse` 提供命令行参数
- 输出图片保存到 `output/` 目录
- 不跟踪生成的输出文件

## 提交信息格式

```
类型(范围): 简短描述

类型: feat / fix / docs / test / refactor
范围: perception / slam / control / shared / ...
```

示例：
- `feat(perception): 添加粒子滤波教程`
- `fix(shared): 修复可视化模块的中文乱码`
- `docs(slam): 补充 SLAM 导航方向的英文镜像`
