# Contributing Guide

中文： [CONTRIBUTING.md](./CONTRIBUTING.md)

Thank you for your interest in Embodied AI Lab. This guide explains how to contribute.

## Contribution Types

| Type | Description |
|------|-------------|
| New experiment | Add a runnable experiment for a research direction |
| Tutorial | Add introductory or advanced documentation for existing experiments |
| Bug fix | Fix code errors, path issues, or documentation inconsistencies |
| Bilingual mirror | Add English translations for Chinese documentation |
| Documentation | Improve explanations, add examples, or correct errors |

## Directory Structure

```
experiments/
├── 01-perception/          # Research direction
│   ├── level-1-python/     # Level 1: Python introduction
│   │   ├── scripts/        # Runnable scripts
│   │   ├── tutorials/      # Tutorials
│   │   ├── filters/        # Filter implementations
│   │   └── output/         # Runtime output (not tracked)
│   ├── level-2-ros2-bridge/  # Level 2: ROS2 bridge
│   └── level-3-research/   # Level 3: Research exit
├── 02-slam-navigation/
│   ...
shared/                     # Shared libraries (visualization, math tools, etc.)
```

## Submission Process

1. Fork this repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Describe your changes"`
4. Push branch: `git push origin feature/your-feature`
5. Create a Pull Request

## Bilingual Rules

- Chinese documentation is primary, English documentation is the mirror
- Every `.md` file should have a corresponding `.en.md` file
- Each file must have a language switch link at the top
- Literal translation is not required, but information equivalence is

## Code Standards

- Python scripts must be independently runnable: `python scripts/xxx.py`
- Use `argparse` for command-line arguments
- Save output images to the `output/` directory
- Generated output files are not tracked

## Commit Message Format

```
type(scope): short description

Type: feat / fix / docs / test / refactor
Scope: perception / slam / control / shared / ...
```

Examples:
- `feat(perception): Add particle filter tutorial`
- `fix(shared): Fix Chinese encoding in visualization module`
- `docs(slam): Add English mirror for SLAM navigation direction`
