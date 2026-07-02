# Xidian University · Embodied Intelligence Micro-Major Notes

中文： [README.md](./README.md)

## Background

Xidian University offers an embodied intelligence micro-major for undergraduate students. This directory holds course notes, experiment records, and study summaries from that program.

This is one of the earliest systematic embodied AI courses for undergraduates in China. Recording these notes serves several purposes:

- Preserve first-hand learning paths and troubleshooting experience
- Provide note templates for future learners
- Cross-reference with Embodied AI Lab experiments

## Usage

1. Create `.md` files in this directory by course or topic
2. After writing, Claude helps polish formatting and add cross-references
3. Completed notes can optionally be synced to the public repository

## Directory Structure

```text
docs/xidian-micro-major/
├── README.md
├── README.en.md
├── CLAUDE_WORKFLOW.md
├── Embodied Perception Basics: ML & DL/        ← Module 1 (Theory)
│   ├── Embodied Perception Fundamentals: Machine Vision & DL.md
│   ├── What_is_neural_network.md
│   ├── From_traditional_image_processing_to_deep_learning.md
│   ├── What_is_Convolution_network.md
│   ├── Commonly_used_convolution_operators.md
│   └── *.en.md                                  ← English mirrors
├── Advanced Cognition: Multimodal LLMs & Semantic Understanding/  ← Module 2 (Theory)
│   ├── How_to_Build_and_Train_a_Neural_Network_p1.md
│   ├── How_to_Build_and_Train_a_Neural_Network_p2.md
│   └── *.en.md                                  ← English mirrors
```

> Practice modules have been moved to [`labs/`](../../labs/): `python-mnist/` and `image-super-res/`.

## Related Documentation

- [Embodied AI Lab Course Panorama](../../README.md)
- [Python-first, ROS2-ready Path](../curriculum/python-first-ros2-ready.md)
