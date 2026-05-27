# Shared Modules

中文： [README.md](./README.md)

## Purpose

This directory is reserved for infrastructure that is genuinely reusable across multiple directions and experiments. Its job is not to make folders look tidy; its job is to pull stable, reusable capabilities out of single experiments once they deserve to become shared assets.

## Inclusion Rules

### what belongs in `shared/`

Code or documentation should move into `shared/` only when:

- it is used by at least two directions
- its interface is stable enough to serve as a dependency
- it can be understood without relying on one experiment's teaching storyline

### what should stay in direction experiments

The following should not be moved into `shared/` just for cosmetic cleanup:

- single-experiment scripts
- private interfaces that are still changing rapidly
- conclusions, reward shaping, tuning logic, or experiment-specific patterns that only make sense inside one direction

## Stability and Reuse Expectation

Anything placed in `shared/` should be treated as a cross-direction contract. Its naming, file layout, and behavioral stability should be stronger than local experiment helper code.

## Current Modules

- `sim2d/`: lightweight simulation foundations shared across navigation, control, and sim-to-real work
- `viz/`: reusable plotting, animation, and presentation helpers
- `math/`: stable state-estimation, geometry, and utility math modules
- `ros2_interfaces/`: ROS2-facing boundaries used by mixed-language experiments
- `datasets/`: reusable data, metadata, and download guidance across directions
