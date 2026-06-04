# s02-mcl-localization — Monte Carlo Localization

中文： [mcl_localization.md](./mcl_localization.md)

## What problem does this lab solve?

`s01-grid-search` assumes the robot already knows where it is on the map. A real mobile robot must first answer a more basic question:

> Where am I in a known map?

This lab uses Monte Carlo Localization (MCL) to show the loop. The robot moves in a known room with fixed beacons. Odometry drifts, range observations are noisy, and a particle filter keeps a cloud of candidate poses to correct the estimate.

## Quick Start

Run from `experiments/02-slam-navigation/level-1-python/`:

```bash
python scripts/mcl_localization.py
```

If the environment has no plotting support, start with the no-plot path:

```bash
python scripts/mcl_localization.py --no-plot
```

You can also adjust particle count and horizon:

```bash
python scripts/mcl_localization.py --particles 800 --steps 30
```

Default output directory:

```text
output/mcl_localization/
```

Main outputs:

| File | Meaning |
|------|---------|
| `mcl_localization.png` | True trajectory, odometry trajectory, MCL estimate, particle cloud, and error curve |
| `mcl_localization_metrics.json` | Final error, mean error, odometry error, effective particle count |
| `mcl_localization_samples.json` | Per-step true pose, estimated pose, and error |
| `mcl_localization_particles.json` | Final particle cloud |
| `mcl_localization_beacons.json` | Known beacon positions |

## What to inspect

Start with the terminal metrics:

```text
final_error=...m
mean_error=...m
final_odometry_error=...m
effective_particle_count=...
```

Focus on:

- `final_error`: final MCL localization error.
- `final_odometry_error`: accumulated error from odometry alone.
- `effective_particle_count`: whether particle weights collapsed too aggressively.

With the default parameters, MCL should produce a much smaller final error than odometry. This shows how observations pull a drifting pose estimate back toward the correct region.

## Core loop

Each step performs four operations:

1. `predict`: move every particle with control input and motion noise.
2. `observe`: measure distances from the robot to fixed beacons.
3. `weight`: particles whose predicted ranges match the observations get higher weights.
4. `resample`: sample particles by weight so more plausible poses survive.

This lab is not a full kidnapped-robot global localization problem. It uses a rough initial pose prior so beginners can first understand the particle-filter loop before moving to global localization, multi-modal beliefs, and complex maps.

## Parameters to try

```bash
python scripts/mcl_localization.py --particles 200 --no-plot
python scripts/mcl_localization.py --particles 1000 --no-plot
python scripts/mcl_localization.py --sensor-sigma 0.35 --no-plot
python scripts/mcl_localization.py --sensor-sigma 0.12 --no-plot
```

Suggested observations:

- Does localization become less stable with fewer particles?
- Does error increase when sensor noise gets larger?
- When the effective particle count is too low, does the cloud collapse into a wrong region?

## ROS2 connection

Level 1 keeps the algorithmic intuition: particles, weights, resampling, and error metrics. In Level 2, this can bridge into ROS2 concepts such as `map`, `odom`, `base_link`, and laser-based observation models.
