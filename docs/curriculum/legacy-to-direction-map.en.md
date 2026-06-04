# Legacy Archive to Direction Map

中文： [legacy-to-direction-map.md](./legacy-to-direction-map.md)

## Migration Rules

- Legacy experiments are archived under `archive/legacy-experiments/` to avoid mixing old experiment numbers with the 10 new direction entries under top-level `experiments/`.
- These mappings preserve historical sources and new direction ownership during migration; they do not claim that all code has already been fully rearranged.
- Current learning starts from `experiments/`. The archive is mainly for historical reference and future migration work.

## What Planned Identity Means

`Planned identity` refers to the future experiment name that each legacy lab should map to once the direction-first structure is fully in place. It is a naming anchor for the future course structure, not a claim that the renamed lab already exists today.

## Mapping

| Legacy archive path | New direction home | Planned identity |
|---|---|---|
| ~~`01-kalman-filter`~~ | `experiments/01-perception/level-1-python/` | `p01-state-estimation-kalman` (migrated) |
| `archive/legacy-experiments/02-particle-filter-mcl` | `experiments/02-slam-navigation/level-1-python/` | `s02-mcl-localization` (migrated) |
| `archive/legacy-experiments/03-ekf-slam` | `experiments/02-slam-navigation/level-1-python/` | `s03-ekf-slam` (migrated) |
| `archive/legacy-experiments/04-robot-sim` | `experiments/09-sim-to-real/` | `r01-sim2d-foundation` |
| `archive/legacy-experiments/05-pid-control` | `experiments/03-motion-control/` | `c01-pid-control-lab` |
| `archive/legacy-experiments/06-robot-kinematics` | `experiments/07-manipulation/` | `m01-robot-kinematics` |
| `archive/legacy-experiments/07-path-planning` | `experiments/02-slam-navigation/level-1-python/` | `s01-grid-search` (migrated) |
| `archive/legacy-experiments/08-trajectory-optimization` | `experiments/03-motion-control/` | `c02-trajectory-optimization` |
| `archive/legacy-experiments/09-q-learning` | `experiments/04-rl-imitation/` | `l01-q-learning` |
| `archive/legacy-experiments/10-deep-rl` | `experiments/04-rl-imitation/` | `l02-deep-rl` |
| `archive/legacy-experiments/11-imitation-learning` | `experiments/04-rl-imitation/` | `l03-imitation-learning` |
| `archive/legacy-experiments/12-vla-grounding` | `experiments/01-perception/` | `p02-vlm-grounding` |
| `archive/legacy-experiments/g1-grasping` | `experiments/07-manipulation/` | `m03-grasping-and-placing` |
| `archive/legacy-experiments/g2-dynamics` | `experiments/07-manipulation/` | `m02-manipulator-dynamics-control` |
| Missing legacy record: `S1-fullstack-nav` | `experiments/10-vertical-apps/` | `x01-autonomous-inspection-capstone` |
