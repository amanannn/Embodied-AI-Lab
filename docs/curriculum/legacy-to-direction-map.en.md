# Legacy Experiment to Direction Map

中文： [legacy-to-direction-map.md](./legacy-to-direction-map.md)

## Migration Rules

- Phase 1 does not move code directly out of the legacy experiment directories. It first establishes the new direction pages and navigation structure.
- These mappings exist to preserve discoverability during migration, not to claim that the codebase has already been fully rearranged.
- Legacy experiment names should remain searchable and understandable until the corresponding direction homes contain real migrated assets.

## What Planned Identity Means

`Planned identity` refers to the future experiment name that each legacy lab should map to once the direction-first structure is fully in place. It is a naming anchor for the future course structure, not a claim that the renamed lab already exists today.

## Mapping

| Legacy experiment | New direction home | Planned identity |
|---|---|---|
| `01-kalman-filter` | `experiments/01-perception/` | `p01-state-estimation-kalman` |
| `02-particle-filter-mcl` | `experiments/02-slam-navigation/` | `s01-mcl-localization` |
| `03-ekf-slam` | `experiments/02-slam-navigation/` | `s02-ekf-slam` |
| `04-robot-sim` | `experiments/09-sim-to-real/` | `r01-sim2d-foundation` |
| `05-pid-control` | `experiments/03-motion-control/` | `c01-pid-control-lab` |
| `06-robot-kinematics` | `experiments/07-manipulation/` | `m01-robot-kinematics` |
| `07-path-planning` | `experiments/02-slam-navigation/` | `s03-path-planning` |
| `08-trajectory-optimization` | `experiments/03-motion-control/` | `c02-trajectory-optimization` |
| `09-q-learning` | `experiments/04-rl-imitation/` | `l01-q-learning` |
| `10-deep-rl` | `experiments/04-rl-imitation/` | `l02-deep-rl` |
| `11-imitation-learning` | `experiments/04-rl-imitation/` | `l03-imitation-learning` |
| `12-vla-grounding` | `experiments/01-perception/` | `p02-vlm-grounding` |
| `g1-grasping` | `experiments/07-manipulation/` | `m03-grasping-and-placing` |
| `g2-dynamics` | `experiments/07-manipulation/` | `m02-manipulator-dynamics-control` |
| `S1-fullstack-nav` | `experiments/10-vertical-apps/` | `x01-autonomous-inspection-capstone` |
