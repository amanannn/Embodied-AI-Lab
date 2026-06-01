# Kalman Filter from Zero to Mastery — Perception Fundamentals

中文： [kalman.md](./kalman.md)

> Pair with code in `filters/`, `scripts/`, and `utils/`. Assumes only basic Python — no probability or control theory background.

---

## Overview

This tutorial covers the Kalman filter family for state estimation in embodied AI systems. It progresses from intuition to implementation, covering four filter variants.

## Contents

1. **Why sensors can't be trusted directly** — all sensors have noise
2. **Intuition** — use uncertainty to distribute trust automatically
3. **Math prerequisites** — Gaussian distribution, covariance matrix, matrix multiplication
4. **State space model** — describing motion with math (state vector, motion model, observation model, noise model)
5. **Five Kalman filter equations** — predict (2 equations) + update (3 equations)
6. **Code walkthrough** — how equations become Python code
7. **Running demos & parameter tuning** — environment setup, output files, hands-on experiments, `kf_tuning.py` five-group comparison, engineering parameter selection guide
8. **FAQ** — KF vs averaging, P divergence, why "filter", KF vs deep learning
9. **EKF** — when linearity isn't enough (Jacobian, angle wrapping pitfall)
10. **UKF** — no derivatives needed (sigma points, second-order accuracy)
11. **Particle filter** — arbitrary distributions (predict-weight-resample loop, MCL)
12. **Four families compared** — summary table and decision guide
13. **What to learn next** — learning path and relationship with sensors.md

## Quick Start

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt
python scripts/all.py    # Four filters comparison (recommended)
python scripts/kf.py     # Linear KF only
python scripts/kf_tuning.py  # Parameter tuning experiment
python scripts/fusion.py     # Generate output/fused_data.csv for external-data KF
python scripts/kf.py --data output/fused_data.csv
```

## The Five Equations (Cheat Sheet)

```
Predict:
  ① x = F·x              "Use model to guess next state"
  ② P = F·P·Fᵀ + Q      "Uncertainty increases after prediction"

Update:
  ③ K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹   "Compute weight: trust model or sensor?"
  ④ x = x + K·(z - H·x)         "Correct state with sensor data"
  ⑤ P = (I - K·H)·P             "Uncertainty decreases after correction"
```

## Filter Decision Guide

```
Is your system linear?
  ├─ Yes → Use KF (simplest, optimal)
  └─ No → Can you differentiate the observation function?
            ├─ Yes, weak nonlinearity → Use EKF
            ├─ Yes, strong nonlinearity → Use UKF
            └─ No → Is the distribution Gaussian?
                      ├─ Yes → Use UKF
                      └─ No (multi-modal) → Use PF
```

**Practical advice**: Start with UKF. No derivatives, good accuracy, acceptable compute. Only use PF when you need multi-modal distributions.

## Relationship with sensors.md

| Tutorial | Core Question | Key Concepts |
|----------|--------------|--------------|
| `sensors.md` | What does sensor data look like? | Gaussian noise, bias, drift, multi-sensor weighted fusion |
| `kalman.md` (this) | How to use motion model + sensor data for optimal estimation? | State space, predict-update cycle, uncertainty propagation |

Read `sensors.md` first (the raw materials), then this tutorial (the processing). The weighted fusion in `sensors.md` is a special case of the Kalman gain — when there's no motion model (F=I, Q=0), KF reduces to static weighted averaging.

For the current repository version, `kf.py --data` expects a **single Cartesian observation stream** such as `output/fused_data.csv` from `scripts/fusion.py`. The raw mixed `sensor_data.csv` export is meant for inspection and preprocessing, not direct use by `kf.py`.

## Key Takeaways

> **Kalman filter's core is one sentence: predict + correct, with weights automatically determined by comparing uncertainties.**
>
> It first flew in the Apollo lunar module's navigation system. Today it runs inside every autonomous car, industrial robot, and drone.
