# Sensor Fundamentals and Noise Modeling — Understanding Sensor Data from Scratch

中文： [sensors.md](./sensors.md)

> Pair with `noise.py`, `sensors/`, `fusion.py`, and `scripts/`. Assumes only basic Python — no signal processing background.

---

## Overview

This tutorial covers the fundamentals of sensor data: what noise looks like, where it comes from, and how to fuse multiple imperfect sensors.

## Contents

1. **Why sensor readings ≠ true values** — all sensors have errors (random fluctuation, systematic bias, slow drift, occasional outliers)
2. **Four noise types** — Gaussian white noise, bias/drift, random walk, outlier/burst noise
3. **Four sensor types** — GPS, LiDAR, IMU, Odometry (each with different noise characteristics)
4. **Multi-sensor fusion** — weighted average combining multiple imperfect measurements
5. **Code walkthrough** — module structure and data flow
6. **Running experiments** — environment setup and output files
7. **What to learn next** — relationship with kalman.md

## Quick Start

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt
python scripts/sensors.py   # Generate sensor data with noise visualization
python scripts/fusion.py    # Multi-sensor fusion experiment
python scripts/noise.py     # Noise type comparison
```

## The Four Noise Types

| Type | Characteristic | Example |
|------|---------------|---------|
| Gaussian white noise | Random fluctuation around true value | GPS positioning jitter |
| Bias/drift | Systematic offset, always high or low | IMU accelerometer zero bias |
| Random walk | Slow cumulative error over time | Gyroscope integration drift |
| Outlier/burst | Occasional large deviations | LiDAR multi-path reflection |

## The Four Sensor Types

| Sensor | Measures | Noise Characteristic |
|--------|---------|---------------------|
| GPS | Position (x, y) | Moderate noise (~3m), occasional outliers |
| LiDAR | Range + bearing | Low noise (~0.05m), affected by environment |
| IMU | Acceleration + angular velocity | Low noise but bias drift over time |
| Odometry | Relative displacement | Accumulates error over distance |

## Multi-Sensor Fusion

The weighted average fusion is a special case of the Kalman gain:

```
fused = w₁·z₁ + w₂·z₂ + ... + wₙ·zₙ
where wᵢ ∝ 1/σᵢ²  (inverse variance weighting)
```

When there's no motion model (F=I, Q=0), the Kalman filter reduces to this static weighted average.

## Relationship with kalman.md

| Tutorial | Core Question | Key Concepts |
|----------|--------------|--------------|
| `sensors.md` (this) | What does sensor data look like? Where does noise come from? | Gaussian noise, bias, drift, multi-sensor weighted fusion |
| `kalman.md` | How to use motion model + sensor data for optimal estimation? | State space, predict-update cycle, uncertainty propagation |

**Learning path**: Read this tutorial first (understand the raw materials), then `kalman.md` (understand the processing). Run `scripts/sensors.py` to inspect the sensor data layout, then use a **preprocessed single-stream Cartesian observation CSV** with `scripts/kf.py --data`.
