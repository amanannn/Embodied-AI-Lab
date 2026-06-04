"""Run the s02-mcl-localization lab.

Usage:
    python scripts/mcl_localization.py
    python scripts/mcl_localization.py --particles 800 --steps 30
    python scripts/mcl_localization.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from localization.mcl import MclResult, run_mcl_demo  # noqa: E402


DEFAULT_OUTPUT_DIR = BASE_DIR / "output/mcl_localization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="s02-mcl-localization: particle-filter localization on a known beacon map",
    )
    parser.add_argument("--seed", type=int, default=11, help="random seed for repeatable teaching output")
    parser.add_argument("--particles", type=int, default=600, help="number of particles")
    parser.add_argument("--steps", type=int, default=24, help="number of simulated control steps")
    parser.add_argument("--sensor-sigma", type=float, default=0.22, help="range sensor noise standard deviation")
    parser.add_argument("--motion-sigma", type=float, default=0.035, help="particle motion noise standard deviation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for JSON metrics and optional PNG outputs",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip PNG rendering")
    return parser.parse_args()


def write_outputs(result: MclResult, output_dir: Path, no_plot: bool) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization = "disabled" if no_plot else "png"
    if not no_plot:
        prepare_matplotlib_cache(output_dir)
        if not maybe_write_plot(result, output_dir / "mcl_localization.png"):
            visualization = "unavailable"

    metrics = dict(result.metrics)
    metrics["visualization"] = visualization
    (output_dir / "mcl_localization_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "mcl_localization_samples.json").write_text(
        json.dumps(result.samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "mcl_localization_particles.json").write_text(
        json.dumps(result.particles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "mcl_localization_beacons.json").write_text(
        json.dumps(result.beacons, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return visualization


def prepare_matplotlib_cache(output_dir: Path) -> Path:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    return cache_dir


def maybe_write_plot(result: MclResult, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed; skip PNG rendering")
        return False

    true_x = [float(sample["true_x"]) for sample in result.samples]
    true_y = [float(sample["true_y"]) for sample in result.samples]
    est_x = [float(sample["estimate_x"]) for sample in result.samples]
    est_y = [float(sample["estimate_y"]) for sample in result.samples]
    odom_x = [float(sample["odometry_x"]) for sample in result.samples]
    odom_y = [float(sample["odometry_y"]) for sample in result.samples]
    errors = [float(sample["error"]) for sample in result.samples]
    odom_errors = [float(sample["odometry_error"]) for sample in result.samples]

    fig, (ax_map, ax_error) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_map.set_facecolor("#f7efe1")
    ax_map.scatter(
        [float(p["x"]) for p in result.particles],
        [float(p["y"]) for p in result.particles],
        s=8,
        alpha=0.18,
        color="#2563eb",
        label="particles",
    )
    ax_map.scatter(
        [float(b["x"]) for b in result.beacons],
        [float(b["y"]) for b in result.beacons],
        marker="*",
        s=180,
        color="#d97706",
        edgecolors="#111827",
        label="beacons",
    )
    ax_map.plot(true_x, true_y, color="#111827", linewidth=2.6, label="true pose")
    ax_map.plot(est_x, est_y, color="#16a34a", linewidth=2.2, label="MCL estimate")
    ax_map.plot(odom_x, odom_y, color="#ef4444", linestyle="--", linewidth=1.8, label="odometry")
    ax_map.set_title("Known-map localization with particles")
    ax_map.set_xlim(0, 8)
    ax_map.set_ylim(0, 6)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(color="#ffffff", linewidth=0.8)
    ax_map.legend(loc="upper right")

    ax_error.plot(errors, color="#16a34a", linewidth=2.2, label="MCL error")
    ax_error.plot(odom_errors, color="#ef4444", linewidth=2.0, linestyle="--", label="odometry error")
    ax_error.set_title("Localization error")
    ax_error.set_xlabel("step")
    ax_error.set_ylabel("meters")
    ax_error.grid(alpha=0.28)
    ax_error.legend(loc="upper left")

    fig.suptitle(
        "s02-mcl-localization: particles correct odometry drift",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def print_summary(result: MclResult) -> None:
    metrics = result.metrics
    print(
        "  final_error={final_error:.3f}m mean_error={mean_error:.3f}m "
        "final_odometry_error={final_odometry_error:.3f}m particles={particle_count}".format(
            **metrics
        )
    )
    print(
        "  effective_particle_count={effective_particle_count:.1f} "
        "sensor_sigma={sensor_sigma:.2f}".format(**metrics)
    )


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("  s02-mcl-localization: Monte Carlo Localization")
    print("=" * 64)
    print(f"  particles={args.particles}, steps={args.steps}, seed={args.seed}")

    result = run_mcl_demo(
        seed=args.seed,
        particle_count=args.particles,
        steps=args.steps,
        sensor_sigma=args.sensor_sigma,
        motion_sigma=args.motion_sigma,
    )
    print("\nMetrics:")
    print_summary(result)
    visualization = write_outputs(result, args.output_dir, args.no_plot)
    print(f"  visualization={visualization}")
    print(f"\n[OK] outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
