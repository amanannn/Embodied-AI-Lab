"""Run the s03-ekf-slam lab.

Usage:
    python scripts/ekf_slam.py
    python scripts/ekf_slam.py --steps 30 --seed 8
    python scripts/ekf_slam.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from slam.ekf_slam import EkfSlamResult, run_ekf_slam_demo  # noqa: E402


DEFAULT_OUTPUT_DIR = BASE_DIR / "output/ekf_slam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="s03-ekf-slam: EKF-SLAM with range-bearing landmark observations",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed for repeatable output")
    parser.add_argument("--steps", type=int, default=24, help="number of simulated control steps")
    parser.add_argument("--range-sigma", type=float, default=0.12, help="range noise standard deviation")
    parser.add_argument("--bearing-sigma", type=float, default=0.045, help="bearing noise standard deviation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for JSON metrics and optional PNG outputs",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip PNG rendering")
    return parser.parse_args()


def write_outputs(result: EkfSlamResult, output_dir: Path, no_plot: bool) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization = "disabled" if no_plot else "png"
    if not no_plot:
        prepare_matplotlib_cache(output_dir)
        if not maybe_write_plot(result, output_dir / "ekf_slam.png"):
            visualization = "unavailable"

    metrics = dict(result.metrics)
    metrics["visualization"] = visualization
    (output_dir / "ekf_slam_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ekf_slam_samples.json").write_text(
        json.dumps(result.samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ekf_slam_landmarks.json").write_text(
        json.dumps(result.landmarks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ekf_slam_covariance_diagonal.json").write_text(
        json.dumps(result.covariance_diagonal, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return visualization


def prepare_matplotlib_cache(output_dir: Path) -> Path:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    return cache_dir


def maybe_write_plot(result: EkfSlamResult, output_path: Path) -> bool:
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
    errors = [float(sample["pose_error"]) for sample in result.samples]
    traces = [float(sample["covariance_trace"]) for sample in result.samples]

    fig, (ax_map, ax_metrics) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_map.set_facecolor("#eef6f2")
    ax_map.plot(true_x, true_y, color="#111827", linewidth=2.6, label="true pose")
    ax_map.plot(est_x, est_y, color="#2563eb", linewidth=2.2, label="EKF-SLAM estimate")
    ax_map.scatter(
        [float(item["true_x"]) for item in result.landmarks],
        [float(item["true_y"]) for item in result.landmarks],
        marker="*",
        s=220,
        color="#dc2626",
        edgecolors="#111827",
        label="true landmarks",
    )
    ax_map.scatter(
        [float(item["estimate_x"]) for item in result.landmarks],
        [float(item["estimate_y"]) for item in result.landmarks],
        marker="x",
        s=140,
        color="#16a34a",
        linewidths=3,
        label="estimated landmarks",
    )
    for item in result.landmarks:
        ax_map.plot(
            [float(item["true_x"]), float(item["estimate_x"])],
            [float(item["true_y"]), float(item["estimate_y"])],
            color="#64748b",
            linewidth=1.0,
            alpha=0.55,
        )
    ax_map.set_title("Joint pose and landmark estimate")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(color="#ffffff", linewidth=0.8)
    ax_map.legend(loc="upper left")

    ax_metrics.plot(errors, color="#2563eb", linewidth=2.2, label="pose error")
    ax_metrics.set_xlabel("step")
    ax_metrics.set_ylabel("pose error (m)")
    ax_metrics.grid(alpha=0.28)
    ax_trace = ax_metrics.twinx()
    ax_trace.plot(traces, color="#f97316", linewidth=1.8, linestyle="--", label="covariance trace")
    ax_trace.set_ylabel("covariance trace")
    ax_metrics.set_title("Pose error and joint uncertainty")
    lines, labels = ax_metrics.get_legend_handles_labels()
    trace_lines, trace_labels = ax_trace.get_legend_handles_labels()
    ax_metrics.legend(lines + trace_lines, labels + trace_labels, loc="upper right")

    fig.suptitle(
        "s03-ekf-slam: state vector contains robot pose and map",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def print_summary(result: EkfSlamResult) -> None:
    metrics = result.metrics
    print(
        "  final_pose_error={final_pose_error:.3f}m mean_pose_error={mean_pose_error:.3f}m "
        "landmark_rmse={landmark_rmse:.3f}m".format(**metrics)
    )
    print(
        "  landmarks={landmark_count} state_size={state_size} "
        "covariance_trace={final_covariance_trace:.3f}".format(**metrics)
    )


def main() -> None:
    args = parse_args()
    print("=" * 64)
    print("  s03-ekf-slam: Extended Kalman Filter SLAM")
    print("=" * 64)
    print(f"  steps={args.steps}, seed={args.seed}, range_sigma={args.range_sigma}")

    result = run_ekf_slam_demo(
        seed=args.seed,
        steps=args.steps,
        range_sigma=args.range_sigma,
        bearing_sigma=args.bearing_sigma,
    )
    print("\nMetrics:")
    print_summary(result)
    visualization = write_outputs(result, args.output_dir, args.no_plot)
    print(f"  visualization={visualization}")
    print(f"\n[OK] outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
