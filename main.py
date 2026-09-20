import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import POINTS_4, POINTS_6
from pipeline import run_pipeline, run_sensitivity_analysis


def parse_points(value):
    if not value:
        return None
    points = []
    for pair in value.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    return points


def build_parser():
    parser = argparse.ArgumentParser(description="KLT optical flow + PnP box tracking")
    parser.add_argument(
        "--video", type=Path, nargs="+", action="append",
        help="One or more input videos; repeat --video or provide multiple paths",
    )
    parser.add_argument(
        "--video-label", nargs="+", action="append",
        help="Optional label for each video, in the same order as --video",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--points-4", help="Initial points as x,y;x,y;x,y;x,y")
    parser.add_argument("--points-6", help="Initial points as x,y;x,y;x,y;x,y;x,y;x,y")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV windows")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run initial-click jitter sensitivity analysis for both point configurations",
    )
    return parser


def _video_inputs(args):
    video_groups = args.video or [[Path(__file__).with_name("box_video_data.avi")]]
    videos = [video for group in video_groups for video in group]
    label_groups = args.video_label or []
    labels = [label for group in label_groups for label in group]
    return videos, labels


def _safe_label(label):
    return "_".join(label.split())


def plot_mse_over_time(mse_4, mse_6, output_path):
    """Save the per-frame MSE comparison for one video sequence."""
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(mse_4, label=f"4 points (mean={np.mean(mse_4):.3f})")
    axis.plot(mse_6, label=f"6 points (mean={np.mean(mse_6):.3f})")
    axis.set_xlabel("Frame number")
    axis.set_ylabel("MSE (px2)")
    axis.set_title("Mean squared reprojection error - 4 pts vs 6 pts")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _point_label(point_indices):
    return ",".join(f"P{index + 1}" for index in point_indices)


def plot_summary_bar_chart(results, output_path):
    """Save grouped mean-MSE bars for all processed video sequences."""
    labels = [result["label"] for result in results]
    means_4 = [result["mean_4"] for result in results]
    means_6 = [result["mean_6"] for result in results]
    stds_4 = [result["std_4"] for result in results]
    stds_6 = [result["std_6"] for result in results]
    positions = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(max(8, 2.4 * len(labels)), 5))
    bars_4 = axis.bar(
        positions - width / 2, means_4, width,
        yerr=stds_4, capsize=4,
        label=f"4 points ({_point_label(POINTS_4)})",
        color="tab:blue",
    )
    bars_6 = axis.bar(
        positions + width / 2, means_6, width,
        yerr=stds_6, capsize=4,
        label=f"6 points (+{','.join(f'P{index + 1}' for index in POINTS_6[len(POINTS_4):])})",
        color="tab:red",
    )
    axis.bar_label(bars_4, fmt="%.3f", padding=3)
    axis.bar_label(bars_6, fmt="%.3f", padding=3)
    for bar, result in zip(bars_4, results):
        axis.text(
            bar.get_x() + bar.get_width() / 2, -0.13,
            f"n={result['frame_count_4']} frames",
            transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9,
        )
    for bar, result in zip(bars_6, results):
        axis.text(
            bar.get_x() + bar.get_width() / 2, -0.21,
            f"n={result['frame_count_6']} frames",
            transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9,
        )
    axis.axhline(2, color="tab:green", linestyle="--", label="Good AR threshold (2 px²)")
    axis.axhline(5, color="tab:orange", linestyle="--", label="Acceptable AR threshold (5 px²)")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean reprojection MSE (px²)")
    axis.set_title("Mean reprojection error by test sequence")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.subplots_adjust(bottom=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    args = build_parser().parse_args()
    video_paths, sequence_labels = _video_inputs(args)
    if sequence_labels and len(sequence_labels) != len(video_paths):
        raise ValueError(
            f"Broj labela ({len(sequence_labels)}) mora odgovarati broju videa ({len(video_paths)})"
        )
    if not sequence_labels:
        sequence_labels = [video.stem for video in video_paths]
    output_dir = args.output_dir.expanduser().resolve()
    points_4 = parse_points(args.points_4)
    points_6 = parse_points(args.points_6)
    if args.sensitivity and (points_4 is None or points_6 is None):
        raise ValueError("--sensitivity zahteva --points-4 i --points-6 bazne klikove")
    results = []
    for video_path, sequence_label in zip(video_paths, sequence_labels):
        video_path = video_path.expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        safe_label = _safe_label(sequence_label)
        mse_4, frame_count_4 = run_pipeline(
            video_path, POINTS_4, f"{safe_label}_4pts", output_dir, initial_points=points_4,
            display=not args.no_display, max_frames=args.max_frames,
        )
        mse_6, frame_count_6 = run_pipeline(
            video_path, POINTS_6, f"{safe_label}_6pts", output_dir, debug=True, initial_points=points_6,
            display=not args.no_display, max_frames=args.max_frames,
        )
        mean_4, std_4 = np.mean(mse_4), np.std(mse_4)
        mean_6, std_6 = np.mean(mse_6), np.std(mse_6)
        print(f"{sequence_label} - 4 points: mean={mean_4:.3f}, std={std_4:.3f}, n={frame_count_4} frames")
        print(f"{sequence_label} - 6 points: mean={mean_6:.3f}, std={std_6:.3f}, n={frame_count_6} frames")
        results.append({
            "label": sequence_label,
            "mean_4": mean_4,
            "mean_6": mean_6,
            "std_4": std_4,
            "std_6": std_6,
            "frame_count_4": frame_count_4,
            "frame_count_6": frame_count_6,
        })
        line_name = "eqm_comparison.png" if len(video_paths) == 1 else f"eqm_comparison_{safe_label}.png"
        plot_mse_over_time(mse_4, mse_6, output_dir / line_name)

        if args.sensitivity:
            sensitivity_4 = run_sensitivity_analysis(
                video_path, POINTS_4, points_4, f"{safe_label}_4pts"
            )
            sensitivity_6 = run_sensitivity_analysis(
                video_path, POINTS_6, points_6, f"{safe_label}_6pts"
            )
            print(
                f"{sequence_label} - sensitivity 4 points: "
                f"mean={np.mean(sensitivity_4):.3f} +/- std={np.std(sensitivity_4):.3f}"
            )
            print(
                f"{sequence_label} - sensitivity 6 points: "
                f"mean={np.mean(sensitivity_6):.3f} +/- std={np.std(sensitivity_6):.3f}"
            )

    plot_summary_bar_chart(results, output_dir / "eqm_summary.png")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()