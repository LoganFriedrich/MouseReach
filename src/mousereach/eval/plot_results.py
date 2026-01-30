"""Matplotlib figure generation for MouseReach eval reports."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

from .collect_results import CorpusResults, OUTCOME_CLASSES


# Color palette
C_PRIMARY = "#2563EB"
C_GREEN = "#059669"
C_AMBER = "#D97706"
C_RED = "#DC2626"
C_PURPLE = "#7C3AED"
C_GRAY = "#6B7280"


def setup_style():
    """Configure matplotlib for clean scientific figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def _short_name(video_name: str) -> str:
    """Shorten video name for axis labels."""
    # e.g. 20250624_CNT0115_P2 -> CNT0115_P2
    parts = video_name.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        return "_".join(parts[1:])
    return video_name


def plot_seg_accuracy(results: CorpusResults, output_dir: Path) -> Path:
    """Per-video horizontal bar chart of segmentation boundary recall."""
    data = sorted(results.seg_results, key=lambda r: r.recall)
    if not data:
        return None

    names = [_short_name(r.video_name) for r in data]
    recalls = [r.recall * 100 for r in data]

    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))
    bars = ax.barh(names, recalls, color=C_PRIMARY, height=0.6)

    mean_val = np.mean(recalls)
    ax.axvline(mean_val, color=C_RED, linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.1f}%")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Boundary Recall (%)")
    ax.set_title("Segmentation Accuracy by Video")
    ax.legend(loc="lower right")

    # Annotate values on bars
    for bar, val in zip(bars, recalls):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                va="center", fontsize=9, color=C_GRAY)

    path = output_dir / "seg_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reach_metrics(results: CorpusResults, output_dir: Path) -> Path:
    """Per-video grouped bar chart of reach precision, recall, F1."""
    data = sorted(results.reach_results, key=lambda r: r.f1)
    if not data:
        return None

    names = [_short_name(r.video_name) for r in data]
    precisions = [r.precision * 100 for r in data]
    recalls = [r.recall * 100 for r in data]
    f1s = [r.f1 * 100 for r in data]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.8), 6))
    ax.bar(x - width, precisions, width, label="Precision", color=C_PRIMARY)
    ax.bar(x, recalls, width, label="Recall", color=C_GREEN)
    ax.bar(x + width, f1s, width, label="F1", color=C_PURPLE)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Reach Detection Metrics ({len(data)} videos with reach GT)")
    ax.legend()

    path = output_dir / "reach_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_outcome_accuracy(results: CorpusResults, output_dir: Path) -> Path:
    """Per-video horizontal bar chart of outcome accuracy, color-coded."""
    data = sorted(results.outcome_results, key=lambda r: r.accuracy)
    if not data:
        return None

    names = [_short_name(r.video_name) for r in data]
    accs = [r.accuracy * 100 for r in data]

    colors = []
    for a in accs:
        if a >= 90:
            colors.append(C_GREEN)
        elif a >= 70:
            colors.append(C_AMBER)
        else:
            colors.append(C_RED)

    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))
    bars = ax.barh(names, accs, color=colors, height=0.6)

    mean_val = np.mean(accs)
    ax.axvline(mean_val, color=C_GRAY, linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.1f}%")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Outcome Classification Accuracy by Video")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, accs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                va="center", fontsize=9, color=C_GRAY)

    path = output_dir / "outcome_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_boundary_error_histogram(results: CorpusResults, output_dir: Path) -> Path:
    """Histogram of signed boundary frame errors."""
    errors = results.all_boundary_errors
    if not errors:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=range(min(errors) - 1, max(errors) + 2), color=C_PRIMARY, edgecolor="white", alpha=0.85)
    ax.axvline(0, color=C_RED, linestyle="-", linewidth=1.5, label="Exact match")

    mean_e = np.mean(errors)
    std_e = np.std(errors)
    ax.axvline(mean_e, color=C_AMBER, linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_e:+.2f} (std: {std_e:.2f})")

    ax.set_xlabel("Frame Error (algo - GT)")
    ax.set_ylabel("Count")
    ax.set_title(f"Boundary Timing Errors (n={len(errors)})")
    ax.legend()

    path = output_dir / "boundary_errors.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_reach_timing_histogram(results: CorpusResults, output_dir: Path) -> Path:
    """Side-by-side histograms of reach start and end frame errors."""
    starts = results.all_start_errors
    ends = results.all_end_errors
    if not starts and not ends:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, errors, title, color in [
        (ax1, starts, "Start Frame Errors", C_PRIMARY),
        (ax2, ends, "End Frame Errors", C_PURPLE),
    ]:
        if not errors:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        bins = range(min(errors) - 1, max(errors) + 2)
        ax.hist(errors, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0, color=C_RED, linestyle="-", linewidth=1.5)

        mean_e = np.mean(errors)
        std_e = np.std(errors)
        ax.axvline(mean_e, color=C_AMBER, linestyle="--", linewidth=1.2,
                   label=f"Mean: {mean_e:+.2f} (std: {std_e:.2f})")

        exact = sum(1 for e in errors if e == 0)
        within_1 = sum(1 for e in errors if abs(e) <= 1)
        ax.set_xlabel("Frame Error (algo - GT)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} (n={len(errors)}, exact={exact}, ±1={within_1})")
        ax.legend(fontsize=9)

    fig.suptitle("Reach Timing Accuracy", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = output_dir / "reach_timing_errors.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(results: CorpusResults, output_dir: Path) -> Path:
    """Heatmap of outcome confusion matrix."""
    cm = results.confusion_matrix
    if not cm:
        return None

    # Build matrix using only classes that appear
    present_classes = []
    for cls in OUTCOME_CLASSES:
        row_sum = sum(cm.get(cls, {}).values())
        col_sum = sum(cm.get(other, {}).get(cls, 0) for other in OUTCOME_CLASSES)
        if row_sum > 0 or col_sum > 0:
            present_classes.append(cls)

    if not present_classes:
        return None

    n = len(present_classes)
    matrix = np.zeros((n, n), dtype=int)
    for i, gt_cls in enumerate(present_classes):
        for j, algo_cls in enumerate(present_classes):
            matrix[i, j] = cm.get(gt_cls, {}).get(algo_cls, 0)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    # Row-normalized for color
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    # Labels
    short_labels = [c.replace("displaced_", "disp_").replace("outside", "out") for c in present_classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Algorithm Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Outcome Confusion Matrix")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if norm_matrix[i, j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, label="Row-normalized proportion", shrink=0.8)

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_summary_dashboard(results: CorpusResults, output_dir: Path) -> Path:
    """Multi-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Aggregate metrics
    ax = axes[0, 0]
    metrics = {}
    if results.seg_results:
        metrics["Seg\nRecall"] = np.mean([r.recall * 100 for r in results.seg_results])
    if results.reach_results:
        metrics["Reach\nPrecision"] = np.mean([r.precision * 100 for r in results.reach_results])
        metrics["Reach\nRecall"] = np.mean([r.recall * 100 for r in results.reach_results])
        metrics["Reach\nF1"] = np.mean([r.f1 * 100 for r in results.reach_results])
    if results.outcome_results:
        metrics["Outcome\nAccuracy"] = np.mean([r.accuracy * 100 for r in results.outcome_results])

    if metrics:
        colors_list = [C_PRIMARY, C_GREEN, C_AMBER, C_PURPLE, C_RED]
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      color=colors_list[:len(metrics)], width=0.5)
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
    ax.set_title("Aggregate Metrics")

    # Panel 2: Boundary error histogram
    ax = axes[0, 1]
    if results.all_boundary_errors:
        errors = results.all_boundary_errors
        bins = range(min(errors) - 1, max(errors) + 2)
        ax.hist(errors, bins=bins, color=C_PRIMARY, edgecolor="white", alpha=0.85)
        ax.axvline(0, color=C_RED, linewidth=1.5)
        mean_e = np.mean(errors)
        ax.axvline(mean_e, color=C_AMBER, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Frame Error")
    ax.set_title(f"Boundary Errors (n={len(results.all_boundary_errors)})")

    # Panel 3: Per-video outcome accuracy
    ax = axes[1, 0]
    if results.outcome_results:
        data = sorted(results.outcome_results, key=lambda r: r.accuracy)
        names = [_short_name(r.video_name) for r in data]
        accs = [r.accuracy * 100 for r in data]
        colors = [C_GREEN if a >= 90 else C_AMBER if a >= 70 else C_RED for a in accs]
        ax.barh(names, accs, color=colors, height=0.6)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Accuracy (%)")
    ax.set_title("Outcome Accuracy by Video")

    # Panel 4: Confusion matrix (compact)
    ax = axes[1, 1]
    cm = results.confusion_matrix
    present_classes = [cls for cls in OUTCOME_CLASSES
                       if sum(cm.get(cls, {}).values()) > 0 or
                       sum(cm.get(o, {}).get(cls, 0) for o in OUTCOME_CLASSES) > 0]
    if present_classes:
        n = len(present_classes)
        matrix = np.zeros((n, n), dtype=int)
        for i, gt_cls in enumerate(present_classes):
            for j, algo_cls in enumerate(present_classes):
                matrix[i, j] = cm.get(gt_cls, {}).get(algo_cls, 0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = matrix / row_sums
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        short_labels = [c.replace("displaced_", "d_").replace("outside", "out") for c in present_classes]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    tc = "white" if norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            fontsize=10, fontweight="bold", color=tc)
    ax.set_title("Confusion Matrix")

    n_seg = len(results.seg_results)
    n_reach = len(results.reach_results)
    n_out = len(results.outcome_results)
    n_skip = len(results.skipped_reach)
    fig.suptitle(
        f"MouseReach Eval Report — {n_seg} seg, {n_reach} reach (+{n_skip} skipped), {n_out} outcome",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "summary_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_plots(results: CorpusResults, output_dir: Path) -> List[Path]:
    """Generate all figures, return list of saved file paths."""
    setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for plot_fn in [
        plot_seg_accuracy,
        plot_reach_metrics,
        plot_outcome_accuracy,
        plot_boundary_error_histogram,
        plot_reach_timing_histogram,
        plot_confusion_matrix,
        plot_summary_dashboard,
    ]:
        p = plot_fn(results, output_dir)
        if p is not None:
            paths.append(p)

    return paths
