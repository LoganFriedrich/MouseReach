"""
Reach detection grapher (algo 2). Reads reach_detection_scalars.json.
Writes:
  <snapshot>/figures/reach_detection_summary.png
  <snapshot>/figures/reach_detection_summary_legend.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mousereach.improvement.lib.inputs import load_snapshot_paths, read_scalars


def graph(snapshot_dir: Path) -> Path:
    paths = load_snapshot_paths(snapshot_dir)
    s = read_scalars(paths.metrics_dir, "reach_detection_scalars.json")
    if not s:
        raise FileNotFoundError(f"reach_detection_scalars.json missing")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    n_tp, n_fp, n_fn, n_tri = s["n_tp"], s["n_fp"], s["n_fn"], s.get("triage_count", 0)

    # Counts bar
    ax = axes[0]
    cats = ["TP", "FP", "FN", "triaged"]
    vals = [n_tp, n_fp, n_fn, n_tri]
    colors = ["#43A047", "#E53935", "#FB8C00", "#FFEB3B"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Reach detection counts", fontweight="bold")
    ax.set_ylabel("Count")
    n_gt = n_tp + n_fn; n_algo = n_tp + n_fp
    ax.text(0.5, -0.18, f"GT reaches: {n_gt} | Algo reaches: {n_algo}",
            ha="center", va="top", transform=ax.transAxes, fontsize=9, color="#555555")

    # Start delta
    sds = [m["start_delta"] for m in s.get("matches", []) if m.get("status") == "tp"]
    ax = axes[1]
    if sds:
        bins = np.arange(-3.5, 3.5 + 1, 1)
        ax.hist(sds, bins=bins, color="#5C6BC0", edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.text(0.02, 0.95, f"n={len(sds)}\nmed={s['start_delta_median']}  |med|={s['start_delta_abs_median']}\n"
                f"range=[{s['start_delta_min']},{s['start_delta_max']}]",
                transform=ax.transAxes, fontsize=9, va="top", family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    ax.set_xlabel("Start delta (algo - gt) [frames]")
    ax.set_title("Start delta on TPs", fontweight="bold")

    # Span delta
    spds = [m["span_delta"] for m in s.get("matches", []) if m.get("status") == "tp"]
    ax = axes[2]
    if spds:
        lo, hi = int(min(spds)) - 1, int(max(spds)) + 1
        bins = np.arange(lo - 0.5, hi + 0.5 + 1, 1)
        ax.hist(spds, bins=bins, color="#26A69A", edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.text(0.02, 0.95, f"n={len(spds)}\nmed={s['span_delta_median']}  |med|={s['span_delta_abs_median']}\n"
                f"range=[{s['span_delta_min']},{s['span_delta_max']}]",
                transform=ax.transAxes, fontsize=9, va="top", family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    ax.set_xlabel("Span delta (algo - gt) [frames]")
    ax.set_title("Span delta on TPs", fontweight="bold")

    plt.tight_layout()
    out = paths.figures_dir / "reach_detection_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)

    legend = f"""# Reach detection evaluator (algo 2)

TP iff |algo.start - gt.start| <= 2 AND span tolerance
(|delta| <= max(0.5*gt_span, 5)). NOT P/R/F1; counts + delta dists.

n_tp={n_tp}  n_fp={n_fp}  n_fn={n_fn}  triage={n_tri}
start delta: med={s.get('start_delta_median')}  |med|={s.get('start_delta_abs_median')}  range=[{s.get('start_delta_min')},{s.get('start_delta_max')}]
span delta:  med={s.get('span_delta_median')}  |med|={s.get('span_delta_abs_median')}  range=[{s.get('span_delta_min')},{s.get('span_delta_max')}]
"""
    (paths.figures_dir / "reach_detection_summary_legend.md").write_text(legend, encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.reach_detection.graph <snapshot_dir>")
        sys.exit(1)
    out = graph(Path(sys.argv[1]))
    print(f"Wrote: {out}")
