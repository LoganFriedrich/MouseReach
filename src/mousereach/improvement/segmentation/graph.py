"""
Segmentation grapher (algo 1).

Reads <snapshot>/metrics/segmentation_scalars.json. Writes:
  <snapshot>/figures/segmentation_summary.png
  <snapshot>/figures/segmentation_summary_legend.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mousereach.improvement.lib.inputs import load_snapshot_paths, read_scalars


def graph(snapshot_dir: Path) -> Path:
    paths = load_snapshot_paths(snapshot_dir)
    s = read_scalars(paths.metrics_dir, "segmentation_scalars.json")
    if not s:
        raise FileNotFoundError(
            f"{paths.metrics_dir / 'segmentation_scalars.json'} missing -- run analyze first")

    deltas = [m["delta"] for m in s.get("matches", []) if m.get("status") == "matched"]

    fig, (ax_v, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Panel 1: boundary delta violin
    if deltas:
        parts = ax_v.violinplot(
            [deltas], positions=[0], vert=False, showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor("#5C6BC0")
            pc.set_alpha(0.55)
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            if k in parts:
                parts[k].set_edgecolor("#5C6BC0")
        rng = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(deltas))
        ax_v.scatter(deltas, rng, alpha=0.25, s=10, color="#5C6BC0", zorder=3)
    ax_v.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax_v.set_yticks([0]); ax_v.set_yticklabels(["matched\nboundaries"], fontsize=10)
    ax_v.set_xlabel("Boundary delta (algo - gt) [frames]")
    ax_v.set_title("Boundary delta on matched", fontweight="bold")
    if deltas:
        lo = min(deltas) - 5
        hi = max(deltas) + 5
        ax_v.set_xlim(lo, hi)

    # Panel 2: TP/FP/FN bar + triage
    cats = ["matched", "fp", "fn", "triaged"]
    vals = [s["n_matched"], s["n_fp"], s["n_fn"], s.get("triage_count", 0)]
    colors = ["#43A047", "#E53935", "#FB8C00", "#FFEB3B"]
    bars = ax_b.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax_b.set_title("Boundary status", fontweight="bold")
    ax_b.set_ylabel("Count")
    for b, v in zip(bars, vals):
        ax_b.text(b.get_x() + b.get_width()/2, v, f"{v}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_b.text(0.5, -0.18,
              f"Videos with GT boundaries: {s['n_videos_with_gt_boundaries']}/{s['n_videos']}",
              ha="center", va="top", transform=ax_b.transAxes, fontsize=9, color="#555555")

    plt.tight_layout()
    out = paths.figures_dir / "segmentation_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)

    legend = f"""# Segmentation evaluator (algo 1)

Boundary delta on matched (algo - gt) within +/- 50 frames; greedy
nearest-neighbor. Unmatched algo boundaries -> FP. Unmatched GT
boundaries -> FN. Triage = algo's own boundary_flags entries indicating
low-confidence boundaries flagged for manual review.

n_matched: {s['n_matched']}
n_fp:      {s['n_fp']}
n_fn:      {s['n_fn']}
triage:    {s.get('triage_count', 0)}
delta median / |median| / range:
   {s.get('delta_median')} / {s.get('delta_abs_median')} / [{s.get('delta_min')}, {s.get('delta_max')}]
"""
    (paths.figures_dir / "segmentation_summary_legend.md").write_text(legend, encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.segmentation.graph <snapshot_dir>")
        sys.exit(1)
    out = graph(Path(sys.argv[1]))
    print(f"Wrote: {out}")
