"""
Reach detection v8 evaluation figures.

Per the user-mandated reporting standard: TP/FP/FN counts + start-delta
+ span-delta distributions on the TPs. NOT P/R/F1.

Inputs: the per-fold or aggregate result JSON from
`reach.v8.train.FoldResult.summary` plus the raw match list. Outputs:
PNG figures + a markdown legend in `<snapshot>/figures/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_v8_reach_figures(
    snapshot_dir: Path,
    raw_results: Sequence[dict],
    summary: dict,
    title_suffix: str = "",
) -> None:
    """Render the canonical reach-detection v8 figures into snapshot/figures/.

    Files written:
      - reach_detection_summary.png  (TP/FP/FN bar + dual-panel deltas)
      - reach_detection_legend.md
    """
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tps = [r for r in raw_results if r["status"] == "tp"]
    n_tp = summary["n_tp"]
    n_fp = summary["n_fp"]
    n_fn = summary["n_fn"]
    start_deltas = np.array([r["start_delta"] for r in tps], dtype=int)
    span_deltas = np.array([r["span_delta"] for r in tps], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel 1: TP/FP/FN bar
    ax = axes[0]
    cats = ["TP", "FP", "FN"]
    vals = [n_tp, n_fp, n_fn]
    colors = ["#43A047", "#E53935", "#FB8C00"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title(f"Reach detection counts{title_suffix}", fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    n_gt = n_tp + n_fn
    n_algo = n_tp + n_fp
    ax.text(0.5, -0.18,
            f"GT reaches: {n_gt} | Algo reaches: {n_algo}",
            ha="center", va="top", transform=ax.transAxes, fontsize=9, color="#555555")

    # Panel 2: Start delta histogram (TPs only)
    ax = axes[1]
    if len(start_deltas) > 0:
        bins = np.arange(-3.5, 3.5 + 1, 1)
        ax.hist(start_deltas, bins=bins, color="#5C6BC0",
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Start delta (algo - gt) [frames]")
        ax.set_ylabel("Count of TP reaches")
        ax.set_title("Start delta on TPs", fontweight="bold")
        med = int(np.median(start_deltas))
        absmed = int(np.median(np.abs(start_deltas)))
        ax.text(0.02, 0.95,
                f"n={len(start_deltas)}\nmedian={med}\n|median|={absmed}\n"
                f"range=[{int(start_deltas.min())},{int(start_deltas.max())}]",
                transform=ax.transAxes, fontsize=9, va="top",
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    # Panel 3: Span delta histogram (TPs only)
    ax = axes[2]
    if len(span_deltas) > 0:
        lo = int(span_deltas.min()) - 1
        hi = int(span_deltas.max()) + 1
        bins = np.arange(lo - 0.5, hi + 0.5 + 1, 1)
        ax.hist(span_deltas, bins=bins, color="#26A69A",
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Span delta (algo.span - gt.span) [frames]")
        ax.set_ylabel("Count of TP reaches")
        ax.set_title("Span delta on TPs", fontweight="bold")
        med = int(np.median(span_deltas))
        absmed = int(np.median(np.abs(span_deltas)))
        ax.text(0.02, 0.95,
                f"n={len(span_deltas)}\nmedian={med}\n|median|={absmed}\n"
                f"range=[{int(span_deltas.min())},{int(span_deltas.max())}]",
                transform=ax.transAxes, fontsize=9, va="top",
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    plt.tight_layout()
    out_png = fig_dir / "reach_detection_summary.png"
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)

    legend_md = f"""# Reach Detection v8 -- Evaluation Summary

## What this figure shows

Per the user-mandated reach-detection reporting standard:

**Panel 1 -- TP/TN/FP/FN counts.** A detected reach counts as **TP** iff
its start_frame is within +/- 2 frames of a GT reach start_frame AND
its span is within max(0.5*gt_span, 5 frames) of GT. Otherwise, an
algo reach with no GT match is **FP**, and a GT reach with no algo
match is **FN**.

**Panel 2 -- Start delta distribution (TPs only).** Histogram of
`algo_start - gt_start` for matched reaches. A peak at 0 with tight
spread means the model is calling reach starts at the right frame.
Negative bias = early; positive = late.

**Panel 3 -- Span delta distribution (TPs only).** Histogram of
`algo.span - gt.span`. Negative = under-extends reach windows;
positive = over-extends.

## Numbers (reproducible from `summary`)

- TP: {n_tp} | FP: {n_fp} | FN: {n_fn}
- GT reaches present: {n_gt} | Algo reaches emitted: {n_algo}
- Start delta on TPs: median={summary['tp_start_delta']['median']}f,
  |median|={summary['tp_start_delta']['abs_median']}f,
  range=[{summary['tp_start_delta']['min']}, {summary['tp_start_delta']['max']}]f
- Span delta on TPs: median={summary['tp_span_delta']['median']}f,
  |median|={summary['tp_span_delta']['abs_median']}f,
  range=[{summary['tp_span_delta']['min']}, {summary['tp_span_delta']['max']}]f

## What is intentionally NOT in this figure

Precision, recall, and F1. They are derivable from the counts above
but are not the lead metric per
`reach_outcome_evaluation_format.md`. F1 is banned outright per
`feedback_no_f1.md`.
"""
    (fig_dir / "reach_detection_legend.md").write_text(
        legend_md, encoding="utf-8")
