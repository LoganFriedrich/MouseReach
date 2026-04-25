"""
Runner script: executes violin + summary_table logic for reach detection snapshots.

This is the CLI-executable equivalent of running both notebooks with SAVE=True.
It exists because the mousereach env lacks papermill/nbconvert.

Usage:
    python _run_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the mousereach source to path so we can import the palette
src_root = Path(__file__).resolve().parents[3]  # -> MouseReach/src
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from mousereach.improvement.lib.palette import (
    REACH_DETECTION_COLORS,
    REACH_DETECTION_LABELS,
    REACH_DETECTION_DELTA_ORDER,
)

SNAPSHOTS = [
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\reach_v6.0.0_state_machine"),
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\reach_v7.1.0_visibility_direction_reversal"),
]

DPI = 300


# ---- violin ----------------------------------------------------------------

def run_violin(snapshot_dir: Path) -> None:
    print(f"\n=== VIOLIN: {snapshot_dir.name} ===")

    matches_path = snapshot_dir / "metrics" / "reach_matches.csv"
    df_all = pd.read_csv(matches_path)
    df_matched = df_all[df_all["status"] == "matched"].copy()
    df_matched["start_delta"] = df_matched["start_delta"].astype(int)
    df_matched["end_delta"] = df_matched["end_delta"].astype(int)

    n_fp = int((df_all["status"] == "fp").sum())
    n_fn = int((df_all["status"] == "fn").sum())

    print(f"  Loaded {len(df_all)} rows, {len(df_matched)} matched, {n_fp} FP, {n_fn} FN")

    figsize = (12, 5)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=DPI, sharey=False)

    for ax_idx, key in enumerate(REACH_DETECTION_DELTA_ORDER):
        ax = axes[ax_idx]
        data = df_matched[key].values
        color = REACH_DETECTION_COLORS[key]
        label = REACH_DETECTION_LABELS[key]

        if len(data) > 1:
            parts = ax.violinplot(
                data, positions=[0], vert=True, showmeans=True, showmedians=True,
                widths=0.7,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
                pc.set_edgecolor(color)
            for partname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
                if partname in parts:
                    parts[partname].set_edgecolor(color)
                    parts[partname].set_linewidth(1.5)

        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data))
        ax.scatter(jitter, data, alpha=0.15, s=6, color=color, zorder=3,
                   edgecolors="none")

        ax.axhline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.6, zorder=1)
        ax.set_xticks([])
        ax.set_ylabel("Delta (frames): algo − GT", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        median_val = float(np.median(data))
        mean_val = float(np.mean(data))
        ax.text(0.02, 0.98, f"median={median_val:.1f}\nmean={mean_val:.1f}\nn={len(data)}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                fontfamily="monospace", color="#555555")

    fig.text(0.5, -0.02, f"N FP: {n_fp} | N FN: {n_fn}",
             fontsize=10, ha="center", fontfamily="monospace", color="#555555")

    fig.suptitle("Reach boundary deltas (matched only)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "violin.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Reach Boundary Delta -- Violin Plot

## What question this answers

For every algorithm-detected reach that was matched to a ground-truth reach,
how far off were the start and end frames -- and in which direction?

- **start_delta** = `algo_start - gt_start`
- **end_delta** = `algo_end - gt_end`

## What improvement looks like

- Mean and median converge toward 0.
- Distributions get tighter.
- FP and FN counts drop.

## Rendering params

- SNAPSHOT_DIR: `{snapshot_dir}`
- FIGSIZE: {figsize}
- DPI: {DPI}

## Data summary

- Matched reaches: {len(df_matched)}
- FP (phantom): {n_fp}
- FN (miss): {n_fn}
"""
    legend_path = fig_dir / "violin_legend.md"
    legend_path.write_text(legend_md, encoding="utf-8")
    print(f"  Saved: {legend_path}")
    plt.close(fig)


# ---- summary_table ---------------------------------------------------------

def run_summary_table(snapshot_dir: Path) -> None:
    print(f"\n=== SUMMARY TABLE: {snapshot_dir.name} ===")

    matches_path = snapshot_dir / "metrics" / "reach_matches.csv"
    scalars_path = snapshot_dir / "metrics" / "scalars.json"

    df_all = pd.read_csv(matches_path)
    with open(scalars_path, "r") as f:
        scalars = json.load(f)

    df_matched = df_all[df_all["status"] == "matched"].copy()
    df_matched["start_delta"] = df_matched["start_delta"].astype(int)
    df_matched["end_delta"] = df_matched["end_delta"].astype(int)
    print(f"  Loaded {len(df_all)} rows, {len(df_matched)} matched")

    n_fp = scalars["total"]["n_fp"]
    n_fn = scalars["total"]["n_fn"]
    n_matched = scalars["total"]["n_matched"]
    n_gt = scalars["total"]["n_gt"]
    n_algo = scalars["total"]["n_algo"]
    n_videos = scalars["n_videos"]
    n_perfect = scalars["n_perfect_videos"]

    rows = []
    for delta_col, label in [("start_delta", "Start delta"), ("end_delta", "End delta")]:
        deltas = df_matched[delta_col].values
        abs_d = np.abs(deltas)

        n_d0 = int((abs_d == 0).sum())
        n_d1 = int((abs_d == 1).sum())
        n_d2_5 = int(((abs_d >= 2) & (abs_d <= 5)).sum())
        n_d6_10 = int(((abs_d >= 6) & (abs_d <= 10)).sum())

        def pct(n):
            return round(100 * n / n_matched, 1) if n_matched > 0 else 0.0

        median_abs = float(np.median(abs_d)) if len(abs_d) > 0 else float("nan")
        mean_signed = float(np.mean(deltas)) if len(deltas) > 0 else float("nan")

        rows.append({
            "Delta Type": label,
            "delta=0 (%)": pct(n_d0),
            "|delta|=1 (%)": pct(n_d1),
            "2-5 (%)": pct(n_d2_5),
            "6-10 (%)": pct(n_d6_10),
            "FP": n_fp,
            "FN": n_fn,
            "med|d|": median_abs,
            "mean d": round(mean_signed, 2),
        })
        print(f"  {label:15s}  delta=0={pct(n_d0):.1f}%  [OK]")

    table_df = pd.DataFrame(rows)

    figsize = (16, 5)
    header_color = "#E8EAF6"

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, dpi=DPI,
                                          gridspec_kw={"height_ratios": [1, 2.5]})

    # --- Top section: overall counts ---
    ax_top.axis("off")
    header_data = [
        ["n_videos", "n_gt_total", "n_algo_total", "n_matched", "n_perfect_videos"],
        [str(n_videos), str(n_gt), str(n_algo), str(n_matched), str(n_perfect)],
    ]
    header_table = ax_top.table(
        cellText=[header_data[1]],
        colLabels=header_data[0],
        colColours=[header_color] * 5,
        loc="center",
        cellLoc="center",
    )
    header_table.auto_set_font_size(False)
    header_table.set_fontsize(10)
    header_table.scale(1, 1.6)
    for (row_idx, col_idx), cell in header_table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_edgecolor("#CCCCCC")

    # --- Bottom section: delta table ---
    ax_bot.axis("off")

    col_labels = list(table_df.columns)
    cell_text = []
    cell_colors = []

    good_color = "#C8E6C9"
    neutral_color = "#FFFFFF"

    good_high_cols = {"delta=0 (%)"}
    good_low_cols = {"2-5 (%)", "6-10 (%)"}

    for _, row in table_df.iterrows():
        row_text = []
        row_colors = []
        for col in col_labels:
            val = row[col]
            if isinstance(val, float):
                if col in ("med|d|", "mean d"):
                    row_text.append(f"{val:.1f}" if not (val != val) else "--")
                else:
                    row_text.append(f"{val:.1f}")
            else:
                row_text.append(str(val))

            if col in good_high_cols:
                frac = min(float(val) / 100.0, 1.0) if isinstance(val, (int, float)) else 0
                r = int(255 - frac * (255 - 200))
                g = int(255 - frac * (255 - 230))
                b = int(255 - frac * (255 - 201))
                row_colors.append(f"#{r:02X}{g:02X}{b:02X}")
            elif col in good_low_cols:
                frac = min(float(val) / 20.0, 1.0) if isinstance(val, (int, float)) else 0
                if frac < 0.01:
                    row_colors.append(good_color)
                else:
                    r = int(200 + frac * (255 - 200))
                    g = int(230 - frac * (230 - 205))
                    b = int(201 - frac * (201 - 210))
                    row_colors.append(f"#{r:02X}{g:02X}{b:02X}")
            else:
                row_colors.append(neutral_color)

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax_bot.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=[header_color] * len(col_labels),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle("Reach detection accuracy summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "summary_table.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Reach Detection Accuracy Summary Table

## What question this answers

What fraction of reach boundaries land exactly on GT, within 1 frame, within 5,
within 10, or beyond 10 -- and how many reaches are FP or FN?

## Layout

- **Top section**: overall counts (n_videos, n_gt_total, n_algo_total, n_matched, n_perfect_videos)
- **Bottom section**: two rows (Start delta, End delta) with non-overlapping accuracy buckets

## Column definitions (NON-overlapping buckets)

| Column | Definition |
|--------|------------|
| delta=0 (%) | Matched with exactly 0 frame error |
| |delta|=1 (%) | Matched with exactly 1 frame error |
| 2-5 (%) | Matched with 2-5 frames error |
| 6-10 (%) | Matched with 6-10 frames error |
| FP | False positives (algo reaches with no GT match) |
| FN | False negatives (GT reaches with no algo match) |
| med|d| | Median absolute error (matched only) |
| mean d | Mean signed error (matched only) |

## Rendering params

- SNAPSHOT_DIR: `{snapshot_dir}`
- FIGSIZE: {figsize}
- DPI: {DPI}
"""
    legend_path = fig_dir / "summary_table_legend.md"
    legend_path.write_text(legend_md, encoding="utf-8")
    print(f"  Saved: {legend_path}")
    plt.close(fig)


# ---- main ------------------------------------------------------------------

if __name__ == "__main__":
    for snap in SNAPSHOTS:
        if not snap.exists():
            print(f"SKIP (not found): {snap}")
            continue
        run_violin(snap)
        run_summary_table(snap)

    print("\n=== ALL DONE ===")
