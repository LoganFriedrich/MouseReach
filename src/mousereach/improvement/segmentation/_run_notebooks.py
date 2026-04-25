"""
Runner script: executes violin + summary_table logic for all snapshots.

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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Add the mousereach source to path so we can import the palette
src_root = Path(__file__).resolve().parents[3]  # -> MouseReach/src
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from mousereach.improvement.lib.palette import (
    SEGMENTATION_COLORS,
    SEGMENTATION_LABELS,
    SEGMENTATION_SUBSET_ORDER,
)

SNAPSHOTS = [
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\segmentation\seg_v2.1.3_phantom_first_post_validation"),
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\segmentation\seg_v2.2.0_multi_proposer"),
]

DPI = 300


# ---- violin ----------------------------------------------------------------

def run_violin(snapshot_dir: Path) -> None:
    print(f"\n=== VIOLIN: {snapshot_dir.name} ===")

    deltas_path = snapshot_dir / "metrics" / "boundary_deltas.csv"
    df_all = pd.read_csv(deltas_path)
    df_matched = df_all[df_all["status"] == "matched"].copy()
    df_matched["signed_delta"] = df_matched["signed_delta"].astype(int)
    print(f"  Loaded {len(df_all)} rows, {len(df_matched)} matched")

    # Compute subsets
    subsets = {}
    for key in SEGMENTATION_SUBSET_ORDER:
        if key == "all":
            matched = df_matched
            full = df_all
        else:
            matched = df_matched[df_matched["subset_tag"] == key]
            full = df_all[df_all["subset_tag"] == key]

        n_phantom = int((full["status"] == "phantom").sum())
        n_miss = int((full["status"] == "miss").sum())

        subsets[key] = {
            "deltas": matched["signed_delta"].values,
            "n_phantom": n_phantom,
            "n_miss": n_miss,
            "label": SEGMENTATION_LABELS[key],
            "color": SEGMENTATION_COLORS[key],
            "n_matched": len(matched),
        }

    for k, v in subsets.items():
        print(f"  {v['label']:30s}  n_matched={v['n_matched']:4d}  phantom={v['n_phantom']}  miss={v['n_miss']}")

    # Render
    figsize = (10, 6)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    positions = list(range(len(SEGMENTATION_SUBSET_ORDER)))

    labels_for_y = []
    for i, key in enumerate(SEGMENTATION_SUBSET_ORDER):
        s = subsets[key]
        data = s["deltas"]
        color = s["color"]
        labels_for_y.append(s["label"])

        if len(data) > 1:
            parts = ax.violinplot(
                data, positions=[i], vert=False, showmeans=True, showmedians=True,
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
        ax.scatter(data, i + jitter, alpha=0.25, s=8, color=color, zorder=3,
                   edgecolors="none")

    ax.axvline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels_for_y, fontsize=11)
    ax.set_xlabel("Signed delta (frames): algo − GT", fontsize=12)
    ax.set_title("Boundary signed delta (matched only)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    footer_lines = []
    for key in SEGMENTATION_SUBSET_ORDER:
        s = subsets[key]
        footer_lines.append(f"{s['label']}: N phantom: {s['n_phantom']} | N miss: {s['n_miss']}")
    footer_text = "\n".join(footer_lines)
    fig.text(0.12, -0.02, footer_text, fontsize=8, fontfamily="monospace",
             verticalalignment="top", color="#555555")

    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "violin.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Boundary Signed Delta -- Violin Plot

## What question this answers

For every algorithm-emitted boundary that was matched to a ground-truth boundary,
how far off was it -- and in which direction? The **signed delta** is
`algo_frame - gt_frame`, so negative = algorithm fires early, positive = late.

## What improvement looks like

- Mean and median converge toward 0 (no systematic bias).
- The distribution gets tighter (fewer frames of error).
- Fewer outliers in the tails.
- Phantom and miss counts drop to 0.

## Red-flag patterns

- **Mean shift away from 0**: systematic bias (algorithm consistently early or late).
- **Long tails**: outlier boundaries that are very far from GT.
- **Many phantom/miss**: fundamental segment-count error, not just boundary placement.

## Rendering params

- SNAPSHOT_DIR: `{snapshot_dir}`
- FIGSIZE: {figsize}
- DPI: {DPI}
- Subsets: {', '.join(SEGMENTATION_SUBSET_ORDER)}

## Data summary

"""
    for key in SEGMENTATION_SUBSET_ORDER:
        s = subsets[key]
        legend_md += f"- **{s['label']}**: {s['n_matched']} matched, {s['n_phantom']} phantom, {s['n_miss']} miss\n"

    legend_path = fig_dir / "violin_legend.md"
    legend_path.write_text(legend_md, encoding="utf-8")
    print(f"  Saved: {legend_path}")
    plt.close(fig)


# ---- summary_table ---------------------------------------------------------

def run_summary_table(snapshot_dir: Path) -> None:
    print(f"\n=== SUMMARY TABLE: {snapshot_dir.name} ===")

    deltas_path = snapshot_dir / "metrics" / "boundary_deltas.csv"
    scalars_path = snapshot_dir / "metrics" / "scalars.json"

    df_all = pd.read_csv(deltas_path)
    with open(scalars_path, "r") as f:
        scalars = json.load(f)

    df_matched = df_all[df_all["status"] == "matched"].copy()
    df_matched["signed_delta"] = df_matched["signed_delta"].astype(int)
    df_matched["abs_delta"] = df_matched["signed_delta"].abs()
    print(f"  Loaded {len(df_all)} rows, {len(df_matched)} matched")

    rows = []
    for key in SEGMENTATION_SUBSET_ORDER:
        sc = scalars[key] if key != "all" else scalars["all"]
        if key == "all":
            matched = df_matched
            full = df_all
        else:
            matched = df_matched[df_matched["subset_tag"] == key]
            full = df_all[df_all["subset_tag"] == key]

        n_gt = int(sc["n_gt_boundaries"])
        n_algo = int(sc["n_algo_boundaries"])
        n_phantom = int(sc["n_phantom"])
        n_miss = int(sc["n_miss"])
        n_matched = len(matched)

        total = n_matched + n_phantom + n_miss

        abs_d = matched["abs_delta"].values if len(matched) > 0 else np.array([])

        n_d0 = int((abs_d == 0).sum())
        n_d1 = int((abs_d == 1).sum())
        n_d2_5 = int(((abs_d >= 2) & (abs_d <= 5)).sum())
        n_d6_10 = int(((abs_d >= 6) & (abs_d <= 10)).sum())
        n_d_gt10 = int((abs_d > 10).sum())

        assert n_d0 + n_d1 + n_d2_5 + n_d6_10 + n_d_gt10 == n_matched, (
            f"Bucket sum mismatch for {key}: "
            f"{n_d0}+{n_d1}+{n_d2_5}+{n_d6_10}+{n_d_gt10} != {n_matched}"
        )

        def pct(n):
            return round(100 * n / total, 1) if total > 0 else 0.0

        pct_d0 = pct(n_d0)
        pct_d1 = pct(n_d1)
        pct_d2_5 = pct(n_d2_5)
        pct_d6_10 = pct(n_d6_10)
        pct_d_gt10 = pct(n_d_gt10)
        pct_miss = pct(n_miss)
        pct_phantom = pct(n_phantom)

        pct_sum = pct_d0 + pct_d1 + pct_d2_5 + pct_d6_10 + pct_d_gt10 + pct_miss + pct_phantom
        assert abs(pct_sum - 100.0) < 0.5, (
            f"Percentage sum for {key} = {pct_sum:.1f}%, expected ~100%"
        )
        print(f"  {SEGMENTATION_LABELS[key]:30s}  pct_sum={pct_sum:.1f}%  [OK]")

        median_abs = float(np.median(abs_d)) if len(abs_d) > 0 else float("nan")
        mean_signed = float(np.mean(matched["signed_delta"].values)) if len(matched) > 0 else float("nan")

        rows.append({
            "Subset": SEGMENTATION_LABELS[key],
            "N_GT": n_gt,
            "N_algo": n_algo,
            "delta=0 (%)": pct_d0,
            "|delta|=1 (%)": pct_d1,
            "2-5 (%)": pct_d2_5,
            "6-10 (%)": pct_d6_10,
            ">10 (%)": pct_d_gt10,
            "miss (%)": pct_miss,
            "phantom (%)": pct_phantom,
            "med|d|": median_abs,
            "mean d": round(mean_signed, 2),
        })

    table_df = pd.DataFrame(rows)

    # Render
    figsize = (16, 4)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    ax.axis("off")

    col_labels = list(table_df.columns)
    cell_text = []
    cell_colors = []

    good_color = "#C8E6C9"
    neutral_color = "#FFFFFF"
    bad_color = "#FFCDD2"
    header_color = "#E8EAF6"

    good_high_cols = {"delta=0 (%)"}
    good_low_cols = {"2-5 (%)", "6-10 (%)", ">10 (%)", "miss (%)", "phantom (%)"}

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

    table = ax.table(
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

    ax.set_title("Boundary accuracy summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "summary_table.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Boundary Accuracy Summary Table

## What question this answers

What fraction of algorithm boundaries land exactly on GT, within 1 frame, within 5,
within 10, or beyond 10 -- and how many boundaries are phantom (no GT match) or
missed (GT with no algo match)?

## Column definitions (NON-overlapping buckets)

Percentage columns partition ALL boundaries into exhaustive, mutually exclusive buckets
that sum to 100%.

| Column | Definition |
|--------|------------|
| N_GT | Number of ground-truth boundaries |
| N_algo | Number of algorithm-emitted boundaries |
| delta=0 (%) | Matched with exactly 0 frame error |
| abs(delta)=1 (%) | Matched with exactly 1 frame error |
| 2-5 (%) | Matched with 2-5 frames error |
| 6-10 (%) | Matched with 6-10 frames error |
| >10 (%) | Matched with >10 frames error |
| miss (%) | GT boundaries with no algo match |
| phantom (%) | Algo boundaries with no GT match |
| median abs(delta) | Median absolute error (matched only) |
| mean signed delta | Mean signed error (matched only) |

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
