"""
Runner script: executes sankey + interaction_frame_violin + summary_table
logic for all outcome snapshots.

This is the CLI-executable equivalent of running the notebooks with SAVE=True.
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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add the mousereach source to path so we can import the palette
src_root = Path(__file__).resolve().parents[3]  # -> MouseReach/src
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from mousereach.improvement.lib.palette import (
    OUTCOME_COLORS,
    OUTCOME_CLASS_ORDER,
    OUTCOME_VERDICT_COLORS,
    OUTCOME_VERDICT_LABELS,
    OUTCOME_VERDICT_ORDER,
)

SNAPSHOTS = [
    Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\outcome\outcome_v2.4.4_pre_new_dlc"),
]

DPI = 300

EXHAUSTIVE_WARNING = (
    "NOTE: Videos not marked exhaustive=True have potentially inflated "
    "label_correct_wrong_reach counts (algo may catch real reaches GT did not label)."
)


# ---- sankey ----------------------------------------------------------------

def run_sankey(snapshot_dir: Path) -> None:
    print(f"\n=== SANKEY: {snapshot_dir.name} ===")

    scalars_path = snapshot_dir / "metrics" / "scalars.json"
    if not scalars_path.exists():
        print(f"  SKIP: no scalars.json in {snapshot_dir.name}")
        return

    with open(scalars_path, "r") as f:
        scalars = json.load(f)

    cm = scalars.get("outcome_label", {}).get("confusion_matrix", {})
    if not cm:
        print("  SKIP: empty confusion matrix")
        return

    # Build flow data: (gt_outcome, algo_outcome, count)
    flows = []
    for key, count in cm.items():
        parts = key.split("__")
        if len(parts) == 2:
            flows.append((parts[0], parts[1], count))

    # Sort by GT outcome order, then algo outcome order
    gt_order = {o: i for i, o in enumerate(OUTCOME_CLASS_ORDER)}
    algo_order = {o: i for i, o in enumerate(OUTCOME_CLASS_ORDER)}
    flows.sort(key=lambda x: (gt_order.get(x[0], 99), algo_order.get(x[1], 99)))

    # Get unique outcomes on each side
    gt_outcomes = []
    algo_outcomes = []
    for gt, algo, _ in flows:
        if gt not in gt_outcomes:
            gt_outcomes.append(gt)
        if algo not in algo_outcomes:
            algo_outcomes.append(algo)

    # Compute totals for positioning
    gt_totals = {}
    algo_totals = {}
    for gt, algo, count in flows:
        gt_totals[gt] = gt_totals.get(gt, 0) + count
        algo_totals[algo] = algo_totals.get(algo, 0) + count

    total_segments = sum(c for _, _, c in flows)

    # Layout: left column = GT, right column = algo
    figsize = (10, 7)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    bar_width = 0.12
    gap = 0.02
    left_x = 0.15
    right_x = 0.85
    y_start = 0.92
    y_scale = 0.80 / total_segments  # normalize to fit

    # Draw GT bars (left)
    gt_positions = {}  # {outcome: (y_top, y_bottom)}
    y_cursor = y_start
    for gt_out in gt_outcomes:
        h = gt_totals[gt_out] * y_scale
        gt_positions[gt_out] = (y_cursor, y_cursor - h)
        color = OUTCOME_COLORS.get(gt_out, "#CCCCCC")
        rect = plt.Rectangle((left_x - bar_width/2, y_cursor - h), bar_width, h,
                              facecolor=color, edgecolor="white", linewidth=0.5,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(left_x - bar_width/2 - 0.02, y_cursor - h/2,
                f"{gt_out}\n({gt_totals[gt_out]})",
                ha="right", va="center", fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        y_cursor -= h + gap

    # Draw algo bars (right)
    algo_positions = {}
    y_cursor = y_start
    for algo_out in algo_outcomes:
        h = algo_totals[algo_out] * y_scale
        algo_positions[algo_out] = (y_cursor, y_cursor - h)
        color = OUTCOME_COLORS.get(algo_out, "#CCCCCC")
        rect = plt.Rectangle((right_x - bar_width/2, y_cursor - h), bar_width, h,
                              facecolor=color, edgecolor="white", linewidth=0.5,
                              transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(right_x + bar_width/2 + 0.02, y_cursor - h/2,
                f"{algo_out}\n({algo_totals[algo_out]})",
                ha="left", va="center", fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        y_cursor -= h + gap

    # Draw flows
    # Track cursor within each bar for flow positioning
    gt_cursors = {o: gt_positions[o][0] for o in gt_outcomes}
    algo_cursors = {o: algo_positions[o][0] for o in algo_outcomes}

    # Track placed mismatch labels as bboxes (x0, y0, x1, y1) in axes coords.
    # Each new label walks along its own ribbon's t parameter trying
    # positions outward from t=0.5; the first position whose bbox does not
    # intersect any already-placed bbox wins.
    placed_bboxes = []

    for gt_out, algo_out, count in flows:
        if count == 0:
            continue
        h = count * y_scale

        gt_top = gt_cursors[gt_out]
        gt_bot = gt_top - h
        gt_cursors[gt_out] = gt_bot

        algo_top = algo_cursors[algo_out]
        algo_bot = algo_top - h
        algo_cursors[algo_out] = algo_bot

        # Determine flow color. Correct flows: source class color, soft alpha.
        # Mismatch flows: source (GT) class color, slightly stronger alpha so
        # the source class is still readable at a glance. The label inherits
        # the same color so labels and their ribbons always look the same.
        if gt_out == algo_out:
            color = OUTCOME_COLORS.get(gt_out, "#CCCCCC")
            alpha = 0.4
        else:
            color = OUTCOME_COLORS.get(gt_out, "#666666")
            alpha = 0.55

        # Draw as a polygon (simplified ribbon)
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.path import Path as MplPath
        import matplotlib.patches as mpatches

        x_left = left_x + bar_width / 2
        x_right = right_x - bar_width / 2
        x_mid = (x_left + x_right) / 2

        verts = [
            (x_left, gt_top),
            (x_mid, gt_top),
            (x_mid, algo_top),
            (x_right, algo_top),
            (x_right, algo_bot),
            (x_mid, algo_bot),
            (x_mid, gt_bot),
            (x_left, gt_bot),
            (x_left, gt_top),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=color, alpha=alpha,
                                   edgecolor="none", transform=ax.transAxes, zorder=1)
        ax.add_patch(patch)

        # Label flows with count (only if mismatched). Jitter to avoid overlap
        # when multiple flow midpoints stack at the same y. Color the label by
        # the GT class so it always matches its ribbon visually, and include a
        # "gt->algo" abbreviation so each label is self-documenting even when
        # jitter pushes it off the ribbon midpoint.
        OUTCOME_ABBR = {
            "retrieved": "ret",
            "displaced_sa": "sa",
            "displaced_outside": "out",
            "untouched": "unt",
            "uncertain": "unc",
            "unknown": "unk",
        }
        if gt_out != algo_out and count >= 1:
            # Place the label ON the ribbon's centerline, at one of several
            # parametric positions along the ribbon's length. Walking t along
            # the actual ribbon (rather than jittering off it) keeps every
            # label visually anchored to its line. When the natural midpoint
            # (t=0.5) collides with an earlier label, slide along the ribbon
            # to the next non-conflicting t.
            gt_mid = (gt_top + gt_bot) / 2
            algo_mid = (algo_top + algo_bot) / 2

            def _ribbon_pos(t):
                # cubic Bezier of the ribbon centerline: smoothstep H(t)
                h = 3.0 * t * t - 2.0 * t * t * t
                rx = x_left + t * (x_right - x_left)
                ry = gt_mid + (algo_mid - gt_mid) * h
                return rx, ry

            # Build the label text first so we know its bbox size.
            label_text = "{0} {1}→{2}".format(
                count, OUTCOME_ABBR.get(gt_out, gt_out[:3]),
                OUTCOME_ABBR.get(algo_out, algo_out[:3]))
            # Estimate bbox in axes coords. Width is character-count-driven
            # (fontsize 7 + pill padding), height is fixed line height.
            bbox_w = 0.0095 * len(label_text) + 0.022
            bbox_h = 0.034

            def _bbox_at(rx, ry):
                return (rx - bbox_w / 2, ry - bbox_h / 2,
                        rx + bbox_w / 2, ry + bbox_h / 2)

            def _intersects(b, others):
                bx0, by0, bx1, by1 = b
                for ox0, oy0, ox1, oy1 in others:
                    if not (bx1 < ox0 or bx0 > ox1 or by1 < oy0 or by0 > oy1):
                        return True
                return False

            # Walk t outward from 0.5 in both directions. Label stays on
            # its own ribbon centerline; only x and y come from t. First
            # bbox that does not collide with any placed_bboxes wins.
            t_grid = [0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70,
                      0.25, 0.75, 0.20, 0.80, 0.15, 0.85]
            label_x, label_y = _ribbon_pos(0.5)
            chosen_bbox = _bbox_at(label_x, label_y)
            for t in t_grid:
                cand_x, cand_y = _ribbon_pos(t)
                cand_bbox = _bbox_at(cand_x, cand_y)
                if not _intersects(cand_bbox, placed_bboxes):
                    label_x, label_y = cand_x, cand_y
                    chosen_bbox = cand_bbox
                    break
            else:
                # Every on-ribbon position conflicts. Fall back: keep at
                # last candidate but nudge y until clear.
                label_x, label_y = cand_x, cand_y
                chosen_bbox = cand_bbox
                while _intersects(chosen_bbox, placed_bboxes):
                    label_y += bbox_h
                    chosen_bbox = _bbox_at(label_x, label_y)

            placed_bboxes.append(chosen_bbox)

            label_color = OUTCOME_COLORS.get(gt_out, "#666666")
            ax.text(label_x, label_y, label_text,
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color=label_color, transform=ax.transAxes, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              alpha=0.92, edgecolor=label_color, linewidth=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title and labels
    ax.text(0.5, 0.98, f"Outcome classification flow (N={total_segments})",
            ha="center", va="top", fontsize=14, fontweight="bold",
            transform=ax.transAxes)
    ax.text(left_x, 0.96, "Ground Truth", ha="center", va="top", fontsize=11,
            fontweight="bold", transform=ax.transAxes)
    ax.text(right_x, 0.96, "Algorithm", ha="center", va="top", fontsize=11,
            fontweight="bold", transform=ax.transAxes)

    # Footer
    n_correct = sum(c for gt, al, c in flows if gt == al)
    strict_acc = scalars.get("outcome_label", {}).get("strict_accuracy")
    committed_acc = scalars.get("outcome_label", {}).get("committed_accuracy")
    footer = f"Correct flows: {n_correct}/{total_segments}"
    if strict_acc is not None:
        footer += f" | Strict acc: {strict_acc:.1%}"
    if committed_acc is not None:
        footer += f" | Committed acc: {committed_acc:.1%}"
    ax.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9,
            color="#555555", transform=ax.transAxes)

    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "sankey.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Outcome Classification Flow -- Sankey Diagram

## What question this answers

How does the algorithm's outcome classification compare to ground truth?
Each flow shows how many segments with a given GT outcome were classified
by the algorithm. Correct flows (GT == algo) are colored by outcome class;
misclassified flows are red with counts labeled.

## What improvement looks like

- Thicker correct flows (same color, GT to algo).
- Thinner or absent red cross-flows.
- Equal bar heights on both sides (no systematic over/under-prediction).

## Red-flag patterns

- **Thick red flows**: systematic misclassification (e.g., retrieved -> displaced_sa).
- **Unbalanced bars**: algorithm over-predicts one class at expense of another.
- **Many flows to uncertain**: algorithm abstaining too often.

## Exhaustive flag

{EXHAUSTIVE_WARNING}

## Rendering params

- SNAPSHOT_DIR: `{snapshot_dir}`
- FIGSIZE: {figsize}
- DPI: {DPI}

## Data summary

- Total segments: {total_segments}
- Correct: {n_correct} ({100*n_correct/total_segments:.1f}%)
"""
    legend_path = fig_dir / "sankey_legend.md"
    legend_path.write_text(legend_md, encoding="utf-8")
    print(f"  Saved: {legend_path}")
    plt.close(fig)


# ---- interaction_frame_violin ----------------------------------------------

def run_interaction_violin(snapshot_dir: Path) -> None:
    print(f"\n=== INTERACTION VIOLIN: {snapshot_dir.name} ===")

    segments_path = snapshot_dir / "metrics" / "outcome_per_segment.csv"
    if not segments_path.exists():
        print(f"  SKIP: no outcome_per_segment.csv in {snapshot_dir.name}")
        return

    df = pd.read_csv(segments_path)

    # Filter to segments where both sides committed and have interaction frames
    touched = df[
        df["gt_outcome"].isin(["retrieved", "displaced_sa", "displaced_outside"]) &
        df["algo_outcome"].isin(["retrieved", "displaced_sa", "displaced_outside"]) &
        df["interaction_frame_delta"].notna()
    ].copy()

    touched["interaction_frame_delta"] = touched["interaction_frame_delta"].astype(int)
    deltas = touched["interaction_frame_delta"].values
    print(f"  Loaded {len(df)} rows, {len(touched)} eligible for interaction delta")

    if len(deltas) < 2:
        print("  SKIP: insufficient data for violin plot")
        return

    # Render -- single horizontal violin
    figsize = (10, 4)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    color = "#5C6BC0"  # Indigo

    parts = ax.violinplot(
        deltas, positions=[0], vert=False, showmeans=True, showmedians=True,
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

    # Jittered points
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(deltas))
    ax.scatter(deltas, 0 + jitter, alpha=0.25, s=8, color=color, zorder=3,
               edgecolors="none")

    ax.axvline(0, color="#333333", linewidth=1, linestyle="--", alpha=0.6, zorder=1)

    ax.set_yticks([0])
    ax.set_yticklabels(["All touched\nsegments"], fontsize=11)
    ax.set_xlabel("Signed delta (frames): algo - GT interaction frame", fontsize=12)
    ax.set_title("Interaction frame delta (both sides committed)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Clip x-axis to reasonable range (exclude extreme outliers for readability)
    p5, p95 = np.percentile(deltas, [2, 98])
    margin = max(abs(p5), abs(p95)) * 0.3
    x_lo = min(p5 - margin, -15)
    x_hi = max(p95 + margin, 15)
    ax.set_xlim(x_lo, x_hi)

    # Stats footer
    med_abs = int(np.median(np.abs(deltas)))
    mean_signed = np.mean(deltas)
    n_exact = int((np.abs(deltas) == 0).sum())
    n_within5 = int((np.abs(deltas) <= 5).sum())
    footer = (f"N={len(deltas)} | median|delta|={med_abs} | mean delta={mean_signed:.1f} | "
              f"exact(0)={n_exact} ({100*n_exact/len(deltas):.0f}%) | "
              f"within 5f={n_within5} ({100*n_within5/len(deltas):.0f}%)")
    fig.text(0.12, -0.02, footer, fontsize=8, fontfamily="monospace",
             verticalalignment="top", color="#555555")

    n_outlier = int((np.abs(deltas) > abs(x_hi)).sum() + (np.abs(deltas) > abs(x_lo)).sum())
    if n_outlier > 0:
        fig.text(0.12, -0.06, f"({n_outlier} outliers beyond axis limits not shown)",
                 fontsize=7, color="#999999")

    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "interaction_frame_violin.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Interaction Frame Delta -- Violin Plot

## What question this answers

For segments where both GT and algorithm identified a touched outcome
(retrieved, displaced_sa, displaced_outside), how close is the algorithm's
interaction_frame to the GT's interaction_frame? The signed delta is
`algo_interaction_frame - gt_interaction_frame`, so negative = algorithm
detects interaction earlier, positive = later.

## What improvement looks like

- Mean and median converge toward 0 (no systematic bias).
- The distribution gets tighter (fewer frames of error).
- Fewer outliers in the tails.

## Red-flag patterns

- **Mean shift away from 0**: systematic timing bias.
- **Long tails**: outlier segments with very wrong interaction timing.
- **Bimodal distribution**: two populations of errors (different failure modes).

## Exhaustive flag

{EXHAUSTIVE_WARNING}

## Rendering params

- SNAPSHOT_DIR: `{snapshot_dir}`
- FIGSIZE: {figsize}
- DPI: {DPI}

## Data summary

- Eligible segments: {len(deltas)}
- Median |delta|: {med_abs} frames
- Mean signed delta: {mean_signed:.1f} frames
- Exact match (delta=0): {n_exact} ({100*n_exact/len(deltas):.0f}%)
- Within 5 frames: {n_within5} ({100*n_within5/len(deltas):.0f}%)
"""
    legend_path = fig_dir / "interaction_frame_violin_legend.md"
    legend_path.write_text(legend_md, encoding="utf-8")
    print(f"  Saved: {legend_path}")
    plt.close(fig)


# ---- summary_table ---------------------------------------------------------

def run_summary_table(snapshot_dir: Path) -> None:
    print(f"\n=== SUMMARY TABLE: {snapshot_dir.name} ===")

    scalars_path = snapshot_dir / "metrics" / "scalars.json"
    segments_path = snapshot_dir / "metrics" / "outcome_per_segment.csv"

    if not scalars_path.exists() or not segments_path.exists():
        print(f"  SKIP: missing metrics files in {snapshot_dir.name}")
        return

    with open(scalars_path, "r") as f:
        scalars = json.load(f)

    df = pd.read_csv(segments_path)

    # ==== Section 1: Label block ====
    label_data = scalars.get("outcome_label", {})
    per_class = label_data.get("per_class", {})

    label_rows = []
    # Overall row
    label_rows.append({
        "Class": "OVERALL",
        "N_GT": scalars.get("n_segments_paired", 0),
        "N_algo": scalars.get("n_segments_paired", 0),
        "Precision": "",
        "Recall": "",
        "F1": "",
        "Strict Acc %": f"{label_data.get('strict_accuracy', 0)*100:.1f}",
        "Committed Acc %": f"{label_data.get('committed_accuracy', 0)*100:.1f}",
        "Abstention %": f"{label_data.get('abstention_rate', 0)*100:.1f}",
    })
    for cls in ["retrieved", "displaced_sa", "displaced_outside", "untouched"]:
        pc = per_class.get(cls, {})
        label_rows.append({
            "Class": cls,
            "N_GT": pc.get("n_gt", 0),
            "N_algo": pc.get("n_algo", 0),
            "Precision": f"{pc.get('precision', 0)*100:.1f}",
            "Recall": f"{pc.get('recall', 0)*100:.1f}",
            "F1": f"{pc.get('f1', 0)*100:.1f}",
            "Strict Acc %": "",
            "Committed Acc %": "",
            "Abstention %": "",
        })

    # ==== Section 2: Interaction frame block ====
    ifr = scalars.get("interaction_frame", {})
    deltas_col = df["interaction_frame_delta"].dropna().astype(int)
    abs_deltas = deltas_col.abs()

    n_d0 = int((abs_deltas == 0).sum())
    n_d1 = int((abs_deltas == 1).sum())
    n_d2_5 = int(((abs_deltas >= 2) & (abs_deltas <= 5)).sum())
    n_d6_10 = int(((abs_deltas >= 6) & (abs_deltas <= 10)).sum())
    n_dgt10 = int((abs_deltas > 10).sum())
    n_null = int(df["interaction_frame_delta"].isna().sum())

    n_ifr_total = len(deltas_col) + n_null

    def pct_ifr(n):
        return f"{100*n/n_ifr_total:.1f}" if n_ifr_total > 0 else "0.0"

    ifr_rows = [{
        "Bucket": "delta=0",
        "Count": n_d0,
        "%": pct_ifr(n_d0),
    }, {
        "Bucket": "|delta|=1",
        "Count": n_d1,
        "%": pct_ifr(n_d1),
    }, {
        "Bucket": "2-5",
        "Count": n_d2_5,
        "%": pct_ifr(n_d2_5),
    }, {
        "Bucket": "6-10",
        "Count": n_d6_10,
        "%": pct_ifr(n_d6_10),
    }, {
        "Bucket": ">10",
        "Count": n_dgt10,
        "%": pct_ifr(n_dgt10),
    }, {
        "Bucket": "null/missing",
        "Count": n_null,
        "%": pct_ifr(n_null),
    }]

    # ==== Section 3: Causal reach block ====
    cr = scalars.get("causal_reach", {})
    cr_overall = cr.get("overall", {})
    cr_per_class = cr.get("per_class", {})

    verdict_keys = ["label_and_reach_correct", "label_correct_wrong_reach",
                    "label_wrong", "abstained"]

    cr_rows = []
    # Overall row
    cr_total = sum(cr_overall.get(v, 0) for v in verdict_keys)
    cr_row = {"Class": "OVERALL (touched)", "N": cr_total}
    for v in verdict_keys:
        n = cr_overall.get(v, 0)
        cr_row[OUTCOME_VERDICT_LABELS.get(v, v)] = f"{n} ({100*n/cr_total:.0f}%)" if cr_total > 0 else "0"
    cr_rows.append(cr_row)

    for cls in ["retrieved", "displaced_sa", "displaced_outside"]:
        cls_data = cr_per_class.get(cls, {})
        cls_total = sum(cls_data.get(v, 0) for v in verdict_keys)
        cr_row = {"Class": cls, "N": cls_total}
        for v in verdict_keys:
            n = cls_data.get(v, 0)
            cr_row[OUTCOME_VERDICT_LABELS.get(v, v)] = f"{n} ({100*n/cls_total:.0f}%)" if cls_total > 0 else "0"
        cr_rows.append(cr_row)

    # ==== Render as three stacked matplotlib tables ====
    figsize = (16, 12)
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=DPI,
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5]})

    header_color = "#E8EAF6"
    good_color = "#C8E6C9"
    bad_color = "#FFCDD2"
    neutral_color = "#FFFFFF"

    # --- Table 1: Label accuracy ---
    ax1 = axes[0]
    ax1.axis("off")
    label_df = pd.DataFrame(label_rows)
    cols1 = list(label_df.columns)
    cell_text1 = [[str(v) for v in row] for _, row in label_df.iterrows()]
    cell_colors1 = [[neutral_color] * len(cols1) for _ in range(len(cell_text1))]

    t1 = ax1.table(cellText=cell_text1, colLabels=cols1,
                   cellColours=cell_colors1,
                   colColours=[header_color] * len(cols1),
                   loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.5)
    for (r, c), cell in t1.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_edgecolor("#CCCCCC")
    ax1.set_title("1. Outcome label accuracy", fontsize=13, fontweight="bold", pad=15)

    # --- Table 2: Interaction frame ---
    ax2 = axes[1]
    ax2.axis("off")
    ifr_df = pd.DataFrame(ifr_rows)
    cols2 = list(ifr_df.columns)
    cell_text2 = [[str(v) for v in row] for _, row in ifr_df.iterrows()]
    cell_colors2 = [[neutral_color] * len(cols2) for _ in range(len(cell_text2))]

    t2 = ax2.table(cellText=cell_text2, colLabels=cols2,
                   cellColours=cell_colors2,
                   colColours=[header_color] * len(cols2),
                   loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.5)
    for (r, c), cell in t2.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_edgecolor("#CCCCCC")

    med_str = str(ifr.get("median_abs_delta", "--"))
    mean_str = f"{ifr.get('mean_signed_delta', 0):.1f}" if ifr.get("mean_signed_delta") is not None else "--"
    ax2.set_title(f"2. Interaction frame accuracy (median |delta|={med_str}, mean delta={mean_str})",
                  fontsize=13, fontweight="bold", pad=15)

    # --- Table 3: Causal reach ---
    ax3 = axes[2]
    ax3.axis("off")
    cr_df = pd.DataFrame(cr_rows)
    cols3 = list(cr_df.columns)
    cell_text3 = [[str(v) for v in row] for _, row in cr_df.iterrows()]
    cell_colors3 = [[neutral_color] * len(cols3) for _ in range(len(cell_text3))]

    t3 = ax3.table(cellText=cell_text3, colLabels=cols3,
                   cellColours=cell_colors3,
                   colColours=[header_color] * len(cols3),
                   loc="center", cellLoc="center")
    t3.auto_set_font_size(False)
    t3.set_fontsize(8)
    t3.scale(1, 1.5)
    for (r, c), cell in t3.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=7)
        cell.set_edgecolor("#CCCCCC")
    ax3.set_title("3. Causal reach matching (touched outcomes only)",
                  fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()

    # Save
    fig_dir = snapshot_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "summary_table.png"
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"  Saved: {png_path}")

    legend_md = f"""# Outcome Accuracy Summary Table

## What question this answers

Three-section summary of outcome classification quality:
1. **Label accuracy**: How often does the algorithm get the outcome right?
   Overall and per-class precision/recall/F1.
2. **Interaction frame accuracy**: When both sides agree on a touched outcome,
   how close is the algorithm's interaction_frame to GT?
3. **Causal reach matching**: For correctly-labeled touched segments, does the
   algorithm identify the same causal reach as GT?

## Column definitions

### Section 1 -- Label
| Column | Definition |
|--------|------------|
| Class | Outcome class or OVERALL |
| N_GT | Ground truth count |
| N_algo | Algorithm count |
| Precision | TP / (TP + FP) for this class |
| Recall | TP / (TP + FN) for this class |
| F1 | Harmonic mean of P and R |
| Strict Acc % | Correct / total (overall only) |
| Committed Acc % | Correct / committed (overall only) |
| Abstention % | Uncertain / total (overall only) |

### Section 2 -- Interaction frame
Non-overlapping buckets that sum to 100% of all paired segments.

### Section 3 -- Causal reach
Verdict breakdown for touched (non-untouched) segments.

## Exhaustive flag

{EXHAUSTIVE_WARNING}

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
        if not (snap / "metrics" / "scalars.json").exists():
            print(f"SKIP (no metrics): {snap}")
            continue
        run_sankey(snap)
        run_interaction_violin(snap)
        run_summary_table(snap)

    print("\n=== ALL DONE ===")
