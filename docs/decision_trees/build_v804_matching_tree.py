"""Strict binary decision tree for the v8.0.4 matching pipeline.

Documents how detected algo reaches get matched to ground-truth reaches
and classified into topology categories (TP, TOLERANCE_ERROR, FP, FN,
MERGED, FRAGMENTED), plus the post-match filters that produce the
headline counts.

Built by reading every conditional in:
  - mousereach.improvement.reach_detection.metrics (matching + filters)
  - scripts/restart_phase_b_matcher_aware_topology.py
    (canonical topology rules locked 2026-05-22)

Stages:
  1. Greedy matching (asymmetric -2/+5 start tolerance + relative span tolerance)
  2. Connected components (overlap graph, min_overlap=1)
  3. Matcher-aware topology classification (6-category)
  4. Post-match filters (MIN_REPORTED_SPAN, outside_gt_segmentation)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).parent


# ---- Drawing helpers (mirror the detection-tree conventions) ----

def add_process_box(ax, x, y, text, color="#cfe2ff", width=5.2, height=0.85,
                     sub_text=None):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=1.4)
    ax.add_patch(box)
    if sub_text:
        ax.text(x, y + 0.18, text, ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(x, y - 0.30, sub_text, ha="center", va="center",
                fontsize=5.8, style="italic", color="#555555")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, weight="bold")


def add_decision_tunable(ax, x, y, text, width=4.5, height=1.2,
                           sub_text=None):
    pts = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x, y - height/2],
        [x - width/2, y],
    ])
    diamond = Polygon(pts, fc="#f8d7da", ec="black", linewidth=1.5)
    ax.add_patch(diamond)
    if sub_text:
        ax.text(x, y + 0.18, text, ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(x, y - 0.30, sub_text, ha="center", va="center",
                fontsize=5.8, style="italic", color="#555555")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=7.8)


def add_decision_structural(ax, x, y, text, width=4.5, height=1.2,
                              sub_text=None):
    pts = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x, y - height/2],
        [x - width/2, y],
    ])
    diamond = Polygon(pts, fc="#e0e0e0", ec="dimgray", linewidth=1.2,
                       linestyle="--")
    ax.add_patch(diamond)
    if sub_text:
        ax.text(x, y + 0.18, text, ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text(x, y - 0.30, sub_text, ha="center", va="center",
                fontsize=5.8, style="italic", color="#555555")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                style="italic")


def add_leaf(ax, x, y, text, color, width=3.8, height=0.85):
    skew = 0.3
    pts = np.array([
        [x - width/2 + skew, y - height/2],
        [x + width/2, y - height/2],
        [x + width/2 - skew, y + height/2],
        [x - width/2, y + height/2],
    ])
    poly = Polygon(pts, fc=color, ec="black", linewidth=1.5)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.3, weight="bold")


def add_edge(ax, x1, y1, x2, y2, label=None, label_xoff=0.0, label_yoff=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    if label:
        mx = (x1 + x2) / 2 + label_xoff
        my = (y1 + y2) / 2 + label_yoff
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                weight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="darkblue", linewidth=0.8))


def add_loop_edge(ax, x1, y1, x2, y2, label, rad=-0.45):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", lw=1.3, color="darkorange",
                            linestyle="--",
                            connectionstyle=f"arc3,rad={rad}",
                            mutation_scale=12)
    ax.add_patch(arr)


# ---- Build figure ----

def main():
    fig, ax = plt.subplots(figsize=(28, 60))
    ax.set_xlim(-14, 18)
    ax.set_ylim(-15, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(2, 79.3,
            "v8.0.4 Matching Pipeline: Strict Binary Decision Tree",
            ha="center", va="center", fontsize=18, weight="bold")
    ax.text(2, 78.4,
            "How algo reaches are matched to GT reaches and classified into "
            "topology categories. Pink = tunable, gray dashed = structural.",
            ha="center", va="center", fontsize=10, style="italic")

    # ===== INPUT =====
    y = 76.5
    add_process_box(ax, 2, y,
                     "INPUT:\nAlgo reaches + GT reaches + GT segmentation boundaries",
                     color="#e7d4f5", width=8.0, height=1.2)
    add_edge(ax, 2, y - 0.65, 2, y - 1.1)

    # ===== STAGE 1: GREEDY MATCHING =====
    y -= 1.9
    ax.text(-10, y, "STAGE 1\nGreedy matching",
            ha="left", va="center", fontsize=13, weight="bold",
            style="italic", color="#444444")
    add_process_box(ax, 2, y,
                     "Build candidate list:\nFOR each algo a, FOR each GT g",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="candidates = []; for ai, a in enumerate(algos): for gi, g in enumerate(gts):")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    y -= 1.8
    add_process_box(ax, 2, y,
                     "Compute deltas of gt/algo matches",
                     color="#cfe2ff", width=6.5, height=0.9,
                     sub_text="start_delta = algo_start - gt_start; span_delta = algo_span - gt_span")
    add_edge(ax, 2, y - 0.5, 2, y - 1.1)

    # ----- Tunable: start_delta early bound -----
    y -= 1.8
    y_pair_top = y  # for the candidate-rejection loop-back
    add_decision_tunable(ax, 2, y,
                          "Did the algo start too early?\n(>2 frames before GT)",
                          sub_text="start_delta < -STRICT_START_TOL_EARLY (=2)? (TUNABLE)")
    add_leaf(ax, -4, y, "Skip pair,\ntry next combination",
              color="#fff3e0", width=4.5)
    add_edge(ax, -0.25, y, -1.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # ----- Tunable: start_delta late bound -----
    y -= 1.8
    add_decision_tunable(ax, 2, y,
                          "Did the algo start too late?\n(>5 frames after GT)",
                          sub_text="start_delta > STRICT_START_TOL_LATE (=5)? (TUNABLE)")
    add_leaf(ax, -4, y, "Skip pair,\ntry next combination",
              color="#fff3e0", width=4.5)
    add_edge(ax, -0.25, y, -1.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # ----- Tunable: span_delta -----
    y -= 1.8
    add_decision_tunable(ax, 2, y,
                          "Span check?\n(0.5 GT span, 5 frame min)",
                          width=6.0, height=1.4,
                          sub_text="abs(span_delta) > max(SPAN_TOL_FRAC=0.5 * gt_span, SPAN_TOL_MIN=5)?")
    add_leaf(ax, -4, y, "Skip pair,\ntry next combination",
              color="#fff3e0", width=4.5)
    add_edge(ax, -1, y, -1.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.7, 2, y - 1.2, label="no", label_xoff=0.35)

    # ----- Add to candidates -----
    y -= 1.8
    add_process_box(ax, 2, y,
                     "Pair passed checks - add to candidates",
                     color="#cfe2ff", width=6.5, height=0.85,
                     sub_text="candidates.append((abs(start_delta), ai, gi, start_delta, span_delta))")
    add_edge(ax, 2, y - 0.45, 2, y - 1.0)

    # ----- Sort candidates -----
    y -= 1.7
    add_process_box(ax, 2, y,
                     "Sort and match candidate list\nby abs(start_delta)",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="candidates.sort()")
    add_edge(ax, 2, y - 0.55, 2, y - 1.0)

    # ----- Greedy assign loop -----
    y -= 1.7
    add_process_box(ax, 2, y,
                     "Loop through candidates in priority order\n(best match first)",
                     color="#cfe2ff", width=7.0, height=1.1,
                     sub_text="for _, ai, gi, sd, pd_ in candidates:")
    y_assign_loop_top = y
    add_edge(ax, 2, y - 0.6, 2, y - 1.1)

    # ----- Already-claimed structural check -----
    y -= 1.7
    y_assign_skip = y
    add_decision_structural(ax, 2, y,
                              "Is the algo or GT\nalready matched?",
                              sub_text="ai in used_a OR gi in used_g?")
    add_leaf(ax, -4, y, "Skip\n(already claimed)",
              color="#fff3e0", width=4.0)
    add_edge(ax, -0.25, y, -2, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # ----- Claim both + emit matched -----
    y -= 1.8
    add_process_box(ax, 2, y,
                     "Mark this pair as a confirmed match\n(lock algo and GT)",
                     color="#d4edda", width=6.5, height=1.0,
                     sub_text="used_a.add(ai); used_g.add(gi); matched.add((ai, gi)); record tp_sd, tp_pd")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    # ===== STAGE 2: CONNECTED COMPONENTS =====
    y -= 2.5
    ax.text(-10, y, "STAGE 2\nConnected components",
            ha="left", va="center", fontsize=13, weight="bold",
            style="italic", color="#444444")
    add_process_box(ax, 2, y,
                     "Build overlap graph for\nevery algo/gt pair",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="for i, a in enumerate(algos): for j, g in enumerate(gts):")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "Do the two spans\noverlap at all?",
                              width=6.0, height=1.4,
                              sub_text="overlap_exists: NOT (a.end < g.start OR a.start > g.end)?")
    add_leaf(ax, -4, y, "Skip pair,\ntry next combination",
              color="#fff3e0", width=4.5)
    add_edge(ax, -1, y, -2, y, label="no", label_yoff=0.3)
    add_edge(ax, 2, y - 0.7, 2, y - 1.2, label="yes", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 2, y,
                     "Merge into same component (union-find)",
                     color="#cfe2ff", width=6.5, height=0.85,
                     sub_text="union(('a', i), ('g', j))")
    add_edge(ax, 2, y - 0.45, 2, y - 1.0)

    y -= 1.7
    add_process_box(ax, 2, y,
                     "Collect components:\neach = (algo set, GT set)",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="by_root = defaultdict(list); comps = [(a_idx, g_idx) for nodes per root]")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    # ===== STAGE 3: TOPOLOGY CLASSIFICATION =====
    y -= 2.0
    ax.text(-10, y, "STAGE 3\nTopology classification",
            ha="left", va="center", fontsize=13, weight="bold",
            style="italic", color="#444444")
    add_process_box(ax, 2, y,
                     "For each component, count algos and GTs",
                     color="#cfe2ff", width=7.0, height=0.9,
                     sub_text="for a_set, g_set in comps: na = len(a_set); ng = len(g_set)")
    add_edge(ax, 2, y - 0.5, 2, y - 1.0)

    # ----- Cardinality cascade -----
    # 1:0 stranded FP
    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "Algo with no\noverlapping GT?",
                              sub_text="na == 1 and ng == 0?")
    add_leaf(ax, 8, y, "FP",
              color="#f8d7da", width=3.5)
    add_edge(ax, 4.25, y, 5.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # 0:1 stranded FN
    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "GT with no\noverlapping algo?",
                              sub_text="na == 0 and ng == 1?")
    add_leaf(ax, 8, y, "FN",
              color="#f8d7da", width=3.5)
    add_edge(ax, 4.25, y, 5.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # 1:1 check
    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "1 algo to 1 GT?",
                              sub_text="na == 1 and ng == 1?")
    add_edge(ax, 4.25, y, 6.25, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)
    # 1:1 sub-decision: was (a, g) matched?
    add_decision_structural(ax, 9.5, y,
                              "Did this pair\npass tolerance?",
                              sub_text="(ai, gi) in matched_pairs?")
    add_leaf(ax, 14.0, y - 1.2, "TP",
              color="#d4edda", width=3.5)
    add_leaf(ax, 14.0, y + 1.2, "Tolerance Error",
              color="#fff3cd", width=4.0)
    add_edge(ax, 9.5, y + 0.6, 11.5, y + 1.2, label="no", label_xoff=0.5)
    add_edge(ax, 9.5, y - 0.6, 11.5, y - 1.2, label="yes", label_xoff=0.5)

    # 1:N MERGED
    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "Did the algo span\ncover multiple GTs?",
                              sub_text="na == 1 and ng >= 2?")
    add_leaf(ax, 8, y, "Merged",
              color="#fff3cd", width=4.0)
    add_edge(ax, 4.25, y, 5.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # N:1 FRAGMENTED
    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "Multiple algos\nspanning one GT?",
                              sub_text="na >= 2 and ng == 1?")
    add_leaf(ax, 8, y, "Fragmented (Split)",
              color="#fff3cd", width=4.5)
    add_edge(ax, 4.25, y, 5.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # ----- N:M decomposition -----
    y -= 1.7
    add_process_box(ax, 2, y,
                     "Complex cases.\nDecompose to individual events",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="N:M component (na >= 2 AND ng >= 2): decompose")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    y -= 1.7
    add_process_box(ax, 2, y,
                     "Pull out matched as TPs",
                     color="#d4edda", width=6.5, height=0.85,
                     sub_text="for ai, gi in local_matched: events.append('TP')")
    add_edge(ax, 2, y - 0.45, 2, y - 1.0)

    y -= 1.7
    add_process_box(ax, 2, y,
                     "For each algo with no match,\nlook for leftover GTs",
                     color="#cfe2ff", width=6.5, height=1.0,
                     sub_text="for ai in sorted(unmatched_a):")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    # Soft-pair structural: any unmatched GT with overlap?
    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "Does the algo overlap\nwith any unmatched GT?",
                              width=6.0, height=1.4,
                              sub_text="best_gi = argmax(overlap_frames); best_ol > 0?")
    add_leaf(ax, 8, y, "Tolerance Error",
              color="#fff3cd", width=4.0)
    add_edge(ax, 5, y, 6.0, y, label="yes", label_yoff=0.3)
    add_leaf(ax, -4, y, "FP",
              color="#f8d7da", width=3.5)
    add_edge(ax, -1, y, -2.25, y, label="no", label_yoff=0.3)
    add_edge(ax, 2, y - 0.7, 2, y - 1.2, label="iter", label_xoff=0.5)

    # Unmatched GT loop
    y -= 1.7
    add_process_box(ax, 2, y,
                     "Unclaimed GTs in this component -> FN",
                     color="#f8d7da", width=7.0, height=0.9,
                     sub_text="for gi in sorted(unmatched_g - soft_paired): events.append('FALSE_NEGATIVE')")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    # ===== STAGE 4: POST-MATCH FILTERS =====
    y -= 2.2
    ax.text(-10, y, "STAGE 4\nPost-match filters",
            ha="left", va="center", fontsize=13, weight="bold",
            style="italic", color="#444444")
    add_process_box(ax, 2, y,
                     "For each event, apply\nheadline filter rules",
                     color="#cfe2ff", width=7.0, height=1.0,
                     sub_text="count_filtered_metrics(results, min_span=MIN_REPORTED_SPAN=4, gt_boundaries)")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    # Status branches
    y -= 1.8
    add_decision_structural(ax, 2, y,
                              "Is this event a match?",
                              sub_text="r.status == 'matched'?")
    add_edge(ax, 4.25, y, 6.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 2.9, label="no", label_xoff=0.35)

    # Matched -> both spans >= 4?
    add_decision_tunable(ax, 9.5, y,
                          "Do both the algo and GT\nhave spans >= 4 frames?",
                          width=6.0, height=1.4,
                          sub_text="a_span >= MIN_REPORTED_SPAN=4 AND g_span >= MIN_REPORTED_SPAN=4? (TUNABLE)")
    add_edge(ax, 9.5, y - 0.7, 9.5, y - 1.5, label="yes", label_xoff=0.5)
    add_leaf(ax, 15.0, y, "Exclude",
              color="#fff3e0", width=3.5)
    add_edge(ax, 12.5, y, 13.25, y, label="no", label_yoff=0.3)
    add_leaf(ax, 9.5, y - 2.0, "TP",
              color="#d4edda", width=2.5)

    y -= 3.5
    add_decision_structural(ax, 2, y,
                              "Is this an FP event?",
                              sub_text="r.status == 'fp'?")
    add_edge(ax, 4.25, y, 6.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 4.7, label="no", label_xoff=0.35)

    # FP -> algo_span >= 4?
    add_decision_tunable(ax, 9.5, y,
                          "Algo span at least 4 frames?\n(FP check)",
                          width=5.5, height=1.3,
                          sub_text="a_span >= MIN_REPORTED_SPAN=4? (TUNABLE)")
    add_leaf(ax, 15.0, y, "Exclude",
              color="#fff3e0", width=3.5)
    add_edge(ax, 12.25, y, 13.25, y, label="no", label_yoff=0.3)
    add_edge(ax, 9.5, y - 0.65, 9.5, y - 1.4, label="yes", label_xoff=0.5)

    y -= 1.8
    add_decision_tunable(ax, 9.5, y,
                          "Did the algo start outside\nsegmentation (the tray)?",
                          width=6.5, height=1.4,
                          sub_text="is_outside_gt_segmentation(algo_start, gt_boundaries)?")
    add_leaf(ax, 15.5, y, "Exclude",
              color="#fff3e0", width=3.5)
    add_edge(ax, 12.75, y, 13.75, y, label="yes", label_yoff=0.3)
    add_leaf(ax, 9.5, y - 1.6, "FP",
              color="#f8d7da", width=2.5)
    add_edge(ax, 9.5, y - 0.7, 9.5, y - 1.3, label="no", label_xoff=0.5)

    # FN branch
    y -= 3.5
    add_decision_structural(ax, 2, y,
                              "Is this an FN event?",
                              sub_text="r.status == 'fn'?")
    add_edge(ax, 4.25, y, 6.5, y, label="yes", label_yoff=0.3)
    add_leaf(ax, -4, y, "Skip (unreachable)",
              color="#e0e0e0", width=4.0)
    add_edge(ax, -0.25, y, -2, y, label="no", label_yoff=0.3)

    # FN -> gt_span >= 4?
    add_decision_tunable(ax, 9.5, y,
                          "GT span at least 4 frames?",
                          width=5.5, height=1.2,
                          sub_text="g_span >= MIN_REPORTED_SPAN=4? (TUNABLE)")
    add_leaf(ax, 15.0, y, "Exclude",
              color="#fff3e0", width=3.5)
    add_edge(ax, 12.25, y, 13.25, y, label="no", label_yoff=0.3)
    add_leaf(ax, 9.5, y - 1.6, "FN",
              color="#f8d7da", width=2.5)
    add_edge(ax, 9.5, y - 0.6, 9.5, y - 1.3, label="yes", label_xoff=0.5)

    # ===== Legend =====
    legend_x = -13
    legend_y = 75
    ax.text(legend_x, legend_y + 0.7, "Legend",
            ha="left", va="center", fontsize=11, weight="bold")
    legend_items = [
        ("Input / process step", "#cfe2ff"),
        ("Tunable parameter (pink diamond)", "#f8d7da"),
        ("Structural / cardinality check (gray dashed diamond)", "#e0e0e0"),
        ("TP / matched outcome", "#d4edda"),
        ("MERGED / FRAGMENTED / TOLERANCE_ERROR", "#fff3cd"),
        ("FALSE_POSITIVE / FALSE_NEGATIVE", "#f8d7da"),
        ("Excluded / skipped event", "#fff3e0"),
    ]
    for i, (label, color) in enumerate(legend_items):
        ly = legend_y - i * 0.55
        ax.add_patch(Rectangle((legend_x - 0.05, ly - 0.18), 0.35, 0.35,
                                fc=color, ec="black", linewidth=1))
        ax.text(legend_x + 0.45, ly, label, ha="left", va="center", fontsize=8.5)

    # Save
    plt.tight_layout()
    png_path = OUT_DIR / "v804_matching_tree.png"
    svg_path = OUT_DIR / "v804_matching_tree.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
