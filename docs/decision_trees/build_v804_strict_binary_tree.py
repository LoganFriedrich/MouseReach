"""Strict binary decision tree for v8.0.4 reach detection.

Constraint: every internal node has exactly 2 children. Every leaf is a
unique terminal outcome. Includes BOTH tunable decisions AND structural
decisions (NaN/OOB guards, peak count check, norm_pos computability).

The HGBM stays one black-box node (not expanded).
The trim loops are represented as binary chains with the "advance + recurse"
outcome shown as a leaf that loops back via a labeled curved edge -- the
only non-tree elements in the diagram (cycles, marked clearly).

Outputs:
  - v804_strict_binary_tree.png
  - v804_strict_binary_tree.svg
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).parent


# ---- Drawing helpers ----

def add_process_box(ax, x, y, text, color="#cfe2ff", width=4.5, height=0.85):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=1.4)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold")


def add_model_box(ax, x, y, text, sub_text=None, color="#fff3cd",
                   width=5.5, height=1.3):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=2.0)
    ax.add_patch(box)
    if sub_text:
        ax.text(x, y + 0.22, text, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x, y - 0.28, sub_text, ha="center", va="center",
                fontsize=8, style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=10, weight="bold")


def add_decision_tunable(ax, x, y, text, width=4.2, height=1.15):
    """Pink diamond for tunable parameter decisions."""
    pts = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x, y - height/2],
        [x - width/2, y],
    ])
    diamond = Polygon(pts, fc="#f8d7da", ec="black", linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.3)


def add_decision_structural(ax, x, y, text, width=4.2, height=1.15):
    """Gray diamond with dashed border for structural (non-tunable) decisions."""
    pts = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x, y - height/2],
        [x - width/2, y],
    ])
    diamond = Polygon(pts, fc="#e0e0e0", ec="dimgray", linewidth=1.3,
                       linestyle="--")
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.0,
            style="italic")


def add_leaf(ax, x, y, text, color, width=3.6, height=0.85):
    """Parallelogram leaf (terminal outcome)."""
    skew = 0.3
    pts = np.array([
        [x - width/2 + skew, y - height/2],
        [x + width/2, y - height/2],
        [x + width/2 - skew, y + height/2],
        [x - width/2, y + height/2],
    ])
    poly = Polygon(pts, fc=color, ec="black", linewidth=1.5)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, weight="bold")


def add_edge(ax, x1, y1, x2, y2, label=None, label_xoff=0.0, label_yoff=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    if label:
        mx = (x1 + x2) / 2 + label_xoff
        my = (y1 + y2) / 2 + label_yoff
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.5,
                weight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="darkblue", linewidth=0.8))


def add_loop_edge(ax, x1, y1, x2, y2, label):
    """Curved edge for loop-back (cycle - the one place strict binary tree breaks)."""
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", lw=1.4, color="darkorange",
                            linestyle="--",
                            connectionstyle="arc3,rad=-0.5",
                            mutation_scale=15)
    ax.add_patch(arr)
    mx = (x1 + x2) / 2 - 1.8
    my = (y1 + y2) / 2
    ax.text(mx, my, label, ha="center", va="center", fontsize=8,
            weight="bold", color="darkorange", style="italic",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="#fff3e0", ec="darkorange", linewidth=1))


# ---- Build figure ----

def build_trim_subtree(ax, root_x, root_y, label, threshold_label,
                        leaf_label_left):
    """Build the binary trim-loop subtree at given root coord.
    Returns the y coordinate of the bottom 'continue to next step' point.
    """
    # Header process box (top of trim section)
    add_process_box(ax, root_x, root_y, label,
                     color="#cfe2ff", width=5.0, height=0.85)
    y = root_y - 1.6
    # Iteration entry
    add_decision_structural(ax, root_x, y,
                              "ITER STEP:\nWindow past array bound?")
    add_edge(ax, root_x, root_y - 0.45, root_x, y + 0.6)

    # Yes -> exit loop
    y_exit_marker = y
    # No -> next check
    y -= 1.7
    add_decision_structural(ax, root_x, y,
                              "ITER STEP:\nNaN in window?")
    add_edge(ax, root_x, y_exit_marker - 0.6, root_x, y + 0.6,
              label="no", label_xoff=0.35)

    # Yes -> exit loop too
    y_exit_marker_2 = y
    # No -> next check (tunable)
    y -= 1.7
    add_decision_tunable(ax, root_x, y,
                          threshold_label)
    add_edge(ax, root_x, y_exit_marker_2 - 0.6, root_x, y + 0.6,
              label="no", label_xoff=0.35)

    y_exit_marker_3 = y
    # No (some high-lk found) -> exit loop too
    # Yes (all low-lk) -> advance + loop back
    y_advance = y - 1.8
    add_leaf(ax, root_x - 3.5, y_advance,
              "advance pointer\n+ re-enter ITER STEP",
              color="#fff3e0", width=3.8, height=0.95)
    add_edge(ax, root_x - 1.0, y - 0.6, root_x - 3.5 + 1.9, y_advance + 0.45,
              label="yes\n(advance + recurse)", label_xoff=-0.8, label_yoff=0.3)
    # Curved loop-back arrow
    add_loop_edge(ax, root_x - 3.5 - 1.9, y_advance,
                   root_x - 2.1, root_y - 1.6,
                   "LOOP BACK")

    # The 3 "yes/no" exits from the 3 ITER STEP decisions all converge to
    # "exit loop" -- which in a strict binary tree should be unique. To keep
    # the tree strictly binary, we use: each exit leads INDIVIDUALLY to the
    # post-loop check, but we draw a single post-loop check below all 3.
    # We acknowledge this is the one place the tree's strict shape relaxes
    # (3 paths -> 1 post-loop point).
    y_post = y_advance - 1.5
    add_process_box(ax, root_x + 4.0, y_post,
                     "Exit loop:\ntrimming complete",
                     color="#e7d4f5", width=3.5, height=0.85)
    # Connect the 3 "yes/exit" arrows
    add_edge(ax, root_x + 2.1, root_y - 1.6, root_x + 4.0, y_post + 0.5,
              label="yes (exit)", label_xoff=0.8, label_yoff=0.4)
    add_edge(ax, root_x + 2.1, root_y - 3.3, root_x + 4.0, y_post + 0.45,
              label="yes (exit)", label_xoff=0.6, label_yoff=0.2)
    add_edge(ax, root_x + 2.1, root_y - 5.0, root_x + 4.0, y_post + 0.45,
              label="no (high-lk\nfound, exit)", label_xoff=0.7, label_yoff=0.15)

    # Post-loop check: span >= 3?
    y_check = y_post - 1.5
    add_decision_tunable(ax, root_x + 4.0, y_check,
                          "Span >= min_span = 3\nafter trim?",
                          width=4.5)
    add_edge(ax, root_x + 4.0, y_post - 0.45, root_x + 4.0, y_check + 0.6)

    # No -> DISCARD leaf
    add_leaf(ax, root_x + 4.0 + 4.5, y_check,
              leaf_label_left, color="#f8d7da", width=3.7)
    add_edge(ax, root_x + 4.0 + 2.25, y_check, root_x + 4.0 + 4.5 - 1.9,
              y_check, label="no", label_yoff=0.35)

    return y_check, root_x + 4.0  # bottom of this subtree, x to continue from


def main():
    fig, ax = plt.subplots(figsize=(22, 38))
    ax.set_xlim(-12, 16)
    ax.set_ylim(-2, 48)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(2, 47.3, "v8.0.4 Reach Detection: Strict Binary Decision Tree",
            ha="center", va="center", fontsize=17, weight="bold")
    ax.text(2, 46.5,
            "Pink diamonds: tunable parameter decisions. "
            "Gray dashed diamonds: structural decisions. "
            "Orange curved arrows: loop-back (cycle, not pure tree).",
            ha="center", va="center", fontsize=10, style="italic")

    # ===== Top-of-pipeline boxes =====
    y = 45.0
    add_process_box(ax, 2, y, "Input: Video DLC h5 file",
                     color="#e7d4f5", width=4.5)
    y -= 1.5
    add_process_box(ax, 2, y, "extract_features (405-D / frame)\n"
                                "structural: handle missing bodyparts, NaN")
    y -= 1.7
    add_model_box(ax, 2, y, "HGBM (learned BLACK BOX)",
                   "200 trees, max_depth=6, BSW b=1/w=0.8\n"
                   "~12,800 learned splits not shown")
    y -= 1.5
    add_process_box(ax, 2, y, "proba per frame in [0, 1]")

    add_edge(ax, 2, 44.55, 2, 44.1)
    add_edge(ax, 2, 43.05, 2, 42.4)
    add_edge(ax, 2, 41.05, 2, 40.45)

    # ===== Per-frame threshold (decision 1) =====
    y -= 1.6
    add_decision_tunable(ax, 2, y,
                          "PER-FRAME:\nproba >= 0.5?\n(TUNABLE threshold)")
    add_edge(ax, 2, 39.3, 2, y + 0.6)

    add_leaf(ax, -3.5, y, "frame is\nNOT a reach frame",
              color="#e0e0e0", width=3.5)
    add_edge(ax, 2 - 2.1, y, -3.5 + 1.75, y, label="no", label_yoff=0.35)

    # Process: form runs
    y -= 1.8
    add_process_box(ax, 2, y, "Form contiguous runs of 'yes' frames\n"
                                "merge_gap=0 (TUNABLE): no bridging\n"
                                "structural: edge handling at video bounds",
                     color="#cfe2ff", width=5.5, height=1.1)
    add_edge(ax, 2, y + 1.15, 2, y + 0.55, label="yes", label_xoff=0.4)

    # ===== Run length check =====
    y -= 1.7
    add_decision_tunable(ax, 2, y,
                          "Run length\n>= min_span = 3?")
    add_edge(ax, 2, y + 1.05, 2, y + 0.6)

    add_leaf(ax, 8, y, "DISCARD\n(too brief)", color="#f8d7da", width=3.5)
    add_edge(ax, 2 + 2.1, y, 8 - 1.75, y, label="no", label_yoff=0.35)

    # ===== Leading-trim subtree =====
    y -= 2.0
    add_edge(ax, 2, y + 1.6, 2, y + 0.45, label="yes", label_xoff=0.4)
    y_after_lt, x_after_lt = build_trim_subtree(
        ax, 2, y,
        "LEADING-TRIM: walk new_s forward, find earliest\n"
        "frame where paw_lk indicates real reach start",
        "ITER STEP:\nAll paw_lk < T = 0.60\nin window?\n(TUNABLE)",
        "DISCARD\n(lead-trimmed away)"
    )

    # ===== Trailing-trim subtree =====
    y = y_after_lt - 2.2
    add_edge(ax, x_after_lt, y_after_lt - 0.6, x_after_lt, y + 0.45,
              label="yes", label_xoff=0.4)
    y_after_tt, x_after_tt = build_trim_subtree(
        ax, x_after_lt, y,
        "TRAILING-TRIM: walk new_e backward, find latest\n"
        "frame where paw_lk indicates real reach end",
        "ITER STEP:\nAll paw_lk < T = 0.60\nin window?\n(TUNABLE)",
        "DISCARD\n(trail-trimmed away)"
    )

    # ===== Apex-split section =====
    y = y_after_tt - 2.2
    add_edge(ax, x_after_tt, y_after_tt - 0.6, x_after_tt, y + 0.45,
              label="yes", label_xoff=0.4)
    add_process_box(ax, x_after_tt, y,
                     "Compute hand-to-BoxL norm_pos:\n"
                     "structural: smooth + scipy.find_peaks\n"
                     "with prominence=0.12, min_distance=4 (TUNABLE)",
                     color="#cfe2ff", width=5.5, height=1.1)

    # Decision: 2+ peaks?
    y -= 1.7
    add_decision_structural(ax, x_after_tt, y,
                              "Found 2+ peaks?\n(0-1 peaks = no double-hump)")
    add_edge(ax, x_after_tt, y + 1.15, x_after_tt, y + 0.6)

    add_leaf(ax, x_after_tt + 6, y,
              "KEEP single\n(no double-hump)",
              color="#d4edda", width=4.0)
    add_edge(ax, x_after_tt + 2.1, y, x_after_tt + 6 - 2.0, y,
              label="no", label_yoff=0.35)

    # Decision: 3+ peaks branch -> structural choice of pair
    y -= 1.8
    add_decision_structural(ax, x_after_tt, y,
                              "3+ peaks?\n(structural: pick\nconsecutive pair with\ndeepest trough)")
    add_edge(ax, x_after_tt, y + 1.15, x_after_tt, y + 0.6,
              label="yes", label_xoff=0.4)

    # Both no (exactly 2 peaks, use that pair) and yes (3+ peaks, pick pair)
    # both feed into next check. Strict binary tree: each branch leads to
    # the same next check. So we put the next check below.
    # To stay strictly binary we let "no" and "yes" both arrive at the
    # next decision visually; we annotate this is structural.

    # Last peak < 0.85?
    y -= 1.8
    add_decision_tunable(ax, x_after_tt, y,
                          "Last peak at < 0.85\nof span length?\n(TUNABLE peak2_rel_max)")
    add_edge(ax, x_after_tt, y + 1.05, x_after_tt, y + 0.6,
              label="continue",
              label_xoff=0.6, label_yoff=0.2)

    add_leaf(ax, x_after_tt + 6, y,
              "KEEP single\n(end-grab artifact)",
              color="#d4edda", width=4.0)
    add_edge(ax, x_after_tt + 2.1, y, x_after_tt + 6 - 2.0, y,
              label="no", label_yoff=0.35)

    # Trough depth
    y -= 1.8
    add_decision_tunable(ax, x_after_tt, y,
                          "Trough depth\n>= depth_min = 0.5?\n(TUNABLE)")
    add_edge(ax, x_after_tt, y + 1.05, x_after_tt, y + 0.6,
              label="yes", label_xoff=0.4)

    add_leaf(ax, x_after_tt + 6, y,
              "KEEP single\n(trough too shallow)",
              color="#d4edda", width=4.0)
    add_edge(ax, x_after_tt + 2.1, y, x_after_tt + 6 - 2.0, y,
              label="no", label_yoff=0.35)

    # Both halves >= 3?
    y -= 1.8
    add_decision_tunable(ax, x_after_tt, y,
                          "Both halves\n>= min_span = 3?")
    add_edge(ax, x_after_tt, y + 1.05, x_after_tt, y + 0.6,
              label="yes", label_xoff=0.4)

    add_leaf(ax, x_after_tt + 6, y,
              "KEEP single\n(halves too short)",
              color="#d4edda", width=4.0)
    add_edge(ax, x_after_tt + 2.1, y, x_after_tt + 6 - 2.0, y,
              label="no", label_yoff=0.35)

    # SPLIT outcome
    y -= 1.6
    add_leaf(ax, x_after_tt, y,
              "SPLIT into 2 reaches\n(apex-split fires)",
              color="#fff3cd", width=4.5, height=0.95)
    add_edge(ax, x_after_tt, y + 1.4, x_after_tt, y + 0.5,
              label="yes", label_xoff=0.4)

    # ===== Legend =====
    legend_x = -10.5
    legend_y_start = 44.5
    ax.text(legend_x, legend_y_start + 0.7, "Legend",
            ha="left", va="center", fontsize=12, weight="bold")
    legend_items = [
        ("Input / process step", "#cfe2ff", "round"),
        ("Learned model (black box)", "#fff3cd", "round"),
        ("Tunable decision (pink diamond)", "#f8d7da", "diamond"),
        ("Structural decision (gray dashed)", "#e0e0e0", "diamond_dashed"),
        ("DISCARD outcome", "#f8d7da", "para"),
        ("KEEP single reach", "#d4edda", "para"),
        ("SPLIT outcome", "#fff3cd", "para"),
        ("Below threshold (not reach)", "#e0e0e0", "para"),
        ("Loop intermediate", "#fff3e0", "para"),
    ]
    for i, (label, color, _shape) in enumerate(legend_items):
        ly = legend_y_start - i * 0.55
        ax.add_patch(Rectangle((legend_x - 0.05, ly - 0.18), 0.35, 0.35,
                                fc=color, ec="black", linewidth=1))
        ax.text(legend_x + 0.45, ly, label, ha="left", va="center", fontsize=8.5)

    ax.text(legend_x, legend_y_start - 0.55 * 9 - 0.3,
            "Note: Trim loops have 3 'exit'\n"
            "branches that all lead to the\n"
            "post-loop check. Drawn as 3 -> 1\n"
            "convergence (strict binary tree\n"
            "doesn't naturally express loops).\n\n"
            "Orange dashed arrow = loop back\n"
            "(the only cycle in the diagram).",
            ha="left", va="top", fontsize=8, style="italic")

    plt.tight_layout()
    png_path = OUT_DIR / "v804_strict_binary_tree.png"
    svg_path = OUT_DIR / "v804_strict_binary_tree.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
