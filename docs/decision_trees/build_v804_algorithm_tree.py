"""Render the v8.0.4 reach detection algorithm as a decision tree image.

Option A scope: full pipeline at the algorithm level. The HGBM model is
shown as a single black-box node (its 200 internal trees aren't expanded).
Per-frame decisions and per-span decisions are both included.

Outputs:
  - v804_algorithm_decision_tree.png
  - v804_algorithm_decision_tree.svg
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle
import numpy as np

OUT_DIR = Path(__file__).parent


# ---- Drawing helpers ----

def add_process_box(ax, x, y, text, color="#cfe2ff", width=4.0, height=0.9):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold")


def add_input_box(ax, x, y, text, color="#e7d4f5", width=3.5, height=0.8):
    add_process_box(ax, x, y, text, color=color, width=width, height=height)


def add_model_box(ax, x, y, text, sub_text=None, color="#fff3cd",
                   width=5.0, height=1.4):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=2.0)
    ax.add_patch(box)
    if sub_text:
        ax.text(x, y + 0.25, text, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x, y - 0.25, sub_text, ha="center", va="center",
                fontsize=8, style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold")


def add_decision_diamond(ax, x, y, text, color="#f8d7da", width=4.0, height=1.2):
    # Diamond shape using Polygon
    pts = np.array([
        [x, y + height/2],
        [x + width/2, y],
        [x, y - height/2],
        [x - width/2, y],
    ])
    diamond = Polygon(pts, fc=color, ec="black", linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5)


def add_leaf(ax, x, y, text, color="#d4edda", width=3.5, height=0.8):
    # Parallelogram for leaves
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


def add_edge(ax, x1, y1, x2, y2, label=None, label_xoff=0.0, label_yoff=0.0,
              label_side="mid"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    if label:
        if label_side == "mid":
            mx = (x1 + x2) / 2 + label_xoff
            my = (y1 + y2) / 2 + label_yoff
        elif label_side == "start":
            mx = x1 + (x2 - x1) * 0.25 + label_xoff
            my = y1 + (y2 - y1) * 0.25 + label_yoff
        elif label_side == "end":
            mx = x1 + (x2 - x1) * 0.75 + label_xoff
            my = y1 + (y2 - y1) * 0.75 + label_yoff
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                weight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="darkblue", linewidth=1))


# ---- Build figure ----

def main():
    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(-9, 9)
    ax.set_ylim(-1, 28)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(0, 27.3, "v8.0.4 Reach Detection Algorithm",
            ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0, 26.4,
            "Full decision tree, per-frame to per-span. The HGBM model "
            "is a learned ensemble (200 trees x depth 6) shown as one node.",
            ha="center", va="center", fontsize=10, style="italic")

    # ===== Pipeline section (top of figure) =====
    # Input
    y = 24.8
    add_input_box(ax, 0, y, "Input: Video DLC h5 file")

    # Feature extraction
    y -= 1.6
    add_process_box(ax, 0, y, "extract_features\n405 features per frame (18 bp x 14 + 153 dists)")

    # HGBM
    y -= 2.0
    add_model_box(ax, 0, y,
                   "HGBM (learned model, black box)",
                   sub_text="HistGradientBoostingClassifier\nmax_iter=200, lr=0.05, max_depth=6, BSW b=1/w=0.8")

    # Proba output
    y -= 1.8
    add_process_box(ax, 0, y, "proba per frame in [0, 1]")

    # Edges along the input -> model chain
    add_edge(ax, 0, 24.4, 0, 23.9)
    add_edge(ax, 0, 22.8, 0, 22.4)
    add_edge(ax, 0, 20.3, 0, 19.9)
    add_edge(ax, 0, 18.6, 0, 18.2)

    # ===== Per-frame decision: threshold =====
    y -= 1.9
    add_decision_diamond(ax, 0, y,
                          "Per-frame:\nproba >= 0.5?\n(threshold)")
    add_edge(ax, 0, 16.3, 0, 15.4)

    # No path: not a reach frame
    add_leaf(ax, -5.5, y, "Not a reach frame",
              color="#e0e0e0")
    add_edge(ax, -2.0, y, -3.75, y, label="no", label_xoff=0, label_yoff=0.4)

    # Form contiguous runs (synthesis from per-frame to per-span)
    y -= 2.0
    add_process_box(ax, 2.0, y,
                     "Form contiguous runs of\n'yes' frames per video\n(merge_gap = 0)")
    add_edge(ax, 0, 13.6, 1.0, 13.3, label="yes", label_xoff=0, label_yoff=0.3)

    # ===== Per-span decisions start =====
    y -= 1.8
    add_decision_diamond(ax, 2.0, y, "Run length\n>= min_span = 3?")
    add_edge(ax, 2.0, 11.7, 2.0, 10.9)

    # No -> DISCARD: brief
    add_leaf(ax, 7.0, y, "DISCARD: too brief\n(< 3 frames)", color="#f8d7da")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Leading-trim decision
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "After leading-trim\n(paw_lk T=0.60, N=3):\nspan >= 3?")
    add_edge(ax, 2.0, 9.9, 2.0, 9.1)

    add_leaf(ax, 7.0, y, "DISCARD: leading-trim\nate the whole span",
              color="#f8d7da")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Trailing-trim decision
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "After trailing-trim\n(paw_lk T=0.60, N=3):\nspan >= 3?")
    add_edge(ax, 2.0, 8.1, 2.0, 7.3)

    add_leaf(ax, 7.0, y, "DISCARD: trailing-trim\nate the whole span",
              color="#f8d7da")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # ===== Apex-split branch =====
    # 2+ peaks?
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "norm_pos:\n2+ peaks at prom >= 0.12?")
    add_edge(ax, 2.0, 6.3, 2.0, 5.5)

    add_leaf(ax, 7.0, y, "KEEP single reach\n(no double-hump)",
              color="#d4edda")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Last peak < 0.85?
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "Last peak at\n< 0.85 of span?")
    add_edge(ax, 2.0, 4.5, 2.0, 3.7)

    add_leaf(ax, 7.0, y, "KEEP single\n(end-grab artifact)",
              color="#d4edda")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Trough depth?
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "Trough depth\n>= 0.5?")
    add_edge(ax, 2.0, 2.7, 2.0, 1.9)

    add_leaf(ax, 7.0, y, "KEEP single\n(trough too shallow)",
              color="#d4edda")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Both halves >= 3?
    y -= 1.8
    add_decision_diamond(ax, 2.0, y,
                          "Both halves\n>= 3 frames?")

    add_leaf(ax, 7.0, y, "KEEP single\n(halves too short)",
              color="#d4edda")
    add_edge(ax, 4.0, y, 5.25, y, label="no", label_yoff=0.3)

    # Final SPLIT outcome
    y -= 1.7
    add_leaf(ax, 2.0, y, "SPLIT into 2 reaches\n(apex-split fires)",
              color="#fff3cd", width=4.5, height=0.9)
    add_edge(ax, 2.0, 0.7, 2.0, y + 0.45, label="yes", label_xoff=0.3, label_yoff=0.0)

    # "yes" labels on the vertical chain
    yes_segments = [(2.0, 11.7, 2.0, 10.9),
                     (2.0, 9.9, 2.0, 9.1),
                     (2.0, 8.1, 2.0, 7.3),
                     (2.0, 6.3, 2.0, 5.5),
                     (2.0, 4.5, 2.0, 3.7),
                     (2.0, 2.7, 2.0, 1.9)]
    for x1, y1, x2, y2 in yes_segments:
        my = (y1 + y2) / 2
        ax.text(x1 + 0.45, my, "yes", ha="left", va="center", fontsize=9,
                weight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="darkblue", linewidth=1))

    # Legend (top-right corner)
    legend_x = -7.5
    legend_y_start = 24.5
    ax.text(legend_x, legend_y_start + 0.7, "Legend",
            ha="left", va="center", fontsize=11, weight="bold")
    legend_items = [
        ("Input/process step", "#cfe2ff"),
        ("Learned model (black box)", "#fff3cd"),
        ("Decision (per-frame/span)", "#f8d7da"),
        ("DISCARD outcome", "#f8d7da"),
        ("KEEP single reach", "#d4edda"),
        ("SPLIT outcome", "#fff3cd"),
        ("Below threshold (not reach)", "#e0e0e0"),
    ]
    for i, (label, color) in enumerate(legend_items):
        ly = legend_y_start - i * 0.55
        ax.add_patch(Rectangle((legend_x - 0.05, ly - 0.18), 0.35, 0.35,
                                fc=color, ec="black", linewidth=1))
        ax.text(legend_x + 0.45, ly, label, ha="left", va="center", fontsize=8.5)

    # Save
    plt.tight_layout()
    png_path = OUT_DIR / "v804_algorithm_decision_tree.png"
    svg_path = OUT_DIR / "v804_algorithm_decision_tree.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
