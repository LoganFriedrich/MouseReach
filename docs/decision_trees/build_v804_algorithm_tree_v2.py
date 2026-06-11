"""v2: v8.0.4 algorithm decision tree with structural checks added.

Adds the category-2 structural decisions: NaN handling, boundary checks,
peak-detection internals, etc. The HGBM stays as a single black-box node
(category 3 not expanded; too many learned splits to be readable).

Outputs:
  - v804_algorithm_decision_tree_v2.png
  - v804_algorithm_decision_tree_v2.svg
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import numpy as np

OUT_DIR = Path(__file__).parent


# ---- Drawing helpers ----

def add_process_box(ax, x, y, text, color="#cfe2ff", width=4.0, height=0.9,
                     fontsize=9.5, weight="bold"):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            weight=weight)


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


def add_struct_note(ax, x, y, text, color="#f0f0f0", width=4.0, height=0.7,
                     fontsize=7.5):
    """Smaller hexagonal-ish box for structural (non-tunable) checks."""
    pts = np.array([
        [x - width/2 + 0.25, y + height/2],
        [x + width/2 - 0.25, y + height/2],
        [x + width/2, y],
        [x + width/2 - 0.25, y - height/2],
        [x - width/2 + 0.25, y - height/2],
        [x - width/2, y],
    ])
    poly = Polygon(pts, fc=color, ec="dimgray", linewidth=1.0,
                    linestyle="--")
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            style="italic", color="#333333")


def add_edge(ax, x1, y1, x2, y2, label=None, label_xoff=0.0, label_yoff=0.0,
              style="solid", color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=color,
                                  linestyle=style))
    if label:
        mx = (x1 + x2) / 2 + label_xoff
        my = (y1 + y2) / 2 + label_yoff
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                weight="bold", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="darkblue", linewidth=1))


def add_struct_edge(ax, x1, y1, x2, y2):
    """Dashed gray line connecting a structural note to its tunable decision."""
    ax.plot([x1, x2], [y1, y2], color="dimgray", linewidth=1.0,
            linestyle="--", zorder=1)


# ---- Build figure ----

def main():
    fig, ax = plt.subplots(figsize=(20, 26))
    ax.set_xlim(-12, 10)
    ax.set_ylim(-1, 32)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(-1, 31.3, "v8.0.4 Reach Detection Algorithm (full structural + tunable view)",
            ha="center", va="center", fontsize=17, weight="bold")
    ax.text(-1, 30.3,
            "Tunable decisions (pink/green/red boxes) and structural decisions "
            "(dashed gray, not tunable). HGBM is one black-box node.",
            ha="center", va="center", fontsize=10, style="italic")

    # Layout columns:
    #   x = -8: structural notes column
    #   x = -1: main pipeline chain
    #   x = +5: decision outcome leaves (no branches)
    MAIN_X = -1
    STRUCT_X = -8
    LEAF_X = 5

    # ===== TOP: Input + features + HGBM =====
    y = 28.8
    add_process_box(ax, MAIN_X, y, "Input: Video DLC h5 file",
                     color="#e7d4f5")

    y -= 1.8
    add_process_box(ax, MAIN_X, y,
                     "extract_features\n405 features/frame (18 bp x 14 + 153 dists)")
    # Structural notes for feature extraction
    add_struct_note(ax, STRUCT_X, y + 0.35,
                     "Missing bodypart -> zero out\nall its 14 features")
    add_struct_note(ax, STRUCT_X, y - 0.45,
                     "Smoothing: 5-frame centered MA\nVelocity: centered DT=2 finite diff")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.0, y)

    y -= 2.3
    add_model_box(ax, MAIN_X, y,
                   "HGBM (learned model, BLACK BOX)",
                   sub_text="HistGradientBoostingClassifier, 200 trees, max_depth=6\n"
                            "BSW b=1/w=0.8, ~12,800 learned splits not expanded")

    y -= 1.8
    add_process_box(ax, MAIN_X, y, "proba per frame in [0, 1]")

    # Edges in top section
    add_edge(ax, MAIN_X, 28.4, MAIN_X, 27.9)
    add_edge(ax, MAIN_X, 26.2, MAIN_X, 25.5)
    add_edge(ax, MAIN_X, 23.5, MAIN_X, 22.7)

    # ===== Per-frame threshold =====
    y -= 1.8
    add_decision_diamond(ax, MAIN_X, y,
                          "Per-frame:\nproba >= 0.5?\n(TUNABLE threshold)")
    add_edge(ax, MAIN_X, 20.3, MAIN_X, 19.6)

    add_leaf(ax, STRUCT_X, y, "Not a reach frame",
              color="#e0e0e0", width=3.0)
    add_edge(ax, MAIN_X - 2.0, y, STRUCT_X + 1.5, y, label="no",
              label_yoff=0.35)

    # ===== Form contiguous runs =====
    y -= 2.0
    add_process_box(ax, MAIN_X, y,
                     "Form contiguous runs of 'yes' frames\n(merge_gap = 0)",
                     width=4.5)
    add_struct_note(ax, STRUCT_X, y + 0.3,
                     "If merge_gap > 0:\nbridge sub-threshold gaps\nof length <= merge_gap")
    add_struct_note(ax, STRUCT_X, y - 0.55,
                     "Edge handling:\nruns touching frame 0 or last\nare clipped to video bounds")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.25, y)
    add_edge(ax, MAIN_X, 17.6, MAIN_X, 16.8,
              label="yes", label_xoff=0.4)

    # ===== Min span check =====
    y -= 1.9
    add_decision_diamond(ax, MAIN_X, y,
                          "Run length >= min_span = 3?\n(TUNABLE)")
    add_edge(ax, MAIN_X, 15.7, MAIN_X, 15.0)

    add_leaf(ax, LEAF_X, y, "DISCARD\n(too brief, < 3 frames)",
              color="#f8d7da", width=4.0)
    add_edge(ax, MAIN_X + 2.0, y, LEAF_X - 2.0, y, label="no",
              label_yoff=0.35)

    # ===== Leading-trim =====
    y -= 1.9
    add_process_box(ax, MAIN_X, y,
                     "Leading-trim LOOP:\nwalk new_s forward through low-lk frames",
                     color="#cfe2ff", width=5.5)
    add_edge(ax, MAIN_X, 13.8, MAIN_X, 13.0,
              label="yes", label_xoff=0.4)

    add_struct_note(ax, STRUCT_X, y + 0.55,
                     "Loop guard: lookahead window\n[new_s : new_s + N - 1] past end?\n-> stop, exit loop")
    add_struct_note(ax, STRUCT_X, y - 0.55,
                     "Loop guard: any NaN in\nlookahead window? -> stop")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.75, y)

    y -= 1.6
    add_decision_diamond(ax, MAIN_X, y,
                          "Lookahead window:\nAll paw_lk < T = 0.60?\n(TUNABLE)",
                          width=4.5)
    add_edge(ax, MAIN_X, 11.0, MAIN_X - 1.5, 10.55)
    ax.text(MAIN_X - 1.4, 11.0, "yes:\nadvance\nnew_s,\nloop",
            ha="center", va="center", fontsize=7,
            color="darkblue", weight="bold")
    # Loop arrow back to top of trim loop
    ax.annotate("", xy=(MAIN_X - 2.25, 11.8),
                xytext=(MAIN_X - 2.25, 10.55),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="darkblue",
                                  connectionstyle="arc3,rad=-0.3"))

    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "After leading-trim:\nspan >= 3?",
                          width=4.0)
    add_edge(ax, MAIN_X + 2.0, 11.0, MAIN_X + 0.75, y + 0.5,
              label="no:\ntrim done", label_xoff=1.0, label_yoff=0.4)
    add_edge(ax, MAIN_X, y - 0.6, MAIN_X, y - 1.2)

    add_leaf(ax, LEAF_X, y, "DISCARD\n(lead-trimmed away)",
              color="#f8d7da", width=4.0)
    add_edge(ax, MAIN_X + 2.0, y, LEAF_X - 2.0, y, label="no",
              label_yoff=0.35)

    # ===== Trailing-trim =====
    y -= 1.6
    add_process_box(ax, MAIN_X, y,
                     "Trailing-trim LOOP:\nwalk new_e backward",
                     color="#cfe2ff", width=5.0)
    add_struct_note(ax, STRUCT_X, y,
                     "Symmetric to leading-trim:\nsame NaN + boundary guards,\nT=0.60, N=3 (TUNABLE)")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.5, y)
    add_edge(ax, MAIN_X, 7.5, MAIN_X, 6.7,
              label="yes", label_xoff=0.4)

    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "After trailing-trim:\nspan >= 3?",
                          width=4.0)
    add_edge(ax, MAIN_X, y - 0.6, MAIN_X, y - 1.2)

    add_leaf(ax, LEAF_X, y, "DISCARD\n(trail-trimmed away)",
              color="#f8d7da", width=4.0)
    add_edge(ax, MAIN_X + 2.0, y, LEAF_X - 2.0, y, label="no",
              label_yoff=0.35)

    # ===== Apex-split: compute norm_pos =====
    y -= 1.6
    add_process_box(ax, MAIN_X, y,
                     "Compute hand-to-BoxL norm_pos\n+ smooth + find_peaks",
                     color="#cfe2ff", width=5.0)
    add_struct_note(ax, STRUCT_X, y + 0.55,
                     "norm_pos = dist(hand_centroid, BoxL)\n/ dist(BoxL, BoxR) (apparatus width)")
    add_struct_note(ax, STRUCT_X, y - 0.55,
                     "scipy.signal.find_peaks with\nprominence=0.12, min_distance=4")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.5, y)
    add_edge(ax, MAIN_X, 3.4, MAIN_X, 2.6,
              label="yes", label_xoff=0.4)

    # 2+ peaks?
    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "Found 2+ peaks?",
                          width=4.0)
    add_edge(ax, MAIN_X, y - 0.6, MAIN_X, y - 1.2)

    add_leaf(ax, LEAF_X, y, "KEEP single reach\n(no double-hump)",
              color="#d4edda", width=4.5)
    add_edge(ax, MAIN_X + 2.0, y, LEAF_X - 2.25, y, label="no",
              label_yoff=0.35)

    # Last peak position
    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "Last peak at < 0.85\nof span length?",
                          width=4.5)
    add_struct_note(ax, STRUCT_X, y,
                     "If 3+ peaks, examines each\nconsecutive pair; selects deepest\ntrough overall")
    add_struct_edge(ax, STRUCT_X + 2.0, y, MAIN_X - 2.25, y)
    add_edge(ax, MAIN_X, y - 0.6, MAIN_X, y - 1.2)

    add_leaf(ax, LEAF_X, y, "KEEP single\n(end-grab artifact)",
              color="#d4edda", width=4.5)
    add_edge(ax, MAIN_X + 2.25, y, LEAF_X - 2.25, y, label="no",
              label_yoff=0.35)

    # Trough depth
    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "Trough depth\n>= depth_min = 0.5?",
                          width=4.5)
    add_edge(ax, MAIN_X, y - 0.6, MAIN_X, y - 1.2)

    add_leaf(ax, LEAF_X, y, "KEEP single\n(trough too shallow)",
              color="#d4edda", width=4.5)
    add_edge(ax, MAIN_X + 2.25, y, LEAF_X - 2.25, y, label="no",
              label_yoff=0.35)

    # Halves
    y -= 1.7
    add_decision_diamond(ax, MAIN_X, y,
                          "Both halves\n>= min_span = 3?",
                          width=4.5)

    add_leaf(ax, LEAF_X, y, "KEEP single\n(halves too short)",
              color="#d4edda", width=4.5)
    add_edge(ax, MAIN_X + 2.25, y, LEAF_X - 2.25, y, label="no",
              label_yoff=0.35)

    # Final SPLIT
    y -= 1.6
    add_leaf(ax, MAIN_X, y, "SPLIT into 2 reaches\n(apex-split fires)",
              color="#fff3cd", width=5.0, height=0.95)
    add_edge(ax, MAIN_X, y + 1.4, MAIN_X, y + 0.5,
              label="yes", label_xoff=0.4)

    # Yes labels on the vertical "main spine" between Q4..Q7 (the apex-split chain)
    spine_segments = [
        (MAIN_X, 0.8, MAIN_X, 0.2, ""),  # last leg into SPLIT - already labeled
    ]
    # Add "yes" labels for the apex-split intermediate decisions
    for x_pos, y_top, y_bot in [
        (MAIN_X, 0.6 + 5*1.7, 0.6 + 4*1.7),  # not exact; computed below
    ]:
        pass  # we added them inline above

    # Legend
    legend_x = -10.5
    legend_y_start = 28.5
    ax.text(legend_x, legend_y_start + 0.7, "Legend",
            ha="left", va="center", fontsize=11, weight="bold")
    legend_items = [
        ("Input / process step", "#cfe2ff", "round"),
        ("Learned model (black box)", "#fff3cd", "round"),
        ("Tunable decision", "#f8d7da", "diamond"),
        ("Structural check (not tunable)", "#f0f0f0", "hex"),
        ("DISCARD outcome", "#f8d7da", "para"),
        ("KEEP single reach", "#d4edda", "para"),
        ("SPLIT outcome", "#fff3cd", "para"),
        ("Below threshold (not reach)", "#e0e0e0", "para"),
    ]
    for i, (label, color, _shape) in enumerate(legend_items):
        ly = legend_y_start - i * 0.55
        ax.add_patch(Rectangle((legend_x - 0.05, ly - 0.18), 0.35, 0.35,
                                fc=color, ec="black", linewidth=1))
        ax.text(legend_x + 0.45, ly, label, ha="left", va="center", fontsize=8.5)

    # Save
    plt.tight_layout()
    png_path = OUT_DIR / "v804_algorithm_decision_tree_v2.png"
    svg_path = OUT_DIR / "v804_algorithm_decision_tree_v2.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
