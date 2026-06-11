"""Complete strict binary decision tree for v8.0.4 reach detection.

Built by reading every conditional in:
  - src/mousereach/reach/v8/features.py
  - src/mousereach/reach/v8/postprocess.py
  - src/mousereach/reach/v8/__init__.py::detect_reaches_v8

Includes ALL structural guards (NaN, OOB, empty-input, missing-bodypart,
div-by-zero) AND ALL tunable parameters (the 3 enabled booleans, all
thresholds, all sustain/min_span/distance params).

HGBM stays one black-box node (~12,800 learned splits not expanded).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).parent


# ---- Drawing helpers ----

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


def add_model_box(ax, x, y, text, sub_text=None, color="#fff3cd",
                   width=6.0, height=1.4):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          fc=color, ec="black", linewidth=2.0)
    ax.add_patch(box)
    if sub_text:
        ax.text(x, y + 0.22, text, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x, y - 0.30, sub_text, ha="center", va="center",
                fontsize=7.8, style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=10, weight="bold")


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
        # Plain-English (larger, bold) above the code-speak (smaller, italic)
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


def add_bypass_edge(ax, x_start, y_start, x_end, y_end, x_via, label):
    """Multi-segment dashed bypass arrow.
    Path: (x_start, y_start) -> (x_via, y_start) -> (x_via, y_end) -> (x_end, y_end)
    Used when an 'enabled? = no' outcome skips a whole section and
    rejoins the flow later. Dashed green to distinguish from main spine.
    """
    color = "darkgreen"
    ax.plot([x_start, x_via], [y_start, y_start],
            color=color, linewidth=1.4, linestyle="--")
    ax.plot([x_via, x_via], [y_start, y_end],
            color=color, linewidth=1.4, linestyle="--")
    ax.annotate("", xy=(x_end, y_end), xytext=(x_via, y_end),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=color,
                                  linestyle="--"))
    label_y = (y_start + y_end) / 2
    ax.text(x_via + 0.4, label_y, label,
            ha="left", va="center", fontsize=8.5,
            weight="bold", color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="#e8f5e8", ec=color, linewidth=1))


def add_loop_edge(ax, x1, y1, x2, y2, label):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", lw=1.3, color="darkorange",
                            linestyle="--",
                            connectionstyle="arc3,rad=-0.45",
                            mutation_scale=12)
    ax.add_patch(arr)
    mx = (x1 + x2) / 2 - 1.5
    my = (y1 + y2) / 2 + 0.2
    ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
            weight="bold", color="darkorange", style="italic",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#fff3e0", ec="darkorange", linewidth=1))


# ---- Build figure ----

def main():
    fig, ax = plt.subplots(figsize=(28, 60))
    ax.set_xlim(-14, 18)
    ax.set_ylim(-5, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(2, 79.3,
            "v8.0.4 Reach Detection: COMPLETE Strict Binary Decision Tree",
            ha="center", va="center", fontsize=18, weight="bold")
    ax.text(2, 78.4,
            "Every conditional in features.py + postprocess.py + __init__.py. "
            "Pink = tunable, gray dashed = structural. HGBM remains one black box.",
            ha="center", va="center", fontsize=10, style="italic")

    # ===== FEATURE EXTRACTION SECTION =====
    y = 76.5
    add_process_box(ax, 2, y, "Input: Video DLC h5 file",
                     color="#e7d4f5", width=5.0)
    add_edge(ax, 2, y - 0.45, 2, y - 1.0)

    y -= 1.6
    add_process_box(ax, 2, y,
                     "extract_features:\n"
                     "For each of 18 BODYPARTS (loop)",
                     color="#cfe2ff", width=5.5, height=1.1)
    add_edge(ax, 2, y - 0.55, 2, y - 1.0)

    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "Bodypart's _x column\nin DLC dataframe?")
    add_leaf(ax, -4, y, "Fill all 14 features\nfor this bp with zeros",
              color="#fff3e0", width=4.0)
    add_edge(ax, -0.25, y, -2, y, label="no", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="yes", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 2, y,
                     "Compute x_smooth, y_smooth\n"
                     "(5-frame centered MA, min_periods=1)",
                     color="#cfe2ff", width=5.5)
    add_edge(ax, 2, y - 0.45, 2, y - 1.1)

    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "_centered_diff:\n"
                              "n_frames >= 2*DT+1=5?")
    add_leaf(ax, -4, y, "vx, vy, ax, ay, dlk\nreturn zeros (too short)",
              color="#fff3e0", width=4.0)
    add_edge(ax, -0.25, y, -2, y, label="no", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="yes", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 2, y,
                     "Compute velocity (DT=2), accel, speed,\n"
                     "dlk, speed_max20, speed_max40, lk_min20",
                     color="#cfe2ff", width=5.5, height=1.0)
    add_edge(ax, 2, y - 0.55, 2, y - 1.0)

    y -= 1.5
    add_process_box(ax, 2, y,
                     "End bp loop.\n"
                     "Compute 153 pairwise distances on smoothed (x,y)",
                     color="#cfe2ff", width=5.5, height=1.0)
    add_edge(ax, 2, y - 0.55, 2, y - 1.0)

    # ===== HGBM =====
    y -= 1.7
    add_model_box(ax, 2, y,
                   "HGBM (learned BLACK BOX)",
                   "200 trees, max_depth=6, BSW b=1/w=0.8\n"
                   "~12,800 learned splits not expanded")
    add_edge(ax, 2, y - 0.75, 2, y - 1.3)

    y -= 1.7
    add_process_box(ax, 2, y, "proba per frame in [0, 1]", width=4.5)
    add_edge(ax, 2, y - 0.45, 2, y - 1.0)

    # ===== probabilities_to_reaches =====
    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "Is the probability\narray empty?",
                              sub_text="probabilities_to_reaches: n = len(proba) == 0?")
    add_leaf(ax, -4, y, "Return Zero Reaches",
              color="#f0f0f0", width=3.8)
    add_edge(ax, -0.25, y, -2.1, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 2, y,
                     "Marks frames as \"in-reach\" if above threshold (0.5)\n"
                     "Groups consecutive candidates into spans",
                     color="#cfe2ff", width=5.5, height=1.0,
                     sub_text="mask = proba > threshold (TUNABLE=0.5); _find_runs(mask)")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "No reaches?\n(corruption check)",
                              sub_text="_merge_close_runs: runs list is empty?")
    add_leaf(ax, -4, y, "Return zero reaches",
              color="#f0f0f0", width=3.8)
    add_edge(ax, -0.25, y, -2.1, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_decision_tunable(ax, 2, y,
                          "For 2 spans next to each other:\n"
                          "Check if gap is small enough to\n"
                          "be continuous reach (merge_gap=0)",
                          width=6.0, height=2.0,
                          sub_text="gap = run2.start - run1.end - 1; gap <= merge_gap=0?")
    add_leaf(ax, 8, y, "Merge spans",
              color="#fff3e0", width=3.5)
    add_edge(ax, 4.25, y, 6.1, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)
    y_merge_leaf = y  # save for the merge-back arrow

    y -= 1.8
    add_decision_tunable(ax, 2, y,
                          "Is the span at least\n3 frames long?",
                          sub_text="span = end - start + 1 >= min_span=3?")
    add_leaf(ax, 8, y, "Discard",
              color="#f8d7da", width=3.5)
    add_edge(ax, 4.25, y, 6.1, y, label="no", label_yoff=-0.45)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="yes", label_xoff=0.35)

    # Bypass arrow: "Merge spans" leaf rejoins the main spine just before
    # the min_span check. Shows merged spans continue to length-filter.
    add_edge(ax, 8, y_merge_leaf - 0.45, 4.25, y,
              label="merged span\nrejoins flow", label_xoff=0.5, label_yoff=0.2)

    # ===== leading_trim_enabled gate =====
    y -= 1.9
    y_lte = y  # save for the bypass arrow
    add_decision_tunable(ax, 2, y,
                          "Leading-trim enabled?",
                          sub_text="leading_trim_enabled? (TUNABLE bool, default True)")
    add_leaf(ax, 8, y, "Use current span as-is",
              color="#fff3e0", width=4.0)
    add_edge(ax, 4.25, y, 6, y, label="no", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="yes", label_xoff=0.35)

    # ===== Leading-trim loop =====
    y -= 1.8
    add_process_box(ax, 2, y,
                     "Compute paw confidence per frame\n"
                     "(mean of 4 hand points)",
                     color="#cfe2ff", width=5.5, height=1.0,
                     sub_text="compute_paw_mean_lk(dlc) -> n_frames array of mean lk across 4 hand keypoints")
    add_edge(ax, 2, y - 0.55, 2, y - 1.1)

    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "No spans?\n(corruption check)",
                              sub_text="trim_leading: not reaches? (empty list)")
    add_leaf(ax, -4, y, "Return zero reaches",
              color="#f0f0f0", width=3.8)
    add_edge(ax, -0.25, y, -2.1, y, label="yes", label_yoff=0.3)
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)

    # Loop entry (per reach)
    y -= 1.8
    add_process_box(ax, 2, y,
                     "Trimming Loop\n"
                     "Begin at start of reach span = s\n"
                     "(lk >= 0.6 for 3 frames)",
                     color="#cfe2ff", width=5.5, height=1.2,
                     sub_text="FOR EACH reach r: new_s = s; WHILE new_s <= e LOOP BODY")
    y_loop_top = y
    # Side annotation: explain the loop will continue until a diamond exits
    ax.text(6.0, y, "Loop continues until\na diamond exits",
            ha="left", va="center", fontsize=9, style="italic",
            weight="bold", color="#8B4513",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#fff3e0", ec="#8B4513", linewidth=1.2))
    add_edge(ax, 2, y - 0.65, 2, y - 1.1)

    # ITER STEP guards
    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "Check if 3-frame lookahead\nwindow is invalid",
                              sub_text="window_end = new_s + sustain_n (3);\nwindow_end > n_frames OR window_end > e+1?")
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)
    # yes -> exit loop (to right column)
    y_exit_1 = y

    y -= 1.7
    add_decision_structural(ax, 2, y,
                              "No confidence data?\n(corruption check)",
                              sub_text="window = paw_lk[new_s:window_end];\nAny NaN in window?")
    add_edge(ax, 2, y - 0.6, 2, y - 1.2, label="no", label_xoff=0.35)
    y_exit_2 = y

    y -= 1.7
    add_decision_tunable(ax, 2, y,
                          "Do any of the next 3 frames\nhave >= 0.60 confidence?",
                          sub_text="Any frame in window >=\nthreshold = 0.60? (TUNABLE)")
    add_edge(ax, -1.5, y - 0.5, -4.0, y - 1.0, label="no")
    y_exit_3 = y

    # "Advance new_s" leaf + loop back
    y_advance = y - 1.7
    add_leaf(ax, -4.5, y_advance,
              "Trim this frame (s),\nretry at (s+1)",
              color="#fff3e0", width=4.5)
    # Loop-back arrow now ends at the LEFT side of Diamond 1 (first iter check)
    # instead of the top of the Trimming Loop box -- matches the conceptual
    # "retry the lookahead check at the new start".
    add_loop_edge(ax, -4.5 - 1.9, y_advance, -0.25, y_loop_top - 1.7,
                   "LOOP BACK")

    # The "yes/exit" arrows from the 3 ITER STEP guards -> "exit loop" box (single)
    y_exit = y_exit_3 - 1.3
    add_process_box(ax, 7.5, y_exit, "Trim complete —\nmove to span check",
                     color="#e7d4f5", width=4.8, height=0.95)
    add_edge(ax, 4.25, y_exit_1, 6.5, y_exit + 0.4, label="yes", label_yoff=0.3)
    add_edge(ax, 4.25, y_exit_2, 6.5, y_exit + 0.4, label="yes", label_yoff=0.2)
    add_edge(ax, 4.25, y_exit_3, 6.5, y_exit + 0.4, label="yes", label_yoff=0.1)

    # Post-trim span check
    y = y_exit - 1.6
    add_decision_tunable(ax, 7.5, y,
                          "Is span >= 3 after trim?",
                          sub_text="e - new_s + 1 >= min_span = 3? (TUNABLE)",
                          width=4.5)
    add_edge(ax, 7.5, y_exit - 0.45, 7.5, y + 0.6)

    add_leaf(ax, 13.5, y, "Discard",
              color="#f8d7da", width=4.0)
    add_edge(ax, 9.75, y, 11.5, y, label="no", label_yoff=0.3)
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="yes", label_xoff=0.35)

    # ===== trailing_trim_enabled gate =====
    y -= 1.9
    y_tte = y
    add_decision_tunable(ax, 7.5, y,
                          "Is trailing-trim enabled?",
                          sub_text="trailing_trim_enabled? (TUNABLE bool, default True)")
    add_leaf(ax, 13.5, y, "Use current span as-is",
              color="#fff3e0", width=4.5)
    add_edge(ax, 9.75, y, 11.25, y, label="no", label_yoff=0.3)
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="yes", label_xoff=0.35)

    # Bypass arrow: if leading-trim disabled, skip its whole subtree and
    # rejoin at the "Trim complete" box (post-trim flow continues from there).
    add_bypass_edge(ax, 10, y_lte, 7.5, y_exit + 0.475, x_via=10,
                     label="bypass leading-trim")

    # ===== Trailing-trim (mirror) =====
    y -= 1.8
    add_decision_structural(ax, 7.5, y,
                              "No spans?\n(corruption check)",
                              sub_text="trim_trailing: not reaches? (empty list)")
    add_leaf(ax, 13.5, y, "Return zero reaches",
              color="#f0f0f0", width=4.0)
    add_edge(ax, 9.75, y, 11.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 7.5, y,
                     "Trimming Loop\n"
                     "Begin at end of reach span = e\n"
                     "(lk >= 0.6 for 3 frames)",
                     color="#cfe2ff", width=5.5, height=1.2,
                     sub_text="FOR EACH reach r: new_e = e; WHILE new_e >= s LOOP BODY")
    y_tt_loop_top = y
    add_edge(ax, 7.5, y - 0.65, 7.5, y - 1.1)

    y -= 1.7
    add_decision_structural(ax, 7.5, y,
                              "Check if 3-frame lookback\nwindow is invalid",
                              sub_text="window_start = new_e - sustain_n + 1;\nwindow_start < s?")
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="no", label_xoff=0.35)
    y_tt_exit_1 = y

    y -= 1.7
    add_decision_structural(ax, 7.5, y,
                              "Invalid 3-frame window?\n(corruption check)",
                              sub_text="window_start < 0 OR\nnew_e >= n_frames?")
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="no", label_xoff=0.35)
    y_tt_exit_2 = y

    y -= 1.7
    add_decision_structural(ax, 7.5, y,
                              "No confidence data?\n(corruption check)",
                              sub_text="window = paw_lk[window_start:new_e+1];\nAny NaN in window?")
    add_edge(ax, 7.5, y - 0.6, 7.5, y - 1.2, label="no", label_xoff=0.35)
    y_tt_exit_3 = y

    y -= 1.7
    add_decision_tunable(ax, 7.5, y,
                          "Do any of the previous 3 frames\nhave >= 0.60 confidence?",
                          sub_text="Any frame in window >=\nthreshold = 0.60? (TUNABLE)")
    add_edge(ax, 4.0, y - 0.5, 1.5, y - 1.0, label="no")
    y_tt_exit_4 = y

    y_tt_advance = y - 1.7
    add_leaf(ax, 1.0, y_tt_advance,
              "Trim this frame (e),\nretry at (e-1)",
              color="#fff3e0", width=4.5)
    # Loop-back arrow ends at the LEFT side of Diamond 1 (first iter check)
    # matching the leading-trim pattern.
    add_loop_edge(ax, 1.0 - 2.25, y_tt_advance, 5.25, y_tt_loop_top - 1.7,
                   "LOOP BACK")

    y_tt_exit = y_tt_exit_4 - 1.3
    add_process_box(ax, 13.0, y_tt_exit, "Trim complete —\ncheck span again",
                     color="#e7d4f5", width=4.8, height=0.95)
    add_edge(ax, 9.75, y_tt_exit_1, 12.0, y_tt_exit + 0.4, label="yes", label_yoff=0.3)
    add_edge(ax, 9.75, y_tt_exit_2, 12.0, y_tt_exit + 0.4, label="yes", label_yoff=0.2)
    add_edge(ax, 9.75, y_tt_exit_3, 12.0, y_tt_exit + 0.4, label="yes", label_yoff=0.1)
    add_edge(ax, 9.75, y_tt_exit_4, 12.0, y_tt_exit + 0.4, label="yes", label_yoff=0.0)

    y = y_tt_exit - 1.6
    add_decision_tunable(ax, 13.0, y,
                          "Is span >= 3 after trim?",
                          sub_text="new_e - s + 1 >= min_span = 3? (TUNABLE)",
                          width=4.5)
    add_edge(ax, 13.0, y_tt_exit - 0.45, 13.0, y + 0.6)

    add_leaf(ax, 7.5, y, "Discard",
              color="#f8d7da", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="no", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="yes", label_xoff=0.35)

    # ===== apex_split_enabled gate =====
    y -= 1.9
    y_apex_enabled = y
    add_decision_tunable(ax, 13.0, y,
                          "Is apex-split enabled?",
                          sub_text="apex_split_enabled? (TUNABLE bool, default True)")
    add_leaf(ax, 7.5, y, "Use current span as-is",
              color="#fff3e0", width=4.5)
    add_edge(ax, 10.75, y, 9.75, y, label="no", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="yes", label_xoff=0.35)

    # Bypass arrow: if trailing-trim disabled, skip its whole subtree and
    # rejoin at the trailing-trim "Trim complete" box (matches the
    # leading-trim bypass pattern of pointing at the post-trim convergence).
    add_bypass_edge(ax, 17, y_tte, 13.0, y_tt_exit + 0.475, x_via=17,
                     label="bypass trailing-trim")
    # NOTE: no bypass arrow from "Use current span as-is" (skip apex-split)
    # because apex-split is the final postprocess -- this leaf is genuinely
    # the end of the line for that span.

    # Apex section
    y -= 1.8
    add_process_box(ax, 13.0, y,
                     "Compute hand position relative to\n"
                     "BoxL/R per frame (merge fix)",
                     color="#cfe2ff", width=5.5, height=1.0,
                     sub_text="compute_hand_to_boxl_norm_pos: smooth (5-frame MA); norm_pos = dist(hand,BoxL)/max(dist(BoxL,BoxR), 1e-3)")
    add_edge(ax, 13.0, y - 0.55, 13.0, y - 1.1)

    y -= 1.8
    add_decision_structural(ax, 13.0, y,
                              "No spans?\n(corruption check)",
                              sub_text="apex_split: not reaches? (empty list)")
    add_leaf(ax, 7.5, y, "Return zero reaches",
              color="#f0f0f0", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 13.0, y,
                     "Check reach (r) spans (s, e)",
                     color="#cfe2ff", width=4.5, height=0.8,
                     sub_text="FOR EACH reach r in reaches; s, e = r.start_frame, r.end_frame")
    add_edge(ax, 13.0, y - 0.4, 13.0, y - 1.0)

    y -= 1.6
    add_decision_structural(ax, 13.0, y,
                              "Does the reach end OOB?\n(corruption check)",
                              sub_text="e >= n_frames? (span past video end)")
    add_leaf(ax, 7.5, y, "Keep reach as-is",
              color="#d4edda", width=3.8)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_decision_structural(ax, 13.0, y,
                              "Paw position invalid?\n(corruption check)",
                              sub_text="sig = norm_pos[s:e+1]; len(sig) < 3 OR any NaN in sig?")
    add_leaf(ax, 7.5, y, "Keep reach as-is",
              color="#d4edda", width=3.8)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_process_box(ax, 13.0, y,
                     "Find peaks in hand-to-BoxL distance signal\n"
                     "(prominence >= 0.12, >= 4 frames apart)",
                     color="#cfe2ff", width=6.0, height=1.2,
                     sub_text="peaks, _ = find_peaks(sig, prominence=0.12 (TUNABLE), distance=min_distance=4 (TUNABLE))")
    add_edge(ax, 13.0, y - 0.65, 13.0, y - 1.2)

    y -= 1.8
    add_decision_structural(ax, 13.0, y,
                              "Less than 2 peaks?",
                              sub_text="len(peaks) < 2? (0 or 1 peak found)")
    add_leaf(ax, 7.5, y, "Keep reach as-is\n(nothing to split)",
              color="#d4edda", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.8
    add_decision_tunable(ax, 13.0, y,
                          "Last peak in final 15% of span?\n(likely noise)",
                          sub_text="peak2_rel = peaks[-1]/(len(sig)-1) >= peak2_rel_max=0.85? (TUNABLE)")
    add_leaf(ax, 7.5, y, "Keep reach as-is",
              color="#d4edda", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    # Inner peak-pair loop
    y -= 1.9
    add_process_box(ax, 13.0, y,
                     "Check each pair of consecutive\n"
                     "peaks for a trough",
                     color="#cfe2ff", width=5.5, height=1.0,
                     sub_text="INNER LOOP: FOR i in 0..len(peaks)-2 (consecutive peak pairs)")
    add_edge(ax, 13.0, y - 0.55, 13.0, y - 1.1)

    y -= 1.7
    y_apex_inner_diamond = y  # save for loop-back + exit-loop arrows
    add_decision_structural(ax, 13.0, y,
                              "Pair too close?\n(< 2 frames apart)",
                              width=6.0, height=1.4,
                              sub_text="p2 - p1 < 2?\n(peaks adjacent; no room for trough)")
    add_leaf(ax, 7.0, y, "Skip pair, try next pair\nOR exit loop if last in span",
              color="#fff3e0", width=5.5)
    add_edge(ax, 10.0, y, 9.75, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.7, 13.0, y - 1.2, label="no", label_xoff=0.35)
    # Loop-back arrow: leaf -> LEFT side of diamond (inner-loop continue, next pair).
    # Drawn inline so the LOOP BACK label sits ABOVE the leaf, not inside it.
    _lb_arr = FancyArrowPatch((4.25, y_apex_inner_diamond + 0.2),
                               (10.0, y_apex_inner_diamond + 0.1),
                               arrowstyle="->", lw=1.3, color="darkorange",
                               linestyle="--",
                               connectionstyle="arc3,rad=-0.55",
                               mutation_scale=12)
    ax.add_patch(_lb_arr)
    ax.text(7.0, y_apex_inner_diamond + 1.2, "LOOP BACK",
            ha="center", va="center", fontsize=7.5,
            weight="bold", color="darkorange", style="italic",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#fff3e0", ec="darkorange", linewidth=1))

    y -= 1.8
    add_process_box(ax, 13.0, y,
                     "Measure this pair's dip, compare to best from\n"
                     "other pairs in this reach; keep deeper one\n"
                     "(dip = how far hand retracted toward BoxL\n"
                     "between the two extensions)",
                     color="#cfe2ff", width=7.0, height=1.8,
                     sub_text="between = sig[p1:p2+1]; t_local = argmin(between);\n"
                              "depth = max(sig[p1], sig[p2]) - sig[t_local]; track deepest")
    add_edge(ax, 13.0, y - 0.95, 13.0, y - 1.1)

    y -= 1.9
    y_depth_floor = y  # for exit-loop arrow from "Skip pair..." leaf
    add_decision_tunable(ax, 13.0, y,
                          "Did the hand pull back?\n"
                          "< 0.5 threshold (hand-to-BoxL distance)\n"
                          "or none found",
                          width=6.5, height=1.7,
                          sub_text="best_depth < depth_min = 0.5 OR\nbest_trough_frame is None? (TUNABLE depth_min)")
    add_leaf(ax, 7.5, y, "Keep reach as-is",
              color="#d4edda", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    # Exit-loop arrow from "Skip pair..." leaf bottom down to depth-floor
    # diamond top. Fires when the inner loop exhausts with no remaining
    # pairs to try. Routed left of the depth-tracking process box.
    _exit_x_via = 9.0
    _leaf_bottom_y = y_apex_inner_diamond - 0.4
    _target_top_y = y_depth_floor + 0.85
    ax.plot([7.0, _exit_x_via], [_leaf_bottom_y, _leaf_bottom_y],
            color="black", linewidth=1.4)
    ax.plot([_exit_x_via, _exit_x_via], [_leaf_bottom_y, _target_top_y],
            color="black", linewidth=1.4)
    ax.annotate("", xy=(13.0, _target_top_y), xytext=(_exit_x_via, _target_top_y),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))

    y -= 1.8
    add_decision_tunable(ax, 13.0, y,
                          "Would splitting create a half\n"
                          "shorter than 3 frames?",
                          width=6.0, height=1.4,
                          sub_text="half1_span < min_span=3 OR\nhalf2_span < min_span=3?")
    add_leaf(ax, 7.5, y, "Keep reach as-is",
              color="#d4edda", width=4.0)
    add_edge(ax, 10.75, y, 9.5, y, label="yes", label_yoff=0.3)
    add_edge(ax, 13.0, y - 0.6, 13.0, y - 1.2, label="no", label_xoff=0.35)

    y -= 1.6
    add_leaf(ax, 13.0, y,
              "Split reach at the deepest trough\n"
              "(across all peak pairs in this reach)",
              color="#fff3cd", width=6.0, height=1.1)

    # ===== Legend =====
    legend_x = -12.5
    legend_y_start = 76.0
    ax.text(legend_x, legend_y_start + 0.7, "Legend",
            ha="left", va="center", fontsize=12, weight="bold")
    legend_items = [
        ("Input / process step", "#cfe2ff"),
        ("Learned model (black box)", "#fff3cd"),
        ("Tunable decision (PINK)", "#f8d7da"),
        ("Structural decision (GRAY dashed)", "#e0e0e0"),
        ("DISCARD outcome", "#f8d7da"),
        ("KEEP single reach", "#d4edda"),
        ("SPLIT outcome", "#fff3cd"),
        ("Loop intermediate / fallthrough", "#fff3e0"),
        ("Empty result", "#f0f0f0"),
    ]
    for i, (label, color) in enumerate(legend_items):
        ly = legend_y_start - i * 0.55
        ax.add_patch(Rectangle((legend_x - 0.05, ly - 0.18), 0.35, 0.35,
                                fc=color, ec="black", linewidth=1))
        ax.text(legend_x + 0.45, ly, label, ha="left", va="center", fontsize=9)

    ax.text(legend_x, legend_y_start - 0.55 * 9 - 0.4,
            "Notes:\n"
            "- HGBM internals (200 trees x ~64 leaves) not\n"
            "  expanded (~12,800 splits would be unreadable)\n"
            "- Trim loops: each ITER STEP has 3-4 binary\n"
            "  guards. All 'yes' exits converge to one\n"
            "  EXIT LOOP box (the 3->1 convergence is the\n"
            "  one place strict binary tree shape relaxes)\n"
            "- Orange dashed arrows = loop-back cycles\n"
            "  (also not strict tree shape)\n"
            "- find_peaks call uses prominence AND distance\n"
            "  parameters to filter peaks before returning\n"
            "- All bool 'enabled' switches at start of each\n"
            "  postprocess section are real tunable knobs",
            ha="left", va="top", fontsize=8.5, style="italic")

    plt.tight_layout()
    png_path = OUT_DIR / "v804_complete_binary_tree.png"
    svg_path = OUT_DIR / "v804_complete_binary_tree.svg"
    plt.savefig(png_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
