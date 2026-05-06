"""
Per-reach Sankey renderer v2.

Changes from the v1 renderer (in _run_notebooks.py::run_sankey):
  1. Triaged segments propagate triage to ALL reaches in the segment,
     not just the would-be-causal reach. Conservative: every reach in
     a triaged segment lands in 'triaged' on the algo side.
  2. All canonical categories are always rendered on both sides, even
     if count is 0 (zero-padded blocks). This ensures two Sankeys can
     be visually compared without layout shifts.
  3. A new 'wrong_reach_called' category on the algo side: when the
     algo committed the correct outcome class but attributed it to a
     different reach than GT's causal reach, that reach lands here.
  4. Fixed canonical ordering for all categories.

Canonical category sets:
  Both sides: retrieved, displaced_sa, untouched, abnormal_exception,
              miss, triaged, absent
  Algo-only:  wrong_reach_called

Usage::

    from mousereach.improvement.outcome.sankey_per_reach_v2 import (
        compute_per_reach_confusion_v2,
        render_per_reach_sankey_v2,
    )

    confusion = compute_per_reach_confusion_v2(
        gt_dir=..., algo_dir=...,
    )
    render_per_reach_sankey_v2(confusion, output_path=...)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical category ordering
# ---------------------------------------------------------------------------

# Categories present on BOTH sides.
# `untouched` is intentionally omitted -- per-reach labeling never assigns
# untouched to a reach (untouched is a segment-level concept; reaches in
# untouched segments are labeled `miss`).
_SHARED_CATEGORIES = [
    "retrieved",
    "displaced_sa",
    "abnormal_exception",
    "miss",
    "triaged",
    "absent",
]

# Additional categories on algo side only
_ALGO_ONLY_CATEGORIES = [
    "wrong_reach_called",
]

# Full algo-side order
ALGO_CATEGORY_ORDER = _SHARED_CATEGORIES + _ALGO_ONLY_CATEGORIES

# GT-side order (no wrong_reach_called)
GT_CATEGORY_ORDER = list(_SHARED_CATEGORIES)


# ---------------------------------------------------------------------------
# Colors (extend palette)
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "retrieved": "#4CAF50",           # Green
    "displaced_sa": "#FF9800",        # Orange
    "untouched": "#2196F3",           # Blue
    "abnormal_exception": "#616161",  # Dim gray
    "miss": "#90A4AE",               # Blue-gray
    "triaged": "#FFEB3B",            # Yellow
    "absent": "#424242",             # Dark gray
    "wrong_reach_called": "#E91E63", # Pink-red -- algo attributed to wrong reach
}

# Short abbreviations for flow labels
CATEGORY_ABBR = {
    "retrieved": "ret",
    "displaced_sa": "sa",
    "untouched": "unt",
    "abnormal_exception": "abn",
    "miss": "miss",
    "triaged": "tri",
    "absent": "abs",
    "wrong_reach_called": "wrc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collapse_displaced_outside(outcome: str) -> str:
    """Collapse displaced_outside -> displaced_sa."""
    if outcome == "displaced_outside":
        return "displaced_sa"
    return outcome


def _is_causal_reach(reach: dict, segment: dict) -> bool:
    """Check if a GT reach is the causal reach via IFR containment."""
    ifr = segment.get("interaction_frame")
    if ifr is None:
        return False
    sf = reach.get("start_frame")
    ef = reach.get("end_frame")
    if sf is None or ef is None:
        return False
    return sf <= ifr <= ef


def _normalize_algo_punted_outcome(algo_seg: dict) -> str:
    """Map algo-punted segments to abnormal_exception.

    When the outcome detector emits a non-untouched outcome but
    explicitly indicates in flag_reason that no reach could be
    attributed (e.g., tail-knockover), normalize to abnormal_exception.
    """
    _ABNORMAL_FLAG_PATTERNS = (
        "no reach data",
        "tail-knockover",
        "abnormal_exception",
    )
    raw = algo_seg.get("outcome", "unknown")
    if raw is None:
        return "unknown"
    if raw in ("untouched", "unknown", "uncertain", "abnormal_exception", "triaged"):
        return raw
    if (
        algo_seg.get("causal_reach_id") is None
        and bool(algo_seg.get("flagged_for_review", False))
    ):
        flag_reason = (algo_seg.get("flag_reason") or "").lower()
        if any(p in flag_reason for p in _ABNORMAL_FLAG_PATTERNS):
            return "abnormal_exception"
    return raw


TOUCHED_OUTCOMES = {"retrieved", "displaced_sa", "displaced_outside"}


def _label_gt_reach(reach: dict, gt_seg: dict) -> str:
    """Label a reach from the GT side.

    Causal reach (by IFR containment) -> GT outcome class.
    Non-causal reach -> 'miss'.
    """
    outcome = _collapse_displaced_outside(gt_seg.get("outcome", "miss"))
    if outcome in ("untouched", "uncertain", "unknown", None):
        return "miss"
    if outcome == "abnormal_exception":
        return "abnormal_exception" if _is_causal_reach(reach, gt_seg) else "miss"
    if outcome in ("retrieved", "displaced_sa"):
        return outcome if _is_causal_reach(reach, gt_seg) else "miss"
    return "miss"


def _label_algo_reach_v2(
    reach: dict,
    algo_seg: dict,
    gt_seg: dict,
    is_gt_causal: bool,
) -> str:
    """Label a reach from the algo side (v2 rules).

    Change 1: triaged segments -> ALL reaches get 'triaged' (not just causal).
    Change 3: wrong_reach_called -> committed touched segment where the algo
              outcome matches GT, but this reach is NOT the GT causal reach
              AND this is the reach the algo would have called causal.
              Since the cascade uses GT reaches (no causal_reach_id), we
              detect wrong_reach_called differently: if the algo committed
              a touched outcome on this segment, and the algo seg has an
              interaction_frame that falls inside THIS reach but NOT inside
              the GT causal reach's window, this reach is wrong_reach_called.
    """
    outcome = _normalize_algo_punted_outcome(algo_seg)
    outcome = _collapse_displaced_outside(outcome)
    gt_outcome = _collapse_displaced_outside(gt_seg.get("outcome", "miss"))

    # TRIAGED: all reaches in a triaged segment get 'triaged'
    if outcome == "triaged":
        return "triaged"

    # UNTOUCHED / abstention -> miss
    if outcome in ("untouched", "uncertain", "unknown"):
        return "miss"

    # ABNORMAL_EXCEPTION: only the GT causal reach gets this label
    if outcome == "abnormal_exception":
        return "abnormal_exception" if is_gt_causal else "miss"

    # TOUCHED committed outcomes (retrieved, displaced_sa)
    if outcome in ("retrieved", "displaced_sa"):
        if is_gt_causal:
            return outcome
        # Check for wrong_reach_called: algo committed the correct class
        # but may have attributed to a different reach. Detect by checking
        # if the algo's interaction_frame falls inside this reach's window.
        algo_ifr = algo_seg.get("interaction_frame")
        if algo_ifr is not None and gt_outcome == outcome:
            sf = reach.get("start_frame")
            ef = reach.get("end_frame")
            if sf is not None and ef is not None and sf <= algo_ifr <= ef:
                # Algo's IFR is inside this reach but this reach is NOT the
                # GT causal reach -> wrong reach was called
                return "wrong_reach_called"
        return "miss"

    return "miss"


# ---------------------------------------------------------------------------
# Compute per-reach confusion matrix (v2)
# ---------------------------------------------------------------------------

def compute_per_reach_confusion_v2(
    gt_dir: Path,
    algo_dir: Path,
    video_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute per-reach confusion matrix with v2 rules.

    Uses GT reach windows (the cascade's operating mode). For each GT
    reach, labels both GT and algo sides. Also detects FN/FP reaches
    via the existing reach-matching infrastructure if available, but
    falls back to GT-only iteration when the algo has no separate
    reach detector output.

    Parameters
    ----------
    gt_dir : Path
        Directory with ``*_unified_ground_truth.json`` files.
    algo_dir : Path
        Directory with ``*_pellet_outcomes.json`` files.
    video_ids : list of str, optional
        Explicit list; auto-discovers from GT if None.

    Returns
    -------
    dict with keys:
        n_reaches_universe, confusion_matrix, gt_totals, algo_totals,
        gt_categories, algo_categories
    """
    gt_dir = Path(gt_dir)
    algo_dir = Path(algo_dir)

    if video_ids is None:
        gt_files = sorted(gt_dir.glob("*_unified_ground_truth.json"))
        video_ids = [f.stem.replace("_unified_ground_truth", "") for f in gt_files]

    confusion: Dict[str, int] = defaultdict(int)
    n_universe = 0

    for vid in video_ids:
        gt_path = gt_dir / f"{vid}_unified_ground_truth.json"
        algo_path = algo_dir / f"{vid}_pellet_outcomes.json"
        if not gt_path.exists() or not algo_path.exists():
            continue

        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_segments = {
            s["segment_num"]: s
            for s in gt_data.get("outcomes", {}).get("segments", [])
        }
        gt_reaches = gt_data.get("reaches", {}).get("reaches", [])

        algo_data = json.loads(algo_path.read_text(encoding="utf-8"))
        algo_segs = {
            s["segment_num"]: s for s in algo_data.get("segments", [])
        }

        for reach in gt_reaches:
            seg_num = reach.get("segment_num")
            if seg_num is None:
                continue
            gt_seg = gt_segments.get(seg_num)
            algo_seg = algo_segs.get(seg_num)
            if gt_seg is None or algo_seg is None:
                continue

            gl = _label_gt_reach(reach, gt_seg)
            is_gt_causal = _is_causal_reach(reach, gt_seg)
            al = _label_algo_reach_v2(reach, algo_seg, gt_seg, is_gt_causal)

            confusion[f"{gl}__{al}"] += 1
            n_universe += 1

    # Compute totals per category on each side
    gt_totals: Dict[str, int] = defaultdict(int)
    algo_totals: Dict[str, int] = defaultdict(int)
    for key, count in confusion.items():
        gt_cls, algo_cls = key.split("__")
        gt_totals[gt_cls] += count
        algo_totals[algo_cls] += count

    return {
        "n_reaches_universe": n_universe,
        "confusion_matrix": dict(confusion),
        "gt_totals": dict(gt_totals),
        "algo_totals": dict(algo_totals),
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_per_reach_sankey_v2(
    confusion_data: Dict[str, Any],
    output_path: Path,
    title_suffix: str = "",
    dpi: int = 300,
) -> None:
    """Render a per-reach Sankey diagram with v2 rules.

    All canonical categories are shown on both sides (zero-padded).
    wrong_reach_called appears on algo side only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath

    cm = confusion_data["confusion_matrix"]
    n_universe = confusion_data["n_reaches_universe"]

    # Build flow data
    flows: List[Tuple[str, str, int]] = []
    for key, count in cm.items():
        parts = key.split("__")
        if len(parts) == 2:
            flows.append((parts[0], parts[1], count))

    # Determine which categories are present
    gt_cats_present = set()
    algo_cats_present = set()
    for gt, algo, count in flows:
        gt_cats_present.add(gt)
        algo_cats_present.add(algo)

    # Build ordered category lists, always including all canonical ones
    # GT side: all shared categories, zero-padded
    gt_categories = list(GT_CATEGORY_ORDER)
    # Add any unexpected categories that appeared in the data
    for c in sorted(gt_cats_present):
        if c not in gt_categories:
            gt_categories.append(c)

    # Algo side: all shared + algo-only, zero-padded
    algo_categories = list(ALGO_CATEGORY_ORDER)
    for c in sorted(algo_cats_present):
        if c not in algo_categories:
            algo_categories.append(c)

    # Compute totals (zero-padded)
    gt_totals: Dict[str, int] = {c: 0 for c in gt_categories}
    algo_totals: Dict[str, int] = {c: 0 for c in algo_categories}
    for gt, algo, count in flows:
        gt_totals[gt] = gt_totals.get(gt, 0) + count
        algo_totals[algo] = algo_totals.get(algo, 0) + count

    # Per-class precision on the algo side: of reaches the algo placed in
    # block X, what fraction were actually X per GT? (For meaningful blocks
    # only; triaged / wrong_reach_called / absent / untouched have no
    # "correct" baseline so precision is undefined.)
    algo_correct: Dict[str, int] = {c: 0 for c in algo_categories}
    for gt, algo, count in flows:
        if gt == algo:
            algo_correct[algo] = algo_correct.get(algo, 0) + count
    PRECISION_CLASSES = {"retrieved", "displaced_sa", "untouched",
                         "abnormal_exception", "miss"}

    # Sort flows by canonical order
    gt_order = {o: i for i, o in enumerate(gt_categories)}
    algo_order = {o: i for i, o in enumerate(algo_categories)}
    flows.sort(key=lambda x: (gt_order.get(x[0], 99), algo_order.get(x[1], 99)))

    total_reaches = sum(c for _, _, c in flows)

    # --- Layout ---
    figsize = (14, 11)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bar_width = 0.10
    gap = 0.018
    min_block_height = 0.018  # minimum visible height for zero-count blocks
    left_x = 0.20
    right_x = 0.80
    y_start = 0.93
    usable_height = 0.84

    # Compute total visual height needed including gaps and min-heights
    n_gt_cats = len(gt_categories)
    n_algo_cats = len(algo_categories)

    # For positioning: compute proportional heights but with min_block_height
    # for zero-count categories
    def _compute_positions(categories, totals, n_cats):
        """Compute (y_top, y_bottom) for each category."""
        # Total count for scaling
        total_count = sum(totals.get(c, 0) for c in categories)
        # Number of zero-count categories
        n_zero = sum(1 for c in categories if totals.get(c, 0) == 0)
        n_nonzero = n_cats - n_zero
        # Reserve space for gaps and zero-count blocks
        total_gap = gap * (n_cats - 1)
        zero_space = min_block_height * n_zero
        available = usable_height - total_gap - zero_space
        if available < 0.1:
            available = 0.1

        y_scale = available / total_count if total_count > 0 else 0

        positions = {}
        y_cursor = y_start
        for cat in categories:
            count = totals.get(cat, 0)
            if count == 0:
                h = min_block_height
            else:
                h = count * y_scale
            positions[cat] = (y_cursor, y_cursor - h)
            y_cursor -= h + gap
        return positions

    gt_positions = _compute_positions(gt_categories, gt_totals, n_gt_cats)
    algo_positions = _compute_positions(algo_categories, algo_totals, n_algo_cats)

    # Draw GT bars (left)
    for cat in gt_categories:
        y_top, y_bot = gt_positions[cat]
        h = y_top - y_bot
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        count = gt_totals.get(cat, 0)
        alpha = 1.0 if count > 0 else 0.25
        rect = plt.Rectangle(
            (left_x - bar_width / 2, y_bot), bar_width, h,
            facecolor=color, edgecolor="white", linewidth=0.5,
            alpha=alpha, transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(rect)
        if cat == "absent":
            label = f"{cat}\n({count})\n(algo only)"
        elif cat == "triaged":
            label = f"{cat}\n({count})\n(GT n/a)"
        else:
            label = f"{cat}\n({count})"
        ax.text(
            left_x - bar_width / 2 - 0.02, (y_top + y_bot) / 2,
            label, ha="right", va="center", fontsize=8, fontweight="bold",
            color="#333333" if count > 0 else "#999999",
            transform=ax.transAxes,
        )

    # Draw algo bars (right)
    for cat in algo_categories:
        y_top, y_bot = algo_positions[cat]
        h = y_top - y_bot
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        count = algo_totals.get(cat, 0)
        alpha = 1.0 if count > 0 else 0.25
        rect = plt.Rectangle(
            (right_x - bar_width / 2, y_bot), bar_width, h,
            facecolor=color, edgecolor="white", linewidth=0.5,
            alpha=alpha, transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(rect)
        # Build label with per-class precision % when meaningful
        if cat in PRECISION_CLASSES and count > 0:
            corr = algo_correct.get(cat, 0)
            pct = 100.0 * corr / count
            label = f"{cat}\n({count})\n{pct:.1f}% prec"
        elif cat == "triaged":
            label = f"{cat}\n({count})\n(for review)"
        elif cat == "wrong_reach_called":
            label = f"{cat}\n({count})\n(wrong by def.)"
        elif cat == "absent":
            label = f"{cat}\n({count})\n(GT only)"
        else:
            label = f"{cat}\n({count})"
        ax.text(
            right_x + bar_width / 2 + 0.02, (y_top + y_bot) / 2,
            label, ha="left", va="center", fontsize=8, fontweight="bold",
            color="#333333" if count > 0 else "#999999",
            transform=ax.transAxes,
        )

    # Draw flows
    gt_cursors = {c: gt_positions[c][0] for c in gt_categories}
    algo_cursors = {c: algo_positions[c][0] for c in algo_categories}

    placed_bboxes: List[Tuple[float, float, float, float]] = []

    for gt_out, algo_out, count in flows:
        if count == 0:
            continue

        # Get proportional height for this flow within each bar
        gt_total = gt_totals.get(gt_out, 1)
        algo_total = algo_totals.get(algo_out, 1)

        gt_bar_top, gt_bar_bot = gt_positions[gt_out]
        algo_bar_top, algo_bar_bot = algo_positions[algo_out]

        gt_bar_h = gt_bar_top - gt_bar_bot
        algo_bar_h = algo_bar_top - algo_bar_bot

        flow_h_gt = (count / gt_total) * gt_bar_h if gt_total > 0 else 0
        flow_h_algo = (count / algo_total) * algo_bar_h if algo_total > 0 else 0

        gt_top = gt_cursors[gt_out]
        gt_bot = gt_top - flow_h_gt
        gt_cursors[gt_out] = gt_bot

        algo_top = algo_cursors[algo_out]
        algo_bot = algo_top - flow_h_algo
        algo_cursors[algo_out] = algo_bot

        # Color: correct flows use source color softly, mismatches are bolder
        if gt_out == algo_out:
            color = CATEGORY_COLORS.get(gt_out, "#CCCCCC")
            alpha = 0.4
        else:
            color = CATEGORY_COLORS.get(gt_out, "#666666")
            alpha = 0.55

        # Draw ribbon
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
        patch = mpatches.PathPatch(
            path, facecolor=color, alpha=alpha,
            edgecolor="none", transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(patch)

        # Label mismatch flows
        if gt_out != algo_out and count >= 1:
            gt_mid_y = (gt_top + gt_bot) / 2
            algo_mid_y = (algo_top + algo_bot) / 2

            def _ribbon_pos(t):
                h = 3.0 * t * t - 2.0 * t * t * t
                rx = x_left + t * (x_right - x_left)
                ry = gt_mid_y + (algo_mid_y - gt_mid_y) * h
                return rx, ry

            label_text = "{0} {1}->{2}".format(
                count,
                CATEGORY_ABBR.get(gt_out, gt_out[:3]),
                CATEGORY_ABBR.get(algo_out, algo_out[:3]),
            )
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
                label_x, label_y = cand_x, cand_y
                chosen_bbox = cand_bbox
                while _intersects(chosen_bbox, placed_bboxes):
                    label_y += bbox_h
                    chosen_bbox = _bbox_at(label_x, label_y)

            placed_bboxes.append(chosen_bbox)

            label_color = CATEGORY_COLORS.get(gt_out, "#666666")
            ax.text(
                label_x, label_y, label_text,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=label_color, transform=ax.transAxes, zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.18", facecolor="white",
                    alpha=0.92, edgecolor=label_color, linewidth=0.7,
                ),
            )

    # Axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    title = f"Per-reach outcome flow (N={total_reaches})"
    if title_suffix:
        title += f" -- {title_suffix}"
    ax.text(
        0.5, 0.98, title,
        ha="center", va="top", fontsize=14, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        left_x, 0.95, "Ground Truth", ha="center", va="top", fontsize=11,
        fontweight="bold", transform=ax.transAxes,
    )
    ax.text(
        right_x, 0.95, "Algorithm", ha="center", va="top", fontsize=11,
        fontweight="bold", transform=ax.transAxes,
    )

    # Footer
    n_correct = sum(c for gt, al, c in flows if gt == al)
    footer = f"Correct flows: {n_correct}/{total_reaches}"
    if total_reaches > 0:
        footer += f" ({100 * n_correct / total_reaches:.1f}%)"
    ax.text(
        0.5, 0.02, footer, ha="center", va="bottom", fontsize=9,
        color="#555555", transform=ax.transAxes,
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0.15,
        facecolor="white",
    )
    plt.close(fig)
    logger.info("Saved per-reach Sankey v2: %s", output_path)
    print(f"Saved: {output_path}")
