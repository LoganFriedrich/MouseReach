"""
Canonical per-reach evaluation Sankeys: the algo AS RUN vs a reference.

============================================================================
THIS IS HOW YOU GENERATE EVALUATION RESULTS for how the algo is doing /
generalizing, measured PER REACH against a reference -- either ground truth
(``algo_vs_gt``) or human review alone (``algo_vs_review``). Do NOT hand-roll
a per-reach Sankey; use these entry points so every eval is produced the same,
directly-comparable way.
============================================================================

Both figures put the **raw algo output on the LEFT** and the **reference on the
RIGHT**; ribbons flow algo -> reference, so the *disagreement* ribbons ARE the
algo's error. Neither figure shows a "post-correction" panel -- once a human
correction is written into the results file it matches the reference by
construction, which measures nothing about the algo. We only ever show the algo
exactly as it ran vs the truth.

Rendering is the plain single-panel v1 style (only categories that actually
occur are drawn -- no zero-padded blocks), flipped to algo-left.

Entry points
------------
algo_vs_gt
    Reproduces the canonical algo-4 per-reach confusion
    (``mousereach.improvement.outcome.metrics.compute_per_reach_confusion``)
    for a set of ground-truthed videos and renders it algo-left / GT-right.
    Fed the 2026-07-03 LIVE run's own stored ``algo_outcomes/`` + ``algo_reaches/``
    it reproduces that run's numbers exactly -- just flipped.

algo_vs_review
    Over the videos reviewed with the Causal Review tool, per reach:
      * algo side comes straight from the reach-assignment labels
        (miss / causal_<outcome> / triaged);
      * reference side comes from the human review (which reach was causal +
        what the human said the outcome was);
      * TRIAGE is a per-SEGMENT decision (the algo holds the whole segment), so
        each triaged segment contributes ONE 'triaged' mark, RE-ROUTED to the
        human's resolution -- NOT one per reach (labeling every reach 'triaged'
        is the ~10x-inflation anti-pattern documented in assignment/AGENTS.md);
      * 'absent' is DROPPED, except when the human marked a causal reach that
        overlaps NO algo reach (a reach the algo missed entirely) -> that one
        flows absent -> <human class>.

CLI
---
    python -m mousereach.improvement.per_reach_sankey_eval algo-vs-gt
    python -m mousereach.improvement.per_reach_sankey_eval algo-vs-review

Run ``--help`` on either subcommand for options. Sensible defaults point at the
2026-07-03 LIVE run (GT) and the Model40_Review Pending queue (review), and write
a dated snapshot under ``MouseReach_Improvement/model40_eval/``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical paths (Y: NAS). Everything is data-under-improvement; code is here.
# ---------------------------------------------------------------------------
LIVE_GT_RUN = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\model40_eval"
    r"\2026-07-03_algo4_per_reach_sankey_v6.1_LIVE"
)
PENDING_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Model40_Review\Pending"
)
EVAL_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\model40_eval"
)

# ---------------------------------------------------------------------------
# Category colors + ordering come from the canonical improvement palette
# (mousereach.improvement.lib.palette). Per improvement/AGENTS.md: import,
# never hardcode; extend the palette, don't fork it. The renderer draws only
# categories that OCCUR (v1 style), which is what naturally drops 'absent'
# from the review figure unless the absent-exception actually fires.
# ---------------------------------------------------------------------------
from mousereach.improvement.lib.palette import OUTCOME_COLORS, OUTCOME_CLASS_ORDER

# The single 'triaged' algo category splits along two axes (the same two the
# review tool tags): reach-uncertain (outcome known, causal reach not pinned) and
# outcome-uncertain (the algo could not call the outcome). Colors mirror the tool
# tags -- gold = reach, orange = outcome.
TRIAGE_SUBCOLORS = {"triaged_reach": "#FFCC00", "triaged_outcome": "#FFA500"}
CATEGORY_COLORS = {**OUTCOME_COLORS, **TRIAGE_SUBCOLORS}
CATEGORY_ORDER = OUTCOME_CLASS_ORDER
CATEGORY_ABBR = {
    "retrieved": "ret", "displaced_sa": "sa", "abnormal_exception": "abn",
    "miss": "miss", "triaged": "tri", "triaged_reach": "tri-R",
    "triaged_outcome": "tri-O", "absent": "abs",
    "displaced_outside": "out", "untouched": "unt", "uncertain": "unc",
}


def _collapse(outcome: Optional[str]) -> Optional[str]:
    """Collapse displaced_outside -> displaced_sa (project directive: on-tray vs
    off-tray is a kinematic detail; the outcome class is 'displaced')."""
    return "displaced_sa" if outcome == "displaced_outside" else outcome


# ===========================================================================
# Renderer: single panel, ALGO on the LEFT, reference on the RIGHT.
# Adapted from mousereach.improvement.outcome._run_notebooks._draw_sankey_panel
# (the v1 style the 2026-07-03 figure used), flipped and made standalone.
# ===========================================================================

def render_algo_left_sankey(
    confusion: Dict[str, int],
    output_path: Path,
    *,
    ref_label: str,
    title: str,
    footer: str = "",
    dpi: int = 300,
) -> int:
    """Render one per-reach Sankey with algo on the left, reference on the right.

    Parameters
    ----------
    confusion : dict
        Keys ``"<ref>__<algo>"`` -> count (the same key order the outcome
        metrics emit: reference first, algo second).
    output_path : Path
        PNG destination.
    ref_label : str
        Header for the right column (e.g. "Ground Truth" / "Human Review").
    title, footer : str
        Figure title / footer annotation.

    Returns the total flow count (reaches drawn).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath

    # Parse into (algo=LEFT, ref=RIGHT, count). Key is "<ref>__<algo>".
    flows: List[Tuple[str, str, int]] = []
    for key, count in confusion.items():
        parts = key.split("__")
        if len(parts) == 2 and count > 0:
            ref, algo = parts[0], parts[1]
            flows.append((algo, ref, count))

    if not flows:
        raise ValueError("empty confusion -- nothing to render")

    # Present categories on each side, in canonical order (present-only = v1).
    present_algo = {f[0] for f in flows}
    present_ref = {f[1] for f in flows}
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    if "triaged" in order:                         # keep the triaged sub-lanes adjacent
        order["triaged_reach"] = order["triaged"] + 0.3
        order["triaged_outcome"] = order["triaged"] + 0.6
    for c in sorted(present_algo | present_ref):   # tolerate unexpected classes
        order.setdefault(c, len(order))
    algo_cats = sorted(present_algo, key=lambda c: order[c])
    ref_cats = sorted(present_ref, key=lambda c: order[c])

    algo_totals: Dict[str, int] = defaultdict(int)
    ref_totals: Dict[str, int] = defaultdict(int)
    for algo, ref, c in flows:
        algo_totals[algo] += c
        ref_totals[ref] += c

    # Per-class algo accuracy (algo==ref) for the left-column labels.
    algo_correct: Dict[str, int] = defaultdict(int)
    for algo, ref, c in flows:
        if algo == ref:
            algo_correct[algo] += c

    total = sum(c for _, _, c in flows)

    # Sort flows so ribbons cluster by algo (left) then ref (right).
    flows.sort(key=lambda x: (order[x[0]], order[x[1]]))

    fig, ax = plt.subplots(figsize=(14, 12.5), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_width = 0.11
    gap = 0.02
    left_x = 0.20     # ALGO
    right_x = 0.80    # REFERENCE
    y_start = 0.90
    y_bottom = 0.12   # keep the bottom bars clear of the grey footer
    # Fit the proportional bar heights AND the inter-category gaps inside the
    # [y_bottom, y_start] band. Splitting triaged into sub-lanes adds categories,
    # so the gap total must be subtracted here or the bottom bars overflow the
    # band and collide with the footer.
    _n_col_cats = max(len(algo_cats), len(ref_cats), 1)
    _gap_total = gap * (_n_col_cats - 1)
    y_scale = ((y_start - y_bottom) - _gap_total) / total if total else 0.0

    # --- left bars (ALGO) ---
    # Small categories have bars far thinner than their multi-line labels, so
    # adjacent small cats (e.g. the two triaged sub-lanes) would overlap. Keep a
    # minimum vertical spacing between consecutive LABELS -- the bars/ribbons stay
    # strictly proportional; only the text is nudged apart.
    _min_label_gap = 0.045
    algo_pos: Dict[str, Tuple[float, float]] = {}
    y = y_start
    _algo_last_y = None
    for cat in algo_cats:
        h = algo_totals[cat] * y_scale
        algo_pos[cat] = (y, y - h)
        ax.add_patch(plt.Rectangle(
            (left_x - bar_width / 2, y - h), bar_width, h,
            facecolor=CATEGORY_COLORS.get(cat, "#CCCCCC"),
            edgecolor="white", linewidth=0.5, transform=ax.transAxes, zorder=2))
        n = algo_totals[cat]
        if cat in ("retrieved", "displaced_sa", "miss", "abnormal_exception") and n:
            lbl = f"{cat}\n({n})\n{100.0 * algo_correct[cat] / n:.1f}% agree"
        elif cat.startswith("triaged"):
            sub = {"triaged_reach": "reach uncertain",
                   "triaged_outcome": "outcome uncertain"}.get(cat, "for review")
            lbl = f"{cat}\n({n})\n({sub})"
        elif cat == "absent":
            lbl = f"{cat}\n({n})\n(algo missed)"
        else:
            lbl = f"{cat}\n({n})"
        label_y = y - h / 2
        if _algo_last_y is not None and (_algo_last_y - label_y) < _min_label_gap:
            label_y = _algo_last_y - _min_label_gap
        _algo_last_y = label_y
        ax.text(left_x - bar_width / 2 - 0.02, label_y, lbl,
                ha="right", va="center", fontsize=8, fontweight="bold",
                transform=ax.transAxes)
        y -= h + gap

    # --- right bars (REFERENCE) ---
    ref_pos: Dict[str, Tuple[float, float]] = {}
    y = y_start
    _ref_last_y = None
    for cat in ref_cats:
        h = ref_totals[cat] * y_scale
        ref_pos[cat] = (y, y - h)
        ax.add_patch(plt.Rectangle(
            (right_x - bar_width / 2, y - h), bar_width, h,
            facecolor=CATEGORY_COLORS.get(cat, "#CCCCCC"),
            edgecolor="white", linewidth=0.5, transform=ax.transAxes, zorder=2))
        label_y = y - h / 2
        if _ref_last_y is not None and (_ref_last_y - label_y) < _min_label_gap:
            label_y = _ref_last_y - _min_label_gap
        _ref_last_y = label_y
        ax.text(right_x + bar_width / 2 + 0.02, label_y,
                f"{cat}\n({ref_totals[cat]})",
                ha="left", va="center", fontsize=8, fontweight="bold",
                transform=ax.transAxes)
        y -= h + gap

    # --- ribbons ---
    algo_cur = {c: algo_pos[c][0] for c in algo_cats}
    ref_cur = {c: ref_pos[c][0] for c in ref_cats}
    x_left = left_x + bar_width / 2
    x_right = right_x - bar_width / 2
    x_mid = (x_left + x_right) / 2
    placed: List[Tuple[float, float, float, float]] = []

    for algo, ref, count in flows:
        h = count * y_scale
        a_top = algo_cur[algo]; a_bot = a_top - h; algo_cur[algo] = a_bot
        r_top = ref_cur[ref]; r_bot = r_top - h; ref_cur[ref] = r_bot
        # correct (agree) flows soft; disagreements (algo error) bolder + labeled
        if algo == ref:
            color, alpha = CATEGORY_COLORS.get(algo, "#CCCCCC"), 0.40
        else:
            color, alpha = CATEGORY_COLORS.get(algo, "#666666"), 0.60
        verts = [
            (x_left, a_top), (x_mid, a_top), (x_mid, r_top), (x_right, r_top),
            (x_right, r_bot), (x_mid, r_bot), (x_mid, a_bot), (x_left, a_bot),
            (x_left, a_top),
        ]
        codes = [
            MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(mpatches.PathPatch(
            MplPath(verts, codes), facecolor=color, alpha=alpha,
            edgecolor="none", transform=ax.transAxes, zorder=1))

        if algo != ref:
            a_mid = (a_top + a_bot) / 2
            r_mid = (r_top + r_bot) / 2

            def _pos(t):
                sh = 3.0 * t * t - 2.0 * t * t * t
                return x_left + t * (x_right - x_left), a_mid + (r_mid - a_mid) * sh

            text = "{0} {1}->{2}".format(
                count, CATEGORY_ABBR.get(algo, algo[:3]),
                CATEGORY_ABBR.get(ref, ref[:3]))
            bw = 0.0095 * len(text) + 0.022
            bh = 0.034

            def _bbox(px, py):
                return (px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)

            def _hit(b):
                bx0, by0, bx1, by1 = b
                for ox0, oy0, ox1, oy1 in placed:
                    if not (bx1 < ox0 or bx0 > ox1 or by1 < oy0 or by0 > oy1):
                        return True
                return False

            lx, ly = _pos(0.5); box = _bbox(lx, ly)
            for t in [0.50, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70,
                      0.25, 0.75, 0.20, 0.80, 0.15, 0.85]:
                cx, cy = _pos(t); cb = _bbox(cx, cy)
                if not _hit(cb):
                    lx, ly, box = cx, cy, cb
                    break
            else:
                lx, ly, box = cx, cy, cb
                while _hit(box):
                    ly += bh; box = _bbox(lx, ly)
            placed.append(box)
            ax.text(lx, ly, text, ha="center", va="center", fontsize=7,
                    fontweight="bold", color=CATEGORY_COLORS.get(algo, "#666666"),
                    transform=ax.transAxes, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              alpha=0.92, edgecolor=CATEGORY_COLORS.get(algo, "#666666"),
                              linewidth=0.7))

    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=14,
            fontweight="bold", transform=ax.transAxes)
    ax.text(left_x, 0.94, "Algorithm (as run)", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(right_x, 0.94, ref_label, ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)

    n_agree = sum(c for a, r, c in flows if a == r)
    foot = f"Algo agrees with {ref_label.lower()}: {n_agree}/{total}"
    if total:
        foot += f" ({100 * n_agree / total:.1f}%)"
    if footer:
        foot += f"\n{footer}"          # second line -- avoid a long overlapping strip
    ax.text(0.5, 0.035, foot, ha="center", va="bottom", fontsize=8,
            color="#555555", transform=ax.transAxes)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return total


# ===========================================================================
# Part 3 compute: algo vs human review, per reach.
# ===========================================================================

def _algo_label_from_assignment(label: Optional[str]) -> str:
    """Map a reach-assignment ``label`` to a per-reach algo category.

    Assignment labels seen in the wild: ``miss``, ``triaged``,
    ``causal_retrieved``, ``causal_displaced_sa`` (and ``causal_displaced_outside``
    / ``causal_abnormal_exception``). Triaged segments mark EVERY reach
    ``triaged`` -- that is faithful to algo-4 (the whole segment is held for
    review) and is what the review then re-routes.
    """
    if not label or label == "miss":
        return "miss"
    if label == "triaged":
        return "triaged"
    if label.startswith("causal_"):
        oc = _collapse(label[len("causal_"):])
        if oc in ("retrieved", "displaced_sa"):
            return oc
        if oc == "abnormal_exception":
            return "abnormal_exception"
    return "miss"


def _human_label_for_reach(reach: dict, rec: Optional[dict]) -> Tuple[str, bool]:
    """Label a reach from the human-review side; return (label, is_human_causal).

    Non-causal reaches -> 'miss'. The human's causal reach -> the human outcome
    class. The human's causal reach is matched by ``reach_id`` when the human
    picked a detected reach, else by ANY frame overlap (the human typed a frame
    range for a reach the algo may or may not have detected).
    """
    if not rec:
        return "miss", False
    human = rec.get("human") or {}
    hcls = _collapse(human.get("outcome"))
    if hcls == "abnormal":
        hcls = "abnormal_exception"
    if hcls in (None, "untouched", "uncertain", "unknown"):
        return "miss", False

    hcr = human.get("causal_reach") or {}
    rid = hcr.get("reach_id")
    # Match on reach_id OR frame overlap: a human causal reach that overlaps an
    # algo reach IS that reach, even if the ids differ across runs (raw vs
    # assignment numbering). This keeps a detected-but-renumbered reach from
    # spuriously counting as 'absent'.
    id_match = rid is not None and reach.get("reach_id") == rid
    s, e = reach.get("start_frame"), reach.get("end_frame")
    hs, he = hcr.get("start"), hcr.get("end")
    overlap = None not in (s, e, hs, he) and s <= he and hs <= e
    is_causal = id_match or overlap

    if not is_causal:
        return "miss", False
    if hcls in ("retrieved", "displaced_sa", "abnormal_exception"):
        return hcls, True
    return "miss", False


def _is_complete_review(by_seg: Dict[int, dict]) -> bool:
    """A review is complete when every segment carries a real review: a
    non-empty ``answers`` block that is NOT the unreviewed placeholder
    (``answers.reviewed == False``).

    Matches the review tool's own ``_bundle_reviewed`` convention -- a reviewed
    segment omits the ``reviewed`` flag, only placeholders set it False -- and
    additionally rejects an empty ``answers`` block so a never-touched segment
    cannot slip through as 'complete'."""
    if not by_seg:
        return False
    for r in by_seg.values():
        ans = r.get("answers") or {}
        if not ans or ans.get("reviewed") is False:
            return False
    return True


def find_reviewed_bundles(pending_dir: Path) -> List[Path]:
    """All Pending bundles that carry a COMPLETE human review + the assignment."""
    from mousereach.review.causal_review_io import load_causal_review
    out: List[Path] = []
    for b in sorted(Path(pending_dir).iterdir()):
        if not (b.is_dir() and (b / "manifest.json").exists()):
            continue
        if not (b / f"{b.name}_reach_assignments.json").exists():
            continue
        try:
            _, by_seg = load_causal_review(b.name, b)
        except Exception:
            by_seg = {}
        if _is_complete_review(by_seg):
            out.append(b)
    return out


def _human_seg_resolution(rec: Optional[dict]) -> str:
    """The human's segment-level outcome mapped to a per-reach category:
    retrieved / displaced_sa / abnormal_exception (touched), 'triaged' (human
    also left it uncertain), or 'miss' (untouched / not reviewed)."""
    if not rec:
        return "miss"
    hcls = _collapse((rec.get("human") or {}).get("outcome"))
    if hcls == "abnormal":
        return "abnormal_exception"
    if hcls in ("retrieved", "displaced_sa", "abnormal_exception"):
        return hcls
    if hcls == "triaged":
        return "triaged"
    return "miss"


def compute_review_confusion(bundle_dirs: List[Path]) -> Dict[str, Any]:
    """Per-reach confusion (algo vs human review) over reviewed bundles.

    Committed segments are scored per reach (causal reach -> outcome, others ->
    miss). TRIAGE is a per-SEGMENT decision -- the algo holds the WHOLE segment
    (both mechanisms: the v6 cascade can't determine the outcome, or the v2
    agreement gate can't confidently pick the causal reach). So, matching the GT
    Sankey and the standing directive in ``assignment/AGENTS.md``, each triaged
    segment contributes exactly ONE 'triaged' mark (routed to the human's
    resolution) and its remaining reaches are non-causal miss->miss. Labeling
    every reach in a triaged segment 'triaged' inflates the bar ~10x and is the
    documented anti-pattern.

    Universe = every algo reach, plus one 'absent' entry per touched human
    segment whose causal reach overlaps no algo reach. Keys are
    ``"<human>__<algo>"``.
    """
    from mousereach.review.causal_review_io import load_causal_review

    confusion: Dict[str, int] = defaultdict(int)
    per_ref: Dict[str, int] = defaultdict(int)
    per_algo: Dict[str, int] = defaultdict(int)
    per_correct: Dict[str, int] = defaultdict(int)
    n_universe = 0
    n_absent_exceptions = 0
    n_triaged_segments = 0
    n_abnormal_segments = 0

    def _add(hl: str, al: str, count: int = 1) -> None:
        confusion[f"{hl}__{al}"] += count
        per_ref[hl] += count
        per_algo[al] += count
        if hl == al:
            per_correct[hl] += count

    for b in bundle_dirs:
        stem = b.name
        assign = json.loads(
            (b / f"{stem}_reach_assignments.json").read_text(encoding="utf-8"))
        _, review_by_seg = load_causal_review(stem, b)
        review_by_seg = {int(k): v for k, v in (review_by_seg or {}).items()}

        # Group algo reaches by segment so a triaged segment is handled once.
        seg_reaches: Dict[int, List[dict]] = defaultdict(list)
        for r in assign.get("reaches", []):
            seg_reaches[r.get("segment_num")].append(r)

        matched_causal: Dict[int, bool] = defaultdict(bool)  # seg -> matched?

        for sn, reaches in seg_reaches.items():
            rec = review_by_seg.get(sn)
            n = len(reaches)
            if any(r.get("label") == "triaged" for r in reaches):
                # Triaged SEGMENT: one triaged mark -> the human's resolution;
                # remaining reaches are non-causal misses. Split the triaged mark
                # by axis (same as the review tool's tags): if algo-4 kept a
                # touched segment_outcome, the OUTCOME is known and only the reach
                # is uncertain (triaged_reach); if segment_outcome is itself
                # 'triaged', the algo could not call the outcome (triaged_outcome).
                n_triaged_segments += 1
                hl = _human_seg_resolution(rec)
                so = _collapse(reaches[0].get("segment_outcome"))
                algo_tri = ("triaged_reach"
                            if so in ("retrieved", "displaced_sa")
                            else "triaged_outcome")
                if hl == "abnormal_exception":
                    # A NON-reach event (e.g. the tail) caused the outcome, so
                    # EVERY reach in the segment is a genuine miss. abnormal_exception
                    # is a segment-level fact with NO causal reach -- it must not be
                    # attributed to a reach (that would steal a reach's miss slot).
                    # Keep the segment's single algo-triaged mark (now flowing to
                    # miss, since the human found all reaches are misses); the rest
                    # are miss->miss. Tally the abnormal segment separately.
                    _add("miss", algo_tri, 1)
                    if n > 1:
                        _add("miss", "miss", n - 1)
                    n_abnormal_segments += 1
                else:
                    _add(hl, algo_tri, 1)
                    if hl in ("retrieved", "displaced_sa"):
                        matched_causal[sn] = True   # human resolved the causal reach
                    if n > 1:
                        _add("miss", "miss", n - 1)
                n_universe += n
            else:
                # Committed segment: score every reach.
                for r in reaches:
                    al = _algo_label_from_assignment(r.get("label"))
                    hl, is_causal = _human_label_for_reach(r, rec)
                    if is_causal:
                        matched_causal[sn] = True
                    _add(hl, al, 1)
                    n_universe += 1
                if _collapse((rec or {}).get("human", {}).get("outcome")) == "abnormal_exception":
                    n_abnormal_segments += 1   # non-reach event; all reaches miss

        # Absent-exception: touched human segment whose causal reach overlaps
        # no algo reach -> algo missed the reach entirely (algo == 'absent').
        for sn, rec in review_by_seg.items():
            human = rec.get("human") or {}
            hcls = _collapse(human.get("outcome"))
            if hcls in ("retrieved", "displaced_sa") and not matched_causal.get(sn):
                _add(hcls, "absent", 1)
                n_universe += 1
                n_absent_exceptions += 1

    per_class = _per_class_stats(per_ref, per_algo, per_correct)
    return {
        "n_reaches_universe": n_universe,
        "n_absent_exceptions": n_absent_exceptions,
        "n_triaged_segments": n_triaged_segments,
        "n_abnormal_segments": n_abnormal_segments,
        "n_bundles": len(bundle_dirs),
        "confusion_matrix": dict(confusion),
        "per_class": per_class,
    }


def _per_class_stats(per_ref, per_algo, per_correct) -> Dict[str, Dict[str, Any]]:
    """precision = agree / algo-count; recall = agree / ref-count."""
    stats: Dict[str, Dict[str, Any]] = {}
    for cls in sorted(set(per_ref) | set(per_algo)):
        if cls == "abnormal_exception":     # non-evaluable, off the denominators
            continue
        n_ref = per_ref.get(cls, 0)
        n_algo = per_algo.get(cls, 0)
        n_ok = per_correct.get(cls, 0)
        prec = round(n_ok / n_algo, 4) if n_algo else 0.0
        rec = round(n_ok / n_ref, 4) if n_ref else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
        stats[cls] = {"n_ref": n_ref, "n_algo": n_algo,
                      "precision": prec, "recall": rec, "f1": f1}
    return stats


# ===========================================================================
# Entry points
# ===========================================================================

def algo_vs_gt(
    live_dir: Path = LIVE_GT_RUN,
    out_dir: Optional[Path] = None,
    video_ids: Optional[List[str]] = None,
) -> Path:
    """Algo-vs-GT per-reach Sankey (algo-left / GT-right).

    Recomputes the canonical algo-4 confusion from ``live_dir``'s own stored
    ``algo_outcomes/`` + ``algo_reaches/`` and the GT for each video, so it
    reproduces that run's numbers exactly -- just flipped.
    """
    from mousereach.improvement.outcome.metrics import compute_per_reach_confusion
    from mousereach.review.causal_review_io import find_gt

    live_dir = Path(live_dir)
    algo_dir = live_dir / "algo_outcomes"
    reaches_dir = live_dir / "algo_reaches"
    if video_ids is None:
        video_ids = sorted(p.name.replace("_pellet_outcomes.json", "")
                           for p in algo_dir.glob("*_pellet_outcomes.json"))
    if out_dir is None:
        out_dir = EVAL_ROOT / f"{datetime.now():%Y-%m-%d}_algo_vs_gt_per_reach_algoLEFT"
    out_dir = Path(out_dir)

    # Stage the canonical GT into a TEMP dir OUTSIDE the improvement tree, run
    # the compute, then delete it. Critical: the GT index scans
    # MouseReach_Improvement recursively for *_unified_ground_truth.json, so
    # copying GT into the snapshot (which lives under model40_eval/) would
    # POLLUTE the index -- find_gt would then resolve to our own copy (and crash
    # on re-run with src==dst). Staging to temp keeps the index clean.
    import tempfile
    tmp_gt = Path(tempfile.mkdtemp(prefix="algo_vs_gt_"))
    staged, missing = 0, []
    try:
        for vid in video_ids:
            g = find_gt(vid)
            if g:
                shutil.copy2(g, tmp_gt / f"{vid}_unified_ground_truth.json")
                staged += 1
            else:
                missing.append(vid)
        if missing:
            print(f"[algo_vs_gt] WARNING: no GT for {len(missing)} videos: {missing[:5]}")
        res = compute_per_reach_confusion(
            gt_dir=tmp_gt, algo_dir=algo_dir, reaches_dir=reaches_dir,
            video_ids=video_ids)
    finally:
        shutil.rmtree(tmp_gt, ignore_errors=True)

    # GT-side triaged is all OUTCOME-uncertain: the metrics path reads algo-3's
    # outcome ('triaged') and attributes touched outcomes by interaction frame, so
    # it has no reach-uncertain class (that would need algo-4 run on all 67 GT
    # videos). Relabel for consistent terminology with the review Sankey.
    _cm = res["confusion_matrix"]
    res["confusion_matrix"] = {
        (k + "_outcome" if k.endswith("__triaged") else k): v for k, v in _cm.items()
    }

    _write_scalars(out_dir, {
        "eval": "algo_vs_gt",
        "source_live_run": str(live_dir),
        "n_videos": staged,
        "n_reaches_universe": res["n_reaches_universe"],
        "outcome_label": {
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        },
    })
    fig = out_dir / "figures" / "sankey_algo_vs_gt.png"
    render_algo_left_sankey(
        res["confusion_matrix"], fig, ref_label="Ground Truth",
        title=f"Per-reach outcome: algo vs GT  (N={res['n_reaches_universe']} reaches, {staged} videos)",
        footer="algo exactly as run; flows = algo -> GT.   GT triage is "
               "outcome-uncertain only (reach-uncertainty is an algo-4 label, "
               "not modeled in this interaction-frame GT eval)")
    print(f"[algo_vs_gt] done -> {out_dir}")
    return out_dir


def algo_vs_review(
    pending_dir: Path = PENDING_DIR,
    out_dir: Optional[Path] = None,
) -> Path:
    """Algo-vs-review per-reach Sankey (algo-left / review-right) over every
    completely-reviewed bundle in the Pending queue."""
    pending_dir = Path(pending_dir)
    if out_dir is None:
        out_dir = EVAL_ROOT / f"{datetime.now():%Y-%m-%d}_algo_vs_review_per_reach_algoLEFT"
    out_dir = Path(out_dir)

    bundles = find_reviewed_bundles(pending_dir)
    if not bundles:
        raise SystemExit(f"No complete reviews found under {pending_dir}")
    print(f"[algo_vs_review] {len(bundles)} reviewed bundles")

    res = compute_review_confusion(bundles)
    _write_scalars(out_dir, {
        "eval": "algo_vs_review",
        "source_pending": str(pending_dir),
        "reviewed_bundles": [b.name for b in bundles],
        "n_bundles": res["n_bundles"],
        "n_reaches_universe": res["n_reaches_universe"],
        "n_triaged_segments": res["n_triaged_segments"],
        "n_abnormal_segments": res["n_abnormal_segments"],
        "n_absent_exceptions": res["n_absent_exceptions"],
        "outcome_label": {
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        },
    })
    fig = out_dir / "figures" / "sankey_algo_vs_review.png"
    render_algo_left_sankey(
        res["confusion_matrix"], fig, ref_label="Human Review",
        title=f"Per-reach outcome: algo vs review  (N={res['n_reaches_universe']} reaches, {res['n_bundles']} videos)",
        footer=f"triage is per-segment ({res['n_triaged_segments']} triaged segments, re-routed by review); "
               f"{res['n_abnormal_segments']} abnormal-exception segment(s) (all reaches miss); "
               f"{res['n_absent_exceptions']} algo-missed reach(es)")
    print(f"[algo_vs_review] done -> {out_dir}")
    return out_dir


def _write_scalars(out_dir: Path, scalars: Dict[str, Any]) -> None:
    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scalars = dict(scalars)
    scalars["generated_at"] = datetime.now().isoformat()
    (metrics_dir / "scalars.json").write_text(
        json.dumps(scalars, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Canonical per-reach evaluation Sankeys (algo AS RUN vs a "
                    "reference). THE way to measure how the algo does / "
                    "generalizes per reach against GT or human review alone.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("algo-vs-gt", help="algo vs ground truth (algo-left/GT-right)")
    g.add_argument("--live-dir", type=Path, default=LIVE_GT_RUN,
                   help="run dir with algo_outcomes/ + algo_reaches/ (default: 2026-07-03 LIVE)")
    g.add_argument("--out", type=Path, default=None, help="output snapshot dir")

    r = sub.add_parser("algo-vs-review", help="algo vs human review (algo-left/review-right)")
    r.add_argument("--pending", type=Path, default=PENDING_DIR,
                   help="Model40_Review Pending dir (reviewed bundles)")
    r.add_argument("--out", type=Path, default=None, help="output snapshot dir")

    args = p.parse_args(argv)
    if args.cmd == "algo-vs-gt":
        algo_vs_gt(live_dir=args.live_dir, out_dir=args.out)
    elif args.cmd == "algo-vs-review":
        algo_vs_review(pending_dir=args.pending, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
