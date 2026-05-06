"""
Phase E: v6 cascade HOLDOUT GENERALIZATION test.

Mirrors restart_phase_e_stage45_validate.py but runs on the 10 held-out
videos from cv_folds.json -> test_holdout.video_ids instead of train_pool.

Purpose: confirm the deterministic rule-based cascade generalizes to
unseen data before declaring it production-ready.

Component-eval mode: GT segments + GT reaches as input (same mode as
calibration so the comparison is apples-to-apples).

Outputs:
  - algo_outputs/{video}_pellet_outcomes.json for all 10 holdout videos
  - metrics/holdout_per_segment.json
  - metrics/holdout_per_video.json
  - metrics/per_reach_scalars.json
  - figures/sankey_per_reach.png
  - figures/sankey.png (per-segment 4-class)
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v6_cascade.stage_0_short_segment_triage import (
    Stage0ShortSegmentTriage)
from mousereach.outcomes.v6_cascade.stage_1_pellet_position_never_changed import (
    Stage1PelletPositionNeverChanged)
from mousereach.outcomes.v6_cascade.stage_2_pellet_stable_untouched import (
    Stage2PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_3_paw_never_in_pellet_area import (
    Stage3PawNeverInPelletArea)
from mousereach.outcomes.v6_cascade.stage_4_pellet_returns_to_pillar import (
    Stage4PelletReturnsToPillar)
from mousereach.outcomes.v6_cascade.stage_5_pellet_off_pillar_throughout import (
    Stage5PelletOffPillarThroughout)
from mousereach.outcomes.v6_cascade.stage_6_pellet_predominantly_on_pillar import (
    Stage6PelletPredominantlyOnPillar)
from mousereach.outcomes.v6_cascade.stage_7_pellet_settled_off_pillar_late import (
    Stage7PelletSettledOffPillarLate)
from mousereach.outcomes.v6_cascade.stage_8_pellet_displaced_to_sa import (
    Stage8PelletDisplacedToSA)
from mousereach.outcomes.v6_cascade.stage_9_pellet_vanished_after_reach import (
    Stage9PelletVanishedAfterReach)
from mousereach.outcomes.v6_cascade.stage_10_pillar_revealed_after_reach import (
    Stage10PillarRevealedAfterReach)
from mousereach.outcomes.v6_cascade.stage_11_single_reach_clean_displacement import (
    Stage11SingleReachCleanDisplacement)
from mousereach.outcomes.v6_cascade.stage_12_retrieved_pellet_above_slit import (
    Stage12RetrievedPelletAboveSlit)
from mousereach.outcomes.v6_cascade.stage_13_retrieved_via_pillar_lk_transition import (
    Stage13RetrievedViaPillarLkTransition)
from mousereach.outcomes.v6_cascade.stage_14_single_reach_moderate_displacement_evidence import (
    Stage14SingleReachModerateDisplacementEvidence)
from mousereach.outcomes.v6_cascade.stage_15_multi_reach_retrieved_via_above_slit_split import (
    Stage15MultiReachRetrievedViaAboveSlitSplit)
from mousereach.outcomes.v6_cascade.stage_16_displaced_via_max_displacement_reach import (
    Stage16DisplacedViaMaxDisplacement)
from mousereach.outcomes.v6_cascade.stage_17_displaced_via_dominant_max_displacement import (
    Stage17DisplacedViaDominantMaxDisplacement)
from mousereach.outcomes.v6_cascade.stage_18_displaced_via_first_significant_displacement import (
    Stage18DisplacedViaFirstSignificantDisplacement)
from mousereach.outcomes.v6_cascade.stage_19_retrieved_via_pillar_lk_first_reach import (
    Stage19RetrievedViaPillarLkFirstReach)
from mousereach.outcomes.v6_cascade.stage_20_per_bout_classifier_displaced import (
    Stage20PerBoutClassifierDisplaced)
from mousereach.outcomes.v6_cascade.stage_21_causal_reach_via_immediate_on_off_transition import (
    Stage21CausalReachViaImmediateOnOffTransition)
from mousereach.outcomes.v6_cascade.stage_22_retry_with_stabilized_dlc import (
    Stage22RetryWithStabilizedDlc)
from mousereach.outcomes.v6_cascade.stage_23_retrieved_with_pillar_tip_noise import (
    Stage23RetrievedWithPillarTipNoise)
from mousereach.outcomes.v6_cascade.stage_24_transition_triangulation import (
    Stage24TransitionTriangulation)
from mousereach.outcomes.v6_cascade.stage_25_retry_with_strict_pellet_confidence import (
    Stage25RetryWithStrictPelletConfidence)
from mousereach.outcomes.v6_cascade.stage_26_retrieved_via_unique_vanish_reach import (
    Stage26RetrievedViaUniqueVanishReach)
from mousereach.outcomes.v6_cascade.stage_27_displaced_sa_via_unique_high_displacement_reach import (
    Stage27DisplacedSaViaUniqueHighDisplacement)
from mousereach.outcomes.v6_cascade.stage_28_retrieved_via_pillar_visibility_transition import (
    Stage28RetrievedViaPillarVisibilityTransition)
from mousereach.outcomes.v6_cascade.stage_29_displaced_sa_via_pillar_disambiguated_multi_displacement import (
    Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement)
from mousereach.outcomes.v6_cascade.stage_98_lost_in_shadow_triage import (
    Stage98LostInShadowTriage)
from mousereach.outcomes.v6_cascade.stage_99_residual_triage import (
    Stage99ResidualTriage)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import calibrate_stage
from mousereach.reach.v8.features import load_dlc_h5


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\outcome\v6_cascade_holdout_generalization"
)

EXHAUSTIVE_HOLDOUT = {
    "20250626_CNT0102_P4",
    "20250708_CNT0210_P3",
    "20250811_CNT0303_P4",
    "20251024_CNT0402_P4",
}


# ---------------------------------------------------------------------------
# Helpers (same as canonical runner, not imported to avoid coupling)
# ---------------------------------------------------------------------------

def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_gt_segment_bounds(video_id: str) -> Dict[int, tuple]:
    gt = json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))
    gt_b = [int(bd["frame"]) for bd in gt.get("segmentation", {}).get("boundaries", [])]
    return {i + 1: (gt_b[i], gt_b[i + 1] - 1) for i in range(len(gt_b) - 1)}


def load_gt(video_id: str) -> dict:
    return json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))


def gt_reaches_for_segment(gt: dict, segment_num: int) -> List[tuple]:
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == segment_num:
            s = r.get("start_frame")
            e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def collapse(o):
    return "displaced_sa" if o == "displaced_outside" else o


# Same-bout logic from canonical runner
PAW_BPS_FOR_BOUT = ("RightHand", "RHLeft", "RHOut", "RHRight")
REACH_EDGE_TOLERANCE_FORWARD = 0


def _same_bout(seg_input: SegmentInput, algo_ifr, gt_ifr, paw_lk_thr=0.5,
               transition_zone_half=5,
               edge_tolerance: int = REACH_EDGE_TOLERANCE_FORWARD) -> bool:
    if algo_ifr is None or gt_ifr is None:
        return False
    candidates = list(seg_input.reach_windows)
    if candidates:
        def find_reach(abs_f):
            af = int(abs_f)
            for ri, (rs, re) in enumerate(candidates):
                if (rs - edge_tolerance) <= af <= (re + edge_tolerance):
                    return ri
            best, best_d = -1, 10**9
            for ri, (rs, re) in enumerate(candidates):
                d = min(abs(af - rs), abs(af - re))
                if d < best_d:
                    best_d, best = d, ri
            return best
        return find_reach(algo_ifr) == find_reach(gt_ifr)
    return False


# ---------------------------------------------------------------------------
# Build cascade stages (identical to canonical runner)
# ---------------------------------------------------------------------------

def build_stages():
    return [
        ("Stage 0 (short-segment-triage)", Stage0ShortSegmentTriage()),
        ("Stage 1 (position-never-changed)", Stage1PelletPositionNeverChanged()),
        ("Stage 2 (stable-on-pillar)", Stage2PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5)),
        ("Stage 3 (paw-never-in-pellet-area)", Stage3PawNeverInPelletArea()),
        ("Stage 4 (pellet-returns-to-pillar)", Stage4PelletReturnsToPillar()),
        ("Stage 5 (pellet-off-pillar-throughout)", Stage5PelletOffPillarThroughout()),
        ("Stage 6 (predominantly-on-pillar)", Stage6PelletPredominantlyOnPillar()),
        ("Stage 7 (settled-off-pillar-late)", Stage7PelletSettledOffPillarLate()),
        ("Stage 8 (pellet-displaced-to-SA)", Stage8PelletDisplacedToSA()),
        ("Stage 9 (pellet-vanished-after-reach)", Stage9PelletVanishedAfterReach()),
        ("Stage 10 (pillar-revealed-after-reach)", Stage10PillarRevealedAfterReach()),
        ("Stage 11 (single-reach-clean-displacement)", Stage11SingleReachCleanDisplacement()),
        ("Stage 12 (retrieved-pellet-above-slit)", Stage12RetrievedPelletAboveSlit()),
        ("Stage 13 (retrieved-via-pillar-lk-transition)", Stage13RetrievedViaPillarLkTransition()),
        ("Stage 14 (single-reach-moderate-displacement-evidence)", Stage14SingleReachModerateDisplacementEvidence()),
        ("Stage 15 (multi-reach-retrieved-via-above-slit)", Stage15MultiReachRetrievedViaAboveSlitSplit()),
        ("Stage 16 (displaced-via-max-displacement-reach)", Stage16DisplacedViaMaxDisplacement()),
        ("Stage 17 (displaced-via-dominant-max-displacement)", Stage17DisplacedViaDominantMaxDisplacement()),
        ("Stage 18 (displaced-via-first-significant-displacement)", Stage18DisplacedViaFirstSignificantDisplacement()),
        ("Stage 19 (retrieved-via-pillar-lk-first-reach)", Stage19RetrievedViaPillarLkFirstReach()),
        ("Stage 20 (per-bout-classifier-displaced)", Stage20PerBoutClassifierDisplaced()),
        ("Stage 21 (causal-reach-via-immediate-on-off-transition)", Stage21CausalReachViaImmediateOnOffTransition()),
        ("Stage 22 (retry-with-stabilized-dlc)", Stage22RetryWithStabilizedDlc()),
        ("Stage 23 (retrieved-with-pillar-tip-noise)", Stage23RetrievedWithPillarTipNoise()),
        ("Stage 24 (transition-triangulation)", Stage24TransitionTriangulation()),
        ("Stage 25 (retry-with-strict-pellet-confidence)", Stage25RetryWithStrictPelletConfidence()),
        ("Stage 26 (retrieved-via-unique-vanish-reach)", Stage26RetrievedViaUniqueVanishReach()),
        ("Stage 27 (displaced-sa-via-unique-high-displacement)", Stage27DisplacedSaViaUniqueHighDisplacement()),
        ("Stage 28 (retrieved-via-pillar-visibility-transition)", Stage28RetrievedViaPillarVisibilityTransition()),
        ("Stage 29 (displaced-sa-via-pillar-disambiguated-multi-disp)", Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement()),
        ("Stage 98 (lost-in-shadow-triage)", Stage98LostInShadowTriage(
            video_dir=QUARANTINE / "videos")),
        ("Stage 99 (residual-triage)", Stage99ResidualTriage()),
    ]


# ---------------------------------------------------------------------------
# Loader: holdout videos only
# ---------------------------------------------------------------------------

def build_holdout_seg_inputs_and_gt():
    """Component-eval loader for holdout videos only."""
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    holdout_ids = folds["test_holdout"]["video_ids"]
    seg_inputs = []
    gt_lookup = {}
    for vid in holdout_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_gt_segment_bounds(vid)
        gt = load_gt(vid)
        gt_outs = {s["segment_num"]: s
                   for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (s_start, s_end) in seg_bounds.items():
            seg_inputs.append(SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=s_start, seg_end=s_end,
                dlc_df=dlc,
                reach_windows=gt_reaches_for_segment(gt, sn),
            ))
            seg = gt_outs.get(sn, {})
            gt_outcome = collapse(seg.get("outcome"))
            expected_triage = bool(seg.get("expected_triage", False))
            if gt_outcome == "abnormal_exception":
                expected_triage = True
            gt_lookup[(vid, sn)] = {
                "gt_outcome": gt_outcome,
                "gt_outcome_known_frame": seg.get("outcome_known_frame"),
                "gt_interaction_frame": seg.get("interaction_frame"),
                "expected_triage": expected_triage,
                "expected_triage_reason": seg.get("expected_triage_reason"),
            }
    return seg_inputs, gt_lookup, holdout_ids


# ---------------------------------------------------------------------------
# Run cascade and collect results
# ---------------------------------------------------------------------------

def run_cascade(seg_inputs, gt_lookup, stages):
    """Run cascade, return per-segment results and calibration cases."""
    seg_lookup = {(s.video_id, s.segment_num): s for s in seg_inputs}
    all_cases = []
    consumed = set()

    for label, stage in stages:
        stage_inputs = [s for s in seg_inputs
                        if (s.video_id, s.segment_num) not in consumed]
        cal = calibrate_stage(
            stage=stage, seg_inputs=stage_inputs, gt_lookup=gt_lookup,
            okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
        for c in cal.cases:
            if c.decision in ("commit", "triage"):
                consumed.add((c.video_id, c.segment_num))
                all_cases.append((label, c))
        print(f"  {label}: {sum(1 for c in cal.cases if c.decision=='commit')} committed, "
              f"{sum(1 for c in cal.cases if c.decision=='triage')} triaged, "
              f"{sum(1 for c in cal.cases if c.decision=='continue')} deferred", flush=True)

    # Residual segments (not consumed) are auto-triaged
    for s in seg_inputs:
        key = (s.video_id, s.segment_num)
        if key not in consumed:
            consumed.add(key)

    return all_cases, seg_lookup


def trust_pass_fn(c, gt_lookup, seg_lookup):
    """Same trust logic as canonical runner."""
    gt = gt_lookup.get((c.video_id, c.segment_num), {})
    expected_triage = gt.get("expected_triage", False)
    if c.decision == "triage":
        return expected_triage
    if c.decision != "commit":
        return False
    if expected_triage:
        return False
    if not c.class_match:
        return False
    if c.committed_class == "untouched":
        return c.okf_within_tol
    seg = seg_lookup.get((c.video_id, c.segment_num))
    if seg is None:
        return False
    return _same_bout(seg, c.committed_interaction_frame,
                      c.gt_interaction_frame)


# ---------------------------------------------------------------------------
# Save pellet_outcomes.json per video
# ---------------------------------------------------------------------------

def save_pellet_outcomes(all_cases, gt_lookup, seg_inputs, out_dir: Path):
    """Save per-video pellet_outcomes.json for the v2 Sankey renderer."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group cases by video
    video_segments = defaultdict(dict)
    for label, c in all_cases:
        key = (c.video_id, c.segment_num)
        if c.decision == "commit":
            video_segments[c.video_id][c.segment_num] = {
                "segment_num": c.segment_num,
                "outcome": c.committed_class,
                "outcome_known_frame": c.committed_outcome_known_frame,
                "interaction_frame": c.committed_interaction_frame,
                "stage": label,
                "flagged_for_review": False,
            }
        elif c.decision == "triage":
            video_segments[c.video_id][c.segment_num] = {
                "segment_num": c.segment_num,
                "outcome": "triaged",
                "outcome_known_frame": None,
                "interaction_frame": None,
                "stage": label,
                "flagged_for_review": True,
                "flag_reason": c.reason,
            }

    # Also add residual (auto-triaged) segments
    consumed_keys = set()
    for label, c in all_cases:
        consumed_keys.add((c.video_id, c.segment_num))
    for s in seg_inputs:
        key = (s.video_id, s.segment_num)
        if key not in consumed_keys:
            video_segments[s.video_id][s.segment_num] = {
                "segment_num": s.segment_num,
                "outcome": "triaged",
                "outcome_known_frame": None,
                "interaction_frame": None,
                "stage": "residual (auto-triage)",
                "flagged_for_review": True,
                "flag_reason": "residual after all cascade stages",
            }

    for vid, segs in video_segments.items():
        out_path = out_dir / f"{vid}_pellet_outcomes.json"
        data = {
            "video_id": vid,
            "detector": "v6_cascade",
            "mode": "holdout_generalization_component_eval",
            "segments": [segs[sn] for sn in sorted(segs.keys())],
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Score: per-segment metrics
# ---------------------------------------------------------------------------

def score_per_segment(all_cases, gt_lookup, seg_lookup, seg_inputs,
                      video_filter=None):
    """Compute per-segment metrics. If video_filter is set, only those."""
    # Collect per-segment outcomes
    seg_outcomes = {}  # (vid, sn) -> algo_outcome
    seg_stages = {}
    for label, c in all_cases:
        key = (c.video_id, c.segment_num)
        if video_filter and c.video_id not in video_filter:
            continue
        if c.decision == "commit":
            seg_outcomes[key] = c.committed_class
            seg_stages[key] = label
        elif c.decision == "triage":
            seg_outcomes[key] = "triaged"
            seg_stages[key] = label

    # Residual segments
    for s in seg_inputs:
        key = (s.video_id, s.segment_num)
        if video_filter and s.video_id not in video_filter:
            continue
        if key not in seg_outcomes:
            seg_outcomes[key] = "triaged"
            seg_stages[key] = "residual (auto-triage)"

    # Confusion matrix
    confusion = defaultdict(int)
    n_by_class = defaultdict(int)
    n_by_class_excl_triage = defaultdict(int)
    correct_commits = defaultdict(int)
    wrong_commits = []
    triage_count = 0
    expected_triage_handled = 0
    expected_triage_total = 0
    n_segments = 0

    for key, algo_outcome in seg_outcomes.items():
        gt = gt_lookup.get(key, {})
        gt_outcome = gt.get("gt_outcome")
        expected_triage = gt.get("expected_triage", False)
        n_segments += 1
        n_by_class[gt_outcome] += 1

        if expected_triage:
            expected_triage_total += 1
            if algo_outcome == "triaged":
                expected_triage_handled += 1
            continue  # Skip from dev metrics

        n_by_class_excl_triage[gt_outcome] += 1
        confusion[f"{gt_outcome}__{algo_outcome}"] += 1

        if algo_outcome == "triaged":
            triage_count += 1
            continue

        # Check trust for commits
        # Find the matching calibration case
        matching_case = None
        for label, c in all_cases:
            if (c.video_id, c.segment_num) == key and c.decision == "commit":
                matching_case = c
                break

        if matching_case is not None:
            passes = trust_pass_fn(matching_case, gt_lookup, seg_lookup)
            if passes:
                correct_commits[algo_outcome] += 1
            else:
                wrong_commits.append({
                    "video_id": key[0],
                    "segment_num": key[1],
                    "gt_outcome": gt_outcome,
                    "algo_outcome": algo_outcome,
                    "stage": seg_stages.get(key, "?"),
                    "reason": matching_case.reason if matching_case else "?",
                })

    return {
        "n_segments": n_segments,
        "confusion_matrix": dict(confusion),
        "n_by_class": dict(n_by_class),
        "n_by_class_excl_triage": dict(n_by_class_excl_triage),
        "correct_commits": dict(correct_commits),
        "wrong_commits": wrong_commits,
        "triage_count": triage_count,
        "expected_triage_total": expected_triage_total,
        "expected_triage_handled": expected_triage_handled,
    }


def score_per_video(all_cases, gt_lookup, seg_lookup, seg_inputs, video_ids):
    """Score each video individually."""
    results = {}
    for vid in video_ids:
        vid_filter = {vid}
        vid_seg_inputs = [s for s in seg_inputs if s.video_id == vid]
        r = score_per_segment(all_cases, gt_lookup, seg_lookup, vid_seg_inputs,
                              video_filter=vid_filter)
        r["video_id"] = vid
        r["is_exhaustive"] = vid in EXHAUSTIVE_HOLDOUT
        results[vid] = r
    return results


# ---------------------------------------------------------------------------
# Per-segment Sankey (4-class)
# ---------------------------------------------------------------------------

def render_per_segment_sankey(confusion_data, output_path, title_suffix=""):
    """Render a simple per-segment 4-class Sankey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath

    cm = confusion_data["confusion_matrix"]

    # Parse flows
    flows = []
    for key, count in cm.items():
        parts = key.split("__")
        if len(parts) == 2:
            flows.append((parts[0], parts[1], count))

    CATEGORY_COLORS = {
        "retrieved": "#4CAF50",
        "displaced_sa": "#FF9800",
        "untouched": "#2196F3",
        "abnormal_exception": "#616161",
        "triaged": "#FFEB3B",
    }

    CATEGORY_ORDER = ["untouched", "displaced_sa", "retrieved",
                      "abnormal_exception", "triaged"]

    # Compute totals
    gt_totals = defaultdict(int)
    algo_totals = defaultdict(int)
    for gt, algo, count in flows:
        gt_totals[gt] += count
        algo_totals[algo] += count

    gt_categories = [c for c in CATEGORY_ORDER if gt_totals.get(c, 0) > 0]
    algo_categories = [c for c in CATEGORY_ORDER if algo_totals.get(c, 0) > 0]
    # Ensure we have all that appear
    for c in sorted(gt_totals.keys()):
        if c not in gt_categories:
            gt_categories.append(c)
    for c in sorted(algo_totals.keys()):
        if c not in algo_categories:
            algo_categories.append(c)

    total_segs = sum(c for _, _, c in flows)

    # Layout
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    bar_width = 0.10
    gap = 0.025
    left_x = 0.22
    right_x = 0.78
    y_start = 0.92
    usable_height = 0.80

    def _compute_positions(categories, totals):
        total_count = sum(totals.get(c, 0) for c in categories)
        total_gap = gap * (len(categories) - 1)
        available = usable_height - total_gap
        y_scale = available / total_count if total_count > 0 else 0
        positions = {}
        y_cursor = y_start
        for cat in categories:
            count = totals.get(cat, 0)
            h = max(count * y_scale, 0.012)
            positions[cat] = (y_cursor, y_cursor - h)
            y_cursor -= h + gap
        return positions

    gt_positions = _compute_positions(gt_categories, gt_totals)
    algo_positions = _compute_positions(algo_categories, algo_totals)

    # Draw bars
    for cat in gt_categories:
        y_top, y_bot = gt_positions[cat]
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        count = gt_totals.get(cat, 0)
        rect = plt.Rectangle(
            (left_x - bar_width/2, y_bot), bar_width, y_top - y_bot,
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(left_x - bar_width/2 - 0.02, (y_top + y_bot)/2,
                f"{cat}\n({count})", ha="right", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)

    for cat in algo_categories:
        y_top, y_bot = algo_positions[cat]
        color = CATEGORY_COLORS.get(cat, "#CCCCCC")
        count = algo_totals.get(cat, 0)
        rect = plt.Rectangle(
            (right_x - bar_width/2, y_bot), bar_width, y_top - y_bot,
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(right_x + bar_width/2 + 0.02, (y_top + y_bot)/2,
                f"{cat}\n({count})", ha="left", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)

    # Draw flows
    gt_cursors = {c: gt_positions[c][0] for c in gt_categories}
    algo_cursors = {c: algo_positions[c][0] for c in algo_categories}

    flows.sort(key=lambda x: (
        CATEGORY_ORDER.index(x[0]) if x[0] in CATEGORY_ORDER else 99,
        CATEGORY_ORDER.index(x[1]) if x[1] in CATEGORY_ORDER else 99))

    for gt_out, algo_out, count in flows:
        if count == 0:
            continue
        gt_bar_top, gt_bar_bot = gt_positions[gt_out]
        algo_bar_top, algo_bar_bot = algo_positions[algo_out]
        gt_bar_h = gt_bar_top - gt_bar_bot
        algo_bar_h = algo_bar_top - algo_bar_bot
        gt_total = gt_totals.get(gt_out, 1)
        algo_total = algo_totals.get(algo_out, 1)

        flow_h_gt = (count / gt_total) * gt_bar_h
        flow_h_algo = (count / algo_total) * algo_bar_h

        gt_top = gt_cursors[gt_out]
        gt_bot = gt_top - flow_h_gt
        gt_cursors[gt_out] = gt_bot

        algo_top = algo_cursors[algo_out]
        algo_bot = algo_top - flow_h_algo
        algo_cursors[algo_out] = algo_bot

        color = CATEGORY_COLORS.get(gt_out, "#CCCCCC")
        alpha = 0.4 if gt_out == algo_out else 0.6

        x_left = left_x + bar_width/2
        x_right = right_x - bar_width/2
        x_mid = (x_left + x_right) / 2

        verts = [
            (x_left, gt_top), (x_mid, gt_top), (x_mid, algo_top),
            (x_right, algo_top), (x_right, algo_bot), (x_mid, algo_bot),
            (x_mid, gt_bot), (x_left, gt_bot), (x_left, gt_top)]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(
            path, facecolor=color, alpha=alpha, edgecolor="none",
            transform=ax.transAxes, zorder=1)
        ax.add_patch(patch)

        # Label mismatches
        if gt_out != algo_out:
            mid_y = ((gt_top + gt_bot)/2 + (algo_top + algo_bot)/2) / 2
            ax.text(0.5, mid_y, f"{count}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color,
                    transform=ax.transAxes, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.9, edgecolor=color, linewidth=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = f"Per-segment outcome flow (N={total_segs})"
    if title_suffix:
        title += f" -- {title_suffix}"
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=13,
            fontweight="bold", transform=ax.transAxes)
    ax.text(left_x, 0.94, "Ground Truth", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(right_x, 0.94, "Algorithm", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)

    n_correct = sum(c for g, a, c in flows if g == a)
    footer = f"Correct: {n_correct}/{total_segs}"
    if total_segs > 0:
        footer += f" ({100*n_correct/total_segs:.1f}%)"
    ax.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9,
            color="#555555", transform=ax.transAxes)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("v6 CASCADE HOLDOUT GENERALIZATION TEST")
    print("=" * 70)
    print()

    # Load holdout data
    print("Loading holdout GT segments + GT reaches (component eval) ...", flush=True)
    seg_inputs, gt_lookup, holdout_ids = build_holdout_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} GT segments from {len(holdout_ids)} holdout videos")

    exhaustive_segs = [s for s in seg_inputs if s.video_id in EXHAUSTIVE_HOLDOUT]
    nonexhaustive_segs = [s for s in seg_inputs if s.video_id not in EXHAUSTIVE_HOLDOUT]
    print(f"  Exhaustive subset: {len(exhaustive_segs)} segments from "
          f"{len(EXHAUSTIVE_HOLDOUT)} videos")
    print(f"  Non-exhaustive: {len(nonexhaustive_segs)} segments from "
          f"{len(holdout_ids) - len(EXHAUSTIVE_HOLDOUT)} videos")
    print()

    # Build and run cascade
    stages = build_stages()
    print("Running cascade (stages 0-29 + 98 + 99) ...", flush=True)
    all_cases, seg_lookup = run_cascade(seg_inputs, gt_lookup, stages)
    print()

    # Save algo outputs
    algo_out_dir = SNAPSHOT_DIR / "algo_outputs"
    print("Saving pellet_outcomes.json per video ...", flush=True)
    save_pellet_outcomes(all_cases, gt_lookup, seg_inputs, algo_out_dir)
    print()

    # ---------------------------------------------------------------------------
    # 1. EXHAUSTIVE HOLDOUT HEADLINE METRICS
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("HEADLINE: EXHAUSTIVE HOLDOUT (4 videos, primary metric)")
    print("=" * 70)
    exhaustive_scores = score_per_segment(
        all_cases, gt_lookup, seg_lookup, seg_inputs,
        video_filter=EXHAUSTIVE_HOLDOUT)

    print(f"  Total segments: {exhaustive_scores['n_segments']}")
    print(f"  GT class distribution (excl expected_triage):")
    for cls in ("untouched", "displaced_sa", "retrieved", "abnormal_exception"):
        n = exhaustive_scores["n_by_class_excl_triage"].get(cls, 0)
        ok = exhaustive_scores["correct_commits"].get(cls, 0)
        pct = 100 * ok / n if n > 0 else 0
        print(f"    {cls:>22s}: {ok}/{n} ({pct:.1f}%)")

    print(f"  Triage count: {exhaustive_scores['triage_count']} "
          f"(rate: {100*exhaustive_scores['triage_count']/max(sum(exhaustive_scores['n_by_class_excl_triage'].values()),1):.1f}%)")
    print(f"  Wrong commits: {len(exhaustive_scores['wrong_commits'])}")
    for wc in exhaustive_scores["wrong_commits"]:
        print(f"    {wc['video_id']} seg {wc['segment_num']}: "
              f"gt={wc['gt_outcome']} algo={wc['algo_outcome']} "
              f"stage={wc['stage']}")
    print(f"  Expected triage: {exhaustive_scores['expected_triage_handled']}/"
          f"{exhaustive_scores['expected_triage_total']} handled")
    print()

    # ---------------------------------------------------------------------------
    # 2. ALL 10 HOLDOUT VIDEOS
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("ALL 10 HOLDOUT VIDEOS")
    print("=" * 70)
    all_scores = score_per_segment(
        all_cases, gt_lookup, seg_lookup, seg_inputs, video_filter=None)

    print(f"  Total segments: {all_scores['n_segments']}")
    print(f"  GT class distribution (excl expected_triage):")
    for cls in ("untouched", "displaced_sa", "retrieved", "abnormal_exception"):
        n = all_scores["n_by_class_excl_triage"].get(cls, 0)
        ok = all_scores["correct_commits"].get(cls, 0)
        pct = 100 * ok / n if n > 0 else 0
        print(f"    {cls:>22s}: {ok}/{n} ({pct:.1f}%)")
    print(f"  Triage count: {all_scores['triage_count']}")
    print(f"  Wrong commits: {len(all_scores['wrong_commits'])}")
    for wc in all_scores["wrong_commits"]:
        print(f"    {wc['video_id']} seg {wc['segment_num']}: "
              f"gt={wc['gt_outcome']} algo={wc['algo_outcome']} "
              f"stage={wc['stage']}")
    print(f"  Expected triage: {all_scores['expected_triage_handled']}/"
          f"{all_scores['expected_triage_total']} handled")
    print()

    # ---------------------------------------------------------------------------
    # 3. PER-VIDEO BREAKDOWN
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("PER-VIDEO BREAKDOWN")
    print("=" * 70)
    per_video = score_per_video(all_cases, gt_lookup, seg_lookup,
                                seg_inputs, holdout_ids)
    for vid in holdout_ids:
        r = per_video[vid]
        tag = "[EXHAUSTIVE]" if r["is_exhaustive"] else "[non-exhaust]"
        n_segs = r["n_segments"]
        n_wrong = len(r["wrong_commits"])
        n_tri = r["triage_count"]
        n_correct = sum(r["correct_commits"].values())
        n_scorable = sum(r["n_by_class_excl_triage"].values())
        pct = 100 * n_correct / n_scorable if n_scorable > 0 else 0
        print(f"  {tag} {vid}: {n_segs} segs, {n_correct}/{n_scorable} correct "
              f"({pct:.1f}%), {n_tri} triaged, {n_wrong} wrong")
    print()

    # ---------------------------------------------------------------------------
    # 4. NON-EXHAUSTIVE SUPPORTING METRICS
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("NON-EXHAUSTIVE (6 videos, supporting only)")
    print("=" * 70)
    nonexh_filter = set(holdout_ids) - EXHAUSTIVE_HOLDOUT
    nonexh_scores = score_per_segment(
        all_cases, gt_lookup, seg_lookup, seg_inputs,
        video_filter=nonexh_filter)
    print(f"  Total segments: {nonexh_scores['n_segments']}")
    for cls in ("untouched", "displaced_sa", "retrieved", "abnormal_exception"):
        n = nonexh_scores["n_by_class_excl_triage"].get(cls, 0)
        ok = nonexh_scores["correct_commits"].get(cls, 0)
        pct = 100 * ok / n if n > 0 else 0
        print(f"    {cls:>22s}: {ok}/{n} ({pct:.1f}%)")
    print(f"  Triage count: {nonexh_scores['triage_count']}")
    print(f"  Wrong commits: {len(nonexh_scores['wrong_commits'])}")
    for wc in nonexh_scores["wrong_commits"]:
        print(f"    {wc['video_id']} seg {wc['segment_num']}: "
              f"gt={wc['gt_outcome']} algo={wc['algo_outcome']} "
              f"stage={wc['stage']}")
    print()

    # ---------------------------------------------------------------------------
    # 5. SAVE METRICS
    # ---------------------------------------------------------------------------
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Per-segment headline
    holdout_per_seg = {
        "corpus": "test_holdout",
        "mode": "component_eval_gt_segments_gt_reaches",
        "exhaustive_only": {
            "n_videos": len(EXHAUSTIVE_HOLDOUT),
            "video_ids": sorted(EXHAUSTIVE_HOLDOUT),
            **exhaustive_scores,
        },
        "all_10_holdout": {
            "n_videos": len(holdout_ids),
            "video_ids": holdout_ids,
            **all_scores,
        },
        "non_exhaustive_supporting": {
            "n_videos": len(nonexh_filter),
            "video_ids": sorted(nonexh_filter),
            **nonexh_scores,
        },
    }
    (metrics_dir / "holdout_per_segment.json").write_text(
        json.dumps(holdout_per_seg, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {metrics_dir / 'holdout_per_segment.json'}")

    # Per-video
    holdout_per_video = {vid: per_video[vid] for vid in holdout_ids}
    (metrics_dir / "holdout_per_video.json").write_text(
        json.dumps(holdout_per_video, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {metrics_dir / 'holdout_per_video.json'}")

    # ---------------------------------------------------------------------------
    # 6. FIGURES
    # ---------------------------------------------------------------------------
    figures_dir = SNAPSHOT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Per-segment Sankey (exhaustive holdout)
    render_per_segment_sankey(
        exhaustive_scores, figures_dir / "sankey.png",
        title_suffix="Holdout exhaustive (4 videos)")

    # Per-reach Sankey v2 (exhaustive holdout)
    from mousereach.improvement.outcome.sankey_per_reach_v2 import (
        compute_per_reach_confusion_v2, render_per_reach_sankey_v2)

    pr_confusion = compute_per_reach_confusion_v2(
        gt_dir=GT_DIR,
        algo_dir=algo_out_dir,
        video_ids=sorted(EXHAUSTIVE_HOLDOUT),
    )
    render_per_reach_sankey_v2(
        pr_confusion,
        output_path=figures_dir / "sankey_per_reach.png",
        title_suffix="Holdout exhaustive (4 videos)",
    )

    # Save per-reach scalars
    (metrics_dir / "per_reach_scalars.json").write_text(
        json.dumps(pr_confusion, indent=2), encoding="utf-8")
    print(f"Saved: {metrics_dir / 'per_reach_scalars.json'}")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Snapshot: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
