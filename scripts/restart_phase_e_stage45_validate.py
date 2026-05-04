"""
Phase E: cascade Stages 1-5 component evaluation.

INPUT (user 2026-05-02): GT segments + GT reaches. The outcome
detector (this cascade) is evaluated unconfounded by upstream
segmentation/reach-detection errors. Each algo decomposes; each is
evaluated independently with perfect upstream input. End-to-end eval
with algo segments/reaches is a separate later concern.

Stages:
  1: pellet stable on pillar throughout segment    -> untouched
  2: paw never enters pellet area                   -> untouched
  3: pellet returns to pillar after off-pillar      -> untouched
  4: pellet off-pillar throughout / co-detection   -> untouched | triage
  5: pellet visibly displaced from pillar to SA    -> displaced_sa
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
ALGO_DIR = QUARANTINE / "algo_outputs"


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


# Component-eval design (user 2026-05-02): use GT segments + GT reaches
# as inputs to outcome detection and algo 4. This isolates outcome
# detection's quality from upstream segmentation/reach detection errors.
# Each algo decomposes; each is evaluated independently with perfect
# upstream input. End-to-end eval with algo segments/reaches is a
# separate later concern.
def load_gt_segment_bounds(video_id: str) -> Dict[int, tuple]:
    """Return GT segment bounds keyed by GT segment_num (1-indexed,
    matching how GT entries label their segments)."""
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


# User-mandated trust definition for touched-class commits
# (`touched_stage_trust_definition.md`): trust = class match AND same
# paw-past-y-line bout. OKF/IFR exact frame match is informational only.
PAW_BPS_FOR_BOUT = ("RightHand", "RHLeft", "RHOut", "RHRight")


REACH_EDGE_TOLERANCE_FORWARD = 0  # 2026-05-03: tested fwd tolerance
                                   # but boundary spillover hurt yields.
                                   # Real fix per user: GT IFR is a
                                   # guess derived from OKF; the actual
                                   # causal reach could be ANY reach
                                   # before GT-OKF, not just the one
                                   # containing GT IFR. Better trust
                                   # criterion: algo's reach must end
                                   # before GT-OKF and be the closest
                                   # such reach to GT-OKF.


def _same_bout(seg_input: SegmentInput, algo_ifr, gt_ifr, paw_lk_thr=0.5,
               transition_zone_half=5,
               edge_tolerance: int = REACH_EDGE_TOLERANCE_FORWARD) -> bool:
    """True if algo_ifr and gt_ifr fall in (or are nearest to) the
    same REACH WINDOW in this segment.

    User 2026-05-02: real GT reaches are deduplicated, noise-free
    candidates -- match by GT reach, not by paw-past-y-line bout.
    Falls back to paw-past-y-line bouts if no reach windows available.

    User 2026-05-03: IFR within +/- edge_tolerance frames of a reach
    window's edge counts as "in" that reach. Loosens exact-frame
    matching while preserving the same-reach requirement.
    """
    if algo_ifr is None or gt_ifr is None:
        return False
    candidates = list(seg_input.reach_windows)  # absolute frames
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

    # Fallback: paw-past-y-line bouts (production mode without reaches)
    from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
    from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
    import numpy as _np

    clean_end = seg_input.seg_end - transition_zone_half
    sub_raw = seg_input.dlc_df.iloc[seg_input.seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return False
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    slit_y_line = geom["pillar_cy"].to_numpy() + geom["pillar_r"].to_numpy()
    paw_past_y = _np.zeros(n, dtype=bool)
    for bp in PAW_BPS_FOR_BOUT:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past_y |= (py <= slit_y_line) & (pl >= paw_lk_thr)
    bouts = []
    rs = -1
    for i in range(n):
        if paw_past_y[i]:
            if rs < 0: rs = i
        else:
            if rs >= 0:
                bouts.append((rs, i - 1)); rs = -1
    if rs >= 0:
        bouts.append((rs, n - 1))
    if not bouts:
        return False
    def find_bout(absolute_frame):
        local = int(absolute_frame) - seg_input.seg_start
        for bi, (s, e) in enumerate(bouts):
            if s <= local <= e:
                return bi
        best, best_d = -1, 10**9
        for bi, (s, e) in enumerate(bouts):
            d = min(abs(local - s), abs(local - e))
            if d < best_d:
                best_d = d; best = bi
        return best
    return find_bout(algo_ifr) == find_bout(gt_ifr)


def build_seg_inputs_and_gt():
    """Component-eval loader: feeds outcome detection with GT
    segmentation + GT reaches. This is the user-mandated 2026-05-02
    approach -- evaluate the outcome algo unconfounded by upstream
    segmentation/reach detection errors.
    """
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    seg_inputs = []
    gt_lookup = {}
    for vid in train_pool_ids:
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
            # Cases the algo cannot reasonably classify get
            # `expected_triage`. abnormal_exception class is treated as
            # implicit expected_triage (per its semantic definition).
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
    return seg_inputs, gt_lookup


def main():
    print("=" * 70)
    print("PHASE E STAGE 4+5 VALIDATION")
    print("=" * 70)
    print()

    print("Loading GT segments + GT reaches (component eval mode) ...", flush=True)
    seg_inputs, gt_lookup = build_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} GT segments")
    print()

    stages = [
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

    seg_lookup = {(s.video_id, s.segment_num): s for s in seg_inputs}

    def trust_pass(c) -> bool:
        """Algo decision is correct iff:
          - For triages: GT marked the segment as expected_triage
            (i.e. data-quality / abnormal -- algo correctly punted)
          - For commits: NOT expected_triage AND class_match AND
            (touched: same paw-past-y-line bout; untouched: OKF tol)
          - For continue: never passes (still in residual)
        """
        gt = gt_lookup.get((c.video_id, c.segment_num), {})
        expected_triage = gt.get("expected_triage", False)
        if c.decision == "triage":
            return expected_triage
        if c.decision != "commit":
            return False
        if expected_triage:
            return False  # algo committed when it should have triaged
        if not c.class_match:
            return False
        if c.committed_class == "untouched":
            return c.okf_within_tol
        seg = seg_lookup.get((c.video_id, c.segment_num))
        if seg is None:
            return False
        return _same_bout(seg, c.committed_interaction_frame,
                          c.gt_interaction_frame)

    cals = []
    inputs = seg_inputs
    consumed = set()
    for label, stage in stages:
        stage_inputs = [s for s in inputs
                        if (s.video_id, s.segment_num) not in consumed]
        print(f"Running {label} on {len(stage_inputs)} input segments ...", flush=True)
        cal = calibrate_stage(
            stage=stage, seg_inputs=stage_inputs, gt_lookup=gt_lookup,
            okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
        committed = sum(1 for c in cal.cases if c.decision == "commit")
        triaged = sum(1 for c in cal.cases if c.decision == "triage")
        deferred = sum(1 for c in cal.cases if c.decision == "continue")
        print(f"  {label}: {committed} committed, {triaged} triaged, {deferred} deferred")
        # Per-class commit + triage breakdown using user-mandated metric.
        # User 2026-05-04: expected_triage cases are IGNORE cases during
        # dev -- algo-unsolvable, excluded from pass-rate metrics. They
        # neither count as correct nor as wrong; they're tracked
        # separately for documentation only.
        commit_by_gt = defaultdict(int)
        commit_by_gt_pass = defaultdict(int)
        triage_by_gt = defaultdict(int)
        triage_correct = 0
        triage_wrong = 0
        ignored_commits = 0
        ignored_triages = 0
        for c in cal.cases:
            gt_meta = gt_lookup.get((c.video_id, c.segment_num), {})
            exp_tri = gt_meta.get("expected_triage", False)
            if exp_tri:
                # IGNORE: don't count toward stage's pass-rate metrics.
                if c.decision == "commit":
                    ignored_commits += 1
                elif c.decision == "triage":
                    ignored_triages += 1
                continue
            if c.decision == "commit":
                commit_by_gt[c.gt_class] += 1
                if trust_pass(c):
                    commit_by_gt_pass[c.gt_class] += 1
                else:
                    extra = ""
                    # For touched-class wrong commits, show algo-IFR
                    # vs GT-IFR + the bouts to expose the bout-pick gap.
                    extra_info = ""
                    if c.committed_class in ("displaced_sa", "retrieved"):
                        seg_obj = seg_lookup.get((c.video_id, c.segment_num))
                        gt_ifr = c.gt_interaction_frame
                        algo_ifr = c.committed_interaction_frame
                        if seg_obj is not None and gt_ifr is not None and algo_ifr is not None:
                            # Compute paw bouts and find which bout each IFR is in
                            from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
                            from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
                            import numpy as _np
                            clean_end = seg_obj.seg_end - 5
                            sub_raw = seg_obj.dlc_df.iloc[seg_obj.seg_start:clean_end + 1]
                            n_local = len(sub_raw)
                            sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
                            geom = compute_pillar_geometry_series(sub)
                            slit = geom["pillar_cy"].to_numpy() + geom["pillar_r"].to_numpy()
                            paw_past = _np.zeros(n_local, dtype=bool)
                            for bp in PAW_BPS_FOR_BOUT:
                                py = sub[f"{bp}_y"].to_numpy(dtype=float)
                                pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
                                paw_past |= (py <= slit) & (pl >= 0.5)
                            bouts_local = []
                            rs = -1
                            for i in range(n_local):
                                if paw_past[i]:
                                    if rs < 0: rs = i
                                else:
                                    if rs >= 0:
                                        bouts_local.append((rs, i - 1)); rs = -1
                            if rs >= 0: bouts_local.append((rs, n_local - 1))
                            def find_b(abs_f):
                                lo = int(abs_f) - seg_obj.seg_start
                                for bi, (s, e) in enumerate(bouts_local):
                                    if s <= lo <= e:
                                        return bi, s, e
                                # nearest
                                best, best_d = -1, 10**9
                                for bi, (s, e) in enumerate(bouts_local):
                                    d = min(abs(lo - s), abs(lo - e))
                                    if d < best_d: best_d, best = d, bi
                                if best >= 0:
                                    return best, bouts_local[best][0], bouts_local[best][1]
                                return -1, -1, -1
                            algo_b, algo_bs, algo_be = find_b(algo_ifr)
                            gt_b, gt_bs, gt_be = find_b(gt_ifr)
                            extra_info = (f"  algo_IFR={algo_ifr} (bout {algo_b}: "
                                          f"{seg_obj.seg_start+algo_bs}..{seg_obj.seg_start+algo_be})  "
                                          f"GT_IFR={gt_ifr} (bout {gt_b}: "
                                          f"{seg_obj.seg_start+gt_bs}..{seg_obj.seg_start+gt_be})  "
                                          f"n_bouts={len(bouts_local)}")
                    print(f"    WRONG COMMIT{extra}: {c.video_id} seg {c.segment_num}  "
                          f"gt={c.gt_class}  algo={c.committed_class}{extra_info}")
            elif c.decision == "triage":
                triage_by_gt[c.gt_class] += 1
                if trust_pass(c):
                    triage_correct += 1
                else:
                    triage_wrong += 1
        if commit_by_gt:
            print(f"  {label} commits by GT class:")
            for cls, n in sorted(commit_by_gt.items()):
                ok = commit_by_gt_pass.get(cls, 0)
                print(f"    {cls:>22s}: {ok}/{n} pass trust ({100*ok/n:.1f}%)")
        if ignored_commits or ignored_triages:
            print(f"  {label} ignored: {ignored_commits} commits + "
                  f"{ignored_triages} triages on expected_triage cases "
                  f"(excluded from dev metrics; algo-unsolvable)")
        if triage_by_gt:
            n_tri = sum(triage_by_gt.values())
            print(f"  {label} triages: {triage_correct}/{n_tri} correct (expected_triage match)  "
                  f"{triage_wrong} non-expected (algo triaged but GT didn't mark expected)")
            # Suppress the per-case dump for the residual catchall
            # stage (would be hundreds of lines and not informative).
            if "residual" not in label.lower():
                for c in cal.cases:
                    if c.decision == "triage":
                        exp = gt_lookup.get((c.video_id, c.segment_num), {}).get("expected_triage", False)
                        tag = "OK" if exp else "NON-EXPECTED"
                        print(f"    [{tag}] {c.video_id} seg {c.segment_num:2d}  "
                              f"gt={c.gt_class}  expected_triage={exp}")
            else:
                # For Stage 99, just show the GT-class breakdown.
                for cls, n in sorted(triage_by_gt.items(), key=lambda x: -x[1]):
                    print(f"    {cls:>22s}: {n}")
        # Mark commits AND triages as consumed (no further stages see them)
        for c in cal.cases:
            if c.decision in ("commit", "triage"):
                consumed.add((c.video_id, c.segment_num))
        cals.append((label, cal))
        print()

    # Cumulative residual
    residual = defaultdict(int)
    residual_by_class = defaultdict(list)
    for s in seg_inputs:
        if (s.video_id, s.segment_num) not in consumed:
            gt = gt_lookup.get((s.video_id, s.segment_num), {})
            cls = gt.get("gt_outcome")
            residual[cls] += 1
            residual_by_class[cls].append(s)
    print(f"Cumulative residual after all 6 stages: {sum(residual.values())} segments")
    for cls, n in sorted(residual.items(), key=lambda x: -x[1]):
        print(f"  {cls:>22s}: {n}")
    print()

    # Per-stage deferral reason for residual untouched cases
    if residual_by_class.get("untouched"):
        print("=" * 70)
        print("RESIDUAL UNTOUCHED CASES (with each stage's deferral reason)")
        print("=" * 70)
        for s in residual_by_class["untouched"]:
            print(f"\n{s.video_id} seg {s.segment_num}  bounds=({s.seg_start}, {s.seg_end})  "
                  f"n_reaches={len(s.reach_windows)}")
            for label, stage in stages:
                d = stage.decide(s)
                short = d.reason[:140] + ("..." if len(d.reason) > 140 else "")
                print(f"  {label}: {d.decision}  {short}")
        print()

    # For each RESIDUAL displaced_sa case, get the deferral reason
    # from BOTH Stage 7 and Stage 8 to understand what's blocking.
    print("=" * 70)
    print("RESIDUAL displaced_sa: deferral reasons from Stage 7 and Stage 8")
    print("=" * 70)
    s7 = next(stage for label, stage in stages
              if "(settled-off-pillar-late)" in label)
    s8 = next(stage for label, stage in stages
              if "(pellet-displaced-to-SA)" in label)
    # Identify cases triaged by Stage 99 (residual triage). These are
    # the cases that fell through everything else.
    s99_triaged = set()
    for label, cal in cals:
        if "Stage 99" not in label:
            continue
        for c in cal.cases:
            if c.decision == "triage":
                s99_triaged.add((c.video_id, c.segment_num))
    s7_reasons = defaultdict(int)
    s8_reasons = defaultdict(int)
    n_residual_disp = 0
    for s in seg_inputs:
        if (s.video_id, s.segment_num) not in s99_triaged:
            continue
        gt = gt_lookup.get((s.video_id, s.segment_num), {})
        if gt.get("gt_outcome") != "displaced_sa":
            continue
        n_residual_disp += 1
        d7 = s7.decide(s)
        d8 = s8.decide(s)
        s7_reasons[d7.reason.split("(")[0].strip()] += 1
        s8_reasons[d8.reason.split("(")[0].strip()] += 1
    print(f"  {n_residual_disp} residual GT-displaced_sa cases")
    print()
    print("  Stage 7 deferral reasons:")
    for r, n in sorted(s7_reasons.items(), key=lambda x: -x[1]):
        print(f"    {n:4d}: {r}")
    print()
    print("  Stage 8 deferral reasons:")
    for r, n in sorted(s8_reasons.items(), key=lambda x: -x[1]):
        print(f"    {n:4d}: {r}")
    print()

    # For each RESIDUAL retrieved case, get Stage 9 deferral reason.
    print("=" * 70)
    print("RESIDUAL retrieved: deferral reasons from Stage 9")
    print("=" * 70)
    s9 = next(stage for label, stage in stages
              if "(pellet-vanished-after-reach)" in label)
    s9_reasons = defaultdict(int)
    s9_residual_cases = []
    n_residual_ret = 0
    for s in seg_inputs:
        if (s.video_id, s.segment_num) not in s99_triaged:
            continue
        gt = gt_lookup.get((s.video_id, s.segment_num), {})
        if gt.get("gt_outcome") != "retrieved":
            continue
        n_residual_ret += 1
        d9 = s9.decide(s)
        reason_short = d9.reason.split("(")[0].strip()
        s9_reasons[reason_short] += 1
        s9_residual_cases.append((s.video_id, s.segment_num, reason_short))
    print(f"  {n_residual_ret} residual GT-retrieved cases")
    print()
    print("  Stage 9 deferral reasons:")
    for r, n in sorted(s9_reasons.items(), key=lambda x: -x[1]):
        print(f"    {n:4d}: {r}")
    print()
    print("  Per-case (residual retrieved, Stage 9 reason):")
    for vid, sn, r in s9_residual_cases:
        print(f"    {vid} seg {sn:2d}  {r}")
    print()

    # Cumulative coverage of each target class. Denominators exclude
    # expected_triage cases (those should be triaged, not committed).
    print("=" * 70)
    print("CUMULATIVE PER-CLASS YIELDS (denominator excludes expected_triage)")
    print("=" * 70)
    n_by_class = defaultdict(int)
    n_by_class_excl_triage = defaultdict(int)
    for k, g in gt_lookup.items():
        cls = g.get("gt_outcome")
        n_by_class[cls] += 1
        if not g.get("expected_triage", False):
            n_by_class_excl_triage[cls] += 1
    for target_class in ("untouched", "displaced_sa", "retrieved", "abnormal_exception"):
        total = n_by_class.get(target_class, 0)
        total_excl = n_by_class_excl_triage.get(target_class, 0)
        correct_commits = 0
        for label, cal in cals:
            for c in cal.cases:
                if c.decision == "commit" and c.committed_class == target_class:
                    if trust_pass(c):
                        correct_commits += 1
        print(f"  {target_class:>22s}: {correct_commits} / {total_excl} non-triage GT "
              f"({100*correct_commits/max(total_excl,1):.1f}%)  "
              f"[GT total {total}, of which {total - total_excl} expected_triage]")


    # Expected-triage performance. Residual cases are auto-triaged at
    # the end of the cascade per the user's design (anything left over
    # is handed to human review), so an expected_triage case that lands
    # in residual = correctly triaged.
    n_expected = sum(1 for k, g in gt_lookup.items() if g.get("expected_triage", False))
    n_explicit_triage = 0
    n_wrongly_committed = 0
    expected_keys = {k for k, g in gt_lookup.items() if g.get("expected_triage", False)}
    explicit_triaged_keys = set()
    committed_keys = set()
    for label, cal in cals:
        for c in cal.cases:
            key = (c.video_id, c.segment_num)
            if key not in expected_keys:
                continue
            if c.decision == "triage":
                explicit_triaged_keys.add(key)
            elif c.decision == "commit":
                committed_keys.add(key)
    n_explicit_triage = len(explicit_triaged_keys)
    n_wrongly_committed = len(committed_keys)
    # Residual = expected_triage cases that were neither committed nor
    # explicitly triaged by any stage. These auto-triage at end.
    n_residual_triage = len(expected_keys - explicit_triaged_keys - committed_keys)

    print()
    print("=" * 70)
    print("EXPECTED-TRIAGE CASES (IGNORED IN DEV METRICS)")
    print("=" * 70)
    print(f"  Total expected_triage cases: {n_expected}")
    print(f"  -- These are algo-unsolvable cases (apparatus failures,")
    print(f"     fast-streak displaced_outs, etc). User 2026-05-04: ignore")
    print(f"     in dev metrics; only matter in aggregate end-stage stats.")
    print(f"  Disposition during this run:")
    print(f"    Explicitly triaged by a cascade stage: {n_explicit_triage}/{n_expected}")
    print(f"    Implicitly triaged (in residual at end): {n_residual_triage}/{n_expected}")
    print(f"    Committed by algo (ignored, not penalized): {n_wrongly_committed}/{n_expected}")

    # Show the residual cases by GT class with expected_triage status,
    # so we can see what's still in residual and whether it's an
    # expected-triage (good) or a missed scorable case (still to handle).
    print()
    print("=" * 70)
    print("RESIDUAL (auto-triaged at end of cascade)")
    print("=" * 70)
    residual_expected = defaultdict(int)
    residual_unexpected = defaultdict(int)
    for s in seg_inputs:
        key = (s.video_id, s.segment_num)
        if key in consumed:
            continue
        gt = gt_lookup.get(key, {})
        cls = gt.get("gt_outcome")
        if gt.get("expected_triage", False):
            residual_expected[cls] += 1
        else:
            residual_unexpected[cls] += 1
    print(f"  Expected-triage residuals (correctly auto-triaged): {sum(residual_expected.values())}")
    for cls, n in sorted(residual_expected.items(), key=lambda x: -x[1]):
        print(f"    {cls:>22s}: {n}")
    print(f"  Non-expected-triage residuals (algo couldn't classify, sent to manual review): {sum(residual_unexpected.values())}")
    for cls, n in sorted(residual_unexpected.items(), key=lambda x: -x[1]):
        print(f"    {cls:>22s}: {n}")


if __name__ == "__main__":
    main()
