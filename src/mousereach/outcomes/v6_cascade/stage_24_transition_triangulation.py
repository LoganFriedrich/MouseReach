"""
Stage 24: Transition triangulation + class via post-transition pellet visibility.

Combines techniques mined from older mousereach versions:
  - v4.0.0-step-5 sliding-min-max pillar-lk transition detector
    (transition_detector.py Signal 2)
  - v3.0.1 tie-breaker: post-interaction-frame pellet visibility
    distinguishes displaced (still seen) from retrieved (not seen)
  - Reach proximity matching: pick the GT reach closest to the
    triangulated transition frame

Pipeline:
  1. Get all transition candidates from transition_detector.
  2. Pick the BEST transition frame: prefer pillar_lk_rise (most
     robust signal), fall back to pellet_left_pillar then
     pellet_vanished. Require strength >= 0.6.
  3. Find GT reach containing or closest-to (within +/- 50 frames)
     the transition frame.
  4. Class via post-transition pellet visibility:
     - >= 5 sustained confident pellet observations after the
       transition frame -> displaced_sa
     - else -> retrieved
  5. Defenses:
     - Defer if no transition candidate qualifies (no signal)
     - Defer if transition frame > 50 frames from any GT reach
       (don't trust the bout-pick)
     - Defer if multiple high-strength transition candidates from
       DIFFERENT signals disagree by > 30 frames (signal conflict)

Cascade emit on commit:
  - committed_class: retrieved or displaced_sa
  - whens["interaction_frame"]: middle of chosen GT reach
  - whens["outcome_known_frame"]: transition frame + small offset
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision
from .transition_detector import detect_transition_moments


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
# 2026-05-03: disabled. Aggressive class commit (post-transition pellet
# visibility + cluster check) yields high commits but breaks trust.
# Conservative version (require very clear signal) yields too few
# commits with 50% trust to be useful. The class discriminator is
# fundamentally noisy for the residual cases.
MIN_TRANSITION_STRENGTH = 100.0  # impossible threshold = effectively disabled
MAX_FRAMES_FROM_REACH = 50
MIN_POST_PELLET_FOR_DISPLACED = 5  # sustained-3 pellet observations
                                    # after transition -> displaced
MIN_SUSTAINED_RUN = 3
SIGNAL_DISAGREEMENT_TOLERANCE = 30  # frames

# Signal priority for picking the best transition frame.
SIGNAL_PRIORITY = {
    'pillar_lk_rise': 1,
    'pellet_vanished': 2,
    'pellet_left_pillar': 3,
    'pellet_jumped': 4,
}

OKF_OFFSET = 6


def _sustained_count(arr, min_run):
    total = 0
    run = 0
    for v in arr:
        if v:
            run += 1
        else:
            if run >= min_run:
                total += run
            run = 0
    if run >= min_run:
        total += run
    return total


class Stage24TransitionTriangulation(Stage):
    name = "stage_24_transition_triangulation"
    target_class = None  # commits retrieved or displaced_sa

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - TRANSITION_ZONE_HALF
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short")

        candidates = detect_transition_moments(seg)
        feats = {
            "n_transition_candidates": len(candidates),
            "candidates": [(c.signal_type, c.frame, c.strength) for c in candidates],
        }
        if not candidates:
            return StageDecision(
                decision="continue",
                reason="no_transition_candidates",
                features=feats,
            )

        # Filter by strength threshold.
        strong = [c for c in candidates if c.strength >= MIN_TRANSITION_STRENGTH]
        if not strong:
            return StageDecision(
                decision="continue",
                reason="no_transition_candidates_above_strength_threshold",
                features=feats,
            )

        # Group strong candidates by frame proximity (within
        # SIGNAL_DISAGREEMENT_TOLERANCE). If they disagree, defer.
        # If they agree, pick the highest-priority signal's frame as
        # the consensus.
        # Sort by priority then frame.
        strong_sorted = sorted(
            strong,
            key=lambda c: (SIGNAL_PRIORITY.get(c.signal_type, 99), c.frame),
        )
        primary = strong_sorted[0]
        # Check that other strong candidates agree (within tolerance).
        for c in strong_sorted[1:]:
            if abs(c.frame - primary.frame) > SIGNAL_DISAGREEMENT_TOLERANCE:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"transition_signals_disagree "
                        f"({primary.signal_type}@{primary.frame} vs "
                        f"{c.signal_type}@{c.frame}; delta>"
                        f"{SIGNAL_DISAGREEMENT_TOLERANCE})"
                    ),
                    features=feats,
                )
        transition_frame = primary.frame
        feats["transition_frame"] = transition_frame
        feats["primary_signal"] = primary.signal_type

        # Find GT reach containing or closest to the transition frame.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(clean_end - seg.seg_start, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        if not reach_windows_local:
            return StageDecision(
                decision="continue",
                reason="no_gt_reaches",
                features=feats,
            )

        trans_local = transition_frame - seg.seg_start
        # Find reach containing trans_local; if none, find nearest.
        chosen_idx = -1
        chosen_dist = 10**9
        for ri, (rs, re) in enumerate(reach_windows_local):
            if rs <= trans_local <= re:
                chosen_idx = ri
                chosen_dist = 0
                break
            d = min(abs(trans_local - rs), abs(trans_local - re))
            if d < chosen_dist:
                chosen_dist = d
                chosen_idx = ri
        feats["chosen_reach_idx"] = chosen_idx
        feats["dist_to_chosen_reach"] = chosen_dist
        if chosen_dist > MAX_FRAMES_FROM_REACH:
            return StageDecision(
                decision="continue",
                reason=(
                    f"transition_too_far_from_any_gt_reach "
                    f"(dist={chosen_dist} > {MAX_FRAMES_FROM_REACH})"
                ),
                features=feats,
            )

        # Class determination via post-transition pellet visibility.
        # (v3.0.1 tie-breaker logic from older mousereach.)
        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        geom = compute_pillar_geometry_series(sub)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line = pillar_cy + pillar_r

        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py <= slit_y_line) & (pl >= PAW_LK_THR)

        # Confident pellet observations AFTER the transition frame,
        # paw not past slit.
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                              + (pellet_y - pillar_cy) ** 2)
                      / np.maximum(pillar_r, 1e-6))
        post_pellet_visible = (pellet_lk >= PELLET_LK_HIGH) & (~paw_past_y)
        trans_local_clipped = max(0, min(n - 1, trans_local))
        # Off-pillar observations only (dist > 1 radius).
        post_off_pillar = post_pellet_visible & (dist_radii > ON_PILLAR_RADII)
        post_off_obs = post_off_pillar[trans_local_clipped + 1:]
        n_post_pellet = _sustained_count(post_off_obs, MIN_SUSTAINED_RUN)
        feats["n_post_transition_off_pillar_sustained"] = n_post_pellet

        # Cluster check: if post-transition off-pillar observations
        # cluster TIGHTLY near pillar tip (within 2.5 radii, std<=15px),
        # they're DLC label-switching to revealed pillar -- retrieved.
        # Real displaced has cluster at >= 3 radii.
        is_pillar_tip_noise = False
        if n_post_pellet >= 5:
            post_idx = np.where(post_off_pillar[trans_local_clipped + 1:])[0] + (trans_local_clipped + 1)
            obs_x = pellet_x[post_idx]
            obs_y = pellet_y[post_idx]
            med_x = float(np.median(obs_x))
            med_y = float(np.median(obs_y))
            std_xy = float(np.sqrt(np.std(obs_x) ** 2 + np.std(obs_y) ** 2))
            med_pillar_cx = float(np.nanmedian(pillar_cx))
            med_pillar_cy = float(np.nanmedian(pillar_cy))
            med_pillar_r = float(np.nanmedian(pillar_r))
            cluster_dist_radii = float(
                np.sqrt((med_x - med_pillar_cx) ** 2
                        + (med_y - med_pillar_cy) ** 2)
                / max(med_pillar_r, 1e-6))
            feats.update({
                "cluster_std_px": std_xy,
                "cluster_dist_radii": cluster_dist_radii,
            })
            if std_xy <= 15.0 and cluster_dist_radii <= 2.5:
                is_pillar_tip_noise = True

        # Conservative class call: require clear signal.
        # - Confident displaced: 30+ sustained off-pillar AND not pillar-tip noise
        # - Confident retrieved: 0 sustained off-pillar OR pillar-tip noise cluster
        # - Anything in between: defer
        if n_post_pellet >= 30 and not is_pillar_tip_noise:
            committed_class = "displaced_sa"
        elif n_post_pellet == 0 or is_pillar_tip_noise:
            committed_class = "retrieved"
        else:
            return StageDecision(
                decision="continue",
                reason=(
                    f"class_signal_borderline "
                    f"(post-transition off-pillar = {n_post_pellet}; "
                    f"is_pillar_tip_noise={is_pillar_tip_noise}; "
                    f"defer)"
                ),
                features=feats,
            )

        # Also defer multi-reach segments to avoid bout-pick ambiguity.
        if len(reach_windows_local) > 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"multi_gt_reach_in_segment "
                    f"({len(reach_windows_local)} reaches; bout-pick "
                    f"ambiguous, defer)"
                ),
                features=feats,
            )

        # IFR/OKF emit. IFR = middle of chosen GT reach.
        bs, be = reach_windows_local[chosen_idx]
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = min(trans_local_clipped + OKF_OFFSET, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "committed_class": committed_class,
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
        })
        return StageDecision(
            decision="commit",
            committed_class=committed_class,
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"transition_triangulation "
                f"(primary signal: {primary.signal_type} @ frame "
                f"{transition_frame}; chosen GT reach idx {chosen_idx}; "
                f"post-transition pellet sustained {n_post_pellet}f -> "
                f"class={committed_class})"
            ),
            features=feats,
        )
