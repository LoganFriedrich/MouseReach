"""
Stage 23: Retrieved when Stage 9's "sustained off-pillar" observations
are clustered tightly near the pillar tip (= DLC label-switching to
the revealed pillar after retrieval).

Question this stage answers:
    "Did Stage 9 reject this segment as 'sustained off-pillar' but
    those off-pillar observations are actually DLC noise on the
    revealed pillar tip (clustered tightly near pillar center)?"

Whittling logic:
    Stage 9's anti-displaced gate fires when post-first-reach pellet
    observations are sustained off-pillar (>1.5 pillar radii). For
    real retrievals where DLC label-switches the Pellet bodypart to
    the now-revealed pillar tip, those observations cluster TIGHTLY
    near the pillar center (typically 1-2.5 radii). Real displaced
    pellets sit at >=3 radii (per Stage 5 empirical block).

    This stage commits retrieved when:
      - Off-pillar pellet observations exist (Stage 9 deferred for
        this reason)
      - Those observations cluster tightly (low position spread) AND
        near the pillar tip (median dist <= 2.5 radii)
      - Single GT reach (avoid bout-pick ambiguity)
      - Pellet was on pillar before the reach (sanity)

Defenses against false-commit:
    - Strict cluster spread (std <= 8px from cluster median)
    - Strict near-pillar location (median dist <= 2.5 radii)
    - Single GT reach restriction
    - Pre-reach on-pillar evidence required

Cascade emit on commit:
    - committed_class: "retrieved"
    - whens["interaction_frame"]: middle of GT reach
    - whens["outcome_known_frame"]: clean zone end
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.95
PELLET_LK_OFF_PILLAR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
NEAR_PILLAR_TIP_RADII = 2.5     # observations within this dist from
                                # pillar center could be tip-noise
# 2026-05-03: Stage 23 disabled. Pillar-tip cluster signature is real
# but bout-pick across multi-reach cases requires GT visual judgment
# we can't replicate. All commit attempts produce wrong-bout errors.
CLUSTER_STD_PX_MAX = 0          # disabled (impossible threshold)
MIN_PRE_ON_PILLAR = 10
MIN_SUSTAINED_RUN = 3


class Stage23RetrievedWithPillarTipNoise(Stage):
    name = "stage_23_retrieved_with_pillar_tip_noise"
    target_class = "retrieved"

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - TRANSITION_ZONE_HALF
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short")

        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line = pillar_cy + pillar_r

        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                              + (pellet_y - pillar_cy) ** 2)
                      / np.maximum(pillar_r, 1e-6))

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py <= slit_y_line) & (pl >= PAW_LK_THR)

        # Single GT reach only.
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        feats = {
            "n_clean_zone_frames": int(n),
            "n_gt_reaches": len(reach_windows_local),
        }
        if not reach_windows_local:
            return StageDecision(
                decision="continue",
                reason="no_gt_reaches",
                features=feats,
            )

        # Pick reach where pillar-tip cluster FIRST appears post-end:
        # walk through reaches, find first whose post-window has a
        # sustained run at the eventual cluster center (= the reach
        # that caused the DLC to start mistakenly tracking the
        # revealed pillar tip).
        # Compute cluster center from ALL post-pellet-pillar-area
        # observations across the segment.
        all_off_eligible_for_seed = (
            (pellet_lk >= PELLET_LK_OFF_PILLAR)
            & (dist_radii > ON_PILLAR_RADII)
            & (~paw_past_y)
        )
        if all_off_eligible_for_seed.sum() < 5:
            return StageDecision(
                decision="continue",
                reason="too_few_off_pillar_obs_in_segment",
                features=feats,
            )
        seed_med_x = float(np.median(pellet_x[all_off_eligible_for_seed]))
        seed_med_y = float(np.median(pellet_y[all_off_eligible_for_seed]))
        # Per-frame: at-cluster = within 8 px of seed median.
        d_to_seed = np.sqrt((pellet_x - seed_med_x) ** 2
                            + (pellet_y - seed_med_y) ** 2)
        at_cluster = (
            (pellet_lk >= PELLET_LK_OFF_PILLAR)
            & (~paw_past_y)
            & (d_to_seed <= 8.0)
        )
        # Walk through reaches: first reach where post-end has 3+
        # sustained at-cluster frames.
        chosen_idx = None
        for ri, (rs_local, re_local) in enumerate(reach_windows_local):
            run = 0
            found = False
            for j in range(re_local + 1, n):
                if at_cluster[j]:
                    run += 1
                    if run >= 3:
                        found = True
                        break
                else:
                    run = 0
            if found:
                chosen_idx = ri
                break
        if chosen_idx is None:
            return StageDecision(
                decision="continue",
                reason="no_reach_seeded_cluster_emergence",
                features=feats,
            )
        bs, be = reach_windows_local[chosen_idx]
        feats["chosen_reach_idx"] = int(chosen_idx)

        # Pre-reach on-pillar evidence.
        on_pillar = (
            (pellet_lk >= PELLET_LK_HIGH)
            & (dist_radii <= ON_PILLAR_RADII)
            & (~paw_past_y)
        )
        pre_on_pillar_count = int(on_pillar[:bs].sum())
        feats["pre_on_pillar_count"] = pre_on_pillar_count
        if pre_on_pillar_count < MIN_PRE_ON_PILLAR:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_pre_reach_on_pillar "
                    f"({pre_on_pillar_count} < {MIN_PRE_ON_PILLAR})"
                ),
                features=feats,
            )

        # Off-pillar observations post-reach (Stage 9's anti-displaced
        # signal). Confident, paw not past slit, dist > 1 radius.
        off_pillar_post = (
            (pellet_lk >= PELLET_LK_OFF_PILLAR)
            & (dist_radii > ON_PILLAR_RADII)
            & (~paw_past_y)
        )
        post_mask = np.zeros(n, dtype=bool)
        post_mask[be + 1:] = True
        post_off_eligible = off_pillar_post & post_mask
        n_post_off = int(post_off_eligible.sum())
        feats["n_post_off_pillar"] = n_post_off
        if n_post_off < 5:
            # Not enough off-pillar observations to even matter; should
            # have been Stage 9's domain. Defer.
            return StageDecision(
                decision="continue",
                reason=f"too_few_post_off_pillar_obs ({n_post_off}); not Stage 23's case",
                features=feats,
            )

        # Cluster analysis: are the off-pillar observations TIGHTLY
        # clustered AND near the pillar tip?
        post_x_obs = pellet_x[post_off_eligible]
        post_y_obs = pellet_y[post_off_eligible]
        med_x = float(np.median(post_x_obs))
        med_y = float(np.median(post_y_obs))
        std_x = float(np.std(post_x_obs))
        std_y = float(np.std(post_y_obs))
        cluster_std = float(np.sqrt(std_x ** 2 + std_y ** 2))
        feats.update({
            "cluster_med_x": med_x, "cluster_med_y": med_y,
            "cluster_std_px": cluster_std,
        })

        # Median distance from pillar in radii.
        med_pillar_cx = float(np.nanmedian(pillar_cx))
        med_pillar_cy = float(np.nanmedian(pillar_cy))
        med_pillar_r = float(np.nanmedian(pillar_r))
        cluster_dist_radii = float(np.sqrt((med_x - med_pillar_cx) ** 2
                                           + (med_y - med_pillar_cy) ** 2)
                                   / max(med_pillar_r, 1e-6))
        feats["cluster_dist_radii"] = cluster_dist_radii

        if cluster_std > CLUSTER_STD_PX_MAX:
            return StageDecision(
                decision="continue",
                reason=(
                    f"off_pillar_cluster_too_spread "
                    f"(std {cluster_std:.1f} > {CLUSTER_STD_PX_MAX}; "
                    f"not a single fixed mis-detection)"
                ),
                features=feats,
            )

        if cluster_dist_radii > NEAR_PILLAR_TIP_RADII:
            return StageDecision(
                decision="continue",
                reason=(
                    f"cluster_too_far_from_pillar_for_tip_noise "
                    f"({cluster_dist_radii:.2f} radii > "
                    f"{NEAR_PILLAR_TIP_RADII}; this is real displaced "
                    f"position, not pillar-tip noise)"
                ),
                features=feats,
            )

        # All gates passed: tightly-clustered off-pillar observations
        # near pillar tip = DLC label-switch to revealed pillar after
        # retrieval. Commit retrieved.
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = n - 1
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
        })
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"retrieved_with_pillar_tip_noise "
                f"(off-pillar cluster: std={cluster_std:.1f}px, "
                f"dist={cluster_dist_radii:.2f} radii from pillar; "
                f"DLC label-switch to revealed pillar tip)"
            ),
            features=feats,
        )
