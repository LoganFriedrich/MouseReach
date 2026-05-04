"""
Stage 21: Causal reach via immediate on->off pillar transition (per
user's 2026-05-03 physics walkthrough).

Question this stage answers:
    "Is there ONE reach where the pellet was confidently on the
    pillar IMMEDIATELY before AND NOT on the pillar IMMEDIATELY
    after? That's the causal reach. Class is then determined by
    whether the pellet appears later in the SA (displaced) or never
    appears stably (retrieved)."

Physics (per user, 2026-05-03):
    - Same bout vs same reach are different. Paw-past-slit bouts
      are kinematic. Reaches are behavioral. The CAUSAL reach is
      uniquely identifiable: pellet on-pillar immediately pre,
      not-on-pillar immediately post.
    - Only ONE reach can be causal in a segment. Once pellet is off
      pillar, no subsequent reach has the pellet-on-pillar
      precondition. If multiple reaches appear to satisfy the
      transition, our on/off detection has noise -- defer.
    - Trust in pellet/pillar position drops near paw. Pre and post
      checks must be done in paw-CLEAR windows (paw not past slit),
      adjacent to but not overlapping the reach itself.
    - Real pellets are STABLE in position. DLC noise "dances around".
      Position stability is a discriminator.

Class determination after causal reach identified:
    - Pellet appears confidently and stably in SA after the causal
      reach: displaced_sa.
    - Pellet never appears stably anywhere off-pillar in SA: retrieved.
    - "Stably" means low position variance across confident frames.

Whittling logic:
    - This stage targets cases where prior stages couldn't pick the
      right causal reach. Stages 7/8/16/17 use longer windows or
      max-displacement; this stage uses the user's exact physics
      criterion.
    - Restricts to cases where exactly ONE reach has the on->off
      transition (= unambiguous causal). Cases where multiple reaches
      show the transition (= DLC noise) defer.

Cascade emit on commit:
    - committed_class: "displaced_sa" or "retrieved" (determined per
      class rule above)
    - whens["interaction_frame"]: middle of the causal reach window
    - whens["outcome_known_frame"]: causal reach end + small offset
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_HIGH = 0.9     # confident pellet detection
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0    # standard on-pillar zone

# IMMEDIATE pre/post windows -- short, per the physics. Looking for
# the transient on->off transition adjacent to the reach.
IMMEDIATE_WINDOW_FRAMES = 10     # search this many frames adjacent
                                 # to the reach for paw-clear frames
MIN_PAW_CLEAR_FRAMES_REQUIRED = 3  # must find >= this many paw-clear
                                   # frames in the window
MIN_ON_PILLAR_IN_WINDOW = 2      # in those paw-clear frames, >=
                                 # this many must be on-pillar (for
                                 # pre-bout); for post-bout, must
                                 # have ZERO on-pillar (NOT-on-pillar
                                 # state)

# Class determination thresholds.
# Stable off-pillar evidence in SA = displaced.
SA_STABLE_MIN_FRAMES = 30        # >= this many frames at confident,
                                 # tightly-clustered, off-pillar in SA
SA_STABLE_TOLERANCE_PX = 15.0    # tight position cluster (real pellet
                                 # at rest doesn't dance around)
PELLET_OFF_PILLAR_RADII_FOR_SA = 1.5

# Late-zone retrieval check: late zone has near-zero confident pellet.
LATE_FRACTION = 0.5
MAX_LATE_OFF_PILLAR_FOR_RETRIEVED = 3

# IFR / OKF emit.
OKF_SETTLE_OFFSET = 6


def _find_paw_past_y_line_bouts(paw_past_y):
    n = len(paw_past_y)
    bouts = []
    rs = -1
    for i in range(n):
        if paw_past_y[i]:
            if rs < 0:
                rs = i
        else:
            if rs >= 0:
                bouts.append((rs, i - 1))
                rs = -1
    if rs >= 0:
        bouts.append((rs, n - 1))
    return bouts


def _check_on_pillar_in_window(
    on_pillar_mask: np.ndarray,
    paw_past_y: np.ndarray,
    window_start: int,
    window_end: int,
    direction: str,  # "before" or "after"
    immediate_window_frames: int,
    min_paw_clear_required: int,
) -> Tuple[Optional[bool], int, int]:
    """Within the window, find paw-clear frames adjacent to the
    direction boundary, then count on-pillar frames among them.

    Returns (is_on_pillar, n_paw_clear, n_on_pillar) where
    is_on_pillar is True/False/None (None = insufficient paw-clear
    frames to decide).
    """
    n = len(on_pillar_mask)
    if direction == "before":
        # Walk backward from window_end (= reach start) looking for
        # paw-clear frames.
        clear_frames = []
        i = window_end - 1
        end_search = max(0, window_end - immediate_window_frames * 3)
        while i >= end_search and len(clear_frames) < immediate_window_frames:
            if i >= 0 and not paw_past_y[i]:
                clear_frames.append(i)
            i -= 1
    else:  # "after"
        # Walk forward from window_start (= reach end + 1).
        clear_frames = []
        i = window_start
        end_search = min(n, window_start + immediate_window_frames * 3)
        while i < end_search and len(clear_frames) < immediate_window_frames:
            if i < n and not paw_past_y[i]:
                clear_frames.append(i)
            i += 1
    if len(clear_frames) < min_paw_clear_required:
        return None, len(clear_frames), 0
    n_on = sum(1 for j in clear_frames if on_pillar_mask[j])
    # Pellet "on pillar" requires UNAMBIGUOUS evidence: all clear
    # frames on-pillar => True. NO clear frames on-pillar => False.
    # Mixed (some on, some off) => None (ambiguous, defer).
    if n_on == len(clear_frames):
        is_on = True
    elif n_on == 0:
        is_on = False
    else:
        is_on = None  # ambiguous
    return is_on, len(clear_frames), n_on


class Stage21CausalReachViaImmediateOnOffTransition(Stage):
    name = "stage_21_causal_reach_via_immediate_on_off_transition"
    target_class = None  # commits either displaced_sa or retrieved

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        immediate_window_frames: int = IMMEDIATE_WINDOW_FRAMES,
        min_paw_clear_frames_required: int = MIN_PAW_CLEAR_FRAMES_REQUIRED,
        sa_stable_min_frames: int = SA_STABLE_MIN_FRAMES,
        sa_stable_tolerance_px: float = SA_STABLE_TOLERANCE_PX,
        pellet_off_pillar_radii_for_sa: float = PELLET_OFF_PILLAR_RADII_FOR_SA,
        late_fraction: float = LATE_FRACTION,
        max_late_off_pillar_for_retrieved: int = MAX_LATE_OFF_PILLAR_FOR_RETRIEVED,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.immediate_window_frames = immediate_window_frames
        self.min_paw_clear_frames_required = min_paw_clear_frames_required
        self.sa_stable_min_frames = sa_stable_min_frames
        self.sa_stable_tolerance_px = sa_stable_tolerance_px
        self.pellet_off_pillar_radii_for_sa = pellet_off_pillar_radii_for_sa
        self.late_fraction = late_fraction
        self.max_late_off_pillar_for_retrieved = max_late_off_pillar_for_retrieved
        self.okf_settle_offset = okf_settle_offset
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
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
            paw_past_y |= (py <= slit_y_line) & (pl >= self.paw_lk_threshold)

        on_pillar = (
            (pellet_lk >= self.pellet_lk_high)
            & (dist_radii <= self.on_pillar_radii)
            & (~paw_past_y)
        )

        # GT reaches.
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

        # For each reach: compute IMMEDIATE pre and post on-pillar
        # state in paw-clear adjacent windows.
        reach_states = []
        for ri, (rs_local, re_local) in enumerate(reach_windows_local):
            pre_state, n_pre_clear, n_pre_on = _check_on_pillar_in_window(
                on_pillar, paw_past_y,
                window_start=0, window_end=rs_local, direction="before",
                immediate_window_frames=self.immediate_window_frames,
                min_paw_clear_required=self.min_paw_clear_frames_required,
            )
            post_state, n_post_clear, n_post_on = _check_on_pillar_in_window(
                on_pillar, paw_past_y,
                window_start=re_local + 1, window_end=n, direction="after",
                immediate_window_frames=self.immediate_window_frames,
                min_paw_clear_required=self.min_paw_clear_frames_required,
            )
            reach_states.append({
                "ri": ri,
                "pre_state": pre_state,
                "post_state": post_state,
                "n_pre_clear": n_pre_clear,
                "n_post_clear": n_post_clear,
                "n_pre_on": n_pre_on,
                "n_post_on": n_post_on,
            })

        # Find reaches with on->off transition: pre_state == True AND
        # post_state == False.
        transition_reaches = [s for s in reach_states
                              if s["pre_state"] is True
                              and s["post_state"] is False]
        feats["n_transition_reaches"] = len(transition_reaches)
        feats["reach_states"] = [
            (s["ri"], s["pre_state"], s["post_state"]) for s in reach_states
        ]
        if len(transition_reaches) == 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_reach_with_on_to_off_transition "
                    f"(no reach satisfied immediate-pre on-pillar AND "
                    f"immediate-post not-on-pillar)"
                ),
                features=feats,
            )
        if len(transition_reaches) > 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"multiple_reaches_with_on_to_off_transition "
                    f"({len(transition_reaches)} -- physically impossible "
                    f"if pellet detection were perfect; DLC noise present, "
                    f"defer)"
                ),
                features=feats,
            )

        causal = transition_reaches[0]
        causal_idx = causal["ri"]
        bs, be = reach_windows_local[causal_idx]
        feats["causal_idx"] = int(causal_idx)
        feats["causal_bout_start_idx"] = int(bs)
        feats["causal_bout_end_idx"] = int(be)

        # Defense: pre-causal-reach pellet must have been on-pillar
        # for sustained frames across the segment leading up to this
        # bout (not just immediately before). Filters cases where
        # DLC briefly saw pellet on-pillar in the immediate-pre window
        # but failed to track it earlier in the segment -- those cases
        # are unreliable, defer.
        pre_causal_on_pillar_total = int(on_pillar[:bs].sum())
        feats["pre_causal_on_pillar_total"] = pre_causal_on_pillar_total
        if pre_causal_on_pillar_total < 30:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_pre_causal_on_pillar_evidence "
                    f"({pre_causal_on_pillar_total} < 30 sustained on-"
                    f"pillar frames before causal reach -- DLC tracking "
                    f"unreliable, defer)"
                ),
                features=feats,
            )

        # Defense: pellet must NEVER return to confident on-pillar
        # after the causal reach. Per the memory rule "pellet cannot
        # return to pillar", any sustained on-pillar observation post-
        # causal-reach contradicts displacement/retrieval -- our
        # causal pick is wrong, defer.
        post_on_pillar_run_max = 0
        run = 0
        for i in range(be + 1, n):
            if on_pillar[i]:
                run += 1
                if run > post_on_pillar_run_max:
                    post_on_pillar_run_max = run
            else:
                run = 0
        feats["post_causal_on_pillar_max_run"] = int(post_on_pillar_run_max)
        # Sustained 3+ frame on-pillar post-causal-reach contradicts
        # the displacement (per "pellet cannot return to pillar" memory).
        # Tighter threshold (1 frame) loses too many true commits to
        # single-frame DLC blips. 3 frames is the empirical sweet spot.
        if post_on_pillar_run_max >= 3:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_returned_to_pillar_post_causal "
                    f"({post_on_pillar_run_max} sustained on-pillar frames "
                    f"post-causal-reach -- impossible for true "
                    f"displacement/retrieval; chosen reach is wrong, defer)"
                ),
                features=feats,
            )

        # Class determination: does the pellet appear stably in SA
        # AFTER the causal reach? If yes -> displaced. If no -> retrieved.
        # Stability check: confident off-pillar in-SA pellet observations
        # cluster in a tight position window.
        sabl_x = sub["SABL_x"].to_numpy(dtype=float)
        sabl_y = sub["SABL_y"].to_numpy(dtype=float)
        sabr_x = sub["SABR_x"].to_numpy(dtype=float)
        sabr_y = sub["SABR_y"].to_numpy(dtype=float)
        satl_x = sub["SATL_x"].to_numpy(dtype=float)
        satl_y = sub["SATL_y"].to_numpy(dtype=float)
        satr_x = sub["SATR_x"].to_numpy(dtype=float)
        satr_y = sub["SATR_y"].to_numpy(dtype=float)
        sa_top_y = (satl_y + satr_y) / 2.0
        sa_bot_y = (sabl_y + sabr_y) / 2.0
        sa_left_x = np.minimum(sabl_x, satl_x)
        sa_right_x = np.maximum(sabr_x, satr_x)
        in_sa = (
            (pellet_y >= sa_top_y) & (pellet_y <= sa_bot_y)
            & (pellet_x >= sa_left_x) & (pellet_x <= sa_right_x)
        )

        post_off_pillar_in_sa = (
            (pellet_lk >= self.pellet_lk_high)
            & (~paw_past_y)
            & (dist_radii > self.pellet_off_pillar_radii_for_sa)
            & in_sa
        )
        # Restrict to post-causal-reach window.
        post_mask = np.zeros(n, dtype=bool)
        post_mask[be + 1:] = True
        post_off_in_sa_eligible = post_off_pillar_in_sa & post_mask
        n_post_off_in_sa = int(post_off_in_sa_eligible.sum())
        feats["n_post_off_in_sa"] = n_post_off_in_sa

        # Stability check: position cluster.
        is_stable_in_sa = False
        cluster_xy = None
        if n_post_off_in_sa >= self.sa_stable_min_frames:
            obs_x = pellet_x[post_off_in_sa_eligible]
            obs_y = pellet_y[post_off_in_sa_eligible]
            med_x = float(np.median(obs_x))
            med_y = float(np.median(obs_y))
            dev = np.sqrt((obs_x - med_x) ** 2 + (obs_y - med_y) ** 2)
            n_near_median = int((dev <= self.sa_stable_tolerance_px).sum())
            is_stable_in_sa = n_near_median >= self.sa_stable_min_frames
            cluster_xy = (med_x, med_y, n_near_median)
            feats["sa_cluster"] = cluster_xy

        if is_stable_in_sa:
            committed_class = "displaced_sa"
            # Defense: if there are LATER GT reaches between the causal
            # reach and the first stable SA observation, the actual
            # displacement might have happened at one of THOSE reaches
            # (DLC briefly lost detection at algo's chosen reach,
            # creating false on->off transition; real displacement at
            # later reach). Defer.
            if cluster_xy is not None:
                first_stable_idx = -1
                obs_x = pellet_x[post_off_in_sa_eligible]
                obs_y = pellet_y[post_off_in_sa_eligible]
                med_x_local, med_y_local, _ = cluster_xy
                # Find first frame post-causal where pellet is at
                # cluster center.
                eligible_idx = np.where(post_off_in_sa_eligible)[0]
                for idx in eligible_idx:
                    dev = ((pellet_x[idx] - med_x_local) ** 2
                           + (pellet_y[idx] - med_y_local) ** 2) ** 0.5
                    if dev <= self.sa_stable_tolerance_px:
                        first_stable_idx = idx
                        break
                feats["first_stable_in_sa_idx"] = int(first_stable_idx)
                if first_stable_idx > 0:
                    n_intervening_reaches = sum(
                        1 for rs2, re2 in reach_windows_local
                        if rs2 > be and re2 < first_stable_idx
                    )
                    feats["intervening_reaches_to_stable"] = n_intervening_reaches
                    if n_intervening_reaches > 0:
                        return StageDecision(
                            decision="continue",
                            reason=(
                                f"intervening_reaches_before_stable_sa "
                                f"({n_intervening_reaches} reaches between "
                                f"causal reach end and first stable SA "
                                f"observation -- actual displacement may "
                                f"be at a later reach)"
                            ),
                            features=feats,
                        )
        else:
            # Check: late zone has near-zero off-pillar observations.
            # Tight (<=2) catches displaced cases where pellet was
            # briefly tracked late in segment but didn't form stable
            # cluster.
            late_start_idx = int(n * (1 - self.late_fraction))
            any_off_pillar = (
                (pellet_lk >= 0.7)
                & (dist_radii > 1.0)
                & (~paw_past_y)
            )
            late_off = int(any_off_pillar[late_start_idx:].sum())
            feats["late_off_pillar_for_retrieved_check"] = late_off
            if late_off > self.max_late_off_pillar_for_retrieved:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"causal_reach_identified_but_class_ambiguous "
                        f"(post-causal pellet not stable in SA but late "
                        f"zone has {late_off} off-pillar observations -- "
                        f"could be displaced with DLC noise)"
                    ),
                    features=feats,
                )
            committed_class = "retrieved"

        # IFR/OKF emit.
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = min(be + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
            "committed_class": committed_class,
        })
        return StageDecision(
            decision="commit",
            committed_class=committed_class,
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"causal_reach_via_immediate_on_off_transition "
                f"(reach {causal_idx}; pre-clear {causal['n_pre_clear']}f "
                f"with {causal['n_pre_on']} on-pillar; post-clear "
                f"{causal['n_post_clear']}f with {causal['n_post_on']} "
                f"on-pillar; class={committed_class})"
            ),
            features=feats,
        )
