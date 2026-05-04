"""
Stage 9: Pellet vanished after reach (retrieved).

Reframe (2026-05-03, user-directed): by the time a segment reaches
this stage, the upstream cascade has already excluded `untouched`. By
apparatus physics the pellet was on the pillar at segment start (ASPA
loads it before each segment). So Stage 9 does NOT need to re-prove
"pellet was on pillar early" -- that's redundant with upstream.

Question this stage answers:
    "Given the segment is touched, does post-reach evidence show the
    pellet vanished (retrieved) rather than landing in the SA
    (displaced)?"

Causal reach pick (GT semantic = "last paw-over-pellet"):
- Walk right-to-left through reaches, picking the LAST reach.
- Chain backward: if the previous reach ends within CHAIN_GAP_THRESHOLD
  frames of the chosen reach's start, treat them as a single retrieval
  action (pick the last in chain as causal, but require the entire
  chain to have post-conditions consistent with retrieval).

Trust-preserving defenses (no wrongful commits):
- Late-visibility: pellet must be essentially gone in late zone.
- Post-causal vanish: from causal reach end onward, pellet must NOT
  be confidently observed (low-lk budget allows brief in-mouth track).
- Anti-displaced: from FIRST reach onward, pellet must NEVER be
  sustained-confidently observed off-pillar inside the SA quadrilateral.
  This is the displaced/retrieved discriminator that replaces the old
  pre-reach on-pillar gate.

Cascade emit on commit:
- committed_class: "retrieved"
- whens["interaction_frame"]: middle of causal reach window
- whens["outcome_known_frame"]: causal reach end + small offset
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

PELLET_LK_THR = 0.95
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0

# Late vanished signature: pellet essentially absent late in segment.
LATE_FRACTION = 0.5
MAX_LATE_VISIBILITY_FRAC = 0.1

# Chain back-to-back reaches into a single retrieval action.
# 20 covers the 13-14 frame gaps GT treats as one retrieval; longer
# gaps mean separate actions.
CHAIN_GAP_THRESHOLD = 20

# Minimum sustained-pellet run (consecutive high-lk frames) for the
# post-reach budget. Brief 1-2 frame DLC flickers don't count -- only
# multi-frame tracks are real pellet observations.
MIN_SUSTAINED_PELLET_RUN = 3

# Post-causal-reach pellet observation budgets, in SUSTAINED-frame
# count. Brief in-mouth track or short SA flickers consume budget;
# truly sustained tracks (longer than budget) reject. Tight (5)
# because any persistent pellet observation post-reach indicates the
# pellet didn't actually leave the apparatus (= displaced).
MAX_POST_OFF_PILLAR_FOR_RETRIEVED = 5
MAX_POST_ANY_PELLET_OBS = 5

# Anti-displaced gate: from first reach onward, sustained off-pillar
# in-SA pellet observations indicate displacement, not retrieval.
# Tight threshold (5 frames) -- the displaced-class signature is even
# brief SA tracking; loose budget admits wrong-class commits.
MAX_INSEG_OFF_PILLAR_IN_SA = 5

# Uncovered (un-annotated) paw activity in the gap between earliest
# pellet-visible frame and the chosen causal reach. Real missed reaches
# have ~10+ consecutive paw-past-y frames.
MAX_UNCOVERED_PAW_IN_GAP = 5
MIN_UNCOVERED_PAW_RUN = 10

OKF_VANISH_OFFSET = 5


class Stage9PelletVanishedAfterReach(Stage):
    name = "stage_9_pellet_vanished_after_reach"
    target_class = "retrieved"

    def __init__(
        self,
        late_fraction: float = LATE_FRACTION,
        max_late_visibility_frac: float = MAX_LATE_VISIBILITY_FRAC,
        pellet_lk_threshold: float = PELLET_LK_THR,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        chain_gap_threshold: int = CHAIN_GAP_THRESHOLD,
        max_post_off_pillar_for_retrieved: int = MAX_POST_OFF_PILLAR_FOR_RETRIEVED,
        max_post_any_pellet_obs: int = MAX_POST_ANY_PELLET_OBS,
        max_inseg_off_pillar_in_sa: int = MAX_INSEG_OFF_PILLAR_IN_SA,
        min_sustained_pellet_run: int = MIN_SUSTAINED_PELLET_RUN,
        max_uncovered_paw_in_gap: int = MAX_UNCOVERED_PAW_IN_GAP,
        min_uncovered_paw_run: int = MIN_UNCOVERED_PAW_RUN,
        okf_vanish_offset: int = OKF_VANISH_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.late_fraction = late_fraction
        self.max_late_visibility_frac = max_late_visibility_frac
        self.pellet_lk_threshold = pellet_lk_threshold
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.chain_gap_threshold = chain_gap_threshold
        self.max_post_off_pillar_for_retrieved = max_post_off_pillar_for_retrieved
        self.max_post_any_pellet_obs = max_post_any_pellet_obs
        self.max_inseg_off_pillar_in_sa = max_inseg_off_pillar_in_sa
        self.min_sustained_pellet_run = min_sustained_pellet_run
        self.max_uncovered_paw_in_gap = max_uncovered_paw_in_gap
        self.min_uncovered_paw_run = min_uncovered_paw_run
        self.okf_vanish_offset = okf_vanish_offset
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

        # Strict paw-past-y (>=2 paw bodyparts at lk>=0.7) -- empirical
        # 2026-05-03: cleanly separates real paw from DLC noise.
        paw_lks = np.stack([sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
                            for bp in PAW_BODYPARTS])
        n_paws_high_lk = (paw_lks >= 0.7).sum(axis=0)
        paw_past_y_strict = paw_past_y & (n_paws_high_lk >= 2)

        confident_pellet = (pellet_lk >= self.pellet_lk_threshold) & ~paw_past_y
        on_pillar = confident_pellet & (dist_radii <= self.on_pillar_radii)

        def sustained_mask(arr: np.ndarray, min_run: int) -> np.ndarray:
            """Boolean mask: True at frames inside a True-run of length
            >= min_run. Brief flickers below min_run drop to False."""
            out = np.zeros_like(arr, dtype=bool)
            run = 0
            for i in range(len(arr)):
                if arr[i]:
                    run += 1
                else:
                    if run >= min_run:
                        out[i - run:i] = True
                    run = 0
            if run >= min_run:
                out[len(arr) - run:] = True
            return out

        # SA quadrilateral mask for off-pillar-in-SA detection.
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
        off_pillar_in_sa_raw = (
            (pellet_lk >= 0.7)
            & (dist_radii > 1.5)
            & ~paw_past_y
            & in_sa
        )
        off_pillar_in_sa = sustained_mask(
            off_pillar_in_sa_raw, self.min_sustained_pellet_run)
        # Off-pillar anywhere (not just SA polygon). Catches displaced
        # cases where pellet lands at SA-edge position outside our
        # quadrilateral approximation. paw_past_y excluded so in-mouth
        # tracks during retrieval don't fire this gate.
        off_pillar_anywhere_raw = (
            (pellet_lk >= 0.7)
            & (dist_radii > 1.5)
            & ~paw_past_y
        )
        off_pillar_anywhere = sustained_mask(
            off_pillar_anywhere_raw, self.min_sustained_pellet_run)
        any_pellet_visible_raw = (pellet_lk >= 0.5) & ~paw_past_y
        any_pellet_visible = sustained_mask(
            any_pellet_visible_raw, self.min_sustained_pellet_run)

        # 1. Late vanished: pellet must be essentially gone in late zone.
        late_start_idx = int(n * (1 - self.late_fraction))
        late_paw_not_past = int((~paw_past_y[late_start_idx:]).sum())
        late_visible = int(confident_pellet[late_start_idx:].sum())
        late_visibility_frac = (late_visible / max(late_paw_not_past, 1))
        feats = {
            "n_clean_zone_frames": n,
            "late_visible_count": late_visible,
            "late_paw_not_past_count": late_paw_not_past,
            "late_visibility_frac": late_visibility_frac,
        }
        if late_visibility_frac > self.max_late_visibility_frac:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_still_visible_late_in_segment "
                    f"(visibility_frac={late_visibility_frac:.3f} > "
                    f"{self.max_late_visibility_frac}); pellet did not "
                    f"vanish -- not retrieved"
                ),
                features=feats,
            )

        # 2. Reaches present.
        reaches_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le < ls:
                continue
            reaches_local.append((ls, le))
        reaches_local.sort()
        feats["n_reaches"] = len(reaches_local)
        if not reaches_local:
            return StageDecision(
                decision="continue",
                reason="no_reaches_in_segment",
                features=feats,
            )

        # 3. Anti-displaced gate: from first reach onward, sustained
        # off-pillar pellet observations (anywhere with paw not past
        # slit) indicate the pellet is still in the apparatus =
        # displaced, not retrieved. Paw-past-slit frames are excluded
        # so brief in-mouth tracks during retrieval don't trigger.
        first_reach_start = reaches_local[0][0]
        inseg_off_pillar_count = int(off_pillar_anywhere[first_reach_start:].sum())
        feats["inseg_off_pillar_anywhere_count"] = inseg_off_pillar_count
        if inseg_off_pillar_count > self.max_inseg_off_pillar_in_sa:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sustained_off_pillar_post_first_reach "
                    f"({inseg_off_pillar_count} frames > "
                    f"{self.max_inseg_off_pillar_in_sa}); looks displaced "
                    f"not retrieved"
                ),
                features=feats,
            )

        # 4. Causal reach: walk LEFT-to-RIGHT, find FIRST reach
        # satisfying post-vanish conditions. After the causal reach the
        # pellet is gone -- so subsequent reaches ALSO trivially pass
        # post-vanish (paw-over-empty-pillar). GT semantic ("last
        # paw-over-pellet event") means the reach AFTER WHICH the
        # pellet first becomes invisible, which is the FIRST such reach.
        causal_first = None
        for ri, (rs_local, re_local) in enumerate(reaches_local):
            post_sustained_on_pillar = int(on_pillar[re_local + 1:].sum())
            post_off_pillar = int(off_pillar_in_sa[re_local + 1:].sum())
            post_any_pellet = int(any_pellet_visible[re_local + 1:].sum())
            if (post_sustained_on_pillar == 0
                    and post_off_pillar <= self.max_post_off_pillar_for_retrieved
                    and post_any_pellet <= self.max_post_any_pellet_obs):
                causal_first = ri
                break
        if causal_first is None:
            return StageDecision(
                decision="continue",
                reason="no_candidate_reach_for_retrieval",
                features=feats,
            )
        # Defense: if MANY earlier reaches were skipped because their
        # post-conditions failed (pellet visible after them), the
        # pellet was around for many reaches -- suggesting displaced
        # (mouse contesting it) rather than retrieved. Real retrievals
        # typically pass on first or near-first reach (small skip
        # budget for DLC noise on the pellet during contest).
        feats["skipped_reaches_before_causal"] = int(causal_first)
        if causal_first > 3:
            return StageDecision(
                decision="continue",
                reason=(
                    f"too_many_skipped_reaches_before_first_passing "
                    f"({causal_first} earlier reaches failed post-conditions; "
                    f"pellet was around for many reaches -- looks like a "
                    f"contested displaced case, not a clean retrieval)"
                ),
                features=feats,
            )

        # Chain forward: back-to-back reaches starting within
        # CHAIN_GAP_THRESHOLD frames of the previous reach's end are a
        # single retrieval action. The IFR anchor is the LAST reach in
        # chain (GT picks last paw-over-pellet within the chain).
        causal_reach_idx = causal_first
        while causal_reach_idx + 1 < len(reaches_local):
            cur_re = reaches_local[causal_reach_idx][1]
            next_rs = reaches_local[causal_reach_idx + 1][0]
            if next_rs - cur_re <= self.chain_gap_threshold:
                causal_reach_idx += 1
            else:
                break
        feats["chain_first_reach_idx"] = int(causal_first)
        feats["causal_reach_idx"] = int(causal_reach_idx)

        # 5. Uncovered-paw defense: between segment start and the chain
        # start, sustained un-annotated paw-past-y activity (using the
        # strict paw filter to ignore DLC noise) suggests an unlabeled
        # reach -- we'd be picking the wrong causal reach. Defer.
        chain_start_local = reaches_local[causal_first][0]
        if chain_start_local > 0:
            covered_mask = np.zeros(n, dtype=bool)
            for rs, re in reaches_local:
                covered_mask[rs:re + 1] = True
            uncovered_paw_mask = paw_past_y_strict & ~covered_mask
            sustained_uncovered = 0
            run = 0
            for i in range(0, chain_start_local):
                if uncovered_paw_mask[i]:
                    run += 1
                else:
                    if run >= self.min_uncovered_paw_run:
                        sustained_uncovered += run
                    run = 0
            if run >= self.min_uncovered_paw_run:
                sustained_uncovered += run
            feats["uncovered_paw_in_gap"] = sustained_uncovered
            if sustained_uncovered > self.max_uncovered_paw_in_gap:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"unannotated_paw_activity_in_gap "
                        f"({sustained_uncovered} sustained paw-past-y frames "
                        f"before chain start are not in any GT reach -- "
                        f"causal reach likely unlabeled, defer)"
                    ),
                    features=feats,
                )

        # 6. Commit retrieved.
        causal_bout_start, causal_bout_end = reaches_local[causal_reach_idx]
        bout_length = causal_bout_end - causal_bout_start + 1
        interaction_idx = causal_bout_start + bout_length // 2
        okf_idx = min(causal_bout_end + self.okf_vanish_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_bout_start_idx": int(causal_bout_start),
            "causal_bout_end_idx": int(causal_bout_end),
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
                f"pellet_vanished_after_reach "
                f"(late visibility: {late_visibility_frac:.3f}; causal "
                f"reach {causal_reach_idx}, chain from {causal_first}; "
                f"in-segment off-pillar-in-SA: {inseg_off_pillar_count})"
            ),
            features=feats,
        )
