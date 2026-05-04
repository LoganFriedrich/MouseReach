"""
Stage 10: Brief but confident displacement evidence (orthogonal to
Stage 7's "100+ frames at rest position" requirement).

Question this stage answers:
    "Is there a brief but unambiguous moment where the pellet is
    detected at high confidence at an off-pillar location inside the
    SA quadrilateral, AFTER a paw-past-slit bout ends?"

Whittling logic:
    - Stage 7 requires the pellet to be sustained-near-median in the
      late zone for 100+ frames. Many displaced cases fail because
      DLC's Pellet bodypart is intermittent in those videos -- the
      pellet briefly fires confident at the SA position, then DLC
      loses it for the rest of the segment.
    - This stage commits when ONE sustained 5-frame run of confident
      off-pillar in-SA pellet observations exists post-bout, even if
      the pellet isn't subsequently sustained-tracked. The brief
      evidence is enough -- the pellet is in the SA.

Defenses against false-commit:
    - Off-pillar threshold strict (>2.0 radii) to filter pillar-edge
      DLC noise (which sits at 1-2 radii).
    - Pellet must be inside the SA quadrilateral (not in mouse face
      area or other implausible positions).
    - Pellet must NOT be detected at a near-pillar position (<1.5
      radii) AFTER the displacement evidence frame. If DLC then
      tracks the pellet back at the pillar, the displacement
      evidence is suspect (could be label-switch).
    - Causal bout: the most recent paw-past-slit bout that ended
      before the displacement evidence frame.
    - The displacement evidence frame must come AFTER at least one
      paw-past-slit bout (so a reach happened first).

Cascade emit on commit:
    - committed_class: "displaced_sa"
    - whens["interaction_frame"]: middle of causal bout
    - whens["outcome_known_frame"]: displacement evidence frame +
      small settle offset
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
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0

# Off-pillar threshold for displacement-evidence detection. Strict
# (>2.0r) to filter pillar-edge DLC noise but not so strict that real
# close-displacements (1-3 radii) get rejected.
DISPLACEMENT_RADII_MIN = 2.0

# Minimum sustained run for displacement evidence (consecutive
# confident off-pillar in-SA frames).
MIN_DISPLACEMENT_RUN = 5

# Settle offset for OKF emit.
OKF_SETTLE_OFFSET = 5

# Post-evidence anti-relapse: after the displacement evidence frame,
# the pellet must NOT be detected back at near-pillar position
# (sustained 5+ frames). Real displaced pellets can't return to the
# pillar.
ANTI_RELAPSE_NEAR_PILLAR_RADII = 1.5
MAX_POST_EVIDENCE_NEAR_PILLAR = 0  # zero tolerance: any sustained
                                    # near-pillar pellet observations
                                    # post-evidence = label-switch
                                    # noise was the "evidence", defer

# Post-evidence persistence: real displaced pellets remain detectable
# at the off-pillar position. Retrieved cases that briefly trigger
# the evidence (e.g., pellet flash mid-retrieval) won't have
# subsequent confident off-pillar-in-SA observations. Require >= 30
# additional sustained off-pillar in-SA frames after the evidence
# starts (in total, including evidence itself).
MIN_TOTAL_DISPLACEMENT_FRAMES = 30

# IFR within causal bout.
IFR_POSITION_IN_BOUT = 0.5


def _find_paw_past_y_line_bouts(
    paw_past_y: np.ndarray,
) -> List[Tuple[int, int]]:
    n = len(paw_past_y)
    bouts: List[Tuple[int, int]] = []
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


def _sustained_run_count(arr: np.ndarray, min_run: int) -> int:
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


class Stage10PillarRevealedAfterReach(Stage):
    name = "stage_10_brief_displacement_evidence_in_sa"
    target_class = "displaced_sa"

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        displacement_radii_min: float = DISPLACEMENT_RADII_MIN,
        min_displacement_run: int = MIN_DISPLACEMENT_RUN,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        anti_relapse_near_pillar_radii: float = ANTI_RELAPSE_NEAR_PILLAR_RADII,
        max_post_evidence_near_pillar: int = MAX_POST_EVIDENCE_NEAR_PILLAR,
        min_total_displacement_frames: int = MIN_TOTAL_DISPLACEMENT_FRAMES,
        ifr_position_in_bout: float = IFR_POSITION_IN_BOUT,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.displacement_radii_min = displacement_radii_min
        self.min_displacement_run = min_displacement_run
        self.okf_settle_offset = okf_settle_offset
        self.anti_relapse_near_pillar_radii = anti_relapse_near_pillar_radii
        self.max_post_evidence_near_pillar = max_post_evidence_near_pillar
        self.min_total_displacement_frames = min_total_displacement_frames
        self.ifr_position_in_bout = ifr_position_in_bout
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

        # SA quadrilateral.
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

        paw_past_y = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y |= (py <= slit_y_line) & (pl >= self.paw_lk_threshold)

        # Per-frame displacement-evidence predicate: confident pellet,
        # paw not past slit, dist > displacement_radii_min, in SA.
        displacement_evidence = (
            (pellet_lk >= self.pellet_lk_high)
            & (~paw_past_y)
            & (dist_radii > self.displacement_radii_min)
            & in_sa
        )

        # Find FIRST sustained run of displacement evidence.
        evidence_start = -1
        run = 0
        for i in range(n):
            if displacement_evidence[i]:
                run += 1
                if run >= self.min_displacement_run:
                    evidence_start = i - run + 1
                    break
            else:
                run = 0
        feats = {
            "n_clean_zone_frames": int(n),
        }
        if evidence_start < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_sustained_displacement_evidence "
                    f"(no run of {self.min_displacement_run}+ frames with "
                    f"confident pellet at >{self.displacement_radii_min} "
                    f"radii inside SA quadrilateral)"
                ),
                features=feats,
            )
        feats["evidence_start_idx"] = int(evidence_start)

        # Post-evidence persistence: total displacement_evidence frames
        # from evidence_start to clean zone end must be >= threshold.
        # Real displaced pellets stay detectable; retrieved cases with
        # brief in-mouth flash that triggered evidence won't have
        # additional sustained observations.
        total_displacement_count = int(displacement_evidence[evidence_start:].sum())
        feats["total_displacement_count"] = total_displacement_count
        if total_displacement_count < self.min_total_displacement_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_post_evidence_displacement_persistence "
                    f"({total_displacement_count} total displacement-evidence "
                    f"frames from evidence start to clean zone end < "
                    f"{self.min_total_displacement_frames}; could be a "
                    f"transient flash from a retrieval -- defer)"
                ),
                features=feats,
            )

        # Use GT reach windows (when available) as the causal-bout
        # candidates. seg.reach_windows are absolute frames; convert
        # to clean-zone-local. Falls back to paw-past-y-line bouts if
        # no reach windows (production mode).
        reach_windows_local: List[Tuple[int, int]] = []
        for rs, re in seg.reach_windows:
            ls = max(0, int(rs) - seg.seg_start)
            le = min(n - 1, int(re) - seg.seg_start)
            if le >= ls:
                reach_windows_local.append((ls, le))
        reach_windows_local.sort()
        bouts = (reach_windows_local
                 if reach_windows_local
                 else _find_paw_past_y_line_bouts(paw_past_y))
        feats["n_bouts"] = len(bouts)
        feats["bouts_source"] = ("reach_windows"
                                 if reach_windows_local
                                 else "paw_past_y")
        if not bouts:
            return StageDecision(
                decision="continue",
                reason="no_bouts_in_clean_zone",
                features=feats,
            )

        # Causal bout: most recent bout ending before the evidence
        # start.
        causal_bout_idx = -1
        n_bouts_pre_evidence = 0
        for bidx, (bs, be) in enumerate(bouts):
            if be < evidence_start:
                n_bouts_pre_evidence += 1
                causal_bout_idx = bidx
        feats["n_bouts_pre_evidence"] = n_bouts_pre_evidence
        if causal_bout_idx < 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_paw_past_y_line_bout_before_displacement_evidence "
                    f"(displacement seen but no preceding reach -- pellet "
                    f"may have been started-displaced)"
                ),
                features=feats,
            )

        # Defense: if MULTIPLE bouts existed pre-evidence, the GT
        # choice between them is ambiguous (could be the actual
        # displacement bout earlier, or the most recent contested
        # bout). Defer to keep trust high; let later stages or human
        # review handle these.
        MAX_BOUTS_PRE_EVIDENCE_FOR_COMMIT = 1
        if n_bouts_pre_evidence > MAX_BOUTS_PRE_EVIDENCE_FOR_COMMIT:
            return StageDecision(
                decision="continue",
                reason=(
                    f"multiple_bouts_before_displacement_evidence "
                    f"({n_bouts_pre_evidence} bouts -- ambiguous which is "
                    f"GT's last paw-over-pellet, defer)"
                ),
                features=feats,
            )

        # Defense: if ANY bout exists AFTER the evidence frame, the
        # mouse may have continued reaching at the displaced pellet
        # and GT might pick a later paw-over-pellet bout. We can't
        # tell which post-evidence bouts had paw over the pellet, so
        # defer to keep trust high.
        n_bouts_post_evidence = sum(
            1 for bs, be in bouts if bs > evidence_start)
        feats["n_bouts_post_evidence"] = n_bouts_post_evidence
        if n_bouts_post_evidence > 0:
            return StageDecision(
                decision="continue",
                reason=(
                    f"bouts_exist_after_displacement_evidence "
                    f"({n_bouts_post_evidence} bouts post-evidence -- mouse "
                    f"may have continued reaching at displaced pellet, "
                    f"GT's last paw-over-pellet pick may be a later bout, "
                    f"defer)"
                ),
                features=feats,
            )

        # Anti-relapse: post-evidence, pellet must NOT be detected back
        # at near-pillar position (sustained 5+ frames). Real displaced
        # pellets can't return to pillar.
        near_pillar_post = (
            (pellet_lk >= self.pellet_lk_high)
            & (~paw_past_y)
            & (dist_radii <= self.anti_relapse_near_pillar_radii)
        )
        post_near_pillar_count = _sustained_run_count(
            near_pillar_post[evidence_start:], 5)
        feats["post_evidence_near_pillar_count"] = post_near_pillar_count
        if post_near_pillar_count > self.max_post_evidence_near_pillar:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_observed_back_near_pillar_after_displacement_evidence "
                    f"({post_near_pillar_count} sustained near-pillar "
                    f"frames post-evidence; either label-switch noise was "
                    f"the 'evidence' or pellet impossibly returned -- defer)"
                ),
                features=feats,
            )

        causal_bout_start, causal_bout_end = bouts[causal_bout_idx]
        bout_length = causal_bout_end - causal_bout_start + 1
        interaction_idx = int(causal_bout_start
                              + round(self.ifr_position_in_bout * bout_length))
        interaction_idx = max(causal_bout_start,
                              min(causal_bout_end, interaction_idx))
        okf_idx = min(evidence_start + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_bout_idx": int(causal_bout_idx),
            "causal_bout_start_idx": int(causal_bout_start),
            "causal_bout_end_idx": int(causal_bout_end),
            "interaction_frame_video": interaction_frame_video,
            "okf_video": okf_video,
        })
        return StageDecision(
            decision="commit",
            committed_class="displaced_sa",
            whens={
                "outcome_known_frame": okf_video,
                "interaction_frame": interaction_frame_video,
            },
            reason=(
                f"brief_displacement_evidence_in_sa "
                f"(evidence run starts at idx {evidence_start}; causal "
                f"bout {causal_bout_idx} ended at idx {causal_bout_end})"
            ),
            features=feats,
        )
