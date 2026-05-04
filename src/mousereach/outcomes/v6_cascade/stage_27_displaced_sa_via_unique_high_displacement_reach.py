"""
Stage 27: Displaced_sa via unique high-displacement reach.

For residual segments where exactly ONE reach in the segment shows a
clear pellet displacement signal (pre-reach pellet position vs
post-reach position differs by >= MIN_DISPLACEMENT_PX) AND no reach
shows the per-reach vanish signal that Stage 26 would have caught,
commit displaced_sa with that reach as causal.

Per-reach signal computation: identical to Stage 26 -- isolated
windows capped at adjacent reach boundaries.

Predicate (commit displaced_sa iff all true):
  P1. Exactly one reach in segment has displacement >= MIN_DISPLACEMENT_PX
      (12 px).
  P2. No reach in segment has the vanish signal (pellet visible pre,
      gone post). Vanish-signal cases are Stage 26 territory; this
      stage only fires when there's NO vanish reach, i.e. the pellet
      stayed visible somewhere after every reach.
  P3. The high-displacement reach is the FIRST reach (chronologically)
      that shows any meaningful displacement (>= 5 px). Rationale: per
      "a pellet cannot move without a reach causing it" -- the causal
      reach is the FIRST one to actually move it. Later reaches just
      bounce/disturb the already-displaced pellet.
  P4. Post-displacement pellet remains visible off-pillar in subsequent
      windows (sustained off-pillar in-SA evidence). Rationale: confirms
      the pellet stayed in the apparatus (= displaced_sa) rather than
      being rapidly retrieved or DLC-lost. This is the inverse of Stage
      26's anti-bounce guard.

Empirical validation (residual touched segments after Stages 0-26):
  - 14/14 cid-known cases pick the correct GT-causal reach (100%)
  - Cross-class firing on retrieved residuals: 0/17 (clean separation)

Cascade emit on commit:
  - committed_class: "displaced_sa"
  - whens["interaction_frame"]: middle of the high-displacement reach
  - whens["outcome_known_frame"]: high-disp reach end + 5
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


PRE_WINDOW = 30
POST_WINDOW = 60
PELLET_CONF_THR = 0.7
MIN_DISPLACEMENT_PX = 10.0
MIN_PRIOR_DISPLACEMENT_PX = 5.0
OFF_PILLAR_RADII = 1.5
MIN_POST_OFF_PILLAR_FRAMES = 5
TRANSITION_ZONE_HALF = 5
OKF_OFFSET = 5
# Minimum pre/post-window frame counts for trusting the displacement
# estimate. Short windows (< this) capped by segment-start or adjacent
# reach edges produce noisy median pellet positions; treat the
# displacement as unmeasurable rather than confidently zero/large.
MIN_WINDOW_FRAMES = 5
# Triage if the chosen causal reach lands within edge buffers of
# segment boundaries. Empirical 2026-05-04: wrong commits cluster at
# d_end < 25 frames (ASPA reload / next-pellet position noise); correct
# commits have d_end >= 595. Start-edge transition zone shorter; correct
# commits land as close as d_start = 38, so start buffer is set lower.
START_EDGE_BUFFER_FRAMES = 30
END_EDGE_BUFFER_FRAMES = 60


class Stage27DisplacedSaViaUniqueHighDisplacement(Stage):
    name = "stage_27_displaced_sa_via_unique_high_displacement_reach"
    target_class = "displaced_sa"

    def __init__(
        self,
        pre_window: int = PRE_WINDOW,
        post_window: int = POST_WINDOW,
        pellet_conf_thr: float = PELLET_CONF_THR,
        min_displacement_px: float = MIN_DISPLACEMENT_PX,
        min_prior_displacement_px: float = MIN_PRIOR_DISPLACEMENT_PX,
        off_pillar_radii: float = OFF_PILLAR_RADII,
        min_post_off_pillar_frames: int = MIN_POST_OFF_PILLAR_FRAMES,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
        okf_offset: int = OKF_OFFSET,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.pellet_conf_thr = pellet_conf_thr
        self.min_displacement_px = min_displacement_px
        self.min_prior_displacement_px = min_prior_displacement_px
        self.off_pillar_radii = off_pillar_radii
        self.min_post_off_pillar_frames = min_post_off_pillar_frames
        self.transition_zone_half = transition_zone_half
        self.okf_offset = okf_offset

    def _per_reach_signal(
        self,
        dlc_df,
        rs: int,
        re: int,
        prev_re: Optional[int],
        next_rs: Optional[int],
        seg_start: int,
        seg_end: int,
    ):
        """Returns (vanish, displacement) for one reach."""
        pre_lo = max(seg_start, rs - self.pre_window)
        if prev_re is not None:
            pre_lo = max(pre_lo, prev_re + 1)
        pre_hi = rs
        post_lo = re + 1
        post_hi = min(seg_end + 1, re + 1 + self.post_window)
        if next_rs is not None:
            post_hi = min(post_hi, next_rs)
        if pre_hi - pre_lo < MIN_WINDOW_FRAMES or post_hi - post_lo < MIN_WINDOW_FRAMES:
            return False, None
        pre_slice = dlc_df.iloc[pre_lo:pre_hi]
        post_slice = dlc_df.iloc[post_lo:post_hi]
        pre_lk = pre_slice["Pellet_likelihood"].to_numpy(dtype=float)
        post_lk = post_slice["Pellet_likelihood"].to_numpy(dtype=float)
        pre_c = pre_lk >= self.pellet_conf_thr
        post_c = post_lk >= self.pellet_conf_thr
        if not pre_c.any():
            return False, None
        if not post_c.any():
            return True, None
        pre_x = pre_slice["Pellet_x"].to_numpy(dtype=float)
        pre_y = pre_slice["Pellet_y"].to_numpy(dtype=float)
        post_x = post_slice["Pellet_x"].to_numpy(dtype=float)
        post_y = post_slice["Pellet_y"].to_numpy(dtype=float)
        pre_mx = float(np.median(pre_x[pre_c]))
        pre_my = float(np.median(pre_y[pre_c]))
        post_mx = float(np.median(post_x[post_c]))
        post_my = float(np.median(post_y[post_c]))
        displacement = float(np.sqrt(
            (post_mx - pre_mx) ** 2 + (post_my - pre_my) ** 2))
        return False, displacement

    def _post_displacement_off_pillar_count(
        self,
        seg: SegmentInput,
        causal_re: int,
    ) -> int:
        """Count of frames after causal_re where pellet at lk >=
        pellet_conf_thr is off-pillar (>off_pillar_radii from pillar
        center). Confirms the pellet stayed in the apparatus."""
        scan_lo = causal_re + 1
        scan_hi = seg.seg_end - self.transition_zone_half
        if scan_hi <= scan_lo:
            return 0
        sub_raw = seg.dlc_df.iloc[scan_lo:scan_hi + 1]
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        try:
            geom = compute_pillar_geometry_series(sub)
        except Exception:
            return 0
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r = geom["pillar_r"].to_numpy(dtype=float)
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
        dist_radii = (np.sqrt((pellet_x - pillar_cx) ** 2
                              + (pellet_y - pillar_cy) ** 2)
                      / np.maximum(pillar_r, 1e-6))
        off_pillar = ((pellet_lk >= self.pellet_conf_thr)
                      & (dist_radii > self.off_pillar_radii))
        return int(off_pillar.sum())

    def decide(self, seg: SegmentInput) -> StageDecision:
        reaches: List[Tuple[int, int]] = sorted(
            [(int(rs), int(re)) for (rs, re) in seg.reach_windows
             if rs is not None and re is not None])
        feats = {"n_reaches": len(reaches)}
        if len(reaches) < 1:
            return StageDecision(
                decision="continue",
                reason="no_reaches_in_segment",
                features=feats,
            )

        signals = []
        for i, (rs, re) in enumerate(reaches):
            prev_re = reaches[i - 1][1] if i > 0 else None
            next_rs = reaches[i + 1][0] if i + 1 < len(reaches) else None
            vanish, disp = self._per_reach_signal(
                seg.dlc_df, rs, re, prev_re, next_rs,
                seg.seg_start, seg.seg_end)
            signals.append((vanish, disp))
        feats["signals"] = [
            (bool(v), None if d is None else round(d, 2))
            for v, d in signals
        ]

        # P2: no vanish reach (Stage 26 territory)
        if any(v for v, _ in signals):
            return StageDecision(
                decision="continue",
                reason="segment_has_vanish_reach (Stage 26 territory)",
                features=feats,
            )

        # P1: exactly one reach with disp >= threshold
        high_disp_idxs = [i for i, (_, d) in enumerate(signals)
                          if d is not None and d >= self.min_displacement_px]
        if len(high_disp_idxs) != 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"requires_exactly_one_high_disp_reach "
                    f"(found {len(high_disp_idxs)})"
                ),
                features=feats,
            )
        causal_idx = high_disp_idxs[0]
        causal_rs, causal_re = reaches[causal_idx]
        feats["causal_reach_idx"] = int(causal_idx)
        feats["causal_displacement"] = round(
            signals[causal_idx][1], 2)

        # P3: causal reach must be the FIRST chronologically with any
        # meaningful displacement. If an earlier reach had moderate
        # displacement, the causal picture is ambiguous.
        for i in range(causal_idx):
            _, d = signals[i]
            if d is not None and d >= self.min_prior_displacement_px:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"earlier_reach_already_showed_displacement "
                        f"(reach idx {i} disp={d:.2f} px >= "
                        f"{self.min_prior_displacement_px}); causal "
                        f"picture ambiguous"
                    ),
                    features=feats,
                )

        # P3.5: triage if causal reach is too near a segment edge.
        # Boundary noise (transition zone, ASPA cycling) makes reach
        # picks unreliable there.
        dist_to_start = causal_rs - seg.seg_start
        dist_to_end = seg.seg_end - causal_re
        if dist_to_start < START_EDGE_BUFFER_FRAMES:
            return StageDecision(
                decision="triage",
                reason=(
                    f"causal_reach_too_near_segment_start "
                    f"(reach starts {dist_to_start} frames into segment "
                    f"< {START_EDGE_BUFFER_FRAMES}; transition-zone noise "
                    f"makes commit unreliable -- send to manual review)"
                ),
                features=feats,
            )
        if dist_to_end < END_EDGE_BUFFER_FRAMES:
            return StageDecision(
                decision="triage",
                reason=(
                    f"causal_reach_too_near_segment_end "
                    f"(reach ends {dist_to_end} frames before segment end "
                    f"< {END_EDGE_BUFFER_FRAMES}; ASPA-reload / next-pellet "
                    f"noise makes commit unreliable -- send to manual review)"
                ),
                features=feats,
            )

        # P4: post-displacement pellet visible off-pillar
        post_off_count = self._post_displacement_off_pillar_count(
            seg, causal_re)
        feats["post_off_pillar_count"] = int(post_off_count)
        if post_off_count < self.min_post_off_pillar_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_post_displacement_off_pillar_evidence "
                    f"({post_off_count} frames at lk>="
                    f"{self.pellet_conf_thr} >{self.off_pillar_radii} "
                    f"radii < {self.min_post_off_pillar_frames}; "
                    f"can't confirm pellet stayed in apparatus)"
                ),
                features=feats,
            )

        # Commit displaced_sa.
        bout_length = causal_re - causal_rs + 1
        interaction_frame = int(causal_rs + bout_length // 2)
        okf = int(causal_re + self.okf_offset)
        feats["interaction_frame"] = interaction_frame
        feats["outcome_known_frame"] = okf
        return StageDecision(
            decision="commit",
            committed_class="displaced_sa",
            whens={
                "outcome_known_frame": okf,
                "interaction_frame": interaction_frame,
            },
            reason=(
                f"unique_high_displacement_reach_with_post_off_pillar_evidence "
                f"(causal reach idx {causal_idx}, disp="
                f"{signals[causal_idx][1]:.2f} px; "
                f"post off-pillar frames = {post_off_count})"
            ),
            features=feats,
        )
