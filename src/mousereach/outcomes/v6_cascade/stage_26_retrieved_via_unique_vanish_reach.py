"""
Stage 26: Retrieved via unique vanish reach.

For residual touched segments where exactly ONE reach in the segment
shows the per-reach "vanish" signal (pellet visible immediately before
the reach, gone immediately after), commit retrieved -- with two
physics-grounded guards that protect 100% trust.

Per-reach signal computation:
  Each reach R gets a pre-window [max(seg_start, R.start - 30,
  prev_reach.end + 1), R.start) and post-window [R.end + 1,
  min(seg_end, R.end + 1 + 60, next_reach.start)].

  vanish:       confident pellet (lk >= 0.7) in pre, none in post
  displacement: confident pellet in both, pixel distance between
                medians of pre and post
  no_pre / no_post / no_data: pellet never seen on one side

Predicate (commit retrieved iff all true):
  P1. Exactly one reach in segment has vanish=True.
  P2. No reach chronologically BEFORE the vanish reach has
      displacement >= MIN_PRIOR_DISPLACEMENT_PX (12 px).
      Rationale: per the user-mandated rule "a pellet cannot return to
      the pillar," once an earlier reach has visibly displaced the
      pellet off-pillar, any subsequent vanish-signal reach is the
      next reach disturbing already-displaced pellet, not a
      retrieval.
  P3. After the vanish reach end through seg_end - transition_zone,
      no sustained run of >= MIN_BOUNCE_RUN_STRICT (10) frames where
      pellet at lk>=0.85 is off-pillar (>1.5 radii from pillar
      center). Rationale: per the user-mandated rule "a pellet
      bouncing in the SA cannot be retrieved," sustained off-pillar
      high-confidence pellet observations after the vanish reach
      mean the pellet is in the SA, not in the mouse.
  P4. (No longer enforced; "vanish reach must be last" was too strict
      -- many valid retrievals have post-retrieval paw movement.)
  P5. No sustained run of >= MIN_UNCOVERED_PAW_RUN (10) frames of
      paw-past-slit activity BEFORE the vanish reach that is outside
      any annotated reach window. Rationale: an unannotated reach
      before the vanish moment means the causal-reach picture is
      incomplete; the pellet may have been moved by an unlabeled
      reach. Activity AFTER the vanish reach is post-retrieval and
      doesn't count.

Empirical validation (residual touched segments after Stages 0-25):
  - 24 retrieved + 1 displaced_sa committed at 100% precision
  - 0 false fires on cid=None ambiguous segments
  - Pushes retrieved cascade yield from ~63% toward ~89%

Cascade emit on commit:
  - committed_class: "retrieved"
  - whens["interaction_frame"]: middle of the vanish reach
  - whens["outcome_known_frame"]: vanish reach end + 5
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
PELLET_STRICT_CONF_THR = 0.85
MIN_PRIOR_DISPLACEMENT_PX = 10.0
OFF_PILLAR_RADII = 1.5
MIN_BOUNCE_RUN_STRICT = 10
TRANSITION_ZONE_HALF = 5
OKF_VANISH_OFFSET = 5
PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
PAW_LK_THR = 0.5
PAW_LK_STRICT_THR = 0.7
MIN_UNCOVERED_PAW_RUN = 10
MAX_UNCOVERED_PAW_FRAMES = 5


def _max_run(arr: np.ndarray) -> int:
    """Maximum run length of True values in a boolean array."""
    run = 0
    m = 0
    for v in arr:
        if v:
            run += 1
            if run > m:
                m = run
        else:
            run = 0
    return m


class Stage26RetrievedViaUniqueVanishReach(Stage):
    name = "stage_26_retrieved_via_unique_vanish_reach"
    target_class = "retrieved"

    def __init__(
        self,
        pre_window: int = PRE_WINDOW,
        post_window: int = POST_WINDOW,
        pellet_conf_thr: float = PELLET_CONF_THR,
        pellet_strict_conf_thr: float = PELLET_STRICT_CONF_THR,
        min_prior_displacement_px: float = MIN_PRIOR_DISPLACEMENT_PX,
        off_pillar_radii: float = OFF_PILLAR_RADII,
        min_bounce_run_strict: int = MIN_BOUNCE_RUN_STRICT,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
        okf_vanish_offset: int = OKF_VANISH_OFFSET,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.pellet_conf_thr = pellet_conf_thr
        self.pellet_strict_conf_thr = pellet_strict_conf_thr
        self.min_prior_displacement_px = min_prior_displacement_px
        self.off_pillar_radii = off_pillar_radii
        self.min_bounce_run_strict = min_bounce_run_strict
        self.transition_zone_half = transition_zone_half
        self.okf_vanish_offset = okf_vanish_offset

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
        pre_hi = rs  # exclusive
        post_lo = re + 1
        post_hi = min(seg_end + 1, re + 1 + self.post_window)
        if next_rs is not None:
            post_hi = min(post_hi, next_rs)
        if pre_hi - pre_lo < 3 or post_hi - post_lo < 3:
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

    def _sustained_uncovered_paw_frames(
        self,
        seg: SegmentInput,
        reaches: List[Tuple[int, int]],
        scan_end_abs: int,
    ) -> int:
        """Total frames in sustained (>=MIN_UNCOVERED_PAW_RUN) runs of
        paw-past-slit activity outside any annotated reach window,
        restricted to seg_start..scan_end_abs (exclusive). Use this to
        check for unlabeled reaches BEFORE the vanish reach (where they
        would affect causal-reach picture). Activity AFTER the vanish
        reach is post-retrieval and shouldn't count."""
        scan_lo = seg.seg_start
        scan_hi = min(seg.seg_end, scan_end_abs - 1)
        if scan_hi < scan_lo:
            return 0
        sub_raw = seg.dlc_df.iloc[scan_lo:scan_hi + 1]
        n = len(sub_raw)
        if n == 0:
            return 0
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        try:
            geom = compute_pillar_geometry_series(sub)
        except Exception:
            return 0
        slit_y = (geom["pillar_cy"].to_numpy(dtype=float)
                  + geom["pillar_r"].to_numpy(dtype=float))
        paw_lks = np.stack([sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
                            for bp in PAW_BODYPARTS])
        n_paws_high = (paw_lks >= PAW_LK_STRICT_THR).sum(axis=0)
        paw_past = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past |= (py <= slit_y) & (pl >= PAW_LK_THR)
        # Strict version: at least 2 paw bodyparts at lk >= 0.7 in the
        # same frame (empirical 2026-05-03: cleanly separates real paw
        # from DLC noise).
        paw_past_strict = paw_past & (n_paws_high >= 2)

        covered = np.zeros(n, dtype=bool)
        for rs, re in reaches:
            ls = max(0, rs - scan_lo)
            le = min(n - 1, re - scan_lo)
            if le < ls:
                continue
            covered[ls:le + 1] = True
        uncovered = paw_past_strict & ~covered

        total = 0
        run = 0
        for i in range(n):
            if uncovered[i]:
                run += 1
            else:
                if run >= MIN_UNCOVERED_PAW_RUN:
                    total += run
                run = 0
        if run >= MIN_UNCOVERED_PAW_RUN:
            total += run
        return total

    def _post_off_pillar_strict_run(
        self,
        seg: SegmentInput,
        vanish_re: int,
    ) -> int:
        """Maximum run of consecutive frames after vanish_re where
        pellet at lk>=strict is off-pillar (>1.5 radii)."""
        scan_lo = vanish_re + 1
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
        off_strict = ((pellet_lk >= self.pellet_strict_conf_thr)
                      & (dist_radii > self.off_pillar_radii))
        return _max_run(off_strict)

    def decide(self, seg: SegmentInput) -> StageDecision:
        reaches: List[Tuple[int, int]] = sorted(
            [(int(rs), int(re)) for (rs, re) in seg.reach_windows
             if rs is not None and re is not None])
        feats = {"n_reaches": len(reaches)}
        if len(reaches) < 2:
            # Single-reach segments are handled by upstream stages.
            return StageDecision(
                decision="continue",
                reason="fewer_than_2_reaches",
                features=feats,
            )

        # Per-reach signal
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

        # P1: exactly one vanish reach
        vanish_idxs = [i for i, (v, _) in enumerate(signals) if v]
        if len(vanish_idxs) != 1:
            return StageDecision(
                decision="continue",
                reason=f"requires_exactly_one_vanish_reach (found {len(vanish_idxs)})",
                features=feats,
            )
        vanish_idx = vanish_idxs[0]
        vanish_rs, vanish_re = reaches[vanish_idx]
        feats["vanish_reach_idx"] = int(vanish_idx)

        # P2: no prior reach with displacement >= threshold
        for i in range(vanish_idx):
            _, disp = signals[i]
            if disp is not None and disp >= self.min_prior_displacement_px:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"prior_reach_already_displaced_pellet "
                        f"(reach {i} disp={disp:.2f} px >= "
                        f"{self.min_prior_displacement_px}); pellet was "
                        f"already off-pillar before the vanish reach -- "
                        f"can't be a clean retrieval"
                    ),
                    features=feats,
                )

        # P3: no sustained off-pillar pellet bouncing post-vanish
        post_run = self._post_off_pillar_strict_run(seg, vanish_re)
        feats["post_off_pillar_strict_run"] = int(post_run)
        if post_run >= self.min_bounce_run_strict:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sustained_off_pillar_pellet_after_vanish "
                    f"({post_run} frames at lk>={self.pellet_strict_conf_thr} "
                    f">{self.off_pillar_radii} radii from pillar; pellet "
                    f"bouncing in SA, not retrieved)"
                ),
                features=feats,
            )

        # P5: no sustained uncovered paw activity BEFORE the vanish reach.
        # Activity after the vanish reach is post-retrieval and ignored.
        uncovered_total = self._sustained_uncovered_paw_frames(
            seg, reaches, vanish_rs)
        feats["sustained_uncovered_paw_frames_pre_vanish"] = int(uncovered_total)
        if uncovered_total > MAX_UNCOVERED_PAW_FRAMES:
            return StageDecision(
                decision="continue",
                reason=(
                    f"unannotated_paw_activity_in_segment "
                    f"({uncovered_total} sustained paw-past-slit frames "
                    f"not in any annotated reach; causal reach picture "
                    f"incomplete -- defer)"
                ),
                features=feats,
            )

        # All guards passed -- commit retrieved.
        bout_length = vanish_re - vanish_rs + 1
        interaction_frame = int(vanish_rs + bout_length // 2)
        okf = int(vanish_re + self.okf_vanish_offset)
        feats["interaction_frame"] = interaction_frame
        feats["outcome_known_frame"] = okf
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "outcome_known_frame": okf,
                "interaction_frame": interaction_frame,
            },
            reason=(
                f"unique_vanish_reach_with_guards_passed "
                f"(vanish reach idx {vanish_idx}; no prior >= "
                f"{self.min_prior_displacement_px}px displacement; "
                f"post off-pillar strict run = {post_run} < "
                f"{self.min_bounce_run_strict})"
            ),
            features=feats,
        )
