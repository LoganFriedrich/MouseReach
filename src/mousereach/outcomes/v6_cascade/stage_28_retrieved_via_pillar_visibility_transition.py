"""
Stage 28: Retrieved via pillar visibility transition with vanish guard.

For residual segments where exactly ONE reach shows a strong pillar-
likelihood transition (pillar becomes confidently visible after the
reach, having been occluded before it) AND the same reach shows the
pellet-vanish signal (pellet visible before, gone after), commit
retrieved with that reach as causal.

Physics rationale:
  When a pellet sits on the pillar, the pellet bodypart is detected
  confidently but the pillar bodypart is occluded (low lk). When the
  mouse retrieves the pellet, the pellet disappears (vanish signal)
  and the pillar tip becomes visible (pillar lk rises). The
  combination of both signals is specific to retrieval -- displacement
  does NOT cause the pellet to vanish (it stays visible in the SA),
  and the pillar-lk-transition alone could fire on either class.

Per-reach signal computation: identical to Stage 26 windowing (isolated
pre/post windows capped at adjacent reach boundaries, 30 pre / 60 post).

Predicate (commit retrieved iff all true):
  P1. Exactly one reach in segment has:
      - pillar_lk_delta > PILLAR_DELTA_THR (0.5)
      - pillar_lk_pre  < PILLAR_PRE_THR  (0.4)
      - pillar_lk_post > PILLAR_POST_THR (0.8)
  P2. That same reach also shows vanish=True (pellet visible pre at
      lk >= 0.7, no confident pellet detection post).
  P3. Minimum pre/post window lengths of MIN_WINDOW_FRAMES (5).

Why the vanish guard works:
  In the dev corpus residual pool, ALL displaced_sa cross-fires (7 cases)
  had vanish=False (pellet remained visible off-pillar). ALL retrieved
  fires that passed the vanish guard were correct (9/9). The vanish
  signal cleanly separates the two classes in this context because
  retrieved = pellet leaves the apparatus (vanishes), displaced =
  pellet stays visible in the SA.

Empirical validation (residual pool after Stages 0-27):
  - 9/9 retrieved correct at 100% precision (all cid-known)
  - 0 cross-class fires on displaced_sa residuals (vanish guard blocks all 7)
  - 12/12 IFR bout matches (SAME_BOUT)

Cascade emit on commit:
  - committed_class: "retrieved"
  - whens["interaction_frame"]: middle of the qualifying reach
  - whens["outcome_known_frame"]: qualifying reach end + 5
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .stage_base import SegmentInput, Stage, StageDecision


PRE_WINDOW = 30
POST_WINDOW = 60
PELLET_CONF_THR = 0.7
PILLAR_DELTA_THR = 0.5
PILLAR_PRE_THR = 0.4
PILLAR_POST_THR = 0.8
MIN_WINDOW_FRAMES = 5
OKF_OFFSET = 5


class Stage28RetrievedViaPillarVisibilityTransition(Stage):
    name = "stage_28_retrieved_via_pillar_visibility_transition"
    target_class = "retrieved"

    def __init__(
        self,
        pre_window: int = PRE_WINDOW,
        post_window: int = POST_WINDOW,
        pellet_conf_thr: float = PELLET_CONF_THR,
        pillar_delta_thr: float = PILLAR_DELTA_THR,
        pillar_pre_thr: float = PILLAR_PRE_THR,
        pillar_post_thr: float = PILLAR_POST_THR,
        min_window_frames: int = MIN_WINDOW_FRAMES,
        okf_offset: int = OKF_OFFSET,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.pellet_conf_thr = pellet_conf_thr
        self.pillar_delta_thr = pillar_delta_thr
        self.pillar_pre_thr = pillar_pre_thr
        self.pillar_post_thr = pillar_post_thr
        self.min_window_frames = min_window_frames
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
        """Returns (pillar_qualifies, vanish) for one reach.

        pillar_qualifies: True if pillar lk rises from low pre to high
        post across this reach (pellet was occluding pillar, then
        removed).

        vanish: True if pellet was confidently detected before the
        reach but not after (pellet left the apparatus).
        """
        pre_lo = max(seg_start, rs - self.pre_window)
        if prev_re is not None:
            pre_lo = max(pre_lo, prev_re + 1)
        pre_hi = rs
        post_lo = re + 1
        post_hi = min(seg_end + 1, re + 1 + self.post_window)
        if next_rs is not None:
            post_hi = min(post_hi, next_rs)
        if (pre_hi - pre_lo < self.min_window_frames
                or post_hi - post_lo < self.min_window_frames):
            return False, False, {}
        pre_slice = dlc_df.iloc[pre_lo:pre_hi]
        post_slice = dlc_df.iloc[post_lo:post_hi]

        # Pillar lk transition
        pillar_qualifies = False
        pillar_pre_mean = None
        pillar_post_mean = None
        pillar_delta = None
        if "Pillar_likelihood" in pre_slice.columns:
            pillar_pre_mean = float(np.mean(
                pre_slice["Pillar_likelihood"].to_numpy(dtype=float)))
            pillar_post_mean = float(np.mean(
                post_slice["Pillar_likelihood"].to_numpy(dtype=float)))
            pillar_delta = pillar_post_mean - pillar_pre_mean
            pillar_qualifies = (
                pillar_delta > self.pillar_delta_thr
                and pillar_pre_mean < self.pillar_pre_thr
                and pillar_post_mean > self.pillar_post_thr
            )

        # Pellet vanish
        pre_lk = pre_slice["Pellet_likelihood"].to_numpy(dtype=float)
        post_lk = post_slice["Pellet_likelihood"].to_numpy(dtype=float)
        pre_conf = pre_lk >= self.pellet_conf_thr
        post_conf = post_lk >= self.pellet_conf_thr
        vanish = bool(pre_conf.any() and not post_conf.any())

        extras = {
            "pillar_pre_mean": pillar_pre_mean,
            "pillar_post_mean": pillar_post_mean,
            "pillar_delta": pillar_delta,
        }
        return pillar_qualifies, vanish, extras

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

        # Compute per-reach signals
        signals = []
        for i, (rs, re) in enumerate(reaches):
            prev_re = reaches[i - 1][1] if i > 0 else None
            next_rs = reaches[i + 1][0] if i + 1 < len(reaches) else None
            pillar_q, vanish, extras = self._per_reach_signal(
                seg.dlc_df, rs, re, prev_re, next_rs,
                seg.seg_start, seg.seg_end)
            signals.append({
                "pillar_qualifies": pillar_q,
                "vanish": vanish,
                "rs": rs, "re": re,
                **extras,
            })

        feats["signals"] = [
            {"pillar_q": s["pillar_qualifies"], "vanish": s["vanish"],
             "pillar_delta": round(s.get("pillar_delta"), 3) if s.get("pillar_delta") is not None else None}
            for s in signals
        ]

        # P1: exactly one reach with pillar-lk transition
        pillar_idxs = [i for i, s in enumerate(signals) if s["pillar_qualifies"]]
        if len(pillar_idxs) != 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"requires_exactly_one_pillar_transition_reach "
                    f"(found {len(pillar_idxs)})"
                ),
                features=feats,
            )
        pidx = pillar_idxs[0]
        sig = signals[pidx]
        feats["qualifying_reach_idx"] = pidx

        # P2: that reach must also show vanish
        if not sig["vanish"]:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pillar_transition_reach_lacks_vanish_signal "
                    f"(pillar delta={sig['pillar_delta']:.3f} qualifies "
                    f"but pellet remained visible after reach -- could be "
                    f"displacement, not retrieval)"
                ),
                features=feats,
            )

        # All predicates passed -- commit retrieved.
        rs, re = sig["rs"], sig["re"]
        bout_length = re - rs + 1
        interaction_frame = int(rs + bout_length // 2)
        okf = int(re + self.okf_offset)
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
                f"unique_pillar_visibility_transition_with_vanish "
                f"(reach idx {pidx}; pillar delta="
                f"{sig['pillar_delta']:.3f}, pre={sig['pillar_pre_mean']:.3f}, "
                f"post={sig['pillar_post_mean']:.3f}; pellet vanished -- "
                f"consistent with retrieval)"
            ),
            features=feats,
        )
