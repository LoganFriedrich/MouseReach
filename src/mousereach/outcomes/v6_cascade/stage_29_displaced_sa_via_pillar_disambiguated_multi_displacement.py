"""
Stage 29: Displaced_sa via pillar-visibility-disambiguated multi-displacement.

For residual segments where MULTIPLE reaches show high displacement
(>= MIN_DISPLACEMENT_PX) -- cases that Stage 27 defers because
uniqueness fails -- use the pillar-visibility-transition signal to
disambiguate which reach is causal. The causal reach is the one that
reveals the pillar (pillar lk transitions from low to high), because
that is the reach that first moved the pellet off the pillar tip.

Physics rationale:
  When a pellet sits on the pillar, the pillar bodypart is occluded
  (low DLC likelihood). The reach that displaces the pellet off the
  pillar is the one after which the pillar becomes visible (high lk).
  Subsequent reaches may further disturb the already-displaced pellet
  (producing additional high-displacement signals), but the pillar was
  already visible before those reaches -- so their pillar_pre is high,
  not low. The low-pre -> high-post transition uniquely identifies
  the first-mover reach.

Per-reach signal computation: identical to Stage 27 windowing.

Predicate (commit displaced_sa iff all true):
  P1. Two or more reaches in segment have displacement >= MIN_DISPLACEMENT_PX.
  P2. Among those high-displacement reaches, exactly ONE has:
      - pillar_lk_delta > PILLAR_DELTA_THR (0.5)
      - pillar_lk_pre  < PILLAR_PRE_THR  (0.4)
      - pillar_lk_post > PILLAR_POST_THR (0.8)
  P3. No reach in segment has the vanish signal (Stage 26 territory).

Empirical validation (residual pool after Stages 0-27):
  - 6/6 correct at 100% precision (all cid-known, all pick GT causal)
  - 0 cross-class fires on retrieved residuals (no retrieved has 2+
    reaches with disp >= 10)
  - Resolves all MULTI_HIGH_DISP residuals

Cascade emit on commit:
  - committed_class: "displaced_sa"
  - whens["interaction_frame"]: middle of the disambiguated reach
  - whens["outcome_known_frame"]: disambiguated reach end + 5
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .stage_base import SegmentInput, Stage, StageDecision


PRE_WINDOW = 30
POST_WINDOW = 60
PELLET_CONF_THR = 0.7
MIN_DISPLACEMENT_PX = 10.0
PILLAR_DELTA_THR = 0.5
PILLAR_PRE_THR = 0.4
PILLAR_POST_THR = 0.8
MIN_WINDOW_FRAMES = 5
OKF_OFFSET = 5


class Stage29DisplacedSaViaPillarDisambiguatedMultiDisplacement(Stage):
    name = "stage_29_displaced_sa_via_pillar_disambiguated_multi_displacement"
    target_class = "displaced_sa"

    def __init__(
        self,
        pre_window: int = PRE_WINDOW,
        post_window: int = POST_WINDOW,
        pellet_conf_thr: float = PELLET_CONF_THR,
        min_displacement_px: float = MIN_DISPLACEMENT_PX,
        pillar_delta_thr: float = PILLAR_DELTA_THR,
        pillar_pre_thr: float = PILLAR_PRE_THR,
        pillar_post_thr: float = PILLAR_POST_THR,
        min_window_frames: int = MIN_WINDOW_FRAMES,
        okf_offset: int = OKF_OFFSET,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.pellet_conf_thr = pellet_conf_thr
        self.min_displacement_px = min_displacement_px
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
        """Returns (vanish, displacement, pillar_qualifies, extras)."""
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
            return False, None, False, {}
        pre_slice = dlc_df.iloc[pre_lo:pre_hi]
        post_slice = dlc_df.iloc[post_lo:post_hi]

        # Pellet signals
        pre_lk = pre_slice["Pellet_likelihood"].to_numpy(dtype=float)
        post_lk = post_slice["Pellet_likelihood"].to_numpy(dtype=float)
        pre_c = pre_lk >= self.pellet_conf_thr
        post_c = post_lk >= self.pellet_conf_thr
        vanish = False
        displacement = None
        if not pre_c.any():
            pass  # no pre pellet detection
        elif not post_c.any():
            vanish = True
        else:
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

        extras = {
            "pillar_pre_mean": pillar_pre_mean,
            "pillar_post_mean": pillar_post_mean,
            "pillar_delta": pillar_delta,
        }
        return vanish, displacement, pillar_qualifies, extras

    def decide(self, seg: SegmentInput) -> StageDecision:
        reaches: List[Tuple[int, int]] = sorted(
            [(int(rs), int(re)) for (rs, re) in seg.reach_windows
             if rs is not None and re is not None])
        feats = {"n_reaches": len(reaches)}
        if len(reaches) < 2:
            return StageDecision(
                decision="continue",
                reason="fewer_than_2_reaches",
                features=feats,
            )

        # Compute per-reach signals
        signals = []
        for i, (rs, re) in enumerate(reaches):
            prev_re = reaches[i - 1][1] if i > 0 else None
            next_rs = reaches[i + 1][0] if i + 1 < len(reaches) else None
            vanish, disp, pillar_q, extras = self._per_reach_signal(
                seg.dlc_df, rs, re, prev_re, next_rs,
                seg.seg_start, seg.seg_end)
            signals.append({
                "vanish": vanish, "disp": disp,
                "pillar_qualifies": pillar_q,
                "rs": rs, "re": re,
                **extras,
            })

        feats["signals"] = [
            {"vanish": s["vanish"],
             "disp": round(s["disp"], 2) if s["disp"] is not None else None,
             "pillar_q": s["pillar_qualifies"],
             "pillar_delta": round(s.get("pillar_delta"), 3) if s.get("pillar_delta") is not None else None}
            for s in signals
        ]

        # P3: no vanish reach (Stage 26 territory)
        if any(s["vanish"] for s in signals):
            return StageDecision(
                decision="continue",
                reason="segment_has_vanish_reach (Stage 26 territory)",
                features=feats,
            )

        # P1: two or more reaches with disp >= threshold
        high_disp_idxs = [i for i, s in enumerate(signals)
                          if s["disp"] is not None
                          and s["disp"] >= self.min_displacement_px]
        if len(high_disp_idxs) < 2:
            return StageDecision(
                decision="continue",
                reason=(
                    f"requires_two_or_more_high_disp_reaches "
                    f"(found {len(high_disp_idxs)})"
                ),
                features=feats,
            )

        # P2: among high-disp reaches, exactly one with pillar transition
        pillar_qual_among_disp = [
            i for i in high_disp_idxs if signals[i]["pillar_qualifies"]
        ]
        if len(pillar_qual_among_disp) != 1:
            return StageDecision(
                decision="continue",
                reason=(
                    f"requires_exactly_one_pillar_transition_among_high_disp "
                    f"(found {len(pillar_qual_among_disp)} pillar-qualifying "
                    f"among {len(high_disp_idxs)} high-disp reaches)"
                ),
                features=feats,
            )

        causal_idx = pillar_qual_among_disp[0]
        sig = signals[causal_idx]
        feats["causal_reach_idx"] = causal_idx
        feats["causal_displacement"] = round(sig["disp"], 2)

        # Commit displaced_sa.
        rs, re = sig["rs"], sig["re"]
        bout_length = re - rs + 1
        interaction_frame = int(rs + bout_length // 2)
        okf = int(re + self.okf_offset)
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
                f"pillar_disambiguated_multi_displacement "
                f"(causal reach idx {causal_idx}, disp="
                f"{sig['disp']:.2f} px; pillar delta="
                f"{sig['pillar_delta']:.3f}, pre={sig['pillar_pre_mean']:.3f}, "
                f"post={sig['pillar_post_mean']:.3f}; unique pillar "
                f"transition among {len(high_disp_idxs)} high-disp reaches)"
            ),
            features=feats,
        )
