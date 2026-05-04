"""
Stage 17: Displaced via dominant max-displacement reach (extends
Stage 16 to multi-displacement-reach cases where one reach clearly
dominates).

Question this stage answers:
    "If multiple GT reaches have displacement >= 1.5 radii, but ONE
    reach's displacement clearly dominates (>= 2x the next-largest),
    pick that one as causal."

Whittling logic:
    - Stage 16 only commits when exactly ONE reach has displacement
      >= threshold. Many displaced segments have multiple reaches
      with measurable displacement (mouse pushes pellet around the
      SA after initial displacement).
    - When one displacement clearly dominates, it's almost certainly
      the actual displacement event. Subsequent smaller-displacement
      reaches are mouse interacting with the already-displaced
      pellet.
    - For displaced GT semantic, GT picks "last paw-over-pellet" --
      typically the actual displacement reach (subsequent reaches
      with the pellet at the same SA position get less visual
      attention from GT).

Defenses against false-commit:
    - Max displacement reach must be >= 2x the next-largest
      displacement (clear dominance).
    - Same pre-causal-at-rest check as Stage 16 (defer if pellet
      observed at rest position before the chosen reach).
    - Same late-zone off-pillar check (pellet stays in SA).

Cascade emit on commit:
    - committed_class: "displaced_sa"
    - whens["interaction_frame"]: middle of dominant max-disp reach
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

PELLET_LK_HIGH = 0.95
PELLET_LK_OFF_PILLAR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0
DISPLACEMENT_RADII_MIN = 1.5
DISPLACEMENT_WINDOW = 30
DOMINANCE_RATIO = 3.0
MIN_LATE_OFF_PILLAR = 50
LATE_FRACTION = 0.3
OKF_SETTLE_OFFSET = 6


class Stage17DisplacedViaDominantMaxDisplacement(Stage):
    name = "stage_17_displaced_via_dominant_max_displacement"
    target_class = "displaced_sa"

    def __init__(
        self,
        pellet_lk_high: float = PELLET_LK_HIGH,
        pellet_lk_off_pillar: float = PELLET_LK_OFF_PILLAR,
        paw_lk_threshold: float = PAW_LK_THR,
        on_pillar_radii: float = ON_PILLAR_RADII,
        displacement_radii_min: float = DISPLACEMENT_RADII_MIN,
        displacement_window: int = DISPLACEMENT_WINDOW,
        dominance_ratio: float = DOMINANCE_RATIO,
        min_late_off_pillar: int = MIN_LATE_OFF_PILLAR,
        late_fraction: float = LATE_FRACTION,
        okf_settle_offset: int = OKF_SETTLE_OFFSET,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.pellet_lk_high = pellet_lk_high
        self.pellet_lk_off_pillar = pellet_lk_off_pillar
        self.paw_lk_threshold = paw_lk_threshold
        self.on_pillar_radii = on_pillar_radii
        self.displacement_radii_min = displacement_radii_min
        self.displacement_window = displacement_window
        self.dominance_ratio = dominance_ratio
        self.min_late_off_pillar = min_late_off_pillar
        self.late_fraction = late_fraction
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

        confident = (pellet_lk >= self.pellet_lk_high) & (~paw_past_y)
        px_conf = np.where(confident, pellet_x, np.nan)
        py_conf = np.where(confident, pellet_y, np.nan)

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

        all_disps = []
        for rs, re in reach_windows_local:
            pre_s = max(0, rs - self.displacement_window)
            post_e = min(n, re + 1 + self.displacement_window)
            pre_x_arr = px_conf[pre_s:rs]
            pre_y_arr = py_conf[pre_s:rs]
            post_x_arr = px_conf[re + 1:post_e]
            post_y_arr = py_conf[re + 1:post_e]
            if np.all(np.isnan(pre_x_arr)) or np.all(np.isnan(post_x_arr)):
                all_disps.append(0.0)
                continue
            pre_mx = float(np.nanmedian(pre_x_arr))
            pre_my = float(np.nanmedian(pre_y_arr))
            post_mx = float(np.nanmedian(post_x_arr))
            post_my = float(np.nanmedian(post_y_arr))
            d_px = float(np.sqrt((post_mx - pre_mx) ** 2
                                 + (post_my - pre_my) ** 2))
            med_pillar_r = float(np.nanmedian(pillar_r[pre_s:post_e]))
            all_disps.append(d_px / max(med_pillar_r, 1e-6))

        sorted_disps = sorted(all_disps, reverse=True)
        max_disp = sorted_disps[0]
        max_disp_idx = int(np.argmax(all_disps))
        second_disp = sorted_disps[1] if len(sorted_disps) > 1 else 0.0
        feats["max_disp_radii"] = max_disp
        feats["second_max_disp_radii"] = second_disp
        feats["max_disp_reach_idx"] = max_disp_idx

        if max_disp < self.displacement_radii_min:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_reach_with_significant_displacement "
                    f"(max {max_disp:.2f})"
                ),
                features=feats,
            )

        # Dominance check: max must be >= dominance_ratio * second.
        if second_disp > 0 and max_disp < self.dominance_ratio * second_disp:
            return StageDecision(
                decision="continue",
                reason=(
                    f"max_displacement_not_dominant "
                    f"(max {max_disp:.2f} < {self.dominance_ratio} * "
                    f"second {second_disp:.2f})"
                ),
                features=feats,
            )

        # Late-zone off-pillar evidence + pre-causal-at-rest defense.
        late_start_idx = int(n * (1 - self.late_fraction))
        late_off_eligible_mask = (
            (pellet_lk >= self.pellet_lk_off_pillar)
            & (~paw_past_y)
            & (dist_radii > self.on_pillar_radii)
        )
        late_off_eligible_mask[:late_start_idx] = False
        late_off_count = int(late_off_eligible_mask.sum())
        feats["late_off_pillar"] = late_off_count
        if late_off_count < self.min_late_off_pillar:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_late_off_pillar_evidence "
                    f"({late_off_count} < {self.min_late_off_pillar})"
                ),
                features=feats,
            )

        if int(late_off_eligible_mask.sum()) >= 5:
            rest_x = float(np.median(pellet_x[late_off_eligible_mask]))
            rest_y = float(np.median(pellet_y[late_off_eligible_mask]))
            chosen_bs = reach_windows_local[max_disp_idx][0]
            pre_causal_dev = np.sqrt(
                (pellet_x - rest_x) ** 2 + (pellet_y - rest_y) ** 2)
            pre_causal_dev_radii = pre_causal_dev / np.maximum(pillar_r, 1e-6)
            pre_at_rest = (
                (pellet_lk >= self.pellet_lk_high)
                & (~paw_past_y)
                & (pre_causal_dev_radii <= 1.5)
                & (dist_radii > self.on_pillar_radii)
            )
            pre_at_rest[chosen_bs:] = False
            pre_at_rest_count = int(pre_at_rest.sum())
            feats["pre_causal_at_rest"] = pre_at_rest_count
            if pre_at_rest_count > 0:
                return StageDecision(
                    decision="continue",
                    reason=(
                        f"pellet_at_rest_position_before_max_disp_reach "
                        f"({pre_at_rest_count} frames)"
                    ),
                    features=feats,
                )

        causal_idx = max_disp_idx
        bs, be = reach_windows_local[causal_idx]
        bout_length = be - bs + 1
        interaction_idx = bs + bout_length // 2
        okf_idx = min(be + self.okf_settle_offset, n - 1)
        interaction_frame_video = int(seg.seg_start + interaction_idx)
        okf_video = int(seg.seg_start + okf_idx)
        feats.update({
            "causal_idx": int(causal_idx),
            "bout_start_idx": int(bs),
            "bout_end_idx": int(be),
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
                f"displaced_via_dominant_max_displacement "
                f"(reach {causal_idx} max {max_disp:.2f} radii dominates "
                f"second {second_disp:.2f}; late off-pillar "
                f"{late_off_count}f)"
            ),
            features=feats,
        )
