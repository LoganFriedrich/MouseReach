"""
Stage 1: Pellet position never changed.

Question this stage answers:
    "Across all confident, non-during-reach frames in the segment's
    clean zone, did the pellet's (x, y) position stay within a tight
    spatial tolerance?"

If yes -> COMMIT untouched (highest confidence). The pellet did not
move, so by physical causality (pellet motion requires reach), no
reach-caused change occurred this segment.

This is the strongest possible untouched signal:
- Catches pellet stable on pillar (subset of old Stage 1 -- now renumbered)
- Catches pellet stationary off-pillar (e.g., pellet sitting on SA tray
  edge, fully visible, never moved) -- territory that the renumbered
  stages 2-5 don't always catch cleanly
- Insensitive to pillar geometry, paw activity, occlusion patterns

Tight criteria (per 2026-05-02 corpus diagnostic of pellet-at-rest
position jitter in displaced_sa cases): real stationary pellets have
xy std p1 = 0.17 px, p50 = 0.32 px. Setting commit threshold at
xy std <= 1.0 px on each axis catches stationary pellets robustly,
well above natural jitter floor and well below any plausible
displacement.

Cascade emit on commit:
- committed_class: "untouched"
- whens["outcome_known_frame"]: seg_end - TRANSITION_ZONE_HALF (the
  last clean-zone frame, matching how renumbered stage 2 emits)
- whens["interaction_frame"]: None (untouched -> no causal reach)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from .stage_base import SegmentInput, Stage, StageDecision


# Pellet must be confidently detected in enough non-during-reach frames
# for the position-stability test to be meaningful. Below this count,
# defer to later stages that can use weaker pellet signal.
MIN_CONFIDENT_FRAMES = 100
# 0.7 is the standard "trust the position" threshold across stages.
# Earlier 0.95 was over-strict and produced zero commits because few
# segments have 30+ frames at that lk in both pre-reach and post-reach
# windows. The position-stability test (std <= 1px) handles spurious
# off-target detections naturally -- a low-lk DLC misread that wanders
# blows up the std and causes the segment to defer.
PELLET_LK_THRESHOLD = 0.7

# Maximum pellet position std (each axis, in pixels) for the segment
# to be classified as "position never changed". Real stationary pellet
# jitter (corpus 2026-05-02): p50 = 0.3 px, p99 = 1.8 px. Setting at
# 1.0 px gives strong precision; anything moving more than this had
# something happen.
MAX_POSITION_STD_PX = 1.0

# Defense against the "all confident frames are post-displacement"
# trap: when reaches exist in the segment, require confident frames
# from BOTH the pre-reach and post-reach windows AND verify their
# median positions agree within tolerance. Otherwise the pellet could
# have been displaced during a reach with us only seeing the post-
# displacement rest position.
MIN_WINDOW_FRAMES = 30                  # min confident frames pre AND post
MAX_PRE_POST_DIST_PX = 2.0              # max pre vs post median distance

# Transition zone matches stages 2-5 (formerly 1-4). Last 5 frames
# of the segment are no-man's-land; exclude.
TRANSITION_ZONE_HALF = 5


def _during_reach_mask(seg_start: int, seg_end: int,
                       reach_windows: List[Tuple[int, int]]) -> np.ndarray:
    n = seg_end - seg_start + 1
    mask = np.zeros(n, dtype=bool)
    for rs, re in reach_windows:
        s = max(seg_start, int(rs))
        e = min(seg_end, int(re))
        if e < s:
            continue
        mask[s - seg_start:e - seg_start + 1] = True
    return mask


class Stage1PelletPositionNeverChanged(Stage):
    name = "stage_1_pellet_position_never_changed"
    target_class = "untouched"

    def __init__(
        self,
        min_confident_frames: int = MIN_CONFIDENT_FRAMES,
        pellet_lk_threshold: float = PELLET_LK_THRESHOLD,
        max_position_std_px: float = MAX_POSITION_STD_PX,
        min_window_frames: int = MIN_WINDOW_FRAMES,
        max_pre_post_dist_px: float = MAX_PRE_POST_DIST_PX,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.min_confident_frames = min_confident_frames
        self.pellet_lk_threshold = pellet_lk_threshold
        self.max_position_std_px = max_position_std_px
        self.min_window_frames = min_window_frames
        self.max_pre_post_dist_px = max_pre_post_dist_px
        self.transition_zone_half = transition_zone_half

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(
                decision="continue",
                reason="segment_too_short_to_have_clean_zone")

        sub = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        # Use pellet position RELATIVE TO PILLAR CENTER, not raw pixel
        # coords. Apparatus tray drifts laterally over the ~30s
        # segment; absolute-coord stds reflect that drift, not pellet
        # motion. Pillar-relative coords cancel the drift.
        geom = compute_pillar_geometry_series(sub)
        pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)

        plk = sub["Pellet_likelihood"].to_numpy(dtype=float)
        px = sub["Pellet_x"].to_numpy(dtype=float) - pillar_cx
        py = sub["Pellet_y"].to_numpy(dtype=float) - pillar_cy

        during = _during_reach_mask(seg.seg_start, clean_end, seg.reach_windows)
        eligible = (~during) & (plk >= self.pellet_lk_threshold)
        n_eligible = int(eligible.sum())

        feats = {
            "n_clean_zone_frames": n,
            "n_during_reach_frames": int(during.sum()),
            "n_confident_non_during_reach_frames": n_eligible,
        }

        if n_eligible < self.min_confident_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_confident_non_during_reach_frames "
                    f"({n_eligible} < {self.min_confident_frames}); "
                    f"position-stability test not meaningful, defer"
                ),
                features=feats,
            )

        x_std = float(px[eligible].std())
        y_std = float(py[eligible].std())
        feats.update({
            "pellet_x_std_px": x_std,
            "pellet_y_std_px": y_std,
        })

        if x_std > self.max_position_std_px or y_std > self.max_position_std_px:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_position_changed_overall "
                    f"(x_std={x_std:.2f}px or y_std={y_std:.2f}px exceeds "
                    f"{self.max_position_std_px}px); defer to later stages"
                ),
                features=feats,
            )

        # Even with low overall std, the displacement could have happened
        # during a reach (which is excluded from `eligible`). If the
        # confident non-during-reach frames are dominated by a single
        # post-displacement rest period, the std is low but the pellet
        # absolutely moved. Defense: require pre-reach AND post-reach
        # confident-frame windows AND verify their median positions
        # match within tolerance.
        feats.update({
            "n_reach_windows": len(seg.reach_windows),
        })

        # No reaches in this segment -> all confident frames are
        # contiguous (no displacement event possible). Std test alone
        # is sufficient.
        if not seg.reach_windows:
            okf = int(seg.seg_end - self.transition_zone_half)
            feats["outcome_known_frame_emitted"] = okf
            return StageDecision(
                decision="commit",
                committed_class="untouched",
                whens={"outcome_known_frame": okf,
                       "interaction_frame": None},
                reason=(
                    f"pellet_position_unchanged_no_reaches "
                    f"(x_std={x_std:.2f}px, y_std={y_std:.2f}px, "
                    f"n_confident={n_eligible}, no reaches in segment)"
                ),
                features=feats,
            )

        # Define pre-reach and post-reach windows.
        first_reach_start_local = min(int(rs) for rs, _ in seg.reach_windows) - seg.seg_start
        last_reach_end_local = max(int(re) for _, re in seg.reach_windows) - seg.seg_start

        pre_mask = np.zeros(n, dtype=bool)
        post_mask = np.zeros(n, dtype=bool)
        if first_reach_start_local > 0:
            pre_mask[:first_reach_start_local] = True
        if last_reach_end_local + 1 < n:
            post_mask[last_reach_end_local + 1:] = True
        pre_eligible = pre_mask & eligible
        post_eligible = post_mask & eligible

        n_pre = int(pre_eligible.sum())
        n_post = int(post_eligible.sum())
        feats.update({
            "n_confident_pre_reach": n_pre,
            "n_confident_post_reach": n_post,
        })

        if n_pre < self.min_window_frames or n_post < self.min_window_frames:
            return StageDecision(
                decision="continue",
                reason=(
                    f"insufficient_pre_or_post_reach_confident_frames "
                    f"(pre={n_pre}, post={n_post}; need "
                    f"{self.min_window_frames}+ each to verify pre==post "
                    f"position); defer"
                ),
                features=feats,
            )

        pre_mx = float(np.median(px[pre_eligible]))
        pre_my = float(np.median(py[pre_eligible]))
        post_mx = float(np.median(px[post_eligible]))
        post_my = float(np.median(py[post_eligible]))
        pre_post_dist = float(np.sqrt(
            (post_mx - pre_mx) ** 2 + (post_my - pre_my) ** 2))
        feats.update({
            "pre_reach_median_x": pre_mx, "pre_reach_median_y": pre_my,
            "post_reach_median_x": post_mx, "post_reach_median_y": post_my,
            "pre_to_post_dist_px": pre_post_dist,
        })

        if pre_post_dist > self.max_pre_post_dist_px:
            return StageDecision(
                decision="continue",
                reason=(
                    f"pellet_moved_between_pre_and_post_reach "
                    f"(pre median=({pre_mx:.1f},{pre_my:.1f}) vs post "
                    f"median=({post_mx:.1f},{post_my:.1f}), "
                    f"dist={pre_post_dist:.2f}px > "
                    f"{self.max_pre_post_dist_px}px); displacement "
                    f"happened during reach activity, defer"
                ),
                features=feats,
            )

        okf = int(seg.seg_end - self.transition_zone_half)
        feats["outcome_known_frame_emitted"] = okf
        return StageDecision(
            decision="commit",
            committed_class="untouched",
            whens={"outcome_known_frame": okf,
                   "interaction_frame": None},
            reason=(
                f"pellet_position_unchanged "
                f"(overall xy_std=({x_std:.2f},{y_std:.2f})px, "
                f"pre_reach median=({pre_mx:.1f},{pre_my:.1f}), "
                f"post_reach median=({post_mx:.1f},{post_my:.1f}), "
                f"pre_to_post_dist={pre_post_dist:.2f}px; "
                f"pellet was at the same location both before and "
                f"after any reach activity)"
            ),
            features=feats,
        )
