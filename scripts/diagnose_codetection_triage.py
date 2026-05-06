"""For each segment that Stage 5 (pellet-off-pillar-throughout)
triaged on the pellet-pillar codetection predicate, dump the firing
frames and surrounding context so we can decide:
  (a) real DLC label-switch artifact (working as intended -> triage)
  (b) legitimate physical proximity (pellet edge of pillar) -> over-triage
  (c) other pattern

For each case prints:
  - segment range and GT class
  - run-lengths of codetection frames within the segment
  - for each codetection run: pellet xy + lk, pillar xy + lk, distance,
    pellet-pillar position offset (whether they're at literally the
    same pixel coords -> label-switch).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"

# Same constants as Stage 5 (pellet-off-pillar-throughout)'s codetection.
# IMPORTANT: Stage 5 uses pellet_lk_thr = 0.7 (not 0.95 like Stage 6).
PELLET_LK_THR = 0.7
PILLAR_LK_THR = 0.7
DISTANCE_RADII_THR = 1.0
SUSTAINED_FRAMES = 3
TRANSITION_ZONE_HALF = 5

CASES = [
    ("20250624_CNT0115_P2", 1, "displaced_sa"),
    ("20250624_CNT0115_P2", 2, "displaced_sa"),
    ("20250627_CNT0105_P1", 2, "retrieved"),
    ("20250627_CNT0105_P1", 6, "displaced_sa"),
    ("20250627_CNT0105_P1", 7, "displaced_sa"),
    ("20250627_CNT0105_P1", 8, "displaced_sa"),
    ("20250627_CNT0105_P1", 14, "displaced_sa"),
    ("20250701_CNT0110_P2", 3, "abnormal_exception"),
    ("20250701_CNT0110_P2", 4, "displaced_sa"),
    ("20250701_CNT0110_P2", 5, "displaced_sa"),
    ("20251023_CNT0401_P4", 5, "retrieved"),
    ("20251028_CNT0404_P4", 5, "displaced_sa"),
    ("20251028_CNT0404_P4", 15, "retrieved"),
    ("20251028_CNT0404_P4", 16, "displaced_sa"),
]


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def load_gt_segment_bounds(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return {i + 1: (gt_b[i], gt_b[i + 1] - 1) for i in range(len(gt_b) - 1)}


def runs_of_true(arr):
    """List of (start_idx, end_idx_inclusive) runs of True."""
    runs = []
    rs = -1
    for i, v in enumerate(arr):
        if v:
            if rs < 0:
                rs = i
        else:
            if rs >= 0:
                runs.append((rs, i - 1))
                rs = -1
    if rs >= 0:
        runs.append((rs, len(arr) - 1))
    return runs


_STAGE6 = None
def get_stage6():
    global _STAGE6
    if _STAGE6 is None:
        from mousereach.outcomes.v6_cascade.stage_6_pellet_displaced_to_sa import (
            Stage6PelletDisplacedToSA)
        _STAGE6 = Stage6PelletDisplacedToSA()
    return _STAGE6


def gt_payload_for_segment(video_id, seg_num):
    """Return (outcome, ifr, okf, comment, determined_by) for the GT
    segment entry."""
    gt = json.loads((GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))
    for s in gt.get("outcomes", {}).get("segments", []) or []:
        if s.get("segment_num") == seg_num:
            return (s.get("outcome"), s.get("interaction_frame"),
                    s.get("outcome_known_frame"), s.get("comment"),
                    s.get("determined_by"))
    return (None, None, None, None, None)


def gt_reaches_for_segment(video_id, seg_num):
    gt = json.loads((GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == seg_num:
            s = r.get("start_frame"); e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def analyze(video_id, seg_num, gt_class):
    dlc = load_dlc_h5(find_dlc(video_id))
    bounds = load_gt_segment_bounds(video_id)
    seg_start, seg_end = bounds[seg_num]
    clean_end = seg_end - TRANSITION_ZONE_HALF

    sub_raw = dlc.iloc[seg_start:clean_end + 1]
    n = len(sub_raw)
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)

    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    pellet_lk_raw = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)

    pillar_x_raw = sub_raw["Pillar_x"].to_numpy(dtype=float)
    pillar_y_raw = sub_raw["Pillar_y"].to_numpy(dtype=float)
    pillar_lk_raw = sub_raw["Pillar_likelihood"].to_numpy(dtype=float)

    pp_dist = np.sqrt((pellet_x - pillar_x_raw) ** 2 + (pellet_y - pillar_y_raw) ** 2)
    pp_dist_radii = pp_dist / np.maximum(pillar_r, 1e-6)

    fire = (
        (pellet_lk_raw >= PELLET_LK_THR)
        & (pillar_lk_raw >= PILLAR_LK_THR)
        & (pp_dist_radii <= DISTANCE_RADII_THR)
    )
    runs = [(s, e) for (s, e) in runs_of_true(fire) if (e - s + 1) >= SUSTAINED_FRAMES]

    gt_outcome, gt_ifr, gt_okf, gt_comment, gt_determined_by = gt_payload_for_segment(video_id, seg_num)
    print(f"\n{'='*78}")
    print(f"{video_id} seg {seg_num}  bounds=({seg_start}, {seg_end})  "
          f"clean_zone_n={n}")
    print(f"  GT: outcome={gt_outcome}  IFR={gt_ifr}  OKF={gt_okf}  by={gt_determined_by}")
    print(f"  GT comment: {gt_comment!r}")
    print(f"  Algo at Stage 5: TRIAGE (codetection)")
    # Simulate Stage 6 to see what algo WOULD commit if Stage 5 didn't triage
    seg = SegmentInput(
        video_id=video_id, segment_num=seg_num,
        seg_start=seg_start, seg_end=seg_end, dlc_df=dlc,
        reach_windows=gt_reaches_for_segment(video_id, seg_num))
    s6 = get_stage6().decide(seg)
    print(f"  Algo at Stage 6 (if Stage 5 didn't triage): {s6.decision}  "
          f"class={s6.committed_class}  reason={s6.reason[:200]}")
    print(f"  -- codetection diagnostic ({int(fire.sum())} firing frames, median pillar_r = {float(np.median(pillar_r)):.2f} px) --")
    if not runs:
        print("  no qualifying runs (this case shouldn't trigger triage; check)")
        return
    print(f"  {len(runs)} qualifying runs (length >= {SUSTAINED_FRAMES}):")
    for ri, (s, e) in enumerate(runs):
        run_len = e - s + 1
        # Sample stats over the run
        avg_pellet_x = float(pellet_x[s:e+1].mean())
        avg_pellet_y = float(pellet_y[s:e+1].mean())
        avg_pillar_x = float(pillar_x_raw[s:e+1].mean())
        avg_pillar_y = float(pillar_y_raw[s:e+1].mean())
        avg_dist_radii = float(pp_dist_radii[s:e+1].mean())
        avg_pellet_lk = float(pellet_lk_raw[s:e+1].mean())
        avg_pillar_lk = float(pillar_lk_raw[s:e+1].mean())
        offset_dist = float(np.sqrt((avg_pellet_x - avg_pillar_x) ** 2
                                    + (avg_pellet_y - avg_pillar_y) ** 2))
        run_frames_abs = (seg_start + s, seg_start + e)
        # Also: variance across run -- if pellet/pillar both stationary
        # at offset, label-switch suspicion grows
        pellet_x_std = float(pellet_x[s:e+1].std())
        pellet_y_std = float(pellet_y[s:e+1].std())
        pillar_x_std = float(pillar_x_raw[s:e+1].std())
        pillar_y_std = float(pillar_y_raw[s:e+1].std())
        # Position relative to fired pillar location
        offset_dx = avg_pellet_x - avg_pillar_x
        offset_dy = avg_pellet_y - avg_pillar_y
        print(f"    run {ri}: len={run_len:4d} frames "
              f"[abs_frames={run_frames_abs[0]}..{run_frames_abs[1]}]")
        print(f"      pellet ({avg_pellet_x:.1f}, {avg_pellet_y:.1f}) "
              f"std=({pellet_x_std:.2f}, {pellet_y_std:.2f}) lk={avg_pellet_lk:.3f}")
        print(f"      pillar ({avg_pillar_x:.1f}, {avg_pillar_y:.1f}) "
              f"std=({pillar_x_std:.2f}, {pillar_y_std:.2f}) lk={avg_pillar_lk:.3f}")
        print(f"      offset (pellet - pillar) = ({offset_dx:+.2f}, {offset_dy:+.2f}) "
              f"= {offset_dist:.2f} px = {avg_dist_radii:.3f} radii")

        # Verdict heuristics
        if offset_dist < 1.0 and pellet_x_std < 0.5 and pillar_x_std < 0.5:
            verdict = "LABEL-SWITCH (pellet ≈ pillar pixel-wise, both stationary)"
        elif offset_dist < 2.0 and pellet_x_std < 1.0 and pillar_x_std < 1.0:
            verdict = "LIKELY LABEL-SWITCH (very close, low jitter)"
        elif avg_dist_radii > 0.5:
            verdict = "PROXIMITY (pellet near pillar edge, separable)"
        else:
            verdict = "AMBIGUOUS"
        print(f"      VERDICT: {verdict}")


for vid, sn, gt in CASES:
    analyze(vid, sn, gt)


# Cross-check: directly invoke the stage's decide() to confirm what's
# actually happening. The 5 "no qualifying runs" cases above should
# either (a) match here (validation must be wrong) or (b) the stage
# uses different inputs than the diagnostic and IS triaging on
# something we missed.
print("\n\n" + "=" * 78)
print("CROSS-CHECK: stage.decide() output for each case")
print("=" * 78)
from mousereach.outcomes.v6_cascade.stage_5_pellet_off_pillar_throughout import (
    Stage5PelletOffPillarThroughout)

stage = Stage5PelletOffPillarThroughout()
for vid, sn, gt_class in CASES:
    dlc = load_dlc_h5(find_dlc(vid))
    bounds = load_gt_segment_bounds(vid)
    seg_start, seg_end = bounds[sn]
    seg = SegmentInput(
        video_id=vid, segment_num=sn,
        seg_start=seg_start, seg_end=seg_end,
        dlc_df=dlc, reach_windows=[])
    d = stage.decide(seg)
    print(f"  {vid} seg {sn:2d}  GT={gt_class:18s}  decision={d.decision}  reason={d.reason[:120]}")

