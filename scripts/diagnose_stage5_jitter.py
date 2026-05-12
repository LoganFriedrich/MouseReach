"""Diagnose Stage 5 commits: dump per-commit features + pellet rest-
position jitter, separating into:
  - correct commits (class match + same-bout match against GT)
  - wrong commits (the 4 stubborn 20250716_CNT0213_P3 cases)

Goal: find a single-segment discriminator that filters wrong commits
without touching correct commits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_1_pellet_position_never_changed import (
    Stage1PelletPositionNeverChanged)
from mousereach.outcomes.v6_cascade.stage_2_pellet_stable_untouched import (
    Stage2PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_3_paw_never_in_pellet_area import (
    Stage3PawNeverInPelletArea)
from mousereach.outcomes.v6_cascade.stage_4_pellet_returns_to_pillar import (
    Stage4PelletReturnsToPillar)
from mousereach.outcomes.v6_cascade.stage_5_pellet_off_pillar_throughout import (
    Stage5PelletOffPillarThroughout)
from mousereach.outcomes.v6_cascade.stage_6_pellet_displaced_to_sa import (
    Stage6PelletDisplacedToSA, PAW_BODYPARTS)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import calibrate_stage
from mousereach.reach.v8.features import load_dlc_h5


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory")
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"


def find_dlc(video_id: str) -> Path:
    return next(DLC_DIR.glob(f"{video_id}DLC_*.h5"))


def load_segment_bounds(video_id: str) -> Dict[int, tuple]:
    seg = json.loads((ALGO_DIR / f"{video_id}_segments.json").read_text(encoding="utf-8"))
    b = seg.get("boundaries", []) or []
    return {i + 1: (int(b[i]), int(b[i + 1]) - 1) for i in range(len(b) - 1)}


def load_gt(video_id: str) -> dict:
    return json.loads((GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))


def gt_reaches_for_segment(gt: dict, segment_num: int):
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == segment_num:
            s, e = r.get("start_frame"), r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def collapse(o):
    return "displaced_sa" if o == "displaced_outside" else o


def compute_jitter_features(seg: SegmentInput, stage: Stage6PelletDisplacedToSA):
    """Re-run Stage 5 logic to recover the rest-position frames, then
    compute pellet-position jitter on those frames."""
    transition_zone_half = stage.transition_zone_half
    clean_end = seg.seg_end - transition_zone_half
    sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return None
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy()
    pillar_cy = geom["pillar_cy"].to_numpy()
    pillar_r = geom["pillar_r"].to_numpy()
    slit_y_line = pillar_cy + pillar_r

    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    pellet_dist_radii = (
        np.sqrt((pellet_x - pillar_cx) ** 2 + (pellet_y - pillar_cy) ** 2)
        / np.maximum(pillar_r, 1e-6))

    paw_past_y = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
        paw_lk = sub[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= stage.paw_lk_threshold)

    off_pillar_eligible = (
        (pellet_lk >= stage.pellet_lk_threshold)
        & (pellet_dist_radii > stage.pellet_off_pillar_radii)
        & (~paw_past_y))

    if off_pillar_eligible.sum() < stage.rest_frames_total:
        return None

    median_x = float(np.median(pellet_x[off_pillar_eligible]))
    median_y = float(np.median(pellet_y[off_pillar_eligible]))
    deviation = np.sqrt((pellet_x - median_x) ** 2 + (pellet_y - median_y) ** 2)
    deviation_radii = deviation / np.maximum(pillar_r, 1e-6)
    near_median = (off_pillar_eligible
                   & (deviation_radii <= stage.near_median_tolerance_radii))
    if near_median.sum() < stage.rest_frames_total:
        return None

    rest_x = pellet_x[near_median]
    rest_y = pellet_y[near_median]
    rest_lk = pellet_lk[near_median]
    rest_pillar_r = float(np.median(pillar_r[near_median]))

    return dict(
        n_rest_frames=int(near_median.sum()),
        rest_x_std_px=float(rest_x.std()),
        rest_y_std_px=float(rest_y.std()),
        rest_x_std_radii=float(rest_x.std() / max(rest_pillar_r, 1e-6)),
        rest_y_std_radii=float(rest_y.std() / max(rest_pillar_r, 1e-6)),
        rest_xy_std_radii=float(np.sqrt(rest_x.var() + rest_y.var()) / max(rest_pillar_r, 1e-6)),
        rest_x_range_px=float(rest_x.max() - rest_x.min()),
        rest_y_range_px=float(rest_y.max() - rest_y.min()),
        rest_lk_min=float(rest_lk.min()),
        rest_lk_mean=float(rest_lk.mean()),
        rest_lk_std=float(rest_lk.std()),
        median_x=median_x, median_y=median_y,
        rest_pillar_r=rest_pillar_r,
    )


def load_gt_segment_bounds(video_id: str):
    gt = json.loads((GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8"))
    gt_b = [int(bd["frame"]) for bd in gt.get("segmentation", {}).get("boundaries", [])]
    return {i + 1: (gt_b[i], gt_b[i + 1] - 1) for i in range(len(gt_b) - 1)}


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    seg_inputs = []
    gt_lookup = {}
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        seg_bounds = load_gt_segment_bounds(vid)
        gt = load_gt(vid)
        gt_outs = {s["segment_num"]: s
                   for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (s_start, s_end) in seg_bounds.items():
            seg_inputs.append(SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=s_start, seg_end=s_end, dlc_df=dlc,
                reach_windows=gt_reaches_for_segment(gt, sn)))
            sgt = gt_outs.get(sn, {})
            gt_lookup[(vid, sn)] = {
                "gt_outcome": collapse(sgt.get("outcome")),
                "gt_outcome_known_frame": sgt.get("outcome_known_frame"),
                "gt_interaction_frame": sgt.get("interaction_frame"),
            }

    # Run cascade through Stage 4 to determine Stage 5's input pool.
    stages = [
        ("Stage 1", Stage1PelletPositionNeverChanged()),
        ("Stage 2", Stage2PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5)),
        ("Stage 3", Stage3PawNeverInPelletArea()),
        ("Stage 4", Stage4PelletReturnsToPillar()),
        ("Stage 5", Stage5PelletOffPillarThroughout()),
    ]
    consumed = set()
    for label, stage in stages:
        stage_inputs = [s for s in seg_inputs
                        if (s.video_id, s.segment_num) not in consumed]
        cal = calibrate_stage(
            stage=stage, seg_inputs=stage_inputs, gt_lookup=gt_lookup,
            okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
        for c in cal.cases:
            if c.decision in ("commit", "triage"):
                consumed.add((c.video_id, c.segment_num))

    stage5 = Stage6PelletDisplacedToSA()
    stage5_inputs = [s for s in seg_inputs
                     if (s.video_id, s.segment_num) not in consumed]

    print(f"Stage 5 input pool: {len(stage5_inputs)} segments")
    print()

    rows = []
    for seg in stage5_inputs:
        decision = stage5.decide(seg)
        if decision.decision != "commit":
            continue
        gt = gt_lookup.get((seg.video_id, seg.segment_num), {})
        gt_class = gt.get("gt_outcome")
        gt_ifr = gt.get("gt_interaction_frame")
        algo_ifr = decision.whens.get("interaction_frame")
        # Same-bout match check: re-derive bouts and check that algo_ifr
        # and gt_ifr fall in the same bout.
        same_bout = None
        if gt_ifr is not None and algo_ifr is not None:
            transition_zone_half = stage5.transition_zone_half
            clean_end = seg.seg_end - transition_zone_half
            sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
            n = len(sub_raw)
            sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
            geom = compute_pillar_geometry_series(sub)
            slit_y_line = geom["pillar_cy"].to_numpy() + geom["pillar_r"].to_numpy()
            paw_past_y = np.zeros(n, dtype=bool)
            for bp in PAW_BODYPARTS:
                paw_y = sub[f"{bp}_y"].to_numpy(dtype=float)
                paw_lk = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
                paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= stage5.paw_lk_threshold)
            algo_local = int(algo_ifr) - seg.seg_start
            gt_local = int(gt_ifr) - seg.seg_start
            algo_bout = paw_past_y[algo_local] if 0 <= algo_local < n else False
            gt_bout = paw_past_y[gt_local] if 0 <= gt_local < n else False
            # Find bout index for both
            bouts = []
            run_start = -1
            for i in range(n):
                if paw_past_y[i]:
                    if run_start < 0: run_start = i
                else:
                    if run_start >= 0:
                        bouts.append((run_start, i - 1)); run_start = -1
            if run_start >= 0: bouts.append((run_start, n - 1))
            def find_bout(idx):
                if idx < 0 or idx >= n:
                    return -1
                for bi, (s, e) in enumerate(bouts):
                    if s <= idx <= e:
                        return bi
                # Find nearest bout
                best = -1; best_d = 1e9
                for bi, (s, e) in enumerate(bouts):
                    d = min(abs(idx - s), abs(idx - e))
                    if d < best_d: best_d = d; best = bi
                return best
            same_bout = find_bout(algo_local) == find_bout(gt_local)

        class_match = (decision.committed_class == gt_class)
        trust_pass = bool(class_match and same_bout)

        jit = compute_jitter_features(seg, stage5)
        if jit is None:
            continue
        rows.append({
            "video_id": seg.video_id,
            "segment_num": seg.segment_num,
            "trust_pass": trust_pass,
            "class_match": class_match,
            "same_bout": same_bout,
            "gt_class": gt_class,
            **jit,
        })

    correct = [r for r in rows if r["trust_pass"]]
    wrong = [r for r in rows if not r["trust_pass"]]
    print(f"Total commits: {len(rows)}  correct: {len(correct)}  wrong: {len(wrong)}")
    print()
    print("=== WRONG commits (full feature dump) ===")
    for r in wrong:
        print(f"  {r['video_id']} seg {r['segment_num']:2d}  gt={r['gt_class']:14s}  "
              f"class_match={r['class_match']}  same_bout={r['same_bout']}")
        print(f"    n_rest={r['n_rest_frames']:4d}  "
              f"xy_std_px=({r['rest_x_std_px']:.2f}, {r['rest_y_std_px']:.2f})  "
              f"xy_range=({r['rest_x_range_px']:.1f}, {r['rest_y_range_px']:.1f})  "
              f"std_radii={r['rest_xy_std_radii']:.4f}  "
              f"lk_min={r['rest_lk_min']:.3f}  lk_mean={r['rest_lk_mean']:.3f}  lk_std={r['rest_lk_std']:.4f}")
    print()
    print("=== CORRECT commits: n_rest distribution ===")
    if correct:
        nr = sorted(r["n_rest_frames"] for r in correct)
        n = len(nr)
        def pct(p): return nr[int(p * (n - 1))]
        print(f"    n_rest: min={nr[0]}  p5={pct(0.05)}  p25={pct(0.25)}  "
              f"p50={pct(0.5)}  p75={pct(0.75)}  p95={pct(0.95)}  max={nr[-1]}")
        print(f"    Wrong n_rest values: {sorted(r['n_rest_frames'] for r in wrong)}")
    print()
    print("=== CORRECT commits: distribution of jitter features ===")
    if correct:
        for k in ("rest_x_std_px", "rest_y_std_px", "rest_xy_std_radii",
                  "rest_x_range_px", "rest_y_range_px",
                  "rest_lk_min", "rest_lk_std"):
            vals = sorted(r[k] for r in correct)
            n = len(vals)
            def pct(p):
                return vals[int(p * (n - 1))]
            print(f"  {k:>22s}: p1={pct(0.01):.4f}  p5={pct(0.05):.4f}  "
                  f"p50={pct(0.50):.4f}  p95={pct(0.95):.4f}  p99={pct(0.99):.4f}  "
                  f"max={vals[-1]:.4f}")
    print()
    # Specifically: candidate cuts that filter all wrong but minimal
    # collateral on correct.
    print("=== Candidate cuts ===")
    for k, op in [("rest_xy_std_radii", "min"), ("rest_x_std_px", "min"),
                  ("rest_y_std_px", "min"), ("rest_lk_std", "min")]:
        wrong_vals = sorted(r[k] for r in wrong)
        correct_vals = sorted(r[k] for r in correct)
        if not wrong_vals or not correct_vals: continue
        # If wrong is generally LOWER than correct, threshold is "must
        # be > X" where X = max(wrong) (or some safety margin)
        max_wrong = max(wrong_vals)
        min_correct = min(correct_vals)
        # Count correct that would be lost at threshold = max_wrong + epsilon
        thr = max_wrong + 1e-6
        lost = sum(1 for v in correct_vals if v <= thr)
        print(f"  Threshold: require {k} > {thr:.4f}")
        print(f"    Wrong values: {[f'{v:.4f}' for v in wrong_vals]}")
        print(f"    Min correct value: {min_correct:.4f}")
        print(f"    Correct lost at this threshold: {lost} / {len(correct_vals)}")


if __name__ == "__main__":
    main()
