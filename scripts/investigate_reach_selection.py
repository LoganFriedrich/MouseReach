"""For each Stage 7 commit (correct + wrong), compute per-reach
features at the GT reach, algo-picked reach, and neighbors. Compare
distributions to find what discriminator separates GT-causal from
non-causal reaches.

Features computed at each reach (start_local, end_local):
  - n_pre_on_pillar in windows [30, 50, 100, 200]
  - max_pre_on_pillar_run (longest consecutive on-pillar run pre-reach)
  - n_post_off_pillar in windows [30, 50, 100, 200]
  - max_post_off_pillar_run
  - frames_to_first_off_pillar after reach end
  - frames_until_pellet_returns_on_pillar after reach end (or -1 if never)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_7_pellet_settled_off_pillar_late import (
    Stage7PelletSettledOffPillarLate, PAW_BODYPARTS)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")
TRANSITION_ZONE_HALF = 5
PELLET_LK_THR = 0.95
PAW_LK_THR = 0.5
PELLET_OFF_PILLAR_RADII = 1.0
ON_PILLAR_RADII = 1.0


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def load_gt(vid):
    return json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))


def reaches_for_segment(gt, sn):
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == sn:
            s, e = r.get("start_frame"), r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def compute_per_reach_features(seg, dlc):
    """Return list of per-reach feature dicts for the segment."""
    clean_end = seg.seg_end - TRANSITION_ZONE_HALF
    sub_raw = dlc.iloc[seg.seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return []
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy()
    pillar_cy = geom["pillar_cy"].to_numpy()
    pillar_r = geom["pillar_r"].to_numpy()
    slit_y_line = pillar_cy + pillar_r

    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    dist_radii = (np.sqrt((pellet_x - pillar_cx)**2 + (pellet_y - pillar_cy)**2)
                  / np.maximum(pillar_r, 1e-6))

    paw_past = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        py = sub[f"{bp}_y"].to_numpy(dtype=float)
        pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
        paw_past |= (py <= slit_y_line) & (pl >= PAW_LK_THR)

    on_pillar = (pellet_lk >= PELLET_LK_THR) & (dist_radii <= ON_PILLAR_RADII) & ~paw_past
    off_pillar = (pellet_lk >= PELLET_LK_THR) & (dist_radii > PELLET_OFF_PILLAR_RADII) & ~paw_past

    def max_run(mask):
        run = 0; best = 0
        for v in mask:
            if v:
                run += 1
                if run > best: best = run
            else:
                run = 0
        return best

    # Pellet position confident-only series (NaN where lk too low)
    confident = pellet_lk >= PELLET_LK_THR
    px_conf = np.where(confident, pellet_x, np.nan)
    py_conf = np.where(confident, pellet_y, np.nan)

    def median_pos_in_window(s, e):
        """Median pellet position over confident frames in [s, e)."""
        if e <= s:
            return None, None
        x = px_conf[s:e]; y = py_conf[s:e]
        if not np.any(~np.isnan(x)):
            return None, None
        return float(np.nanmedian(x)), float(np.nanmedian(y))

    out = []
    for ri, (rs, re) in enumerate(seg.reach_windows):
        ls = max(0, int(rs) - seg.seg_start)
        le = min(n - 1, int(re) - seg.seg_start)
        if le < ls:
            continue
        feats = {"reach_idx": ri, "rs_local": ls, "re_local": le}
        # Pre-reach windows
        for w in [30, 50, 100, 200]:
            ws = max(0, ls - w)
            feats[f"pre_on_count_w{w}"] = int(on_pillar[ws:ls].sum())
            feats[f"pre_off_count_w{w}"] = int(off_pillar[ws:ls].sum())
            feats[f"pre_max_on_run_w{w}"] = max_run(on_pillar[ws:ls])
        # Post-reach windows
        for w in [30, 50, 100, 200]:
            we = min(n, le + 1 + w)
            feats[f"post_off_count_w{w}"] = int(off_pillar[le+1:we].sum())
            feats[f"post_on_count_w{w}"] = int(on_pillar[le+1:we].sum())
            feats[f"post_max_off_run_w{w}"] = max_run(off_pillar[le+1:we])
        # Pellet displacement across this reach: median position
        # before reach minus median position after reach (in pillar
        # radii, so cross-segment comparable).
        for pre_w, post_w in [(30, 30), (50, 50), (100, 100)]:
            pre_s = max(0, ls - pre_w)
            post_e = min(n, le + 1 + post_w)
            pre_mx, pre_my = median_pos_in_window(pre_s, ls)
            post_mx, post_my = median_pos_in_window(le + 1, post_e)
            if pre_mx is None or post_mx is None:
                feats[f"reach_displacement_px_w{pre_w}"] = None
                feats[f"reach_displacement_radii_w{pre_w}"] = None
            else:
                d_px = float(np.sqrt((post_mx - pre_mx)**2 + (post_my - pre_my)**2))
                med_pillar_r = float(np.nanmedian(pillar_r[pre_s:post_e]))
                feats[f"reach_displacement_px_w{pre_w}"] = d_px
                feats[f"reach_displacement_radii_w{pre_w}"] = (
                    d_px / max(med_pillar_r, 1e-6))
        out.append(feats)
    return out


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]

    # Re-run Stage 7 to get its commits, then compare GT vs algo reach picks
    stage7 = Stage7PelletSettledOffPillarLate()

    correct_gt_features = []   # features at GT reach when algo got it right
    wrong_gt_features = []     # features at GT reach when algo got it wrong
    wrong_algo_features = []   # features at algo's reach when wrong
    other_reach_features_in_correct = []  # non-GT reaches in correct segments
    other_reach_features_in_wrong = []    # non-GT, non-algo reaches in wrong segments

    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = load_gt(vid)
        gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
        gt_outs = {s["segment_num"]: s for s in gt.get("outcomes", {}).get("segments", []) or []}
        for i in range(len(gt_b) - 1):
            sn = i + 1
            entry = gt_outs.get(sn, {})
            outcome = entry.get("outcome")
            if outcome != "displaced_sa":
                continue
            gt_ifr = entry.get("interaction_frame")
            if gt_ifr is None:
                continue
            seg = SegmentInput(
                video_id=vid, segment_num=sn,
                seg_start=gt_b[i], seg_end=gt_b[i+1]-1, dlc_df=dlc,
                reach_windows=reaches_for_segment(gt, sn))
            d = stage7.decide(seg)
            if d.decision != "commit":
                continue
            algo_ifr = d.whens.get("interaction_frame")
            if algo_ifr is None:
                continue
            # Find GT reach idx and algo reach idx
            gt_reach_idx = -1
            algo_reach_idx = -1
            for ri, (rs, re) in enumerate(seg.reach_windows):
                if rs <= int(gt_ifr) <= re:
                    gt_reach_idx = ri
                if rs <= int(algo_ifr) <= re:
                    algo_reach_idx = ri
            if gt_reach_idx < 0:
                continue
            per_reach = compute_per_reach_features(seg, dlc)
            for f in per_reach:
                f["video_id"] = vid
                f["segment_num"] = sn
                f["n_reaches"] = len(per_reach)
            if gt_reach_idx == algo_reach_idx:
                # Correct
                for f in per_reach:
                    if f["reach_idx"] == gt_reach_idx:
                        correct_gt_features.append(f)
                    else:
                        other_reach_features_in_correct.append(f)
            else:
                # Wrong
                for f in per_reach:
                    if f["reach_idx"] == gt_reach_idx:
                        wrong_gt_features.append(f)
                    elif f["reach_idx"] == algo_reach_idx:
                        wrong_algo_features.append(f)
                    else:
                        other_reach_features_in_wrong.append(f)

    print(f"Stage 7 commits investigated:")
    print(f"  Correct: {len(correct_gt_features)}")
    print(f"  Wrong: {len(wrong_gt_features)}")
    print(f"  Other reaches in correct segments: {len(other_reach_features_in_correct)}")
    print(f"  Other reaches in wrong segments: {len(other_reach_features_in_wrong)}")
    print()

    def stats(arr, key):
        vals = sorted(a[key] for a in arr)
        if not vals: return "(empty)"
        n = len(vals)
        def pct(p): return vals[max(0, min(n-1, int(p/100*(n-1))))]
        return f"min={pct(0):.1f} p5={pct(5):.1f} p25={pct(25):.1f} p50={pct(50):.1f} p75={pct(75):.1f} p95={pct(95):.1f}"

    keys = ["pre_on_count_w30", "pre_on_count_w100",
            "post_off_count_w30", "post_off_count_w100",
            "reach_displacement_radii_w30", "reach_displacement_radii_w50",
            "reach_displacement_radii_w100"]

    def stats_skipnone(arr, key):
        vals = sorted(a[key] for a in arr if a.get(key) is not None)
        if not vals: return "(empty)"
        n = len(vals)
        def pct(p): return vals[max(0, min(n-1, int(p/100*(n-1))))]
        return f"min={pct(0):.2f} p5={pct(5):.2f} p25={pct(25):.2f} p50={pct(50):.2f} p75={pct(75):.2f} p95={pct(95):.2f}"

    # Override stats for displacement keys (handle None)
    _orig_stats = stats
    def stats(arr, key):
        if "displacement" in key:
            return stats_skipnone(arr, key)
        return _orig_stats(arr, key)

    print("=== AT GT REACH (correct vs wrong) ===")
    for k in keys:
        print(f"  {k}:")
        print(f"    correct: {stats(correct_gt_features, k)}")
        print(f"    wrong  : {stats(wrong_gt_features, k)}")
    print()
    print("=== AT ALGO REACH (when wrong) ===")
    for k in keys:
        print(f"  {k}: {stats(wrong_algo_features, k)}")
    print()
    print("=== OTHER NON-GT REACHES (correct segs vs wrong segs) ===")
    for k in keys:
        print(f"  {k}:")
        print(f"    other in correct: {stats(other_reach_features_in_correct, k)}")
        print(f"    other in wrong  : {stats(other_reach_features_in_wrong, k)}")


if __name__ == "__main__":
    main()
