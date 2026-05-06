"""For each segment in the corpus, compute pellet position std in
pillar-relative coords across confident, non-during-reach frames in
the clean zone. The hypothesis:

  Untouched -> pellet doesn't move -> std is tiny (regardless of seg length)
  Displaced -> pellet must move to a different position -> std is large
  Retrieved -> pellet often disappears or moves to mouth -> std is large
                (or visibility_frac is low)

If untouched has a tight distribution and touched has a wide distribution
with a clean gap, we can build a stage that captures the 4 short-segment
residuals safely.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_7_pellet_displaced_to_sa import PAW_BODYPARTS
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")
PELLET_LK_THR = 0.7
TRANSITION_ZONE_HALF = 5


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def load_gt(vid):
    return json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))


def reach_windows_for(gt, sn):
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == sn:
            s = r.get("start_frame"); e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def during_reach(seg_start, seg_end, reaches):
    n = seg_end - seg_start + 1
    m = np.zeros(n, dtype=bool)
    for rs, re in reaches:
        s = max(seg_start, int(rs)); e = min(seg_end, int(re))
        if e < s: continue
        m[s - seg_start:e - seg_start + 1] = True
    return m


def compute(vid, sn, seg_start, seg_end, dlc, reaches):
    clean_end = seg_end - TRANSITION_ZONE_HALF
    if clean_end <= seg_start: return None
    sub_raw = dlc.iloc[seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0: return None
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pcx = geom["pillar_cx"].to_numpy(dtype=float)
    pcy = geom["pillar_cy"].to_numpy(dtype=float)
    pellet_x = sub["Pellet_x"].to_numpy(dtype=float) - pcx
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float) - pcy
    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    not_during = ~during_reach(seg_start, clean_end, reaches)
    eligible = not_during & (pellet_lk >= PELLET_LK_THR)
    n_eligible = int(eligible.sum())
    n_not_during = int(not_during.sum())
    if n_eligible < 5: return None
    return {
        "n_eligible": n_eligible,
        "n_not_during": n_not_during,
        "visibility_frac": n_eligible / max(n_not_during, 1),
        "x_std_relative": float(pellet_x[eligible].std()),
        "y_std_relative": float(pellet_y[eligible].std()),
        "x_range": float(pellet_x[eligible].max() - pellet_x[eligible].min()),
        "y_range": float(pellet_y[eligible].max() - pellet_y[eligible].min()),
    }


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    rows = []
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = load_gt(vid)
        gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
        gt_outs = {s["segment_num"]: s for s in gt.get("outcomes", {}).get("segments", []) or []}
        for i in range(len(gt_b) - 1):
            sn = i + 1
            seg_start, seg_end = gt_b[i], gt_b[i + 1] - 1
            outcome = gt_outs.get(sn, {}).get("outcome")
            if outcome == "displaced_outside": outcome = "displaced_sa"
            reaches = reach_windows_for(gt, sn)
            m = compute(vid, sn, seg_start, seg_end, dlc, reaches)
            if m is None: continue
            rows.append({"vid": vid, "sn": sn, "outcome": outcome, **m})

    by_oc = defaultdict(list)
    for r in rows: by_oc[r["outcome"]].append(r)

    print("Pellet position std (pillar-relative) by GT class:")
    for oc, rs in sorted(by_oc.items()):
        xs = sorted(r["x_std_relative"] for r in rs)
        ys = sorted(r["y_std_relative"] for r in rs)
        n = len(xs)
        print(f"\n{oc} (n={n}):")
        for label, arr in [("x_std_rel", xs), ("y_std_rel", ys)]:
            line = f"  {label}: "
            for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
                i = max(0, min(n - 1, int(p / 100 * (n - 1))))
                line += f"p{p}={arr[i]:.3f}  "
            print(line)

    print()
    print("4 short-segment residuals -- their std values:")
    SHORT_RES = [("20250806_CNT0311_P2", 17), ("20250806_CNT0311_P2", 19),
                 ("20250806_CNT0311_P2", 20), ("20250806_CNT0312_P2", 18)]
    for vid, sn in SHORT_RES:
        m = next((r for r in rows if r["vid"] == vid and r["sn"] == sn), None)
        if m:
            print(f"  {vid} seg {sn:2d}: x_std={m['x_std_relative']:.2f}  "
                  f"y_std={m['y_std_relative']:.2f}  "
                  f"x_range={m['x_range']:.2f}  "
                  f"y_range={m['y_range']:.2f}  "
                  f"vis={m['visibility_frac']:.3f}")

    print()
    # Find threshold where x_std AND y_std AND vis collectively safely separate
    print("Test: position variance + visibility (no minimum frame count)")
    KNOWN_RESID = {("20250624_CNT0115_P2", 17), ("20250630_CNT0104_P3", 2),
                   ("20250630_CNT0104_P3", 6), ("20250806_CNT0311_P2", 17),
                   ("20250806_CNT0311_P2", 19), ("20250806_CNT0311_P2", 20),
                   ("20250806_CNT0312_P2", 18), ("20250811_CNT0305_P2", 14),
                   ("20250909_CNT0209_P4", 4), ("20250912_CNT0209_P3", 9),
                   ("20251008_CNT0301_P4", 8)}
    SHORT_RES_SET = set(SHORT_RES)
    for std_thr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        for vis_thr in [0.99, 0.95]:
            counts = defaultdict(int)
            short_caught = 0
            resid_caught = 0
            for r in rows:
                if (r["x_std_relative"] <= std_thr
                        and r["y_std_relative"] <= std_thr
                        and r["visibility_frac"] >= vis_thr):
                    counts[r["outcome"]] += 1
                    if (r["vid"], r["sn"]) in KNOWN_RESID: resid_caught += 1
                    if (r["vid"], r["sn"]) in SHORT_RES_SET: short_caught += 1
            contam = counts.get("displaced_sa", 0) + counts.get("retrieved", 0) + counts.get("abnormal_exception", 0)
            print(f"  x_std<={std_thr} AND y_std<={std_thr} AND vis>={vis_thr}: "
                  f"u={counts.get('untouched', 0)}/{len(by_oc.get('untouched', []))}  "
                  f"d={counts.get('displaced_sa', 0)}  r={counts.get('retrieved', 0)}  "
                  f"a={counts.get('abnormal_exception', 0)}  contam={contam}  "
                  f"resid_caught={resid_caught}/11  short_caught={short_caught}/4")


if __name__ == "__main__":
    main()
