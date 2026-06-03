"""
DIAGNOSTIC (not an experiment): measure the pre-vanish pellet trajectory to
test whether "direction before the vanish" separates displaced-that-vanished
(GT displaced, cascade wrongly said retrieved) from REAL retrievals.

Hypothesis (Idea 1): real retrievals carry the pellet UP toward the mouse/slit
(y decreases, pellet rises above the slit line) with little lateral motion;
displaced-out pellets move LATERALLY / stay in the pellet-SA zone, then vanish.

For each segment we find the longest sustained pellet-vanish run after the
first reach, take the last confident pellet positions just before it, and
report displacement from the loaded (on-pillar) baseline:
  dy_toward_mouse  = baseline_y - prevanish_y   (POSITIVE = moved toward mouse/up)
  dx_abs           = |prevanish_x - baseline_x| (lateral)
  above_slit       = prevanish_y < slit_y       (pellet above slit line = toward mouse)
  dist_r           = distance from pillar center at pre-vanish (radii)

Group A (GT displaced, cascade->retrieved) vs CONTROL (GT retrieved &
cascade->retrieved, auto-sampled from the v6.0.4 per-segment csvs).
Read-only; prints only.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_leverA", SCRIPTS_DIR / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lva)
detect_reaches_v8 = lva.detect_reaches_v8
load_dlc_h5 = lva.load_dlc_h5
load_gt_segments = lva.load_gt_segments

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

M31_DLC = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1")
M31_GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")
GEN_DLC = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\algo_outputs_current")
GEN_GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt")
CORPUS = {"m31": (M31_DLC, M31_GT), "gen": (GEN_DLC, GEN_GT)}
SNAP = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.4_leverA_net_displaced_sa_2026-06-03")

GROUP_A = [
    ("m31", "20250710_CNT0215_P4", 8), ("m31", "20251024_CNT0402_P4", 5),
    ("m31", "20250821_CNT0110_P4", 8), ("gen", "20250625_CNT0102_P4", 10),
    ("gen", "20250711_CNT0216_P1", 1), ("gen", "20250715_CNT0209_P2", 17),
    ("gen", "20250718_CNT0214_P1", 2), ("gen", "20251008_CNT0303_P2", 6),
]
LK = 0.7


def find_control_retrieved(max_n=14):
    """Correctly-classified retrieved segments from the v6.0.4 snapshot."""
    out = []
    for corpus, sub in (("m31", "model31"), ("gen", "generalization")):
        csvp = SNAP / sub / "metrics" / "outcome_per_segment.csv"
        for r in csv.DictReader(open(csvp)):
            if r["gt_outcome"] == "retrieved" and r["algo_outcome"] == "retrieved":
                out.append((corpus, r["video_id"], int(r["segment_num"])))
    return out[:max_n]


def vanish_metrics(dlc, segments, seg_num, reaches):
    s, e = segments[seg_num - 1]
    sub_raw = dlc.iloc[s:e + 1]
    n = len(sub_raw)
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pcx = geom["pillar_cx"].to_numpy(float); pcy = geom["pillar_cy"].to_numpy(float)
    pr = geom["pillar_r"].to_numpy(float)
    slit_y = pcy + pr
    px = sub["Pellet_x"].to_numpy(float); py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    conf = plk >= LK
    dist_r = np.sqrt((px - pcx) ** 2 + (py - pcy) ** 2) / np.maximum(pr, 1e-6)
    seg_reaches = sorted((r0 - s, r1 - s) for r0, r1 in reaches if s <= r0 <= e)
    first_reach = seg_reaches[0][0] if seg_reaches else 0
    # baseline: median confident on-pillar pellet before first reach
    base_mask = conf & (dist_r <= 1.2) & (np.arange(n) < max(first_reach, 5))
    if base_mask.sum() < 3:
        base_mask = conf & (np.arange(n) < max(first_reach, 30))
    if base_mask.sum() < 3:
        return None
    bx, by = float(np.median(px[base_mask])), float(np.median(py[base_mask]))
    # longest sustained vanish run (lk<LK, >=30f) after first reach
    low = (~conf).copy(); low[:first_reach] = False
    runs = []; st = -1
    for i in range(n):
        if low[i] and st < 0: st = i
        elif not low[i] and st >= 0:
            if i - st >= 30: runs.append((st, i - 1))
            st = -1
    if st >= 0 and n - st >= 30: runs.append((st, n - 1))
    if not runs:
        return ("no_sustained_vanish", None)
    onset = max(runs, key=lambda ab: ab[1] - ab[0])[0]
    # last up-to-10 confident frames before onset
    pre = [i for i in range(onset) if conf[i]][-10:]
    if len(pre) < 3:
        return ("too_few_preconf", None)
    vx, vy = float(np.median(px[pre])), float(np.median(py[pre]))
    sy = float(np.median(slit_y[pre]))
    vdist = float(np.median(dist_r[pre]))
    return ("ok", dict(dy_toward_mouse=round(by - vy, 1), dx_abs=round(abs(vx - bx), 1),
                       above_slit=bool(vy < sy), dist_r=round(vdist, 2),
                       vanish_onset_local=onset))


def main():
    cache = {}
    def get(corpus, vid):
        key = (corpus, vid)
        if key not in cache:
            dd, gd = CORPUS[corpus]
            dlc = load_dlc_h5(sorted(dd.glob(f"{vid}DLC_*.h5"))[0])
            cache[key] = (dlc, load_gt_segments(gd, vid), detect_reaches_v8(dlc))
        return cache[key]

    print("=" * 92)
    print("GROUP A  (GT=displaced, cascade WRONGLY said retrieved) -- expect lateral / not-toward-mouse")
    print("=" * 92)
    print(f"  {'segment':38s} {'dy_toward_mouse':>15} {'dx_abs':>7} {'above_slit':>11} {'dist_r':>7}")
    for corpus, vid, seg in GROUP_A:
        dlc, segs, reaches = get(corpus, vid)
        status, m = vanish_metrics(dlc, segs, seg, reaches)
        if m is None:
            print(f"  {corpus+' '+vid+' s'+str(seg):38s} {status}")
        else:
            print(f"  {corpus+' '+vid+' s'+str(seg):38s} {m['dy_toward_mouse']:>15} {m['dx_abs']:>7} {str(m['above_slit']):>11} {m['dist_r']:>7}")

    print("\n" + "=" * 92)
    print("CONTROL  (GT=retrieved, cascade CORRECTLY said retrieved) -- expect toward-mouse / above-slit")
    print("=" * 92)
    print(f"  {'segment':38s} {'dy_toward_mouse':>15} {'dx_abs':>7} {'above_slit':>11} {'dist_r':>7}")
    for corpus, vid, seg in find_control_retrieved():
        dlc, segs, reaches = get(corpus, vid)
        try:
            status, m = vanish_metrics(dlc, segs, seg, reaches)
        except Exception as ex:
            print(f"  {corpus+' '+vid+' s'+str(seg):38s} ERR {ex}"); continue
        if m is None:
            print(f"  {corpus+' '+vid+' s'+str(seg):38s} {status}")
        else:
            print(f"  {corpus+' '+vid+' s'+str(seg):38s} {m['dy_toward_mouse']:>15} {m['dx_abs']:>7} {str(m['above_slit']):>11} {m['dist_r']:>7}")


if __name__ == "__main__":
    main()
