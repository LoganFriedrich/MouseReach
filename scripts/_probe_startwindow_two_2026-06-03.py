"""Read-only: start-window / on-pillar-baseline analysis for
20251031_CNT0413_P2 s10 and 20251031_CNT0407_P1 s19. Does the displacing
reach happen so early that the on-pillar baseline window is clipped?"""
import importlib.util
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_dd", S / "outcome_remaining_errors_deepdive_2026-06-03.py")
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

TARGETS = [("20251031_CNT0413_P2", 10), ("20251031_CNT0407_P1", 19)]
cache = {}
for vid, seg_num in TARGETS:
    if vid not in cache:
        cache[vid] = (dd.load_dlc_h5(sorted(dd.M31_DLC.glob(f"{vid}DLC_*.h5"))[0]),
                      dd.load_gt_segments(dd.M31_GT, vid), )
    dlc, segs = cache[vid]
    reaches = dd.detect_reaches_v8(dlc)
    s, e = segs[seg_num - 1]
    seg_r = sorted((r0 - s, r1 - s) for r0, r1 in reaches if s <= r0 <= e)
    sub_raw = dlc.iloc[s:e+1]; sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pcx=geom["pillar_cx"].to_numpy(float); pcy=geom["pillar_cy"].to_numpy(float); pr=geom["pillar_r"].to_numpy(float)
    px=sub["Pellet_x"].to_numpy(float); py=sub["Pellet_y"].to_numpy(float); plk=sub_raw["Pellet_likelihood"].to_numpy(float)
    dist_r=np.sqrt((px-pcx)**2+(py-pcy)**2)/np.maximum(pr,1e-6)
    conf=plk>=0.7; onp=conf&(dist_r<=1.2)
    slit_y=pcy+pr; paw=np.zeros(len(sub),bool)
    for bp in ("RightHand","RHLeft","RHOut","RHRight"):
        paw |= (sub[f"{bp}_y"].to_numpy(float)<=slit_y)&(sub[f"{bp}_likelihood"].to_numpy(float)>=0.5)
    first_paw=next((i for i in range(len(paw)) if paw[i]), -1)
    print("="*72)
    print(f"{vid} s{seg_num}  len={e-s+1}  first 5 reaches(local)={seg_r[:5]}")
    print(f"  first paw-past-slit local={first_paw}   first v8 reach local={seg_r[0][0] if seg_r else None}")
    print(f"  pellet first 12 frames (lk, dist_r, onP):")
    for i in range(min(12,len(sub))):
        print(f"    {i:2d}  lk={plk[i]:.2f} dist={dist_r[i]:5.2f} {'ON' if onp[i] else ''}")
    print(f"  total on-pillar frames in segment: {int(onp.sum())}")
    if first_paw>0:
        lo=max(0,first_paw-30); win=onp[lo:first_paw]
        run=mx=0
        for b in win: run=run+1 if b else 0; mx=max(mx,run)
        print(f"  Stage8 pre-reach window [{lo},{first_paw}): {int(win.sum())} on-pillar, longest run={mx} (needs >=5)")
    else:
        print("  no paw-past-slit found")
    print()
