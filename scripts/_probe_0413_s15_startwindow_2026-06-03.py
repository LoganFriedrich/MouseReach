"""Read-only: where is the pellet on-pillar relative to the first reach in
20251031_CNT0413_P2 s15? Tests whether the on-pillar baseline window is too
short because the pellet is knocked off almost immediately."""
import importlib.util
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_dd", S / "outcome_remaining_errors_deepdive_2026-06-03.py")
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

vid, seg_num = "20251031_CNT0413_P2", 15
dlc = dd.load_dlc_h5(sorted(dd.M31_DLC.glob(f"{vid}DLC_*.h5"))[0])
segs = dd.load_gt_segments(dd.M31_GT, vid)
reaches = dd.detect_reaches_v8(dlc)
s, e = segs[seg_num - 1]
seg_r = sorted((r0 - s, r1 - s) for r0, r1 in reaches if s <= r0 <= e)
print(f"{vid} s{seg_num}  frames[{s},{e}] len={e-s+1}")
print(f"first 6 reaches (local): {seg_r[:6]}")

sub_raw = dlc.iloc[s:e+1]; sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
geom = compute_pillar_geometry_series(sub)
pcx=geom["pillar_cx"].to_numpy(float); pcy=geom["pillar_cy"].to_numpy(float); pr=geom["pillar_r"].to_numpy(float)
px=sub["Pellet_x"].to_numpy(float); py=sub["Pellet_y"].to_numpy(float)
plk=sub_raw["Pellet_likelihood"].to_numpy(float)
dist_r=np.sqrt((px-pcx)**2+(py-pcy)**2)/np.maximum(pr,1e-6)
conf = plk>=0.7
onp = conf & (dist_r<=1.2)

# paw-past-slit (first reach proxy) per Stage 8
slit_y=pcy+pr
paw_past=np.zeros(len(sub),bool)
for bp in ("RightHand","RHLeft","RHOut","RHRight"):
    paw_past |= (sub[f"{bp}_y"].to_numpy(float)<=slit_y)&(sub[f"{bp}_likelihood"].to_numpy(float)>=0.5)
first_paw = next((i for i in range(len(paw_past)) if paw_past[i]), -1)
print(f"first paw-past-slit (Stage 8 'first reach') local frame: {first_paw}")
print(f"first v8 reach start local: {seg_r[0][0] if seg_r else None}")

# pellet timeline, first 60 local frames
print("\nlocal | pellet_lk | dist_r | on_pillar?  (first 60 frames)")
for i in range(min(60, len(sub))):
    mark = " <-- first paw-past-slit" if i==first_paw else ""
    print(f"  {i:3d}   {plk[i]:.2f}      {dist_r[i]:5.2f}    {'ON' if onp[i] else '  '}{mark}")

print(f"\ntotal confident on-pillar frames in segment: {int(onp.sum())}")
# Stage 8 pre-reach window check
if first_paw>0:
    lo=max(0, first_paw-30)
    win = onp[lo:first_paw]
    # longest sustained run
    run=mx=0
    for b in win:
        run=run+1 if b else 0; mx=max(mx,run)
    print(f"Stage 8 pre-reach window [{lo},{first_paw}): {int(win.sum())} on-pillar frames, longest sustained run={mx} (needs >=5)")
