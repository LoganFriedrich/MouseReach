"""Read-only corpus-wide count: segments where a CLEAN on-pillar pellet load
exists at segment start but the Stage-8 pre-reach baseline window is clipped by
an early displacing reach (so displaced_sa stages can't establish the baseline).

Pattern (operational, matches Stage 8's lk>=0.95 / dist<=1.2r on-pillar gate):
  load_run     = longest consecutive on-pillar run starting within first 15 frames
  prewin_run   = longest consecutive on-pillar run in [first_reach-30, first_reach)
  PATTERN      = (load_run >= 5) AND (prewin_run < 5)
                 -> clean load present, but current baseline window clips it.

Cross with GT + current v6.0.4 outcome:
  ADDRESSABLE  = PATTERN & GT=displaced_sa & current != displaced_sa  (would gain)
  RISK         = PATTERN & GT in {untouched,retrieved} & current==GT  (could break)
  already_ok   = PATTERN & GT=displaced_sa & current==displaced_sa
Scans model-3.1 + generalization decision corpora. Prints only.
"""
import csv, importlib.util
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_lva", S / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(spec); spec.loader.exec_module(lva)
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series

SNAP = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.4_leverA_net_displaced_sa_2026-06-03")
def norm(x): return "displaced_sa" if x in ("displaced_outside", "displaced_sa") else x

def outcomes(sub):  # current v6.0.4 per-segment outcomes
    d = {}
    for r in csv.DictReader(open(SNAP / sub / "metrics" / "outcome_per_segment.csv")):
        d[(r["video_id"], int(r["segment_num"]))] = (norm(r["gt_outcome"]), norm(r["algo_outcome"]))
    return d

def longest_run(mask):
    run = mx = 0
    for b in mask:
        run = run + 1 if b else 0; mx = max(mx, run)
    return mx

def analyze(cfg, sub):
    cur = outcomes(sub)
    ids = cfg["ids"] or sorted(p.stem.replace("_unified_ground_truth","") for p in cfg["gt"].glob("*_unified_ground_truth.json"))
    res = {"addressable": [], "risk": [], "already_ok": [], "pattern_total": 0}
    for vid in ids:
        h5 = sorted(cfg["dlc"].glob(f"{vid}DLC_*.h5"))
        if not h5: continue
        dlc = lva.load_dlc_h5(h5[0]); segs = lva.load_gt_segments(cfg["gt"], vid)
        reaches = lva.detect_reaches_v8(dlc)
        for si, (s, e) in enumerate(segs):
            sn = si + 1
            sub_raw = dlc.iloc[s:e+1]; n = len(sub_raw)
            if n < 40: continue
            c = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
            g = compute_pillar_geometry_series(c)
            pcx=g["pillar_cx"].to_numpy(float); pcy=g["pillar_cy"].to_numpy(float); pr=g["pillar_r"].to_numpy(float)
            px=c["Pellet_x"].to_numpy(float); py=c["Pellet_y"].to_numpy(float); plk=sub_raw["Pellet_likelihood"].to_numpy(float)
            dist=np.sqrt((px-pcx)**2+(py-pcy)**2)/np.maximum(pr,1e-6)
            onp=(plk>=0.95)&(dist<=1.2)
            slit=pcy+pr; paw=np.zeros(n,bool)
            for bp in ("RightHand","RHLeft","RHOut","RHRight"):
                paw |= (c[f"{bp}_y"].to_numpy(float)<=slit)&(c[f"{bp}_likelihood"].to_numpy(float)>=0.5)
            fp=next((i for i in range(n) if paw[i]), -1)
            if fp < 0: continue
            # load_run: longest on-pillar run beginning within first 15 frames
            load_run=0; i=0
            while i < min(15, n):
                if onp[i]:
                    j=i
                    while j<n and onp[j]: j+=1
                    load_run=max(load_run, j-i); i=j
                else: i+=1
            prewin = onp[max(0,fp-30):fp]
            prewin_run = longest_run(prewin)
            if load_run >= 5 and prewin_run < 5:
                res["pattern_total"] += 1
                key=(vid, sn); gt, algo = cur.get(key, (None, None))
                tag=f"{vid} s{sn} (GT={gt}, cur={algo}, load={load_run}, prewin={prewin_run}, first_reach={fp})"
                if gt=="displaced_sa" and algo!="displaced_sa": res["addressable"].append(tag)
                elif gt in ("untouched","retrieved") and algo==gt: res["risk"].append(tag)
                elif gt=="displaced_sa" and algo=="displaced_sa": res["already_ok"].append(tag)
    return res

for cfg, sub in ((lva.M31,"model31"), (lva.GEN,"generalization")):
    r = analyze(cfg, sub)
    print("="*78); print(f"{sub}: {r['pattern_total']} segments match the early-clip pattern")
    print(f"  ADDRESSABLE (GT=displaced_sa, currently wrong): {len(r['addressable'])}")
    for t in r["addressable"]: print(f"    + {t}")
    print(f"  RISK (GT=untouched/retrieved, currently correct): {len(r['risk'])}")
    for t in r["risk"]: print(f"    ! {t}")
    print(f"  already-ok (GT=displaced_sa, already committed): {len(r['already_ok'])}")
    for t in r["already_ok"]: print(f"    . {t}")
    print()
