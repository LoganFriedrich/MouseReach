"""For each Stage 7 walkback-with-pre-post-fail case, identify which
specific gate failed (pre-on, post-off, or both) at the walk-back-
chosen reach. Helps decide direction for fixes.
"""
import json, sys
from pathlib import Path as P
from collections import defaultdict
import numpy as np
sys.path.insert(0, 'Y:/2_Connectome/Behavior/MouseReach/src')
from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.stage_7_pellet_settled_off_pillar_late import (
    Stage7PelletSettledOffPillarLate, PAW_BODYPARTS)
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = P('Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough')
CORPUS = P('Y:/2_Connectome/Behavior/MouseReach_Pipeline/Improvement_Snapshots/_corpus/2026-04-30_restart_inventory')

stage = Stage7PelletSettledOffPillarLate()
folds = json.loads((CORPUS / 'cv_folds.json').read_text())
n_failures = 0
fail_categories = defaultdict(int)
sample_failures = []

for vid in folds['train_pool']['video_ids']:
    dlc = load_dlc_h5(next((QUARANTINE / 'dlc').glob(f'{vid}DLC_*.h5')))
    gt = json.loads((QUARANTINE / 'gt' / f'{vid}_unified_ground_truth.json').read_text(encoding='utf-8'))
    gt_b = [int(b['frame']) for b in gt.get('segmentation', {}).get('boundaries', [])]
    gt_outs = {s['segment_num']: s for s in gt.get('outcomes', {}).get('segments', []) or []}
    for i in range(len(gt_b) - 1):
        sn = i + 1
        entry = gt_outs.get(sn, {})
        if entry.get('outcome') != 'displaced_sa':
            continue
        reaches = []
        for r in gt.get('reaches', {}).get('reaches', []) or []:
            if r.get('segment_num') == sn:
                s, e = r.get('start_frame'), r.get('end_frame')
                if s is not None and e is not None:
                    reaches.append((int(s), int(e)))
        seg = SegmentInput(video_id=vid, segment_num=sn, seg_start=gt_b[i], seg_end=gt_b[i+1]-1,
                           dlc_df=dlc, reach_windows=reaches)
        d = stage.decide(seg)
        if d.decision != 'continue' or not d.reason.startswith('no_causal_reach_via_walkback'):
            continue
        n_failures += 1
        # Replicate walk-back logic to find the chosen reach and reason
        clean_end = seg.seg_end - 5
        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=('Pellet',))
        geom = compute_pillar_geometry_series(sub)
        pcx = geom['pillar_cx'].to_numpy(); pcy = geom['pillar_cy'].to_numpy()
        pr = geom['pillar_r'].to_numpy(); slit = pcy + pr
        plk = sub_raw['Pellet_likelihood'].to_numpy(dtype=float)
        px = sub['Pellet_x'].to_numpy(dtype=float)
        py = sub['Pellet_y'].to_numpy(dtype=float)
        dr = np.sqrt((px-pcx)**2 + (py-pcy)**2) / np.maximum(pr, 1e-6)
        paw_past = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            paw_past |= (sub[f'{bp}_y'].to_numpy(dtype=float) <= slit) & \
                        (sub_raw[f'{bp}_likelihood'].to_numpy(dtype=float) >= 0.5)
        on_pillar = (plk >= 0.95) & (dr <= 1.0) & ~paw_past
        off_conf = (plk >= 0.95) & (dr > 1.0) & ~paw_past
        anchor = (plk >= 0.7) & (dr > 0.85) & ~paw_past
        # find first 3-consecutive anchor run
        run = 0; first_anch = -1
        for j in range(n):
            if anchor[j]:
                run += 1
                if run >= 3:
                    first_anch = j - 2
                    break
            else:
                run = 0
        if first_anch < 0:
            fail_categories['no_anchor_run'] += 1
            continue
        # walk back through reaches
        reach_local = []
        for rs, re in reaches:
            ls = max(0, rs - seg.seg_start); le = min(n-1, re - seg.seg_start)
            if le >= ls:
                reach_local.append((ls, le))
        chosen = None
        for ri in range(len(reach_local)-1, -1, -1):
            rs, re = reach_local[ri]
            if re >= first_anch:
                continue
            pre_on = int(on_pillar[max(0, rs-2):rs].sum())
            post_start = re + 1
            post_off = int(off_conf[post_start:min(n, post_start+30)].sum())
            post_imm = int(off_conf[post_start:min(n, post_start+10)].sum())
            chosen = (ri, rs, re, pre_on, post_off, post_imm)
            break
        if chosen is None:
            fail_categories['no_reach_before_anchor'] += 1
            continue
        ri, rs, re, pre_on, post_off, post_imm = chosen
        if pre_on < 1 and post_off < 5:
            fail_categories['pre_AND_post_fail'] += 1
        elif pre_on < 1:
            fail_categories['pre_fail_only'] += 1
        elif post_off < 5:
            fail_categories['post_fail_only'] += 1
        elif post_imm < 3:
            fail_categories['post_imm_fail'] += 1
        else:
            fail_categories['unknown'] += 1
        if len(sample_failures) < 8:
            sample_failures.append((vid, sn, ri, rs, re, pre_on, post_off, post_imm))

print(f"Total walk-back-with-pre-post failures: {n_failures}")
for k, v in sorted(fail_categories.items(), key=lambda x: -x[1]):
    print(f"  {v:3d}: {k}")
print()
print("Sample failures (vid, seg, reach_idx, rs_local, re_local, pre_on/2, post_off/30, post_imm/10):")
for f in sample_failures:
    print(f"  {f}")
