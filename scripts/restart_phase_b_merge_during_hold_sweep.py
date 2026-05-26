"""
v8.0.x experiment: merge-during-hold postprocess sweep.

Direct attack on the hold-during-extension mechanism that creates
FRAGMENTED events. Symmetric to apex-split but in the MERGING direction:
if two consecutive algo reaches are separated by a small gap during
which the hand is held stationary at apex (norm_pos stable, paw_lk
moderate), merge them into a single reach.

Identified earlier today:
  - HOL CNT0303_P2 cid=135: GT span 29, 2 algos with 12f gap. During
    gap: hand_x range 0.3px, norm_pos range 0.05, paw_lk mean 0.36
  - HOL CNT0407_P3 cid=259: GT span 56, 2 algos with 19f gap. During
    gap: hand_x range 1.7px, norm_pos range 0.06, paw_lk mean 0.58

Distinguishing feature vs apex-split's "between-reach trough":
  - Hold gap: norm_pos STABLE at apex value (small range)
  - Apex trough: norm_pos has DEEP TROUGH (variation > prominence)
The two mechanisms are orthogonal signals. Merge-during-hold should
not re-merge legitimate apex-split conversions (those have varying
norm_pos in the gap).

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26 evening):
   - Production v8.0.4 reach detector (BSW + mg=0 + leading-trim +
     trailing-trim + apex-split).
   - Asymmetric matcher -2/+5, matcher-aware topology classifier.
   - Baseline (cumulative best, post 2026-05-26 GT edits, live GT):
       Cal filtered topology: TP=2312 TOL=18 MERGED=10 FRAG=5  FP=9  FN=34
       Hol filtered topology: TP=3655 TOL=10 MERGED=3  FRAG=7  FP=12 FN=27

2. Existing-code-modification check: NO. merge_during_hold implemented
   inline in this runner. If accepted, integrate into v8/postprocess.py.

3. Unverified hypotheses:
   - That the merge fires correctly on the 2 known cases (CNT0303_P2,
     CNT0407_P3) without firing on apex-split-resolved real
     between-reach gaps.
   - That FRAGMENTED count drops on holdout (target class).
   - That MERGED doesn't rise (over-merging risk: combining two truly
     separate reaches that happen to have a stationary frame at the
     boundary). Expected: rare because real adjacent reaches retract
     between them.
   - That TP rises or holds (the conversions from FRAG -> TP).

4. FN-direction-reporting:
   - Lead with FN delta vs cumulative best (v8.0.4 post-2026-05-26 GT).
   - Topology paired with legacy.

5. Framework check:
   - Output to v8.0.4_dev_merge_during_hold_sweep/.
   - sweep_results.json + RESULTS.md.

6. Branch + tag:
   - feature/v8-merge-during-hold
   - Tag: pre-merge-during-hold-2026-05-26

7. Decision rule (per config):
   ACCEPT if:
     - TP rises or holds (drop <= 2 events) on BOTH corpora
     - FRAGMENTED drops on at least one corpus, non-increasing on both
     - MERGED non-increasing on both corpora
     - TP_topo non-decreasing
     - start_delta AND span_delta abs_median both = 0
   REJECT if:
     - TP drops materially (> 2 events) on either corpus
     - MERGED rises
     - FRAGMENTED rises
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, trim_trailing_sustained_lk,
    compute_paw_mean_lk, apex_split_at_trough,
    compute_hand_to_boxl_norm_pos,
)


CAL_LOOCV_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOLDOUT_ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0\algo_outputs_v8.0.0_mg0"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
CAL_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.4_dev_merge_during_hold_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# Production v8.0.4 postprocess params
TRIM_LEADING_T = 0.60; TRIM_LEADING_N = 3
TRIM_TRAILING_T = 0.60; TRIM_TRAILING_N = 3
TRIM_MIN_SPAN = 3
APEX_PROMINENCE = 0.12; APEX_DEPTH_MIN = 0.5
APEX_PEAK2_REL_MAX = 0.85; APEX_MIN_DISTANCE = 4; APEX_MIN_SPAN = 3

# Sweep: merge-during-hold params
# Fixed: min paw_lk mean in gap (distinguishes hold from paw-invisible)
MIN_PAW_LK_MEAN_IN_GAP = 0.20
# Sweep:
SWEEP_CONFIGS = [
    # (max_gap, max_norm_pos_range, label)
    (20, 0.10, "gap20_nr0.10"),
    (25, 0.10, "gap25_nr0.10"),
    (30, 0.10, "gap30_nr0.10"),
    (25, 0.05, "gap25_nr0.05"),
    (25, 0.15, "gap25_nr0.15"),
]

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


# ---- merge-during-hold postprocess ----

def merge_during_hold(reaches, paw_mean_lk, norm_pos,
                      max_gap, max_norm_pos_range,
                      min_paw_lk_mean=MIN_PAW_LK_MEAN_IN_GAP):
    """Merge consecutive algo reaches separated by a hold-pattern gap.

    Signal: 1 <= gap <= max_gap, norm_pos range in gap <= max_norm_pos_range,
    paw_lk mean in gap >= min_paw_lk_mean.

    Iteratively merges chains of consecutive reaches (a, b, c) if a-b and
    b-c both meet the criteria.
    """
    if len(reaches) < 2:
        return list(reaches)
    out = []
    cur = reaches[0]
    i = 1
    while i < len(reaches):
        nxt = reaches[i]
        gap_start = cur.end_frame + 1
        gap_end = nxt.start_frame - 1
        gap = gap_end - gap_start + 1
        should_merge = False
        if (1 <= gap <= max_gap
                and gap_start >= 0
                and gap_end < len(paw_mean_lk)):
            gap_paw = paw_mean_lk[gap_start:gap_end + 1]
            gap_norm = norm_pos[gap_start:gap_end + 1]
            if not (np.any(np.isnan(gap_paw)) or np.any(np.isnan(gap_norm))):
                norm_range = float(gap_norm.max() - gap_norm.min())
                paw_mean = float(gap_paw.mean())
                if norm_range <= max_norm_pos_range and paw_mean >= min_paw_lk_mean:
                    should_merge = True
        if should_merge:
            cur = ReachSpan(start_frame=cur.start_frame, end_frame=nxt.end_frame)
            i += 1
        else:
            out.append(cur)
            cur = nxt
            i += 1
    out.append(cur)
    return out


# ---- Matcher + topology (matcher-aware) ----

def overlap_exists(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            sd = a_s - g_s
            pd_ = algo_span - gt_span
            sp_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
            if (-STRICT_START_TOL_EARLY <= sd <= STRICT_START_TOL_LATE
                    and abs(pd_) <= sp_tol):
                candidates.append((abs(sd), ai, gi, sd, pd_))
    candidates.sort()
    matched = set(); used_a, used_g = set(), set()
    tp_sd, tp_pd = [], []
    for _, ai, gi, sd, pd_ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi)); tp_sd.append(sd); tp_pd.append(pd_)
    return matched, tp_sd, tp_pd


def classify_matcher_aware(algos, gts, matched):
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for i in range(len(algos)): parent[('a', i)] = ('a', i)
    for j in range(len(gts)): parent[('g', j)] = ('g', j)
    for i, a in enumerate(algos):
        for j, g in enumerate(gts):
            if overlap_exists(*a, *g):
                union(('a', i), ('g', j))
    by_root = defaultdict(list)
    for n in parent: by_root[find(n)].append(n)
    counts = defaultdict(int)
    tol_event_count = 0
    for nodes in by_root.values():
        a_idx = {i for k, i in nodes if k == 'a'}
        g_idx = {j for k, j in nodes if k == 'g'}
        na, ng = len(a_idx), len(g_idx)
        if na == 1 and ng == 0:
            counts['FALSE_POSITIVE'] += 1
        elif na == 0 and ng == 1:
            counts['FALSE_NEGATIVE'] += 1
        elif na == 1 and ng == 1:
            i = next(iter(a_idx)); j = next(iter(g_idx))
            if (i, j) in matched:
                counts['TP'] += 1
            else:
                tol_event_count += 2
        elif na == 1 and ng >= 2:
            counts['MERGED'] += 1
        elif na >= 2 and ng == 1:
            counts['FRAGMENTED'] += 1
        elif na >= 2 and ng >= 2:
            matched_in = [(ai, gj) for (ai, gj) in matched
                          if ai in a_idx and gj in g_idx]
            for _ in matched_in:
                counts['TP'] += 1
            unmatched_a = a_idx - {ai for ai, _ in matched_in}
            unmatched_g = g_idx - {gj for _, gj in matched_in}
            soft_paired = set()
            for ai in sorted(unmatched_a):
                best_gj = None; best_ol = 0
                for gj in sorted(unmatched_g - soft_paired):
                    a_s, a_e = algos[ai]; g_s, g_e = gts[gj]
                    s = max(a_s, g_s); e = min(a_e, g_e)
                    ol = max(0, e - s + 1)
                    if ol > best_ol:
                        best_ol = ol; best_gj = gj
                if best_gj is not None and best_ol > 0:
                    tol_event_count += 2
                    soft_paired.add(best_gj)
                else:
                    counts['FALSE_POSITIVE'] += 1
            for gj in sorted(unmatched_g - soft_paired):
                counts['FALSE_NEGATIVE'] += 1
    counts['TOLERANCE_ERROR_pairs'] = tol_event_count // 2
    return dict(counts)


# ---- Data loading ----

def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def compute_norm_pos_from_df(df):
    hand_x = smooth(np.mean([df[f"{kp}_x"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    hand_y = smooth(np.mean([df[f"{kp}_y"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    boxl_x = smooth(df["BOXL_x"].to_numpy())
    boxl_y = smooth(df["BOXL_y"].to_numpy())
    boxr_x = smooth(df["BOXR_x"].to_numpy())
    boxr_y = smooth(df["BOXR_y"].to_numpy())
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
    return dist_boxl / np.maximum(apparatus, 1e-3)


def load_dlc_h5(path):
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def load_live_gt(corpus, vid):
    p = (CAL_GT_DIR if corpus == "cal" else GEN_GT_DIR) / f"{vid}_unified_ground_truth.json"
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set((int(r["start_frame"]), int(r["end_frame"]))
                       for r in rlist if not r.get("exclude_from_analysis")))


def load_calibration():
    print("Loading calibration LOOCV + parquet...", flush=True)
    data = json.loads(CAL_LOOCV_SOURCE.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    algos = defaultdict(set)
    for r in raw:
        if r["algo_start_frame"] >= 0:
            algos[r["video_id"]].add((int(r["algo_start_frame"]), int(r["algo_end_frame"])))
    df = pd.read_parquet(CAL_PARQUET,
                          columns=["video_id", "frame"] + PARQUET_LK_COLS + PARQUET_POS_COLS)
    out = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame").reset_index(drop=True)
        paw_lk_matrix = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32)
        paw_mean = paw_lk_matrix.mean(axis=1)
        mx = int(g["frame"].max())
        lk_arr = np.full(mx + 1, np.nan, dtype=np.float32)
        lk_arr[g["frame"].to_numpy()] = paw_mean
        norm_pos = compute_norm_pos_from_df(g)
        np_arr = np.full(mx + 1, np.nan, dtype=np.float32)
        np_arr[g["frame"].to_numpy()] = norm_pos
        out[vid] = {
            "algos_v801": sorted(algos.get(vid, set())),
            "gts": load_live_gt("cal", vid),
            "paw_lk": lk_arr, "norm_pos": np_arr,
        }
    print(f"  {len(out)} calibration videos loaded")
    return out


def load_holdout():
    print("Loading holdout outputs + DLC + GT...", flush=True)
    out = {}
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not dlc_path.exists(): continue
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v801 = sorted(set((int(r["start_frame"]), int(r["end_frame"]))
                                 for r in adata.get("reaches", [])))
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        out[vid] = {
            "algos_v801": algos_v801,
            "gts": load_live_gt("hol", vid),
            "paw_lk": paw_lk, "norm_pos": norm_pos,
        }
    print(f"  {len(out)} holdout videos loaded")
    return out


def apply_pipeline(algos_v801, paw_lk, norm_pos, mh_max_gap, mh_max_nr):
    """Full v8.0.4 + merge-during-hold (NEW)."""
    spans = [ReachSpan(s, e) for s, e in algos_v801]
    spans = trim_leading_sustained_lk(spans, paw_lk,
                                       threshold=TRIM_LEADING_T,
                                       sustain_n=TRIM_LEADING_N,
                                       min_span=TRIM_MIN_SPAN)
    spans = trim_trailing_sustained_lk(spans, paw_lk,
                                        threshold=TRIM_TRAILING_T,
                                        sustain_n=TRIM_TRAILING_N,
                                        min_span=TRIM_MIN_SPAN)
    spans = apex_split_at_trough(spans, norm_pos,
                                  prominence=APEX_PROMINENCE,
                                  depth_min=APEX_DEPTH_MIN,
                                  peak2_rel_max=APEX_PEAK2_REL_MAX,
                                  min_distance=APEX_MIN_DISTANCE,
                                  min_span=APEX_MIN_SPAN)
    if mh_max_gap > 0:
        spans = merge_during_hold(spans, paw_lk, norm_pos,
                                   max_gap=mh_max_gap,
                                   max_norm_pos_range=mh_max_nr)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


def score_corpus(corpus_data, mh_max_gap, mh_max_nr):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    topo = defaultdict(int)
    start_d = []; span_d = []
    n_merges = 0
    for vid, vd in corpus_data.items():
        # Algos before merge-during-hold (for counting merges)
        spans_pre = apply_pipeline(vd["algos_v801"], vd["paw_lk"], vd["norm_pos"], 0, 0)
        algos = apply_pipeline(vd["algos_v801"], vd["paw_lk"], vd["norm_pos"],
                                mh_max_gap, mh_max_nr)
        n_merges += len(spans_pre) - len(algos)
        gts = vd["gts"]
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        totals["tp"] += len(matched)
        totals["fp"] += len(algos) - len(matched)
        totals["fn"] += len(gts) - len(matched)
        start_d.extend(tp_sd); span_d.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            topo[k] += v
    return {
        "totals": totals, "topology": dict(topo),
        "start_delta_abs_med": int(np.median([abs(d) for d in start_d])) if start_d else None,
        "span_delta_abs_med": int(np.median([abs(d) for d in span_d])) if span_d else None,
        "n_merges": n_merges,
    }


def main():
    print("=" * 70)
    print("MERGE-DURING-HOLD SWEEP")
    print(f"Configs: {[(g, nr) for g, nr, _ in SWEEP_CONFIGS]}")
    print(f"Fixed: min_paw_lk_mean_in_gap = {MIN_PAW_LK_MEAN_IN_GAP}")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    print("Computing baseline (v8.0.4, no merge-during-hold)...", flush=True)
    cal_base = score_corpus(cal_data, 0, 0)
    hol_base = score_corpus(hol_data, 0, 0)
    print(f"  Cal: TP={cal_base['totals']['tp']} FP={cal_base['totals']['fp']} FN={cal_base['totals']['fn']}")
    print(f"  Hol: TP={hol_base['totals']['tp']} FP={hol_base['totals']['fp']} FN={hol_base['totals']['fn']}")
    print(f"  Cal topology: {cal_base['topology']}")
    print(f"  Hol topology: {hol_base['topology']}")
    print()

    results = {"baseline": {"cal": cal_base, "hol": hol_base}, "configs": {}}
    for max_gap, max_nr, label in SWEEP_CONFIGS:
        print(f"--- {label} (max_gap={max_gap}, max_norm_pos_range={max_nr}) ---", flush=True)
        cr = score_corpus(cal_data, max_gap, max_nr)
        hr = score_corpus(hol_data, max_gap, max_nr)
        results["configs"][label] = {"max_gap": max_gap, "max_nr": max_nr,
                                       "cal": cr, "hol": hr}
        cb = cal_base["totals"]; ct = cr["totals"]
        hb = hol_base["totals"]; ht = hr["totals"]
        print(f"  Cal: TP={ct['tp']} ({ct['tp']-cb['tp']:+}) FP={ct['fp']} ({ct['fp']-cb['fp']:+}) "
              f"FN={ct['fn']} ({ct['fn']-cb['fn']:+})  merges={cr['n_merges']}  "
              f"abs_med start={cr['start_delta_abs_med']} span={cr['span_delta_abs_med']}")
        print(f"  Hol: TP={ht['tp']} ({ht['tp']-hb['tp']:+}) FP={ht['fp']} ({ht['fp']-hb['fp']:+}) "
              f"FN={ht['fn']} ({ht['fn']-hb['fn']:+})  merges={hr['n_merges']}  "
              f"abs_med start={hr['start_delta_abs_med']} span={hr['span_delta_abs_med']}")
        for lbl, base, cur in [("CAL", cal_base, cr), ("HOL", hol_base, hr)]:
            deltas = []
            for k in ("TP","TOLERANCE_ERROR_pairs","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE"):
                d = cur["topology"].get(k, 0) - base["topology"].get(k, 0)
                deltas.append(f"{k}={cur['topology'].get(k,0)}({d:+})")
            print(f"    {lbl} topology: {' '.join(deltas)}")
        print()

    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "configs": SWEEP_CONFIGS,
        "fixed_min_paw_lk_mean": MIN_PAW_LK_MEAN_IN_GAP,
        "results": results,
    }, indent=2, default=int), encoding="utf-8")

    # Summary
    print()
    print("=" * 130)
    print("SUMMARY (legacy matcher counts)")
    print("=" * 130)
    print(f"{'config':<20} {'merges_C':>8} {'merges_H':>8}  |  "
          f"{'Cal TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  |  "
          f"{'Hol TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}")
    print("-" * 130)
    cb = cal_base["totals"]; hb = hol_base["totals"]
    print(f"{'BASE':<20} {'':>8} {'':>8}  |  {cb['tp']:>6} {'':>4} {cb['fp']:>4} {'':>4} {cb['fn']:>4} {'':>4}  |  "
          f"{hb['tp']:>6} {'':>4} {hb['fp']:>4} {'':>4} {hb['fn']:>4} {'':>4}")
    for _, _, label in SWEEP_CONFIGS:
        r = results["configs"][label]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{label:<20} {r['cal']['n_merges']:>8} {r['hol']['n_merges']:>8}  |  "
              f"{ct['tp']:>6} {ct['tp']-cb['tp']:>+4} {ct['fp']:>4} {ct['fp']-cb['fp']:>+4} "
              f"{ct['fn']:>4} {ct['fn']-cb['fn']:>+4}  |  "
              f"{ht['tp']:>6} {ht['tp']-hb['tp']:>+4} {ht['fp']:>4} {ht['fp']-hb['fp']:>+4} "
              f"{ht['fn']:>4} {ht['fn']-hb['fn']:>+4}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
