"""v8.0.x experiment: split-at-low-lk-interior-gap sweep.

Postprocess that splits an algo reach when a sustained paw_lk dip is
found in the interior. Targets the residual MERGED events where two
real reaches were absorbed into one algo span because the paw briefly
went out-of-frame between them.

Mechanism is analogous to the leading/trailing trim, but the cut is
interior. The split assigns the dip frames as inter-reach gap (dropped),
giving two cleaner halves.

Per the 2026-05-26 paw_lk-interior-dip diagnostic (see
scripts/diagnose_paw_lk_interior_dip_for_split.py):
  - 13 filtered MERGEDs in cal + hol.
  - 5 of them have interior sustained low-lk runs at T=0.60/n=3.
  - 6 of 13 have a measurable inter-GT gap with paw_lk < 0.60 in the gap.
  - TP false-positive rate of "interior dip present": 0.04-0.13% at
    reasonable thresholds (discrimination ratio 100-1000x vs MERGED).
  - Mechanism: paw briefly invisible (pellet occlusion, retraction
    obscuration) between two real reaches, but GBM stays elevated.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26 evening from master
   at 3050dc6, post v8.0.4 trailing-trim ship):
   - Production stack: BSW b=1/w=0.8 model + mg=0 + leading-trim
     T=0.60/N=3 + trailing-trim T=0.60/N=3 + apex-split
     prom=0.12/depth=0.5/peak2<0.85.
   - Asymmetric strict matcher -2/+5 on live GT.
   - Matcher-aware topology classifier (no COMPLEX).
   - Comparison baseline (verified at experiment time by running the
     identical pipeline with split-at-low-lk DISABLED):
       Cal filtered (post 2026-05-26 GT): TP=2312 TOL=18 MERGED=10 FRAG=5 FP=9 FN=34
       Hol filtered:                       TP=3655 TOL=10 MERGED=3 FRAG=7 FP=12 FN=27

2. Existing-code-modification check: NO.
   - split_at_low_lk_interior_gap implemented inline in this runner.
   - postprocess.py, __init__.py UNTOUCHED.
   - If accepted, will integrate into postprocess.py on the ship branch.

3. Unverified hypotheses:
   - That splitting the 5 catchable MERGEDs (at T=0.60/n=3) will give
     2 clean TPs per split, not over-split into matched-TP + stranded
     FP. The diagnostic gives the dip position but doesn't verify
     boundary alignment with GT.
   - That the TP false-positive rate measured per-reach (0.04-0.13%)
     translates to a small absolute over-split count when scaled to
     the corpus (~7 events at T=0.60/n=3 across 5565 TPs).
   - That over-splits create FRAGMENTED (one half matches, one
     stranded as FP) rather than total-TP-loss (neither half matches).

4. FN-direction-reporting:
   - Both deltas (vs cumulative best v8.0.4 AND vs pure baseline v8.0.0
     mg=2). Lead with FN direction.
   - Topology paired with legacy throughout.

5. Framework check:
   - Output: Improvement_Snapshots/reach_detection/v8.0.4_dev_split_at_low_lk_sweep/
   - sweep_results.json + RESULTS.md when sweep completes.

6. Branch + tag:
   - Pre-experiment tag: pre-split-at-low-lk-2026-05-26 (created at 3050dc6)
   - Feature branch: feature/v8-split-at-low-lk (off 3050dc6)

7. Decision rule (applied per config, then pick Pareto-optimal):
   ACCEPT if (vs cumulative best v8.0.4):
     - TP non-decreasing on BOTH corpora (delta_TP >= 0)
     - MERGED strictly non-increasing on BOTH corpora
     - FRAGMENTED rise <= 3 events per corpus
     - Cardinal Rule: start_delta abs_median = 0 AND span_delta abs_median = 0
       on BOTH corpora
   REJECT if:
     - TP drops on either corpus
     - Either Cardinal Rule axis fails (abs_median > 0)
     - FRAGMENTED rises > 3 on either corpus
     - FN rises on either corpus (per FN-over-FP rule)
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
    compute_paw_mean_lk,
    apex_split_at_trough, compute_hand_to_boxl_norm_pos,
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
    r"\reach_detection\v8.0.4_dev_split_at_low_lk_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# Production v8.0.4 stack (frozen)
TRIM_LEADING_THRESHOLD = 0.60
TRIM_LEADING_SUSTAIN_N = 3
TRIM_TRAILING_THRESHOLD = 0.60
TRIM_TRAILING_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3
APEX_PROMINENCE = 0.12
APEX_DEPTH_MIN = 0.5
APEX_PEAK2_REL_MAX = 0.85
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3

# Split-at-low-lk sweep dimensions
SPLIT_MIN_SPAN = 3       # min frames for each post-split half
SPLIT_EDGE_PROTECT = 3   # don't search in first/last N frames (leading/
                          # trailing trim already cover those)
SPLIT_CONFIGS = [
    # (label, threshold, sustain_n)
    ("base",     None, None),
    ("t.40_n3", 0.40, 3),
    ("t.50_n3", 0.50, 3),
    ("t.50_n4", 0.50, 4),
    ("t.60_n3", 0.60, 3),
    ("t.60_n4", 0.60, 4),
]

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


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


# -----------------------------------------------------------------
# Split-at-low-lk-interior-gap postprocess (this experiment)
# -----------------------------------------------------------------

def split_at_low_lk_interior(reaches, paw_lk, threshold, sustain_n=3,
                               min_span=3, edge_protect=3):
    """For each reach, find the longest sustained run of paw_lk < threshold
    in the interior (excluding the first edge_protect and last edge_protect
    frames). If the run length >= sustain_n, split the reach into two halves:
       half1 = [start, dip_start - 1]
       half2 = [dip_end + 1, end]
    The dip frames themselves are dropped as inter-reach gap.

    Reaches without a qualifying dip pass through unchanged. Halves below
    min_span suppress the split.
    """
    if not reaches:
        return []
    n_frames = len(paw_lk)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        # Interior window (excluding edge_protect frames at each end)
        lo = s + edge_protect
        hi = e - edge_protect
        if hi < lo:
            out.append(r)
            continue
        if hi >= n_frames:
            out.append(r)
            continue
        vals = paw_lk[lo:hi + 1]
        if np.any(np.isnan(vals)):
            # NaN-containing interior is uninformative for splitting
            out.append(r)
            continue
        below = vals < threshold
        if not np.any(below):
            out.append(r)
            continue
        # Find longest run of True
        longest_len = 0
        longest_start = None
        i = 0
        n = len(below)
        while i < n:
            if below[i]:
                j = i
                while j < n and below[j]:
                    j += 1
                run_len = j - i
                if run_len > longest_len:
                    longest_len = run_len
                    longest_start = i
                i = j
            else:
                i += 1
        if longest_len < sustain_n:
            out.append(r)
            continue
        # Convert run indices (relative to interior) to frame indices
        dip_start = lo + longest_start
        dip_end = dip_start + longest_len - 1
        # Build halves; drop the dip frames
        half1_end = dip_start - 1
        half2_start = dip_end + 1
        half1_span = half1_end - s + 1
        half2_span = e - half2_start + 1
        if half1_span < min_span or half2_span < min_span:
            out.append(r)
            continue
        out.append(ReachSpan(start_frame=s, end_frame=half1_end))
        out.append(ReachSpan(start_frame=half2_start, end_frame=e))
    return out


# ---- Matcher + topology (same as the matcher-aware-topology snapshot) ----

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
    matched = set()
    used_a, used_g = set(), set()
    tp_sd, tp_pd = [], []
    for _, ai, gi, sd, pd_ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi))
        tp_sd.append(sd); tp_pd.append(pd_)
    return matched, tp_sd, tp_pd


def classify_matcher_aware(algos, gts, matched):
    """Locked rules from 2026-05-22 topology refactor."""
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
    tol_pair_events = 0
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
                tol_pair_events += 2
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
                    a_s, a_e = algos[ai]
                    g_s, g_e = gts[gj]
                    s = max(a_s, g_s); e = min(a_e, g_e)
                    ol = max(0, e - s + 1)
                    if ol > best_ol:
                        best_ol = ol; best_gj = gj
                if best_gj is not None and best_ol > 0:
                    tol_pair_events += 2
                    soft_paired.add(best_gj)
                else:
                    counts['FALSE_POSITIVE'] += 1
            for gj in sorted(unmatched_g - soft_paired):
                counts['FALSE_NEGATIVE'] += 1
    counts['TOLERANCE_ERROR_pairs'] = tol_pair_events // 2
    return dict(counts)


# ---- Data loading ----

def load_live_gt(corpus_label, video_id):
    if corpus_label == "calibration_loocv":
        gt_path = CAL_GT_DIR / f"{video_id}_unified_ground_truth.json"
    else:
        gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_calibration():
    print("Loading calibration LOOCV + parquet...", flush=True)
    data = json.loads(CAL_LOOCV_SOURCE.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    algos = defaultdict(set)
    for r in raw:
        if r["algo_start_frame"] >= 0:
            algos[r["video_id"]].add((int(r["algo_start_frame"]),
                                       int(r["algo_end_frame"])))
    df = pd.read_parquet(CAL_PARQUET,
                          columns=["video_id", "frame"] + PARQUET_LK_COLS + PARQUET_POS_COLS)
    out = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame").reset_index(drop=True)
        paw_lk_matrix = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32)
        paw_mean_lk = paw_lk_matrix.mean(axis=1)
        mx_frame = int(g["frame"].max())
        lk_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        lk_arr[g["frame"].to_numpy()] = paw_mean_lk
        norm_pos = compute_norm_pos_from_df(g)
        np_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        np_arr[g["frame"].to_numpy()] = norm_pos
        out[vid] = {
            "algos_v801": sorted(algos.get(vid, set())),
            "gts": load_live_gt("calibration_loocv", vid),
            "paw_lk": lk_arr,
            "norm_pos": np_arr,
        }
    print(f"  {len(out)} calibration videos loaded")
    return out


def load_holdout():
    print("Loading holdout outputs + DLC + GT...", flush=True)
    out = {}
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not dlc_path.exists():
            continue
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v801 = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in adata.get("reaches", [])
        ))
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        out[vid] = {
            "algos_v801": algos_v801,
            "gts": load_live_gt("holdout_2026_05_11", vid),
            "paw_lk": paw_lk,
            "norm_pos": norm_pos,
        }
    print(f"  {len(out)} holdout videos loaded")
    return out


# ---- Pipeline ----

def apply_pipeline(algos_v801, paw_lk, norm_pos, split_threshold, split_sustain_n):
    """v8.0.1 algos -> leading trim -> trailing trim -> apex split
       -> (NEW) split-at-low-lk-interior."""
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos_v801]
    spans = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_LEADING_THRESHOLD,
        sustain_n=TRIM_LEADING_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN)
    spans = trim_trailing_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_TRAILING_THRESHOLD,
        sustain_n=TRIM_TRAILING_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN)
    spans = apex_split_at_trough(
        spans, norm_pos,
        prominence=APEX_PROMINENCE,
        depth_min=APEX_DEPTH_MIN,
        peak2_rel_max=APEX_PEAK2_REL_MAX,
        min_distance=APEX_MIN_DISTANCE,
        min_span=APEX_MIN_SPAN)
    if split_threshold is not None and split_sustain_n is not None:
        spans = split_at_low_lk_interior(
            spans, paw_lk,
            threshold=split_threshold,
            sustain_n=split_sustain_n,
            min_span=SPLIT_MIN_SPAN,
            edge_protect=SPLIT_EDGE_PROTECT)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


def score_corpus(corpus_data, split_threshold, split_sustain_n):
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_algo": 0}
    topo = defaultdict(int)
    start_deltas = []
    span_deltas = []
    splits_introduced = 0
    for vid, vd in corpus_data.items():
        algos_pre = apply_pipeline(vd["algos_v801"], vd["paw_lk"], vd["norm_pos"],
                                    None, None)
        algos = apply_pipeline(vd["algos_v801"], vd["paw_lk"], vd["norm_pos"],
                                split_threshold, split_sustain_n)
        splits_introduced += len(algos) - len(algos_pre)
        gts = vd["gts"]
        totals["n_algo"] += len(algos)
        matched, tp_sd, tp_pd = greedy_match(algos, gts)
        totals["tp"] += len(matched)
        totals["fp"] += len(algos) - len(matched)
        totals["fn"] += len(gts) - len(matched)
        start_deltas.extend(tp_sd)
        span_deltas.extend(tp_pd)
        tc = classify_matcher_aware(algos, gts, matched)
        for k, v in tc.items():
            topo[k] += v
    s_abs = int(np.median([abs(d) for d in start_deltas])) if start_deltas else None
    p_abs = int(np.median([abs(d) for d in span_deltas])) if span_deltas else None
    return {
        "totals": totals,
        "topology": dict(topo),
        "start_delta_abs_median": s_abs,
        "span_delta_abs_median": p_abs,
        "splits_introduced": splits_introduced,
    }


def main():
    print("=" * 70)
    print("SPLIT-AT-LOW-LK-INTERIOR SWEEP")
    print(f"Configs: {[c[0] for c in SPLIT_CONFIGS]}")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    results = {"configs": {}}
    base_cal = None
    base_hol = None
    for label, thresh, sustain in SPLIT_CONFIGS:
        if thresh is None:
            print(f"--- {label} (v8.0.4 baseline, no interior split) ---", flush=True)
        else:
            print(f"--- {label} (T={thresh}, n={sustain}) ---", flush=True)
        cal_r = score_corpus(cal_data, thresh, sustain)
        hol_r = score_corpus(hol_data, thresh, sustain)
        if label == "base":
            base_cal = cal_r
            base_hol = hol_r
        results["configs"][label] = {
            "threshold": thresh,
            "sustain_n": sustain,
            "cal": cal_r,
            "hol": hol_r,
        }
        if label == "base":
            cb = cal_r["totals"]; hb = hol_r["totals"]
            print(f"  Cal: TP={cb['tp']} FP={cb['fp']} FN={cb['fn']}  "
                  f"abs_med start={cal_r['start_delta_abs_median']} span={cal_r['span_delta_abs_median']}")
            print(f"  Hol: TP={hb['tp']} FP={hb['fp']} FN={hb['fn']}  "
                  f"abs_med start={hol_r['start_delta_abs_median']} span={hol_r['span_delta_abs_median']}")
            print(f"  Cal topology: {cal_r['topology']}")
            print(f"  Hol topology: {hol_r['topology']}")
        else:
            cb = base_cal["totals"]; ct = cal_r["totals"]
            hb = base_hol["totals"]; ht = hol_r["totals"]
            print(f"  Cal: splits={cal_r['splits_introduced']}  "
                  f"TP={ct['tp']} ({ct['tp']-cb['tp']:+}) "
                  f"FP={ct['fp']} ({ct['fp']-cb['fp']:+}) "
                  f"FN={ct['fn']} ({ct['fn']-cb['fn']:+})  "
                  f"abs_med start={cal_r['start_delta_abs_median']} span={cal_r['span_delta_abs_median']}")
            print(f"  Hol: splits={hol_r['splits_introduced']}  "
                  f"TP={ht['tp']} ({ht['tp']-hb['tp']:+}) "
                  f"FP={ht['fp']} ({ht['fp']-hb['fp']:+}) "
                  f"FN={ht['fn']} ({ht['fn']-hb['fn']:+})  "
                  f"abs_med start={hol_r['start_delta_abs_median']} span={hol_r['span_delta_abs_median']}")
            for lbl, base, cur in [("CAL", base_cal, cal_r), ("HOL", base_hol, hol_r)]:
                b_topo = base["topology"]; t_topo = cur["topology"]
                deltas = []
                for k in ("TP","TOLERANCE_ERROR_pairs","MERGED","FRAGMENTED",
                          "FALSE_POSITIVE","FALSE_NEGATIVE"):
                    d = t_topo.get(k, 0) - b_topo.get(k, 0)
                    deltas.append(f"{k}={t_topo.get(k,0)}({d:+})")
                print(f"    {lbl} topology: {' '.join(deltas)}")
        print()

    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "configs": [{"label": c[0], "threshold": c[1], "sustain_n": c[2]}
                    for c in SPLIT_CONFIGS],
        "results": results,
    }, indent=2, default=int), encoding="utf-8")

    # Summary table (legacy matcher counts)
    print()
    print("=" * 130)
    print("SUMMARY (legacy matcher counts; topology counts in detail above)")
    print("=" * 130)
    print(f"{'label':<10}  {'Cal_TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  |  "
          f"{'Hol_TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}")
    print("-" * 130)
    cb = base_cal["totals"]; hb = base_hol["totals"]
    print(f"{'BASE':<10}  {cb['tp']:>6} {'':>4} {cb['fp']:>4} {'':>4} {cb['fn']:>4} {'':>4}  |  "
          f"{hb['tp']:>6} {'':>4} {hb['fp']:>4} {'':>4} {hb['fn']:>4} {'':>4}")
    for label, _, _ in SPLIT_CONFIGS:
        if label == "base":
            continue
        r = results["configs"][label]
        ct = r["cal"]["totals"]; ht = r["hol"]["totals"]
        print(f"{label:<10}  {ct['tp']:>6} {ct['tp']-cb['tp']:>+4} {ct['fp']:>4} {ct['fp']-cb['fp']:>+4} "
              f"{ct['fn']:>4} {ct['fn']-cb['fn']:>+4}  |  "
              f"{ht['tp']:>6} {ht['tp']-hb['tp']:>+4} {ht['fp']:>4} {ht['fp']-hb['fp']:>+4} "
              f"{ht['fn']:>4} {ht['fn']-hb['fn']:>+4}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
