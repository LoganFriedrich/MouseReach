"""
v8.0.x experiment: apex-split postprocess sweep (6 configs).

After v8.0.2 trim and before the asymmetric-tolerance matcher, apply a
split-at-apex-trough postprocess: for each emitted reach, detect 2+
peaks in dist(hand_centroid -> BoxL) / apparatus_width, and split at the
deepest trough if the trough depth is sufficient AND the second peak
isn't a late-stage end-of-reach artifact.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-22):
   - Production v8.0.2 (BSW b=1/w=0.8 + mg=0 + sustained-trim N=3/T=0.60)
   - Metric convention: asymmetric -2/+5 tolerance
   - Comparison baseline (current cumulative best with live GT):
     Calibration LOOCV unfiltered: TP=2231 / FP=98 / FN=170
     Holdout 19 unfiltered:        TP=3656 / FP=71 / FN=96
   - This experiment stacks apex-split on top of v8.0.2 trim, scored
     under the asymmetric matcher.

2. Existing-code-modification check:
   - NO modifications to src/mousereach/reach/v8/* or to
     mousereach.improvement.reach_detection.metrics during the experiment.
   - All apex-split logic inline in this runner.

3. Unverified hypotheses:
   - Diagnostic showed 67/73 MERGED with detectable 2+ peaks at depth>=0.5
     (corpus-wide), and 11/35 TP false-splits surviving peak2_rel<0.85.
   - Whether MERGED actually convert to TPs (or fall into TOLERANCE_ERROR /
     FRAGMENTED) under the cumulative scoring with asymmetric tolerance.
   - Whether the split point (trough frame) is kinematically accurate
     enough to preserve start_delta abs_median=0.
   - GT was edited 2026-05-22 by Logan; baseline updated to reflect.

4. FN-direction-reporting (planned):
   - Lead with FN delta vs cumulative best (2231 cal / 3656 holdout).
   - Both legacy (TP/FP/FN) and topology (TP / TOLERANCE_ERROR / MERGED /
     FRAGMENTED / FALSE_POSITIVE / FALSE_NEGATIVE / COMPLEX).
   - ASCII output only.

5. Framework check:
   - Output to v8.0.2_dev_apex_split_sweep/
   - JSON per config + summary table.

6. Branch + tag:
   - feature/v8-apex-split-postprocess
   - Tag: v8-pre-apex-split-2026-05-22

7. Decision rule (per config):
   ACCEPT if all of:
     - TP rises or holds vs baseline on BOTH corpora
     - FN drops or holds vs baseline on BOTH corpora
     - start_delta abs_median holds at 0 (Cardinal Rule)
     - MERGED topology drops materially on the corpus where merges exist
     - FRAGMENTED topology does not explode (>2x baseline)
   REJECT if:
     - TP drops AND FN rises
     - start_delta abs_median > 0
     - FRAGMENTED count > 2x baseline (apex split going wrong)

   Best config = highest TP gain with cleanest topology shift.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches, MIN_REPORTED_SPAN, is_kinematically_excluded,
    is_outside_gt_segmentation,
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
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
HOLDOUT_GT_DIR = Path(
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
    r"\reach_detection\v8.0.2_dev_apex_split_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# v8.0.2 trim parameters (production)
TRIM_THRESHOLD = 0.60
TRIM_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

# Apex split sweep parameters
APEX_PROMINENCE = 0.05
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3  # min span of each half after split

SWEEP_CONFIGS = [
    # (trough_depth_min, peak2_rel_max)
    (0.4, 0.80),  # A: aggressive
    (0.4, 0.85),  # B: aggressive+permissive
    (0.5, 0.80),  # C: balanced
    (0.5, 0.85),  # D: USER CHOICE (conservative)
    (0.6, 0.80),  # E: strict
    (0.6, 0.85),  # F: strict+permissive
]

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


# ---------- Helpers ----------

def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def compute_norm_pos_from_df(df):
    """Compute centroid norm_pos (0=BoxL, 1=BoxR) from a df with raw positions."""
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


def apex_split(reach, norm_pos, depth_min, peak2_rel_max,
               prominence=APEX_PROMINENCE, min_distance=APEX_MIN_DISTANCE,
               min_span=APEX_MIN_SPAN):
    """Returns a list of (start, end) tuples. 1 if no split, 2 if split."""
    s, e = reach
    if e >= len(norm_pos):
        return [(s, e)]
    sig = norm_pos[s:e + 1]
    if len(sig) < 3:
        return [(s, e)]
    peaks, _ = find_peaks(sig, prominence=prominence, distance=min_distance)
    if len(peaks) < 2:
        return [(s, e)]
    # Check peak2 relative position (use last peak)
    peak2_rel = peaks[-1] / (len(sig) - 1)
    if peak2_rel >= peak2_rel_max:
        return [(s, e)]
    # Find the deepest trough among all consecutive peak pairs
    best_depth = 0.0
    best_trough_frame = None
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        if p2 - p1 < 2:
            continue
        between = sig[p1:p2 + 1]
        t_local = int(np.argmin(between))
        t_val = float(between[t_local])
        max_p = max(float(sig[p1]), float(sig[p2]))
        depth = max_p - t_val
        if depth > best_depth:
            best_depth = depth
            best_trough_frame = s + p1 + t_local
    if best_depth < depth_min or best_trough_frame is None:
        return [(s, e)]
    # Split at trough: half1 ends at trough, half2 starts at trough+1
    half1 = (s, best_trough_frame)
    half2 = (best_trough_frame + 1, e)
    # Check min_span
    if (half1[1] - half1[0] + 1) < min_span or (half2[1] - half2[0] + 1) < min_span:
        return [(s, e)]
    return [half1, half2]


# ---------- Matcher + topology (asymmetric tolerance) ----------

def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
            if (-STRICT_START_TOL_EARLY <= start_delta <= STRICT_START_TOL_LATE
                    and abs(span_delta) <= span_tol):
                candidates.append((abs(start_delta), ai, gi, start_delta))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []
    tp_sd = []
    for _, ai, gi, sd in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        pairs.append((ai, gi))
        tp_sd.append(sd)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_sd


def classify_topology(algos, gts):
    algo_to_gt = defaultdict(set)
    gt_to_algo = defaultdict(set)
    for i, (a_s, a_e) in enumerate(algos):
        for j, (g_s, g_e) in enumerate(gts):
            if overlap(a_s, a_e, g_s, g_e):
                algo_to_gt[i].add(j)
                gt_to_algo[j].add(i)
    visited_a, visited_g = set(), set()
    comps = []
    for i in range(len(algos)):
        if i in visited_a: continue
        if not algo_to_gt[i]:
            comps.append("FALSE_POSITIVE")
            visited_a.add(i); continue
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in: continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]: queue.append(("g", gj))
            else:
                if idx in gt_in: continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]: queue.append(("a", ai))
        visited_a.update(algo_in); visited_g.update(gt_in)
        na, ng = len(algo_in), len(gt_in)
        if na == 1 and ng == 1:
            a = algos[next(iter(algo_in))]
            g = gts[next(iter(gt_in))]
            sd = a[0] - g[0]
            span_a = a[1] - a[0] + 1; span_g = g[1] - g[0] + 1
            sp_d = span_a - span_g
            sp_tol = max(SPAN_TOL_FRAC * span_g, SPAN_TOL_MIN)
            if -STRICT_START_TOL_EARLY <= sd <= STRICT_START_TOL_LATE and abs(sp_d) <= sp_tol:
                comps.append("TP")
            else:
                comps.append("TOLERANCE_ERROR")
        elif na == 1 and ng >= 2:
            comps.append("MERGED")
        elif na >= 2 and ng == 1:
            comps.append("FRAGMENTED")
        elif na >= 2 and ng >= 2:
            comps.append("COMPLEX")
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append("FALSE_NEGATIVE")
    return comps


# ---------- Data loading ----------

def load_live_gt(corpus_label, video_id):
    if corpus_label == "calibration_loocv":
        gt_path = CAL_GT_DIR / f"{video_id}_unified_ground_truth.json"
    else:
        gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    reaches_obj = data.get("reaches", {})
    rlist = (reaches_obj.get("reaches", [])
             if isinstance(reaches_obj, dict) else [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_calibration():
    """Returns dict[video_id] -> {algos_v801, gts_live, paw_lk, norm_pos}."""
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
        # Align lk array by frame
        mx_frame = int(g["frame"].max())
        lk_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        lk_arr[g["frame"].to_numpy()] = paw_mean_lk
        # norm_pos
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
    """Returns dict[video_id] -> {algos_v801, gts_live, paw_lk, norm_pos}."""
    print("Loading holdout outputs + DLC + GT...", flush=True)
    out = {}
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        dlc_path = HOLDOUT_GT_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not dlc_path.exists():
            continue
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v801 = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in adata.get("reaches", [])
        ))
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_norm_pos_from_df(dlc)
        out[vid] = {
            "algos_v801": algos_v801,
            "gts": load_live_gt("holdout_2026_05_11", vid),
            "paw_lk": paw_lk,
            "norm_pos": norm_pos,
        }
    print(f"  {len(out)} holdout videos loaded")
    return out


# ---------- Scoring ----------

def apply_v802_trim(algos, paw_lk):
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_THRESHOLD,
        sustain_n=TRIM_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN,
    )
    return [(r.start_frame, r.end_frame) for r in trimmed]


def score_corpus(corpus_data, depth_min, peak2_rel_max, apply_apex=True):
    """Score with v8.0.2 trim + optional apex split + asymmetric matcher."""
    total_tp = total_fp = total_fn = 0
    topo_counts = defaultdict(int)
    tp_start_deltas = []
    splits_made = 0
    for vid, vd in corpus_data.items():
        # v8.0.2 trim
        algos_v802 = apply_v802_trim(vd["algos_v801"], vd["paw_lk"])
        # Apex split (if enabled)
        if apply_apex:
            algos_final = []
            for reach in algos_v802:
                parts = apex_split(reach, vd["norm_pos"],
                                    depth_min=depth_min,
                                    peak2_rel_max=peak2_rel_max)
                if len(parts) > 1:
                    splits_made += 1
                algos_final.extend(parts)
        else:
            algos_final = algos_v802
        # Sort + dedup
        algos_final = sorted(set(algos_final))
        # Match against live GT
        pairs, fps, fns, tp_sd = greedy_match(algos_final, vd["gts"])
        total_tp += len(pairs); total_fp += len(fps); total_fn += len(fns)
        tp_start_deltas.extend(tp_sd)
        for c in classify_topology(algos_final, vd["gts"]):
            topo_counts[c] += 1
    abs_med = int(np.median([abs(d) for d in tp_start_deltas])) if tp_start_deltas else None
    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "topology": dict(topo_counts),
        "start_delta_abs_median": abs_med,
        "splits_made": splits_made,
    }


# ---------- Main ----------

def main():
    print("=" * 70)
    print("APEX SPLIT POSTPROCESS SWEEP (6 configs)")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    holdout_data = load_holdout()
    print()

    # Baseline (no apex split)
    print("Computing baseline (v8.0.2 trim + asymmetric tol, NO apex split)...", flush=True)
    cal_baseline = score_corpus(cal_data, 0, 0, apply_apex=False)
    hol_baseline = score_corpus(holdout_data, 0, 0, apply_apex=False)
    print(f"  Calibration baseline: TP={cal_baseline['tp']} FP={cal_baseline['fp']} FN={cal_baseline['fn']}")
    print(f"  Holdout baseline:     TP={hol_baseline['tp']} FP={hol_baseline['fp']} FN={hol_baseline['fn']}")
    print()

    # Sweep
    results = {}
    for (depth, p2_rel) in SWEEP_CONFIGS:
        key = f"depth_{depth}_peak2_{p2_rel}"
        print(f"--- config: depth>={depth}, peak2_rel<{p2_rel} ---", flush=True)
        cal_r = score_corpus(cal_data, depth, p2_rel, apply_apex=True)
        hol_r = score_corpus(holdout_data, depth, p2_rel, apply_apex=True)
        results[key] = {
            "depth_min": depth, "peak2_rel_max": p2_rel,
            "calibration": cal_r, "holdout": hol_r,
        }
        # Print quick comparison
        c = cal_r; h = hol_r
        print(f"  Cal:     TP={c['tp']:>4} ({c['tp']-cal_baseline['tp']:+3})  "
              f"FP={c['fp']:>3} ({c['fp']-cal_baseline['fp']:+3})  "
              f"FN={c['fn']:>4} ({c['fn']-cal_baseline['fn']:+3})  "
              f"abs_med={c['start_delta_abs_median']}  splits={c['splits_made']}")
        print(f"  Holdout: TP={h['tp']:>4} ({h['tp']-hol_baseline['tp']:+3})  "
              f"FP={h['fp']:>3} ({h['fp']-hol_baseline['fp']:+3})  "
              f"FN={h['fn']:>4} ({h['fn']-hol_baseline['fn']:+3})  "
              f"abs_med={h['start_delta_abs_median']}  splits={h['splits_made']}")
        topo_diff = []
        for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE","COMPLEX"):
            cb = cal_baseline["topology"].get(k, 0); cn = c["topology"].get(k, 0)
            hb = hol_baseline["topology"].get(k, 0); hn = h["topology"].get(k, 0)
            topo_diff.append(f"{k}={cn}({cn-cb:+d})|{hn}({hn-hb:+d})")
        print(f"  topo: {'  '.join(topo_diff)}")
        print()

    # Save
    out_json = OUT_DIR / "metrics" / "sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline": {"calibration": cal_baseline, "holdout": hol_baseline},
        "configs": results,
    }, indent=2, default=int), encoding="utf-8")
    print(f"Wrote: {out_json}")

    # Summary table
    print()
    print("=" * 140)
    print("SUMMARY (LIVE GT)")
    print("=" * 140)
    print(f"{'config':<25}  {'CAL TP':>8} {'dTP':>5} {'FP':>5} {'dFP':>4} {'FN':>5} {'dFN':>4}  "
          f"{'CAL MERGED':>10} {'FRAG':>5}  | "
          f"{'HOL TP':>7} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  "
          f"{'HOL MERGED':>10} {'FRAG':>5}")
    print("-" * 140)
    print(f"{'baseline (no apex)':<25}  {cal_baseline['tp']:>8} {'':>5} {cal_baseline['fp']:>5} {'':>4} "
          f"{cal_baseline['fn']:>5} {'':>4}  {cal_baseline['topology'].get('MERGED',0):>10} "
          f"{cal_baseline['topology'].get('FRAGMENTED',0):>5}  | "
          f"{hol_baseline['tp']:>7} {'':>4} {hol_baseline['fp']:>4} {'':>4} "
          f"{hol_baseline['fn']:>4} {'':>4}  {hol_baseline['topology'].get('MERGED',0):>10} "
          f"{hol_baseline['topology'].get('FRAGMENTED',0):>5}")
    for key, r in results.items():
        c = r["calibration"]; h = r["holdout"]
        d_tp_c = c['tp'] - cal_baseline['tp']
        d_fp_c = c['fp'] - cal_baseline['fp']
        d_fn_c = c['fn'] - cal_baseline['fn']
        d_tp_h = h['tp'] - hol_baseline['tp']
        d_fp_h = h['fp'] - hol_baseline['fp']
        d_fn_h = h['fn'] - hol_baseline['fn']
        m_c = c['topology'].get('MERGED', 0); fr_c = c['topology'].get('FRAGMENTED', 0)
        m_h = h['topology'].get('MERGED', 0); fr_h = h['topology'].get('FRAGMENTED', 0)
        print(f"{key:<25}  {c['tp']:>8} {d_tp_c:>+5} {c['fp']:>5} {d_fp_c:>+4} {c['fn']:>5} {d_fn_c:>+4}  "
              f"{m_c:>10} {fr_c:>5}  | "
              f"{h['tp']:>7} {d_tp_h:>+4} {h['fp']:>4} {d_fp_h:>+4} {h['fn']:>4} {d_fn_h:>+4}  "
              f"{m_h:>10} {fr_h:>5}")
    print()


if __name__ == "__main__":
    main()
