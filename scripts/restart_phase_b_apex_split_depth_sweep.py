"""
v8.0.x experiment: apex-split depth sweep at prominence=0.12, peak2_rel<0.85.

Follow-up to v8.0.2_dev_apex_split_prominence_sweep: the prom=0.12 config
is the current candidate (Cal +84/-35/-84, Hol +22/-7/-22, over_splits
2 cal + 1 hol). This run sweeps trough_depth_min at the chosen prominence
to confirm depth=0.5 is the right setting (or find a better one).

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-22):
   - Production v8.0.2 + asymmetric -2/+5 on live GT.
   - Baseline: Cal TP=2231 / FP=98 / FN=170, Hol TP=3656 / FP=71 / FN=96.
   - No reverts on master since v8.0.2 ship.

2. Existing-code-modification check: NO. Inline only.

3. Unverified hypotheses:
   - That depth in [0.4, 0.7] at fixed prominence=0.12 will identify
     whether depth=0.5 is the right cut. Lower depth = more aggressive
     (more MERGED catches, possibly more over-splits). Higher depth =
     fewer of both.

4. FN-direction-reporting: lead with FN delta vs cumulative best (also
   == pure baseline). Pair legacy with topology.

5. Framework: output to v8.0.2_dev_apex_split_depth_sweep/.

6. Branch + tag: feature/v8-apex-split-postprocess + tag
   v8-pre-apex-split-2026-05-22.

7. Decision rule (per config): same as prominence sweep
   ACCEPT if all of:
     - TP rises/holds both corpora
     - FN drops/holds both corpora
     - start_delta abs_median = 0
     - MERGED drops >= 70%
     - FRAGMENTED <= 2x baseline AND over_splits <= 4 per corpus
   Best = highest MERGED catch with lowest over-split count.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
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
    r"\reach_detection\v8.0.2_dev_apex_split_depth_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

TRIM_THRESHOLD = 0.60
TRIM_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

PROMINENCE = 0.12  # fixed (chosen from prior sweep)
PEAK2_REL_MAX = 0.85
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3

DEPTH_VALUES = [0.40, 0.50, 0.60, 0.70]

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


def apex_split(reach, norm_pos, depth_min,
               prominence=PROMINENCE, peak2_rel_max=PEAK2_REL_MAX,
               min_distance=APEX_MIN_DISTANCE, min_span=APEX_MIN_SPAN):
    s, e = reach
    if e >= len(norm_pos):
        return [(s, e)]
    sig = norm_pos[s:e + 1]
    if len(sig) < 3:
        return [(s, e)]
    peaks, _ = find_peaks(sig, prominence=prominence, distance=min_distance)
    if len(peaks) < 2:
        return [(s, e)]
    peak2_rel = peaks[-1] / (len(sig) - 1)
    if peak2_rel >= peak2_rel_max:
        return [(s, e)]
    best_depth = 0.0
    best_trough_frame = None
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        if p2 - p1 < 2:
            continue
        between = sig[p1:p2 + 1]
        t_local = int(np.argmin(between))
        depth = max(float(sig[p1]), float(sig[p2])) - float(between[t_local])
        if depth > best_depth:
            best_depth = depth
            best_trough_frame = s + p1 + t_local
    if best_depth < depth_min or best_trough_frame is None:
        return [(s, e)]
    half1 = (s, best_trough_frame)
    half2 = (best_trough_frame + 1, e)
    if (half1[1] - half1[0] + 1) < min_span or (half2[1] - half2[0] + 1) < min_span:
        return [(s, e)]
    return [half1, half2]


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
    pairs = []; tp_sd = []
    for _, ai, gi, sd in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        pairs.append((ai, gi))
        tp_sd.append(sd)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_sd


def classify_components(algos, gts):
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
        if i in visited_a:
            continue
        if not algo_to_gt[i]:
            comps.append({"algos": {i}, "gts": set(), "label": "FALSE_POSITIVE"})
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
                label = "TP"
            else:
                label = "TOLERANCE_ERROR"
        elif na == 1 and ng >= 2:
            label = "MERGED"
        elif na >= 2 and ng == 1:
            label = "FRAGMENTED"
        elif na >= 2 and ng >= 2:
            label = "COMPLEX"
        comps.append({"algos": algo_in, "gts": gt_in, "label": label})
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append({"algos": set(), "gts": {j}, "label": "FALSE_NEGATIVE"})
    return comps


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


def apply_v802_trim(algos, paw_lk):
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_THRESHOLD,
        sustain_n=TRIM_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN,
    )
    return [(r.start_frame, r.end_frame) for r in trimmed]


def score_corpus(corpus_data, depth_min, apply_apex):
    totals = {"tp": 0, "fp": 0, "fn": 0, "splits_made": 0}
    topo_counts = defaultdict(int)
    tp_start_deltas = []
    per_video = {}
    for vid, vd in corpus_data.items():
        algos_v802 = apply_v802_trim(vd["algos_v801"], vd["paw_lk"])
        if apply_apex:
            algos_final = []
            for reach in algos_v802:
                parts = apex_split(reach, vd["norm_pos"], depth_min=depth_min)
                if len(parts) > 1:
                    totals["splits_made"] += 1
                algos_final.extend(parts)
        else:
            algos_final = algos_v802
        algos_final = sorted(set(algos_final))
        pairs, fps, fns, tp_sd = greedy_match(algos_final, vd["gts"])
        totals["tp"] += len(pairs); totals["fp"] += len(fps); totals["fn"] += len(fns)
        tp_start_deltas.extend(tp_sd)
        comps = classify_components(algos_final, vd["gts"])
        for c in comps:
            topo_counts[c["label"]] += 1
        per_video[vid] = {
            "algos_final": algos_final,
            "gts": vd["gts"],
            "components": [
                {"algos": sorted(c["algos"]),
                 "gts": sorted(c["gts"]),
                 "label": c["label"]}
                for c in comps
            ],
        }
    abs_med = int(np.median([abs(d) for d in tp_start_deltas])) if tp_start_deltas else None
    return {
        "totals": totals,
        "topology": dict(topo_counts),
        "start_delta_abs_median": abs_med,
        "per_video": per_video,
    }


def count_over_splits(baseline_per_video, apex_per_video):
    over_split_count = 0
    from_merged = 0
    from_frag = 0
    from_tol = 0
    for vid in apex_per_video:
        base_comps = baseline_per_video[vid]["components"]
        gt_to_base_label = {}
        for c in base_comps:
            for gj in c["gts"]:
                gt_to_base_label[gj] = c["label"]
        for c in apex_per_video[vid]["components"]:
            if c["label"] != "FRAGMENTED":
                continue
            gj = c["gts"][0]
            base_label = gt_to_base_label.get(gj, "UNKNOWN")
            if base_label == "TP":
                over_split_count += 1
            elif base_label == "MERGED":
                from_merged += 1
            elif base_label == "FRAGMENTED":
                from_frag += 1
            elif base_label == "TOLERANCE_ERROR":
                from_tol += 1
    return {"over_splits": over_split_count,
             "from_merged": from_merged,
             "from_fragmented": from_frag,
             "from_tolerance_error": from_tol}


def main():
    print("=" * 70)
    print(f"APEX SPLIT DEPTH SWEEP (prominence={PROMINENCE}, peak2_rel<{PEAK2_REL_MAX})")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    print("Scoring baselines (no apex)...", flush=True)
    cal_base = score_corpus(cal_data, depth_min=0.5, apply_apex=False)
    hol_base = score_corpus(hol_data, depth_min=0.5, apply_apex=False)
    cbt = cal_base["totals"]; hbt = hol_base["totals"]
    cb_merged = cal_base["topology"].get("MERGED", 0)
    hb_merged = hol_base["topology"].get("MERGED", 0)
    cb_frag = cal_base["topology"].get("FRAGMENTED", 0)
    hb_frag = hol_base["topology"].get("FRAGMENTED", 0)
    print(f"  Cal baseline: TP={cbt['tp']} FP={cbt['fp']} FN={cbt['fn']} MERGED={cb_merged} FRAG={cb_frag}")
    print(f"  Hol baseline: TP={hbt['tp']} FP={hbt['fp']} FN={hbt['fn']} MERGED={hb_merged} FRAG={hb_frag}")
    print()

    results = {}
    for depth in DEPTH_VALUES:
        print(f"--- depth={depth} ---", flush=True)
        cal_r = score_corpus(cal_data, depth_min=depth, apply_apex=True)
        hol_r = score_corpus(hol_data, depth_min=depth, apply_apex=True)
        cal_frag_brk = count_over_splits(cal_base["per_video"], cal_r["per_video"])
        hol_frag_brk = count_over_splits(hol_base["per_video"], hol_r["per_video"])
        ct = cal_r["totals"]; ht = hol_r["totals"]
        cm = cal_r["topology"].get("MERGED", 0); hm = hol_r["topology"].get("MERGED", 0)
        cf = cal_r["topology"].get("FRAGMENTED", 0); hf = hol_r["topology"].get("FRAGMENTED", 0)
        print(f"  Cal: TP={ct['tp']:>4} ({ct['tp']-cbt['tp']:+3})  FP={ct['fp']:>3} ({ct['fp']-cbt['fp']:+3})  "
              f"FN={ct['fn']:>4} ({ct['fn']-cbt['fn']:+3})  splits={ct['splits_made']}  "
              f"MERGED={cm} ({cm-cb_merged:+d})  FRAG={cf} ({cf-cb_frag:+d})  over_splits={cal_frag_brk['over_splits']}")
        print(f"  Hol: TP={ht['tp']:>4} ({ht['tp']-hbt['tp']:+3})  FP={ht['fp']:>3} ({ht['fp']-hbt['fp']:+3})  "
              f"FN={ht['fn']:>4} ({ht['fn']-hbt['fn']:+3})  splits={ht['splits_made']}  "
              f"MERGED={hm} ({hm-hb_merged:+d})  FRAG={hf} ({hf-hb_frag:+d})  over_splits={hol_frag_brk['over_splits']}")
        print()
        results[str(depth)] = {
            "depth_min": depth,
            "calibration": {"totals": ct, "topology": cal_r["topology"],
                             "start_delta_abs_median": cal_r["start_delta_abs_median"],
                             "frag_breakdown": cal_frag_brk},
            "holdout": {"totals": ht, "topology": hol_r["topology"],
                         "start_delta_abs_median": hol_r["start_delta_abs_median"],
                         "frag_breakdown": hol_frag_brk},
        }

    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "fixed_prominence": PROMINENCE,
        "fixed_peak2_rel_max": PEAK2_REL_MAX,
        "baseline": {
            "calibration": {"totals": cbt, "topology": cal_base["topology"],
                             "start_delta_abs_median": cal_base["start_delta_abs_median"]},
            "holdout": {"totals": hbt, "topology": hol_base["topology"],
                         "start_delta_abs_median": hol_base["start_delta_abs_median"]},
        },
        "configs": results,
    }, indent=2, default=int), encoding="utf-8")

    print("=" * 150)
    print(f"SUMMARY: depth sweep at prominence={PROMINENCE}, peak2_rel={PEAK2_REL_MAX} (LIVE GT)")
    print("=" * 150)
    print(f"{'depth':>6}  {'CAL TP':>7} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>5} {'dFN':>4}  "
          f"{'MERGED':>7} {'FRAG':>5} {'overSplit':>9}  | "
          f"{'HOL TP':>7} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  "
          f"{'MERGED':>7} {'FRAG':>5} {'overSplit':>9}")
    print("-" * 150)
    print(f"{'BASE':>6}  {cbt['tp']:>7} {'':>4} {cbt['fp']:>4} {'':>4} {cbt['fn']:>5} {'':>4}  "
          f"{cb_merged:>7} {cb_frag:>5} {'':>9}  | "
          f"{hbt['tp']:>7} {'':>4} {hbt['fp']:>4} {'':>4} {hbt['fn']:>4} {'':>4}  "
          f"{hb_merged:>7} {hb_frag:>5} {'':>9}")
    for key, r in results.items():
        ct = r["calibration"]["totals"]; ht = r["holdout"]["totals"]
        cm = r["calibration"]["topology"].get("MERGED", 0)
        hm = r["holdout"]["topology"].get("MERGED", 0)
        cf = r["calibration"]["topology"].get("FRAGMENTED", 0)
        hf = r["holdout"]["topology"].get("FRAGMENTED", 0)
        co = r["calibration"]["frag_breakdown"]["over_splits"]
        ho = r["holdout"]["frag_breakdown"]["over_splits"]
        print(f"{key:>6}  {ct['tp']:>7} {ct['tp']-cbt['tp']:>+4} {ct['fp']:>4} {ct['fp']-cbt['fp']:>+4} "
              f"{ct['fn']:>5} {ct['fn']-cbt['fn']:>+4}  {cm:>7} {cf:>5} {co:>9}  | "
              f"{ht['tp']:>7} {ht['tp']-hbt['tp']:>+4} {ht['fp']:>4} {ht['fp']-hbt['fp']:>+4} "
              f"{ht['fn']:>4} {ht['fn']-hbt['fn']:>+4}  {hm:>7} {hf:>5} {ho:>9}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
