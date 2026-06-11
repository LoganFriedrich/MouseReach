"""
Phase B experiment: asymmetric matcher tolerance, late side K=5.

Change to the metric-rule: start_delta tolerance becomes [-2, +5] instead
of strict [-2, +2]. Span tolerance unchanged (max(0.5 * gt_span, 5)).

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-22):
   - Current production v8.0.2 = BSW b=1/w=0.8 (model bundle) + mg=0
     postprocess default + sustained-trim N=3, T=0.60 (shipped 2026-05-21).
   - This experiment changes the MATCHER, not the algo. v8.0.2 outputs
     are unchanged. Joins MIN_REPORTED_SPAN=4 and outside_gt_segmentation
     as a metric convention.
   - Comparison baseline:
     - Calibration LOOCV (v8.0.2 outputs, +/-2 tolerance):
         TP=2112  FP=217  FN=263
     - Holdout 19 videos (v8.0.2 outputs, +/-2 tolerance):
         TP=3628  FP=99   FN=123

2. Existing-code-modification check:
   - NO modifications to src/mousereach/reach/v8/* or to
     mousereach.improvement.reach_detection.metrics during the experiment.
   - The runner replicates the matcher inline with the new tolerance.
   - Production change (modifying metrics.match_reaches) only happens
     if/when the experiment is accepted and shipped.

3. Unverified hypotheses (acknowledged):
   - Phase A diagnostic showed the start_late population splits into:
     - Sub-pop A: start_delta 3-5, span_delta moderate, algo INSIDE gt
       (88% cal, 97% holdout) -- "truncated subset" of real reach
     - Sub-pop B: start_delta 10+, span_delta very negative, algo span
       << gt span -- "algo detected only the tail of long GT"
   - K=5 with span_tol kept admits sub-pop A. Hypothesis: sub-pop B
     fails span_tol naturally and stays as TOLERANCE_ERROR.
   - VERIFY in this experiment by checking that no FP/FN/MERGED/etc.
     counts change beyond TP and TOLERANCE_ERROR.
   - The change is symmetric across calibration and holdout: hypothesis
     is that ~56 cal + ~28 holdout events convert to TPs.

4. FN-direction-reporting:
   - Lead with delta FN vs v8.0.2 baseline.
   - Include topology counts (both classes affected: TP rises,
     TOLERANCE_ERROR drops). All other topology classes should be
     invariant -- this is the verification check.
   - ASCII output only.

5. Framework check:
   - Output to Improvement_Snapshots/reach_detection/v8.0.2_dev_asymmetric_tolerance_k5/
   - Calibration + holdout in subdirs.
   - JSON with topology breakdown.

6. Branch + tag:
   - Pre-experiment tag: v8-pre-asymmetric-tolerance-2026-05-22 (already set)
   - Branch: feature/v8-asymmetric-tolerance-k5 (already on it)

7. Decision rule:
   ACCEPT if all of:
     - TP rises on BOTH corpora
     - TOLERANCE_ERROR(start_late) drops by approximately the
       Phase A predicted amount (~56 cal, ~28 holdout)
     - FALSE_POSITIVE, FALSE_NEGATIVE, MERGED, FRAGMENTED, COMPLEX
       counts UNCHANGED (the change is matcher-only)
     - start_delta abs_median rises modestly (admitted events have
       start_delta 3-5, so median shifts; should not exceed +2)

   REJECT if:
     - Any topology class outside (TP, TOLERANCE_ERROR) changes
     - Phase A's "sub-pop B doesn't get admitted" assumption fails
       (TP rises by dramatically more than predicted)
     - start_delta mean shifts dramatically (>+2)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
)


CAL_LOOCV_JSON = Path(
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
    r"\generalization_test_2026-05-11\gt"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_asymmetric_tolerance_k5"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]

# Tolerance configurations under test
SYMMETRIC_TOL = 2          # baseline (strict +/-2)
ASYMMETRIC_TOL_NEG = 2     # late side restricts to algo can start at most 2 frames BEFORE gt
ASYMMETRIC_TOL_POS = 5     # algo can start up to 5 frames AFTER gt

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts, neg_tol, pos_tol, span_tol_frac=SPAN_TOL_FRAC,
                 span_tol_min=SPAN_TOL_MIN):
    """Greedy 1:1 nearest-start matcher with asymmetric tolerance.

    Matches algo to gt iff:
      -neg_tol <= start_delta <= pos_tol AND |span_delta| <= span_tol
    where start_delta = algo_start - gt_start.

    For symmetric matching, set neg_tol = pos_tol.
    """
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(span_tol_frac * gt_span, span_tol_min)
            if (-neg_tol <= start_delta <= pos_tol) and (abs(span_delta) <= span_tol):
                candidates.append((abs(start_delta), ai, gi, start_delta))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []
    tp_start_deltas = []
    for _, ai, gi, sd in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        pairs.append((ai, gi))
        tp_start_deltas.append(sd)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_start_deltas


def classify_topology(algos, gts, neg_tol, pos_tol,
                      span_tol_frac=SPAN_TOL_FRAC, span_tol_min=SPAN_TOL_MIN):
    """Connected-components topology with asymmetric tolerance."""
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
            comps.append({"topology": "FALSE_POSITIVE", "sub": None})
            visited_a.add(i)
            continue
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
        visited_a.update(algo_in)
        visited_g.update(gt_in)
        na, ng = len(algo_in), len(gt_in)
        if na == 1 and ng == 1:
            a_s, a_e = algos[next(iter(algo_in))]
            g_s, g_e = gts[next(iter(gt_in))]
            start_delta = a_s - g_s
            algo_span = a_e - a_s + 1
            gt_span = g_e - g_s + 1
            span_delta = algo_span - gt_span
            span_tol = max(span_tol_frac * gt_span, span_tol_min)
            if (-neg_tol <= start_delta <= pos_tol) and (abs(span_delta) <= span_tol):
                comps.append({"topology": "TP", "sub": None})
            else:
                if start_delta < -neg_tol: sub = "start_early"
                elif start_delta > pos_tol: sub = "start_late"
                elif span_delta > span_tol: sub = "span_over"
                elif span_delta < -span_tol: sub = "span_short"
                else: sub = "unclassified"
                comps.append({"topology": "TOLERANCE_ERROR", "sub": sub})
        elif na == 1 and ng >= 2:
            comps.append({"topology": "MERGED", "sub": f"{ng}_gt"})
        elif na >= 2 and ng == 1:
            comps.append({"topology": "FRAGMENTED", "sub": f"{na}_algo"})
        elif na >= 2 and ng >= 2:
            comps.append({"topology": "COMPLEX", "sub": f"{na}_algo_{ng}_gt"})
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append({"topology": "FALSE_NEGATIVE", "sub": None})
    return comps


def apply_v802_trim(algos, paw_lk):
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(spans, paw_lk,
                                         threshold=0.60, sustain_n=3, min_span=3)
    return [(r.start_frame, r.end_frame) for r in trimmed]


def load_holdout_algo(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [(int(r["start_frame"]), int(r["end_frame"]))
            for r in data.get("reaches", [])]


def load_holdout_gt(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    reaches_obj = data.get("reaches", {})
    rlist = reaches_obj.get("reaches", []) if isinstance(reaches_obj, dict) else []
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_dlc(path):
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def score_corpus(algos_by_video, gts_by_video, neg_tol, pos_tol):
    """Score corpus under given tolerance. Returns (tp, fp, fn, topo, tsub, tp_start_deltas)."""
    total_tp = total_fp = total_fn = 0
    topo = defaultdict(int)
    tsub = defaultdict(int)
    all_tp_sd = []
    for vid in sorted(set(algos_by_video.keys()) | set(gts_by_video.keys())):
        algos = sorted(algos_by_video.get(vid, set()))
        gts = sorted(gts_by_video.get(vid, set()))
        pairs, fps, fns, tp_sd = greedy_match(algos, gts, neg_tol, pos_tol)
        total_tp += len(pairs)
        total_fp += len(fps)
        total_fn += len(fns)
        all_tp_sd.extend(tp_sd)
        for c in classify_topology(algos, gts, neg_tol, pos_tol):
            topo[c["topology"]] += 1
            if c["sub"]:
                tsub[f"{c['topology']}({c['sub']})"] += 1
    return total_tp, total_fp, total_fn, dict(topo), dict(tsub), all_tp_sd


def report(label, tp, fp, fn, topo, tsub, tp_sd, baseline=None):
    tp_sd_abs = [abs(d) for d in tp_sd] if tp_sd else [0]
    abs_med = int(np.median(tp_sd_abs))
    med = int(np.median(tp_sd)) if tp_sd else 0
    print(f"  {label}")
    print(f"    Legacy: TP={tp}  FP={fp}  FN={fn}")
    if baseline:
        print(f"    Deltas: TP={tp-baseline['tp']:+d}  FP={fp-baseline['fp']:+d}  FN={fn-baseline['fn']:+d}")
    print(f"    start_delta on TPs: median={med}  abs_median={abs_med}")
    print(f"    Topology:")
    for k in ("TP", "TOLERANCE_ERROR", "MERGED", "FRAGMENTED",
              "FALSE_POSITIVE", "FALSE_NEGATIVE", "COMPLEX"):
        v = topo.get(k, 0)
        if baseline and k in baseline["topo"]:
            print(f"      {k:<22} {v:>5}   delta={v - baseline['topo'].get(k, 0):+d}")
        else:
            print(f"      {k:<22} {v:>5}")
    for sub in ("start_early", "start_late", "span_over", "span_short"):
        key = f"TOLERANCE_ERROR({sub})"
        v = tsub.get(key, 0)
        if baseline and "tsub" in baseline:
            b_v = baseline["tsub"].get(key, 0)
            print(f"      {key:<28} {v:>4}   delta={v - b_v:+d}")
        else:
            print(f"      {key:<28} {v:>4}")


def main():
    print("=" * 70)
    print("ASYMMETRIC TOLERANCE EXPERIMENT  K_pos=5, K_neg=2")
    print("=" * 70)
    print()

    # ===== Load calibration =====
    print("Loading calibration v8.0.1 LOOCV outputs + applying v8.0.2 trim...", flush=True)
    loocv = json.loads(CAL_LOOCV_JSON.read_text(encoding="utf-8"))
    raw = loocv["raw_results"]
    cal_algos_pre = defaultdict(set)
    cal_gts = defaultdict(set)
    for r in raw:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            cal_algos_pre[vid].add((int(r["algo_start_frame"]),
                                    int(r["algo_end_frame"])))
        if r["gt_start_frame"] >= 0:
            cal_gts[vid].add((int(r["gt_start_frame"]),
                              int(r["gt_end_frame"])))

    df = pd.read_parquet(CAL_PARQUET, columns=["video_id", "frame"] + PARQUET_LK_COLS)
    df["paw_mean_lk"] = df[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    cal_lk = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        mx = int(grp_sorted["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        cal_lk[vid] = arr

    cal_algos_post = defaultdict(set)
    for vid in sorted(cal_algos_pre.keys()):
        if vid not in cal_lk:
            cal_algos_post[vid] = cal_algos_pre[vid]
            continue
        trimmed = apply_v802_trim(sorted(cal_algos_pre[vid]), cal_lk[vid])
        cal_algos_post[vid] = set(trimmed)

    # ===== Load holdout =====
    print("Loading holdout outputs + DLC + GT + applying v8.0.2 trim...", flush=True)
    holdout_algos_post = defaultdict(set)
    holdout_gts = defaultdict(set)
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        gt_path = HOLDOUT_GT_DIR / f"{vid}_unified_ground_truth.json"
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}DLC_resnet50_MPSAOct27shuffle1_100000.h5"
        if not gt_path.exists() or not dlc_path.exists():
            continue
        algos_pre = load_holdout_algo(algo_path)
        gts = load_holdout_gt(gt_path)
        dlc = load_dlc(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        algos_post = apply_v802_trim(algos_pre, paw_lk)
        holdout_algos_post[vid] = set(algos_post)
        holdout_gts[vid] = set(gts)
    print()

    # ===== Score under both tolerances =====
    results = {}
    for corpus_name, algos, gts in (
        ("calibration", cal_algos_post, cal_gts),
        ("holdout", holdout_algos_post, holdout_gts),
    ):
        print("=" * 70)
        print(f"CORPUS: {corpus_name}")
        print("=" * 70)
        # Baseline: symmetric +/-2
        tp_b, fp_b, fn_b, topo_b, tsub_b, sd_b = score_corpus(
            algos, gts, neg_tol=SYMMETRIC_TOL, pos_tol=SYMMETRIC_TOL)
        baseline = {"tp": tp_b, "fp": fp_b, "fn": fn_b,
                    "topo": topo_b, "tsub": tsub_b}
        report("Baseline (+/-2 symmetric)", tp_b, fp_b, fn_b, topo_b, tsub_b, sd_b)
        print()
        # Experiment: asymmetric -2, +5
        tp_e, fp_e, fn_e, topo_e, tsub_e, sd_e = score_corpus(
            algos, gts, neg_tol=ASYMMETRIC_TOL_NEG, pos_tol=ASYMMETRIC_TOL_POS)
        report(f"Experiment ({ASYMMETRIC_TOL_NEG} neg / {ASYMMETRIC_TOL_POS} pos)",
               tp_e, fp_e, fn_e, topo_e, tsub_e, sd_e, baseline=baseline)
        print()
        results[corpus_name] = {
            "baseline_symmetric": {"tp": tp_b, "fp": fp_b, "fn": fn_b,
                                    "topology": topo_b, "topology_sub": tsub_b,
                                    "start_delta_median": int(np.median(sd_b)) if sd_b else 0},
            "asymmetric_-2_+5": {"tp": tp_e, "fp": fp_e, "fn": fn_e,
                                  "topology": topo_e, "topology_sub": tsub_e,
                                  "start_delta_median": int(np.median(sd_e)) if sd_e else 0},
            "deltas": {"tp": tp_e - tp_b, "fp": fp_e - fp_b, "fn": fn_e - fn_b},
        }

    # ===== Save =====
    out_json = OUT_DIR / "metrics" / "asymmetric_tolerance_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print()

    # ===== Decision rule eval =====
    print("=" * 70)
    print("DECISION RULE EVALUATION")
    print("=" * 70)
    invariants_ok = True
    for corpus_name in ("calibration", "holdout"):
        b = results[corpus_name]["baseline_symmetric"]["topology"]
        e = results[corpus_name]["asymmetric_-2_+5"]["topology"]
        for k in ("FALSE_POSITIVE", "FALSE_NEGATIVE", "MERGED", "FRAGMENTED", "COMPLEX"):
            if b.get(k, 0) != e.get(k, 0):
                print(f"  [WARN] {corpus_name}: {k} changed {b.get(k,0)} -> {e.get(k,0)} "
                      f"(should be invariant under tolerance change)")
                invariants_ok = False
    if invariants_ok:
        print("  Invariants OK: FP/FN/MERGED/FRAGMENTED/COMPLEX unchanged on both corpora.")
    print()
    print("  TP/TOLERANCE_ERROR(start_late) shifts:")
    for corpus_name in ("calibration", "holdout"):
        b_tp = results[corpus_name]["baseline_symmetric"]["topology"].get("TP", 0)
        e_tp = results[corpus_name]["asymmetric_-2_+5"]["topology"].get("TP", 0)
        b_se = results[corpus_name]["baseline_symmetric"]["topology_sub"].get("TOLERANCE_ERROR(start_late)", 0)
        e_se = results[corpus_name]["asymmetric_-2_+5"]["topology_sub"].get("TOLERANCE_ERROR(start_late)", 0)
        print(f"    {corpus_name}: TP {b_tp}->{e_tp} (+{e_tp-b_tp}), "
              f"start_late {b_se}->{e_se} ({e_se-b_se:+d})")
    print()


if __name__ == "__main__":
    main()
