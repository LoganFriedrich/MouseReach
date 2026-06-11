"""
Leading-trim sweep extended with configurable matcher tolerance.

Sweeps over (T, K, tolerance) grid:
  T   = paw_mean_lk threshold for trimming a leading frame
  K   = max number of leading frames to trim per algo reach
  tol = matcher tolerance for start_delta (and 0.5*gt_span or 5 for span)

Replaces v8.eval.evaluate_reaches with an inline greedy nearest-start matcher
parameterized by tol. Re-computes both topology counts and legacy TP/FP/FN
from the same matched output.

Per Logan 2026-05-21: "if everything fails we should consider messing with
the tolerance of the matcher cuz I think its a little too strict right now."
This explores whether relaxed tolerance changes the picture for the bounded
trim postprocess.

Important caveat (per CARDINAL_RULE_NUANCE_2026-05-18.md): relaxing
tolerance accepts more events but those events have higher kinematic drift.
A ±3 or ±4 tolerance is defensible for Class A (apex-anchored) features but
introduces meaningful drift for Class B and especially Class C features.
This sweep doesn't address that; it just measures how the TP/FP/FN counts
shift.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PARQUET_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
BASELINE_LOOCV_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_dev_leading_trim_tolerance_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
MIN_SPAN = 3

THRESHOLDS = [0.50, 0.60, 0.70]
MAX_TRIM_VALUES = [1, 2, 3, 4]
TOLERANCES = [2, 3, 4]  # start tolerance, also affects span tolerance via max(0.5*gt_span, span_min)
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5  # absolute floor for span tolerance


def trim_leading(algo_reach, lk_arr, threshold, min_span=MIN_SPAN, max_trim=None):
    s, e = algo_reach
    new_s = s
    trimmed = 0
    while new_s <= e:
        if new_s >= len(lk_arr):
            break
        if max_trim is not None and trimmed >= max_trim:
            break
        lk = lk_arr[new_s]
        if np.isnan(lk) or lk >= threshold:
            break
        new_s += 1
        trimmed += 1
    new_span = e - new_s + 1
    if new_span < min_span:
        return None
    return (new_s, e)


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts, start_tol, span_tol_frac=SPAN_TOL_FRAC,
                 span_tol_min=SPAN_TOL_MIN):
    """Greedy nearest-start 1:1 matcher with configurable tolerance.
    Returns dict with:
      tp_pairs: list of (ai, gi)
      fp_indices: list of unmatched algo indices
      fn_indices: list of unmatched gt indices
      tp_start_deltas: list of start_delta per tp pair (algo - gt)
      tp_span_deltas: list of span_delta per tp pair (algo_span - gt_span)
    """
    candidates = []  # (|start_delta|, ai, gi, start_delta, span_delta)
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(span_tol_frac * gt_span, span_tol_min)
            if abs(start_delta) <= start_tol and abs(span_delta) <= span_tol:
                candidates.append((abs(start_delta), ai, gi, start_delta, span_delta))
    candidates.sort()

    used_algos, used_gts = set(), set()
    tp_pairs = []
    tp_start_deltas = []
    tp_span_deltas = []
    for _, ai, gi, sd, spd in candidates:
        if ai in used_algos or gi in used_gts:
            continue
        used_algos.add(ai)
        used_gts.add(gi)
        tp_pairs.append((ai, gi))
        tp_start_deltas.append(sd)
        tp_span_deltas.append(spd)

    fp_indices = [ai for ai in range(len(algos)) if ai not in used_algos]
    fn_indices = [gi for gi in range(len(gts)) if gi not in used_gts]
    return {
        "tp_pairs": tp_pairs,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "tp_start_deltas": tp_start_deltas,
        "tp_span_deltas": tp_span_deltas,
    }


def classify_topology(algos, gts, start_tol, span_tol_frac=SPAN_TOL_FRAC,
                      span_tol_min=SPAN_TOL_MIN):
    """Connected-components topology with configurable tolerance."""
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
            comps.append({"topology": "FALSE_POSITIVE", "sub": None,
                          "n_algo": 1, "n_gt": 0})
            visited_a.add(i)
            continue
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in:
                    continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]:
                    queue.append(("g", gj))
            else:
                if idx in gt_in:
                    continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]:
                    queue.append(("a", ai))
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
            if abs(start_delta) <= start_tol and abs(span_delta) <= span_tol:
                comps.append({"topology": "TP", "sub": None,
                              "n_algo": 1, "n_gt": 1})
            else:
                if start_delta < -start_tol:
                    sub = "start_early"
                elif start_delta > start_tol:
                    sub = "start_late"
                elif span_delta > span_tol:
                    sub = "span_over"
                elif span_delta < -span_tol:
                    sub = "span_short"
                else:
                    sub = "unclassified"
                comps.append({"topology": "TOLERANCE_ERROR", "sub": sub,
                              "n_algo": 1, "n_gt": 1})
        elif na == 1 and ng >= 2:
            comps.append({"topology": "MERGED", "sub": f"{ng}_gt",
                          "n_algo": 1, "n_gt": ng})
        elif na >= 2 and ng == 1:
            comps.append({"topology": "FRAGMENTED", "sub": f"{na}_algo",
                          "n_algo": na, "n_gt": 1})
        elif na >= 2 and ng >= 2:
            comps.append({"topology": "COMPLEX", "sub": f"{na}_algo_{ng}_gt",
                          "n_algo": na, "n_gt": ng})
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append({"topology": "FALSE_NEGATIVE", "sub": None,
                          "n_algo": 0, "n_gt": 1})
    return comps


def main():
    print("=" * 70)
    print("LEADING-TRIM SWEEP WITH MATCHER-TOLERANCE VARIATIONS")
    print(f"  T = {THRESHOLDS}")
    print(f"  K = {MAX_TRIM_VALUES}")
    print(f"  start_tol = {TOLERANCES}")
    print("=" * 70)
    print()

    print("Loading baseline LOOCV...", flush=True)
    loocv = json.loads(BASELINE_LOOCV_PATH.read_text(encoding="utf-8"))
    raw = loocv["raw_results"]
    algos_by_video = defaultdict(set)
    gts_by_video = defaultdict(set)
    for r in raw:
        vid = r["video_id"]
        if r["algo_start_frame"] >= 0:
            algos_by_video[vid].add((int(r["algo_start_frame"]),
                                     int(r["algo_end_frame"])))
        if r["gt_start_frame"] >= 0:
            gts_by_video[vid].add((int(r["gt_start_frame"]),
                                   int(r["gt_end_frame"])))
    print(f"  videos: {len(set(algos_by_video.keys()) | set(gts_by_video.keys()))}")
    print()

    print("Loading parquet for paw_mean_lk...", flush=True)
    df = pd.read_parquet(PARQUET_PATH, columns=["video_id", "frame"] + HAND_LK_COLS)
    df["paw_mean_lk"] = df[HAND_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    lk_by_vid = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        mx = int(grp_sorted["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        lk_by_vid[vid] = arr
    print()

    all_videos = sorted(set(algos_by_video.keys()) | set(gts_by_video.keys()))

    def score_config(T, K, tol, do_trim):
        """Apply trim (if do_trim) and score under tolerance `tol`."""
        total_tp = total_fp = total_fn = 0
        topo_counts = defaultdict(int)
        topo_sub_counts = defaultdict(int)
        for vid in all_videos:
            algos = sorted(algos_by_video.get(vid, set()))
            gts = sorted(gts_by_video.get(vid, set()))
            lk_arr = lk_by_vid.get(vid)
            if do_trim and lk_arr is not None:
                trimmed = []
                for a_s, a_e in algos:
                    r = trim_leading((a_s, a_e), lk_arr, T, max_trim=K)
                    if r is not None:
                        trimmed.append(r)
                algos = trimmed
            match = greedy_match(algos, gts, start_tol=tol)
            total_tp += len(match["tp_pairs"])
            total_fp += len(match["fp_indices"])
            total_fn += len(match["fn_indices"])
            topo = classify_topology(algos, gts, start_tol=tol)
            for c in topo:
                topo_counts[c["topology"]] += 1
                if c["sub"]:
                    topo_sub_counts[f"{c['topology']}({c['sub']})"] += 1
        return total_tp, total_fp, total_fn, dict(topo_counts), dict(topo_sub_counts)

    # Baseline (no trim) at each tolerance
    print("=" * 105)
    print("BASELINE (no trim) per matcher tolerance")
    print("=" * 105)
    base_per_tol = {}
    for tol in TOLERANCES:
        tp, fp, fn, topo, tsub = score_config(0.0, 0, tol, do_trim=False)
        base_per_tol[tol] = {"tp": tp, "fp": fp, "fn": fn, "topo": topo, "tsub": tsub}
        print(f"  tol=+/-{tol}: TP={tp}  FP={fp}  FN={fn}  "
              f"TolErr={topo.get('TOLERANCE_ERROR',0)} "
              f"(early={tsub.get('TOLERANCE_ERROR(start_early)',0)} "
              f"late={tsub.get('TOLERANCE_ERROR(start_late)',0)})  "
              f"FP_topo={topo.get('FALSE_POSITIVE',0)} FN_topo={topo.get('FALSE_NEGATIVE',0)} "
              f"MERGED={topo.get('MERGED',0)}")
    print()

    # Full sweep
    sweep = []
    for tol in TOLERANCES:
        for K in MAX_TRIM_VALUES:
            for T in THRESHOLDS:
                tp, fp, fn, topo, tsub = score_config(T, K, tol, do_trim=True)
                d_tp = tp - base_per_tol[tol]["tp"]
                d_fp = fp - base_per_tol[tol]["fp"]
                d_fn = fn - base_per_tol[tol]["fn"]
                sweep.append({
                    "T": T, "K": K, "tol": tol,
                    "tp": tp, "fp": fp, "fn": fn,
                    "delta_tp": d_tp, "delta_fp": d_fp, "delta_fn": d_fn,
                    "topology": topo, "topology_sub": tsub,
                })

    # Save
    out_json = OUT_DIR / "sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline_per_tolerance": base_per_tol,
        "sweep": sweep,
    }, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print()

    # ===== Summary tables per tolerance =====
    for tol in TOLERANCES:
        print("=" * 110)
        print(f"SWEEP at matcher tolerance +/-{tol}  (baseline TP={base_per_tol[tol]['tp']}/"
              f"FP={base_per_tol[tol]['fp']}/FN={base_per_tol[tol]['fn']})")
        print("=" * 110)
        print(f"{'T':>5} {'K':>3} {'TP':>5} {'dTP':>5} {'FP':>5} {'dFP':>5} {'FN':>5} {'dFN':>5}  "
              f"{'TolErr':>7} {'StartE':>7} {'StartL':>7} {'Phantom':>8} {'Merged':>7} {'FalsN':>7}")
        base = base_per_tol[tol]
        print(f"{'base':>5} {'-':>3} {base['tp']:>5} {'':>5} {base['fp']:>5} {'':>5} {base['fn']:>5} {'':>5}  "
              f"{base['topo'].get('TOLERANCE_ERROR',0):>7} "
              f"{base['tsub'].get('TOLERANCE_ERROR(start_early)',0):>7} "
              f"{base['tsub'].get('TOLERANCE_ERROR(start_late)',0):>7} "
              f"{base['topo'].get('FALSE_POSITIVE',0):>8} "
              f"{base['topo'].get('MERGED',0):>7} "
              f"{base['topo'].get('FALSE_NEGATIVE',0):>7}")
        print("-" * 110)
        for s in sweep:
            if s["tol"] != tol:
                continue
            print(f"{s['T']:>5.2f} {s['K']:>3} "
                  f"{s['tp']:>5} {s['delta_tp']:>+5} "
                  f"{s['fp']:>5} {s['delta_fp']:>+5} "
                  f"{s['fn']:>5} {s['delta_fn']:>+5}  "
                  f"{s['topology'].get('TOLERANCE_ERROR',0):>7} "
                  f"{s['topology_sub'].get('TOLERANCE_ERROR(start_early)',0):>7} "
                  f"{s['topology_sub'].get('TOLERANCE_ERROR(start_late)',0):>7} "
                  f"{s['topology'].get('FALSE_POSITIVE',0):>8} "
                  f"{s['topology'].get('MERGED',0):>7} "
                  f"{s['topology'].get('FALSE_NEGATIVE',0):>7}")
        print()


if __name__ == "__main__":
    main()
