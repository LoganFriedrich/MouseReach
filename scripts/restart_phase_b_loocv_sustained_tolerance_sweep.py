"""
Sustained-low-lk trim sweep across matcher tolerances.

Grid: T x N x tolerance.
  T = paw_mean_lk threshold
  N = sustain length (consecutive low-lk frames required before trimming)
  tolerance = matcher start_delta tolerance (+/-2, +/-3, +/-4)

K (max trim) is unbounded -- the sustain check is the only gate on trim depth.

Caveat: per CARDINAL_RULE_NUANCE_2026-05-18.md, relaxing matcher tolerance
admits events with measurable kinematic drift (Class B/C features). The
sustained-trim postprocess does NOT have that cost; it only shifts
boundaries within the configured tolerance window. Relaxed tolerance and
postprocess effects are independent.
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
    r"\reach_detection\v8.0.1_dev_sustained_tolerance_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
MIN_SPAN = 3

THRESHOLDS = [0.50, 0.60, 0.70]
SUSTAIN_VALUES = [2, 3, 4]
TOLERANCES = [2, 3, 4]
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def trim_leading_sustained(algo_reach, lk_arr, threshold, sustain_n, min_span=MIN_SPAN):
    s, e = algo_reach
    new_s = s
    while new_s <= e:
        window_end = new_s + sustain_n
        if window_end > len(lk_arr) or window_end > e + 1:
            break
        window = lk_arr[new_s:window_end]
        if np.any(np.isnan(window)):
            break
        if np.any(window >= threshold):
            break
        new_s += 1
    if e - new_s + 1 < min_span:
        return None
    return (new_s, e)


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts, start_tol, span_tol_frac=SPAN_TOL_FRAC,
                 span_tol_min=SPAN_TOL_MIN):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(span_tol_frac * gt_span, span_tol_min)
            if abs(start_delta) <= start_tol and abs(span_delta) <= span_tol:
                candidates.append((abs(start_delta), ai, gi))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []
    for _, ai, gi in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        pairs.append((ai, gi))
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns


def classify_topology(algos, gts, start_tol, span_tol_frac=SPAN_TOL_FRAC,
                      span_tol_min=SPAN_TOL_MIN):
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
            if abs(start_delta) <= start_tol and abs(span_delta) <= span_tol:
                comps.append({"topology": "TP", "sub": None})
            else:
                if start_delta < -start_tol: sub = "start_early"
                elif start_delta > start_tol: sub = "start_late"
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


def main():
    print("=" * 70)
    print("SUSTAINED-LOW-LK TRIM x TOLERANCE SWEEP")
    print(f"  T = {THRESHOLDS}")
    print(f"  N = {SUSTAIN_VALUES}")
    print(f"  tolerance = {TOLERANCES}")
    print("=" * 70)
    print()

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

    df = pd.read_parquet(PARQUET_PATH, columns=["video_id", "frame"] + HAND_LK_COLS)
    df["paw_mean_lk"] = df[HAND_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    lk_by_vid = {}
    for vid, grp in df.groupby("video_id", sort=False):
        grp_sorted = grp.sort_values("frame")
        mx = int(grp_sorted["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[grp_sorted["frame"].to_numpy()] = grp_sorted["paw_mean_lk"].to_numpy()
        lk_by_vid[vid] = arr

    all_videos = sorted(set(algos_by_video.keys()) | set(gts_by_video.keys()))

    def score_config(T, N, tol, do_trim):
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
                    r = trim_leading_sustained((a_s, a_e), lk_arr, T, N)
                    if r is not None:
                        trimmed.append(r)
                algos = trimmed
            pairs, fps, fns = greedy_match(algos, gts, start_tol=tol)
            total_tp += len(pairs)
            total_fp += len(fps)
            total_fn += len(fns)
            for c in classify_topology(algos, gts, start_tol=tol):
                topo_counts[c["topology"]] += 1
                if c["sub"]:
                    topo_sub_counts[f"{c['topology']}({c['sub']})"] += 1
        return (total_tp, total_fp, total_fn,
                dict(topo_counts), dict(topo_sub_counts))

    # Baselines per tolerance
    base_per_tol = {}
    for tol in TOLERANCES:
        tp, fp, fn, topo, tsub = score_config(0.0, 0, tol, do_trim=False)
        base_per_tol[tol] = {"tp": tp, "fp": fp, "fn": fn, "topo": topo, "tsub": tsub}

    print("BASELINES per matcher tolerance (no trim):")
    for tol in TOLERANCES:
        b = base_per_tol[tol]
        print(f"  +/-{tol}: TP={b['tp']} FP={b['fp']} FN={b['fn']}  "
              f"TolErr={b['topo'].get('TOLERANCE_ERROR',0)} "
              f"(early={b['tsub'].get('TOLERANCE_ERROR(start_early)',0)}, "
              f"late={b['tsub'].get('TOLERANCE_ERROR(start_late)',0)})  "
              f"Phantom={b['topo'].get('FALSE_POSITIVE',0)} "
              f"FalseN={b['topo'].get('FALSE_NEGATIVE',0)} "
              f"MERGED={b['topo'].get('MERGED',0)}")
    print()

    # Sweep
    sweep = []
    for tol in TOLERANCES:
        for N in SUSTAIN_VALUES:
            for T in THRESHOLDS:
                tp, fp, fn, topo, tsub = score_config(T, N, tol, do_trim=True)
                b = base_per_tol[tol]
                sweep.append({
                    "T": T, "N": N, "tol": tol,
                    "tp": tp, "fp": fp, "fn": fn,
                    "delta_tp": tp - b["tp"], "delta_fp": fp - b["fp"],
                    "delta_fn": fn - b["fn"],
                    "topology": topo, "topology_sub": tsub,
                })

    out_json = OUT_DIR / "sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline_per_tolerance": base_per_tol,
        "sweep": sweep,
    }, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print()

    # Per-tolerance tables
    for tol in TOLERANCES:
        print("=" * 110)
        b = base_per_tol[tol]
        print(f"TOLERANCE +/-{tol}  (baseline TP={b['tp']}/FP={b['fp']}/FN={b['fn']})")
        print("=" * 110)
        print(f"{'T':>5} {'N':>3} {'TP':>5} {'dTP':>5} {'FP':>5} {'dFP':>5} {'FN':>5} {'dFN':>5}  "
              f"{'TolErr':>7} {'StartE':>7} {'StartL':>7} {'Phantom':>8} {'Merged':>7} {'FalsN':>7}")
        print(f"{'base':>5} {'-':>3} {b['tp']:>5} {'':>5} {b['fp']:>5} {'':>5} {b['fn']:>5} {'':>5}  "
              f"{b['topo'].get('TOLERANCE_ERROR',0):>7} "
              f"{b['tsub'].get('TOLERANCE_ERROR(start_early)',0):>7} "
              f"{b['tsub'].get('TOLERANCE_ERROR(start_late)',0):>7} "
              f"{b['topo'].get('FALSE_POSITIVE',0):>8} "
              f"{b['topo'].get('MERGED',0):>7} "
              f"{b['topo'].get('FALSE_NEGATIVE',0):>7}")
        print("-" * 110)
        for s in sweep:
            if s["tol"] != tol: continue
            print(f"{s['T']:>5.2f} {s['N']:>3} "
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
