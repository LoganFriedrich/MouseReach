"""
Leading-trim sweep with sustained-low-lk requirement (option 2) combined
with bounded trim (K, option 1).

Trim logic:
  At each candidate frame F, only trim if frames [F, F+1, ..., F+N-1] are
  ALL below the threshold T. This prevents trimming isolated low-lk frames
  where the very next frame is confident. Stop after trimming K total.

Parameters:
  T = paw_mean_lk threshold for "low-lk"
  K = max total frames to trim (bounded trim from option 1)
  N = sustain length -- how many consecutive frames must be low to trim

N=1 reduces to the un-sustained version (option 1 alone).

Matcher tolerance held at +/-2 (Cardinal Rule). Decision rule and accept
criteria match the prior sweep.
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
    r"\reach_detection\v8.0.1_dev_leading_trim_sustained_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
MIN_SPAN = 3

THRESHOLDS = [0.50, 0.60, 0.70]
MAX_TRIM_VALUES = [None]  # unbounded -- sustain check is the only gate
SUSTAIN_VALUES = [2, 3, 4, 5]
TOLERANCE = 2  # held strict per Cardinal Rule
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


def trim_leading_sustained(algo_reach, lk_arr, threshold, sustain_n,
                           min_span=MIN_SPAN, max_trim=None):
    """Trim leading frames where the next sustain_n frames are ALL below threshold.
    Cap at max_trim total trimmed frames. Drop reach if remaining span < min_span.
    """
    s, e = algo_reach
    new_s = s
    trimmed = 0
    while new_s <= e:
        if max_trim is not None and trimmed >= max_trim:
            break
        # Check the next sustain_n frames starting at new_s
        window_end = new_s + sustain_n
        if window_end > len(lk_arr) or window_end > e + 1:
            # not enough frames left to satisfy sustain check
            break
        window = lk_arr[new_s:window_end]
        if np.any(np.isnan(window)):
            break
        if np.any(window >= threshold):
            break  # at least one frame in the window is confident; don't trim
        # All frames in window are below threshold; trim ONE frame (the leading)
        new_s += 1
        trimmed += 1
    new_span = e - new_s + 1
    if new_span < min_span:
        return None
    return (new_s, e)


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts, start_tol=TOLERANCE, span_tol_frac=SPAN_TOL_FRAC,
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


def classify_topology(algos, gts, start_tol=TOLERANCE,
                      span_tol_frac=SPAN_TOL_FRAC, span_tol_min=SPAN_TOL_MIN):
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
    print("LEADING-TRIM SUSTAINED-LOW-LK SWEEP")
    print(f"  T = {THRESHOLDS}")
    print(f"  K (max trim) = {MAX_TRIM_VALUES}")
    print(f"  N (sustain) = {SUSTAIN_VALUES}  (N=1 means no sustain check)")
    print(f"  Tolerance = +/-{TOLERANCE} (held strict)")
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

    all_videos = sorted(set(algos_by_video.keys()) | set(gts_by_video.keys()))

    def score_config(T, K, N, do_trim):
        total_tp = total_fp = total_fn = 0
        topo_counts = defaultdict(int)
        topo_sub_counts = defaultdict(int)
        algos_dropped = 0
        algos_trimmed_any = 0
        for vid in all_videos:
            algos = sorted(algos_by_video.get(vid, set()))
            gts = sorted(gts_by_video.get(vid, set()))
            lk_arr = lk_by_vid.get(vid)
            n_orig = len(algos)
            if do_trim and lk_arr is not None:
                trimmed = []
                for a_s, a_e in algos:
                    r = trim_leading_sustained((a_s, a_e), lk_arr, T, N, max_trim=K)
                    if r is not None:
                        trimmed.append(r)
                        if r[0] != a_s:
                            algos_trimmed_any += 1
                    else:
                        algos_dropped += 1
                algos = trimmed
            pairs, fps, fns = greedy_match(algos, gts, start_tol=TOLERANCE)
            total_tp += len(pairs)
            total_fp += len(fps)
            total_fn += len(fns)
            topo = classify_topology(algos, gts, start_tol=TOLERANCE)
            for c in topo:
                topo_counts[c["topology"]] += 1
                if c["sub"]:
                    topo_sub_counts[f"{c['topology']}({c['sub']})"] += 1
        return (total_tp, total_fp, total_fn,
                dict(topo_counts), dict(topo_sub_counts),
                algos_trimmed_any, algos_dropped)

    # Baseline (no trim) for reference
    base_tp, base_fp, base_fn, base_topo, base_tsub, _, _ = score_config(0.0, 0, 1, do_trim=False)
    print(f"BASELINE (no trim): TP={base_tp} FP={base_fp} FN={base_fn}")
    print(f"  Topology: TolErr={base_topo.get('TOLERANCE_ERROR',0)} "
          f"(early={base_tsub.get('TOLERANCE_ERROR(start_early)',0)}, "
          f"late={base_tsub.get('TOLERANCE_ERROR(start_late)',0)})  "
          f"Phantom={base_topo.get('FALSE_POSITIVE',0)} "
          f"FalseN={base_topo.get('FALSE_NEGATIVE',0)} "
          f"MERGED={base_topo.get('MERGED',0)}")
    print()

    # Sweep
    sweep_results = []
    for N in SUSTAIN_VALUES:
        for K in MAX_TRIM_VALUES:
            for T in THRESHOLDS:
                tp, fp, fn, topo, tsub, trimmed, dropped = score_config(T, K, N, do_trim=True)
                d_tp = tp - base_tp
                d_fp = fp - base_fp
                d_fn = fn - base_fn
                sweep_results.append({
                    "T": T, "K": K, "N": N,
                    "tp": tp, "fp": fp, "fn": fn,
                    "delta_tp": d_tp, "delta_fp": d_fp, "delta_fn": d_fn,
                    "algos_trimmed": trimmed, "algos_dropped": dropped,
                    "topology": topo, "topology_sub": tsub,
                })

    # Save
    out_json = OUT_DIR / "sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline": {"tp": base_tp, "fp": base_fp, "fn": base_fn,
                     "topology": base_topo, "topology_sub": base_tsub},
        "sweep": sweep_results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print()

    # Per-N summary tables
    for N in SUSTAIN_VALUES:
        print("=" * 110)
        if N == 1:
            print(f"SWEEP at N={N} (NO sustain check, equivalent to bounded-K only)")
        else:
            print(f"SWEEP at N={N} (require {N} consecutive low-lk frames to trim)")
        print("=" * 110)
        print(f"{'T':>5} {'TP':>5} {'dTP':>5} {'FP':>5} {'dFP':>5} {'FN':>5} {'dFN':>5}  "
              f"{'TolErr':>7} {'StartE':>7} {'StartL':>7} {'Phantom':>8} {'Merged':>7} {'FalsN':>7}  "
              f"{'trim':>5} {'drop':>5}")
        print(f"{'base':>5} {base_tp:>5} {'':>5} {base_fp:>5} {'':>5} {base_fn:>5} {'':>5}  "
              f"{base_topo.get('TOLERANCE_ERROR',0):>7} "
              f"{base_tsub.get('TOLERANCE_ERROR(start_early)',0):>7} "
              f"{base_tsub.get('TOLERANCE_ERROR(start_late)',0):>7} "
              f"{base_topo.get('FALSE_POSITIVE',0):>8} "
              f"{base_topo.get('MERGED',0):>7} "
              f"{base_topo.get('FALSE_NEGATIVE',0):>7}")
        print("-" * 110)
        for s in sweep_results:
            if s["N"] != N: continue
            print(f"{s['T']:>5.2f} "
                  f"{s['tp']:>5} {s['delta_tp']:>+5} "
                  f"{s['fp']:>5} {s['delta_fp']:>+5} "
                  f"{s['fn']:>5} {s['delta_fn']:>+5}  "
                  f"{s['topology'].get('TOLERANCE_ERROR',0):>7} "
                  f"{s['topology_sub'].get('TOLERANCE_ERROR(start_early)',0):>7} "
                  f"{s['topology_sub'].get('TOLERANCE_ERROR(start_late)',0):>7} "
                  f"{s['topology'].get('FALSE_POSITIVE',0):>8} "
                  f"{s['topology'].get('MERGED',0):>7} "
                  f"{s['topology'].get('FALSE_NEGATIVE',0):>7}  "
                  f"{s['algos_trimmed']:>5} {s['algos_dropped']:>5}")
        print()


if __name__ == "__main__":
    main()
