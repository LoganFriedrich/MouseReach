"""
Postprocess experiment: leading-frame paw-likelihood trim, threshold sweep.

Targets TOLERANCE_ERROR(start_early) and FALSE_POSITIVE topology classes
by trimming consecutive leading frames where paw_mean_lk < T from each
emitted algo reach. Reaches reduced below MIN_SPAN are dropped entirely.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking: baseline = v8.0.1_model_3_1_baseline_loocv
   (TP=2068 / FP=309 / FN=307 against parquet GT). All accepted
   improvements (BSW b=1/w=0.8 + mg=0) are baked into the model bundle
   that produced this baseline. The trim is a pure POSTPROCESS layered
   on top.

2. Existing-code-modification: NO. The trim operates on the
   loocv_aggregate.json output. v8 module code is untouched.

3. Unverified hypotheses:
   - Diagnostic showed TOLERANCE_ERROR(start_early) median leading_lk = 0.32
     and FALSE_POSITIVE median leading_lk = 0.45, vs TP_clean = 0.90.
     Hypothesis: a threshold in [0.5, 0.8] separates the classes well.
   - Trimming leading frames preserves TPs because TP leading_lk is
     usually well above any threshold in this range.
   - Edge cases: TPs whose actual reach onset has briefly-low lk could
     be hurt. Severity unknown; the sweep will surface this.

4. FN-direction-reporting: per [[pair_legacy_with_topology]], reports
   topology counts (TP / TOLERANCE_ERROR / MERGED / FRAGMENTED /
   FALSE_POSITIVE / FALSE_NEGATIVE / COMPLEX) alongside legacy
   TP/FP/FN. Lead with delta FALSE_NEGATIVE direction.

5. Framework: outputs to
   Improvement_Snapshots/reach_detection/v8.0.1_dev_leading_trim_postprocess_sweep/
   with one sub-dir per threshold.

6. Branch: feature/v8-leading-trim-postprocess
   Tag: v8-pre-leading-trim-2026-05-21

7. Decision rule (per threshold, compared to baseline):
   ACCEPT if all of:
     - TP rises OR stays equal
     - TOLERANCE_ERROR(start_early) drops materially (>= 30%)
     - FALSE_POSITIVE drops or holds
     - FALSE_NEGATIVE does not rise materially (<= +5 from baseline 30)
     - start_delta abs_median holds at 0
   Best threshold = highest TP at lowest TOLERANCE_ERROR + FALSE_POSITIVE
   while satisfying the FN-not-rising constraint.

   If no threshold accepts, REJECT and move to a different mechanism.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, evaluate_reaches, summarize_results,
)


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
    r"\reach_detection\v8.0.1_dev_leading_trim_postprocess_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAND_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
MIN_SPAN = 3

THRESHOLDS = [0.50, 0.60, 0.70]
MAX_TRIM_VALUES = [1, 2, 3, 4, 5]  # K = bounded trim distance (max frames trimmed from leading)


def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def classify_topology_components(algos, gts):
    """Return list of dicts {topology, sub, n_algo, n_gt} per component."""
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
            span_tol = max(0.5 * gt_span, 5)
            if abs(start_delta) <= 2 and abs(span_delta) <= span_tol:
                comps.append({"topology": "TP", "sub": None,
                              "n_algo": 1, "n_gt": 1})
            else:
                if start_delta < -2:
                    sub = "start_early"
                elif start_delta > 2:
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


def trim_leading(algo_reach, lk_arr, threshold, min_span=MIN_SPAN, max_trim=None):
    """Walk inward from start, trim consecutive frames where lk < threshold.
    If max_trim is set, stop after trimming that many frames.
    Returns (new_start, new_end) or None if trimmed below min_span.
    """
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


def topology_counts(comps):
    counts = defaultdict(int)
    sub_counts = defaultdict(int)
    for c in comps:
        counts[c["topology"]] += 1
        if c["sub"]:
            sub_counts[f"{c['topology']}({c['sub']})"] += 1
    return dict(counts), dict(sub_counts)


def evaluate_video(vid, algos, gt_reaches, lk_arr, threshold, do_trim, max_trim=None):
    """Apply leading-trim (if do_trim) then evaluate against GT.
    Returns: (summary, topology_counts_dict, topology_sub_counts_dict).
    """
    if do_trim:
        trimmed = []
        for a_s, a_e in algos:
            r = trim_leading((a_s, a_e), lk_arr, threshold, max_trim=max_trim)
            if r is not None:
                trimmed.append(r)
        algos = trimmed
    algo_reaches = [
        AlgoReach(start_frame=s, end_frame=e, video_id=vid, index=i)
        for i, (s, e) in enumerate(algos)
    ]
    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=vid)
    summary = summarize_results(results)
    topo_comps = classify_topology_components(algos, [(g.start_frame, g.end_frame) for g in gt_reaches])
    tc, tsc = topology_counts(topo_comps)
    return summary, tc, tsc, algos


def main():
    print("=" * 70)
    print("POSTPROCESS LEADING-TRIM THRESHOLD SWEEP")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Min span after trim: {MIN_SPAN}")
    print("=" * 70)
    print()

    print("Loading baseline LOOCV...", flush=True)
    loocv = json.loads(BASELINE_LOOCV_PATH.read_text(encoding="utf-8"))
    raw = loocv["raw_results"]

    # Reconstruct algo and GT spans per video from raw_results
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

    print(f"  videos: {len(algos_by_video)}")
    print(f"  total algo reaches: {sum(len(s) for s in algos_by_video.values())}")
    print(f"  total GT reaches:   {sum(len(s) for s in gts_by_video.values())}")
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
    print(f"  videos with lk arrays: {len(lk_by_vid)}")
    print()

    # Build GT reach objects per video (for evaluator). Cover videos that
    # have algo reaches but zero GTs by defaulting to an empty list.
    all_videos = set(algos_by_video.keys()) | set(gts_by_video.keys())
    gt_reach_objs_by_vid = {}
    for vid in all_videos:
        gts = sorted(gts_by_video.get(vid, set()))
        gt_reach_objs_by_vid[vid] = [
            GTReach(start_frame=s, end_frame=e, video_id=vid, index=i)
            for i, (s, e) in enumerate(gts)
        ]

    # ===== Baseline pass (no trim) for self-consistency check =====
    print("Re-scoring baseline (no trim) for self-consistency...", flush=True)
    base_tp = base_fp = base_fn = 0
    base_start_deltas = []
    base_topo = defaultdict(int)
    base_topo_sub = defaultdict(int)
    for vid in sorted(algos_by_video.keys()):
        algos = sorted(algos_by_video[vid])
        gt_objs = gt_reach_objs_by_vid[vid]
        summary, tc, tsc, _ = evaluate_video(vid, algos, gt_objs, lk_by_vid.get(vid), 0.0, do_trim=False)
        base_tp += summary["n_tp"]
        base_fp += summary["n_fp"]
        base_fn += summary["n_fn"]
        for t, c in tc.items():
            base_topo[t] += c
        for t, c in tsc.items():
            base_topo_sub[t] += c
    print(f"  Re-scored baseline: TP={base_tp}, FP={base_fp}, FN={base_fn}")
    print(f"  Original LOOCV:     TP={loocv['summary']['n_tp']}, FP={loocv['summary']['n_fp']}, FN={loocv['summary']['n_fn']}")
    print(f"  Baseline topology: {dict(base_topo)}")
    print()

    # ===== Sweep (T x K grid) =====
    sweep_results = []
    for K in MAX_TRIM_VALUES:
        for T in THRESHOLDS:
            print(f"--- T = {T:.2f}, K (max_trim) = {K} ---", flush=True)
            sum_tp = sum_fp = sum_fn = 0
            topo = defaultdict(int)
            topo_sub = defaultdict(int)
            algos_remaining = 0
            algos_dropped = 0
            algos_trimmed = 0
            for vid in sorted(algos_by_video.keys()):
                algos = sorted(algos_by_video[vid])
                gt_objs = gt_reach_objs_by_vid[vid]
                lk_arr = lk_by_vid.get(vid)
                if lk_arr is None:
                    continue
                summary, tc, tsc, new_algos = evaluate_video(
                    vid, algos, gt_objs, lk_arr, T, do_trim=True, max_trim=K)
                sum_tp += summary["n_tp"]
                sum_fp += summary["n_fp"]
                sum_fn += summary["n_fn"]
                for t, c in tc.items():
                    topo[t] += c
                for t, c in tsc.items():
                    topo_sub[t] += c
                algos_remaining += len(new_algos)
                algos_dropped += len(algos) - len(new_algos)
                # Count algos whose start frame actually changed
                new_starts = {(s, e): s for s, e in new_algos}
                for orig_s, orig_e in algos:
                    r = trim_leading((orig_s, orig_e), lk_arr, T, max_trim=K)
                    if r is not None and r[0] != orig_s:
                        algos_trimmed += 1

            delta_tp = sum_tp - base_tp
            delta_fp = sum_fp - base_fp
            delta_fn = sum_fn - base_fn

            print(f"  Legacy: TP={sum_tp} ({delta_tp:+d}) FP={sum_fp} ({delta_fp:+d}) FN={sum_fn} ({delta_fn:+d})")
            print(f"  Algos: remaining={algos_remaining}, trimmed={algos_trimmed}, dropped={algos_dropped}")
            te_subs = {k: v for k, v in topo_sub.items() if k.startswith("TOLERANCE_ERROR(")}
            print(f"  Topology: TP={topo.get('TP',0)}  "
                  f"TolErr={topo.get('TOLERANCE_ERROR',0)} "
                  f"(early={te_subs.get('TOLERANCE_ERROR(start_early)',0)} "
                  f"late={te_subs.get('TOLERANCE_ERROR(start_late)',0)})  "
                  f"FP={topo.get('FALSE_POSITIVE',0)} "
                  f"FN={topo.get('FALSE_NEGATIVE',0)} "
                  f"MERGED={topo.get('MERGED',0)} "
                  f"FRAG={topo.get('FRAGMENTED',0)}")
            print()

            sweep_results.append({
                "threshold": T, "max_trim": K,
                "tp": sum_tp, "fp": sum_fp, "fn": sum_fn,
                "delta_tp": delta_tp, "delta_fp": delta_fp, "delta_fn": delta_fn,
                "algos_remaining": algos_remaining,
                "algos_trimmed": algos_trimmed,
                "algos_dropped": algos_dropped,
                "topology": dict(topo),
                "topology_sub": dict(topo_sub),
            })

    # ===== Save outputs =====
    out_json = OUT_DIR / "sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline": {
            "tp": base_tp, "fp": base_fp, "fn": base_fn,
            "topology": dict(base_topo),
            "topology_sub": dict(base_topo_sub),
        },
        "sweep": sweep_results,
        "thresholds": THRESHOLDS,
        "min_span": MIN_SPAN,
    }, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")

    # ===== Summary table =====
    print()
    print("=" * 105)
    print("SWEEP SUMMARY (T x K grid)")
    print("=" * 105)
    print(f"{'T':>5} {'K':>3} {'TP':>5} {'dTP':>5} {'FP':>5} {'dFP':>5} {'FN':>5} {'dFN':>5}  "
          f"{'TolErr':>7} {'StartE':>7} {'StartL':>7} {'Phantom':>8} {'Merged':>7} {'FalsN':>7}")
    print(f"{'base':>5} {'-':>3} {base_tp:>5} {'':>5} {base_fp:>5} {'':>5} {base_fn:>5} {'':>5}  "
          f"{base_topo.get('TOLERANCE_ERROR',0):>7} "
          f"{base_topo_sub.get('TOLERANCE_ERROR(start_early)',0):>7} "
          f"{base_topo_sub.get('TOLERANCE_ERROR(start_late)',0):>7} "
          f"{base_topo.get('FALSE_POSITIVE',0):>8} "
          f"{base_topo.get('MERGED',0):>7} "
          f"{base_topo.get('FALSE_NEGATIVE',0):>7}")
    print("-" * 105)
    for s in sweep_results:
        print(f"{s['threshold']:>5.2f} {s['max_trim']:>3} "
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
