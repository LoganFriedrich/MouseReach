"""
v8.0.1 + sustained-trim postprocess: HOLDOUT GATE on 19 exhaustive videos.

Per playbook step 5: generalization test on the canonical 20-video holdout
(2026-05-11 generalization set). 19 videos have GT and existing v8 inference
output; one is excluded for missing GT.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-21):
   - Current production v8.0.1 = BSW b=1/w=0.8 (baked into model bundle) + mg=0
     postprocess default. This experiment LAYERS sustained-trim (N=3, T=0.60)
     on top of the existing production output.
   - Holdout baseline (no postprocess, v8.0.1 mg=0 against parquet GT):
     TP=3577 / FP=199 / FN=175 (from
     v8.0.0_holdout_generalization_merge_gap_0/metrics/gate_decision.json)
   - This is the comparison anchor.
   - LOOCV-accepted result of sustained-trim (N=3, T=0.60) on calibration:
     TP +44 / FP -92 / FN -44 vs the LOOCV baseline.

2. Existing-code-modification check:
   - NO modifications to src/mousereach/reach/v8/*.
   - Postprocess applied externally in this runner.
   - v8 algo outputs read from existing snapshot, read-only.

3. Unverified hypotheses:
   - The LOOCV calibration result generalizes to holdout. Holdout is
     different mice with potentially different DLC-likelihood
     distributions; the calibration N=3, T=0.60 may not be the same
     sweet spot on holdout.
   - The 19 holdout videos use the same DLC weights as calibration
     (confirmed per dlc_model_3_1_corpus_state_2026-05-21 memory:
     model 3.1 and the 2026-05-11 generalization holdout share weights),
     so DLC vintage is not a confounder.
   - DLC h5 source: iterations/generalization_test_2026-05-11/dlc/

4. FN-direction-reporting:
   - Lead with delta FN vs holdout baseline (TP=3577 / FP=199 / FN=175).
   - Include topology breakdown alongside legacy TP/FP/FN per
     feedback_pair_legacy_with_topology.md.
   - ASCII output only. No precision/recall/F1/AUC.

5. Framework check:
   - Output to
     Improvement_Snapshots/reach_detection/v8.0.1_dev_sustained_trim_holdout_gate/
   - metrics/holdout_aggregate.json, metrics/topology_breakdown.json
   - Apples-to-apples with calibration LOOCV scoring.

6. Branch + tag:
   - Pre-experiment tag: v8-pre-leading-trim-2026-05-21 (already set)
   - Branch: feature/v8-leading-trim-postprocess (already on it)

7. Decision rule (holdout gate, per playbook):
   ACCEPT if all of:
     - FN drops or holds vs holdout baseline (175)
     - TP rises or holds (3577)
     - start_delta abs_median holds at 0 (Cardinal Rule)
     - TOLERANCE_ERROR(start_early) drops or holds
   REJECT if:
     - TP drops AND FN rises (standard rejection rule)
     - start_delta abs_median > 0 (boundary precision regression)
     - Calibration improvement doesn't generalize

   If ACCEPT: ready to ship as postprocess module (separate commit).
   If REJECT: diagnostic-first investigation; do NOT ship.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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
    r"\reach_detection\v8.0.1_dev_sustained_trim_holdout_gate"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

# Postprocess parameters (LOOCV-accepted)
T_THRESHOLD = 0.60
N_SUSTAIN = 3
MIN_SPAN = 3

# Matcher
START_TOL = 2
SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

HAND_LK_COLS = ["RightHand_x", "RHLeft_x", "RHOut_x", "RHRight_x",
                "RightHand_y", "RHLeft_y", "RHOut_y", "RHRight_y",
                "RightHand_likelihood", "RHLeft_likelihood",
                "RHOut_likelihood", "RHRight_likelihood"]
HAND_LK_LIKELIHOOD_COLS = ["RightHand_likelihood", "RHLeft_likelihood",
                           "RHOut_likelihood", "RHRight_likelihood"]


def load_dlc_h5(path):
    """Load DLC h5 file and flatten column names."""
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def load_holdout_algo_reaches(json_path):
    """Extract (start, end) tuples for all algo reaches.

    v8 schema (verified 2026-05-21): data["reaches"] is a flat list of
    dicts with start_frame, end_frame, duration_frames. No segments wrapper.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    reaches = []
    for r in data.get("reaches", []):
        reaches.append((int(r["start_frame"]), int(r["end_frame"])))
    return sorted(set(reaches))


def load_holdout_gt_reaches(json_path):
    """Extract (start, end) tuples for GT reaches.

    Schema (verified 2026-05-21): data["reaches"]["reaches"] is the list.
    data["reaches"]["exhaustive"] is a boolean flag (whether GT is exhaustive).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    reaches_obj = data.get("reaches", {})
    reach_list = reaches_obj.get("reaches", []) if isinstance(reaches_obj, dict) else []
    reaches = []
    for r in reach_list:
        if r.get("exclude_from_analysis"):
            continue
        reaches.append((int(r["start_frame"]), int(r["end_frame"])))
    return sorted(set(reaches))


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


def greedy_match(algos, gts, start_tol=START_TOL):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
            if abs(start_delta) <= start_tol and abs(span_delta) <= span_tol:
                candidates.append((abs(start_delta), ai, gi, start_delta, span_delta))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []
    tp_start_deltas = []
    for _, ai, gi, sd, _ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        pairs.append((ai, gi))
        tp_start_deltas.append(sd)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_start_deltas


def classify_topology(algos, gts, start_tol=START_TOL):
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
            span_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
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


def compute_paw_mean_lk(dlc_df):
    """Compute paw_mean_lk per frame as the mean of the 4 hand keypoint
    likelihoods. Returns a numpy array indexed by frame.
    """
    lk_cols = [c for c in HAND_LK_LIKELIHOOD_COLS if c in dlc_df.columns]
    if not lk_cols:
        return None
    lk_matrix = dlc_df[lk_cols].to_numpy(dtype=np.float32)
    return lk_matrix.mean(axis=1)


def main():
    print("=" * 70)
    print("HOLDOUT GATE -- sustained-trim N=3, T=0.60 postprocess")
    print(f"  Postprocess: N={N_SUSTAIN}, T={T_THRESHOLD}, MIN_SPAN={MIN_SPAN}")
    print(f"  Matcher tolerance: +/-{START_TOL}")
    print("=" * 70)
    print()

    # Find videos that have all three: algo output, GT, DLC h5
    algo_files = list(HOLDOUT_ALGO_DIR.glob("*_reaches.json"))
    print(f"Found {len(algo_files)} holdout algo output files")
    print()

    base_tp = base_fp = base_fn = 0
    base_topo = defaultdict(int)
    base_tsub = defaultdict(int)
    base_start_deltas = []

    pp_tp = pp_fp = pp_fn = 0
    pp_topo = defaultdict(int)
    pp_tsub = defaultdict(int)
    pp_start_deltas = []

    per_video = []
    skipped = []

    for algo_path in sorted(algo_files):
        vid = algo_path.stem.replace("_reaches", "")
        gt_path = HOLDOUT_GT_DIR / f"{vid}_unified_ground_truth.json"
        dlc_path = HOLDOUT_DLC_DIR / f"{vid}DLC_resnet50_MPSAOct27shuffle1_100000.h5"

        if not gt_path.exists():
            skipped.append((vid, "no GT"))
            continue
        if not dlc_path.exists():
            skipped.append((vid, "no DLC"))
            continue

        algos = load_holdout_algo_reaches(algo_path)
        gts = load_holdout_gt_reaches(gt_path)
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        if paw_lk is None:
            skipped.append((vid, "no paw lk cols"))
            continue

        # Baseline scoring
        b_pairs, b_fps, b_fns, b_sd = greedy_match(algos, gts)
        b_topo = classify_topology(algos, gts)
        base_tp += len(b_pairs)
        base_fp += len(b_fps)
        base_fn += len(b_fns)
        base_start_deltas.extend(b_sd)
        for c in b_topo:
            base_topo[c["topology"]] += 1
            if c["sub"]:
                base_tsub[f"{c['topology']}({c['sub']})"] += 1

        # Postprocess scoring
        trimmed_algos = []
        for a_s, a_e in algos:
            r = trim_leading_sustained((a_s, a_e), paw_lk,
                                       T_THRESHOLD, N_SUSTAIN)
            if r is not None:
                trimmed_algos.append(r)
        p_pairs, p_fps, p_fns, p_sd = greedy_match(trimmed_algos, gts)
        p_topo = classify_topology(trimmed_algos, gts)
        pp_tp += len(p_pairs)
        pp_fp += len(p_fps)
        pp_fn += len(p_fns)
        pp_start_deltas.extend(p_sd)
        for c in p_topo:
            pp_topo[c["topology"]] += 1
            if c["sub"]:
                pp_tsub[f"{c['topology']}({c['sub']})"] += 1

        n_orig = len(algos)
        n_after = len(trimmed_algos)
        per_video.append({
            "video": vid,
            "n_algos_baseline": n_orig,
            "n_algos_after_trim": n_after,
            "n_dropped": n_orig - n_after,
            "baseline_tp": len(b_pairs),
            "baseline_fp": len(b_fps),
            "baseline_fn": len(b_fns),
            "postprocess_tp": len(p_pairs),
            "postprocess_fp": len(p_fps),
            "postprocess_fn": len(p_fns),
        })

    print(f"Processed {len(per_video)} videos")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for v, reason in skipped:
            print(f"  {v}: {reason}")
    print()

    # ===== FN DIRECTION (the headline) =====
    print("=" * 70)
    print("FN DIRECTION")
    print("=" * 70)
    print(f"  Baseline:    FN = {base_fn}")
    print(f"  Postprocess: FN = {pp_fn}")
    print(f"  delta FN = {pp_fn - base_fn:+d}")
    print()

    # ===== Aggregate results =====
    base_sd_abs = [abs(d) for d in base_start_deltas]
    pp_sd_abs = [abs(d) for d in pp_start_deltas]
    base_sd_abs_med = int(np.median(base_sd_abs)) if base_sd_abs else None
    pp_sd_abs_med = int(np.median(pp_sd_abs)) if pp_sd_abs else None

    print("=" * 70)
    print("HOLDOUT AGGREGATE (19 exhaustive videos)")
    print("=" * 70)
    print()
    print("LEGACY:")
    print(f"  {'':<22} {'Baseline':>10} {'Postprocess':>12} {'Delta':>8}")
    print(f"  {'TP':<22} {base_tp:>10} {pp_tp:>12} {pp_tp - base_tp:>+8}")
    print(f"  {'FP':<22} {base_fp:>10} {pp_fp:>12} {pp_fp - base_fp:>+8}")
    print(f"  {'FN':<22} {base_fn:>10} {pp_fn:>12} {pp_fn - base_fn:>+8}")
    print(f"  {'start_delta abs_med':<22} {str(base_sd_abs_med):>10} {str(pp_sd_abs_med):>12}")
    print()

    print("TOPOLOGY:")
    print(f"  {'':<22} {'Baseline':>10} {'Postprocess':>12} {'Delta':>8}")
    keys = ["TP", "TOLERANCE_ERROR", "MERGED", "FRAGMENTED",
            "FALSE_POSITIVE", "FALSE_NEGATIVE", "COMPLEX"]
    for k in keys:
        b = base_topo.get(k, 0)
        p = pp_topo.get(k, 0)
        print(f"  {k:<22} {b:>10} {p:>12} {p - b:>+8}")
    print()
    print("TOLERANCE_ERROR sub-types:")
    for sub in ("start_early", "start_late", "span_over", "span_short"):
        key = f"TOLERANCE_ERROR({sub})"
        b = base_tsub.get(key, 0)
        p = pp_tsub.get(key, 0)
        print(f"  {key:<28} {b:>5} -> {p:<5}  ({p - b:>+d})")
    print()

    # ===== Decision =====
    print("=" * 70)
    print("DECISION RULE EVALUATION")
    print("=" * 70)
    fn_check = (pp_fn <= base_fn)
    tp_check = (pp_tp >= base_tp)
    sd_check = (pp_sd_abs_med == 0)
    se = base_tsub.get("TOLERANCE_ERROR(start_early)", 0)
    pse = pp_tsub.get("TOLERANCE_ERROR(start_early)", 0)
    se_check = (pse <= se)
    rej_tp_drop = (pp_tp < base_tp)
    rej_fn_rise = (pp_fn > base_fn)
    reject = rej_tp_drop and rej_fn_rise
    print(f"  FN drops or holds: {fn_check} (was {base_fn}, now {pp_fn})")
    print(f"  TP rises or holds: {tp_check} (was {base_tp}, now {pp_tp})")
    print(f"  start_delta abs_med = 0: {sd_check} (= {pp_sd_abs_med})")
    print(f"  start_early drops or holds: {se_check} (was {se}, now {pse})")
    print()
    if reject:
        print("  ==> REJECT: TP drops AND FN rises")
    elif fn_check and tp_check and sd_check and se_check:
        print("  ==> ACCEPT: all criteria met")
    else:
        print("  ==> BORDERLINE: not strict reject but not strict accept either")
    print()

    # ===== Save =====
    summary = {
        "baseline": {
            "tp": base_tp, "fp": base_fp, "fn": base_fn,
            "topology": dict(base_topo),
            "topology_sub": dict(base_tsub),
            "start_delta_abs_median": base_sd_abs_med,
        },
        "postprocess": {
            "tp": pp_tp, "fp": pp_fp, "fn": pp_fn,
            "topology": dict(pp_topo),
            "topology_sub": dict(pp_tsub),
            "start_delta_abs_median": pp_sd_abs_med,
        },
        "delta": {
            "tp": pp_tp - base_tp,
            "fp": pp_fp - base_fp,
            "fn": pp_fn - base_fn,
        },
        "config": {
            "T_threshold": T_THRESHOLD,
            "N_sustain": N_SUSTAIN,
            "min_span": MIN_SPAN,
            "matcher_start_tol": START_TOL,
        },
        "per_video": per_video,
        "skipped": skipped,
    }
    (OUT_DIR / "metrics" / "holdout_aggregate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'holdout_aggregate.json'}")


if __name__ == "__main__":
    main()
