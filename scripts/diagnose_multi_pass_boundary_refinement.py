"""Diagnostic for multi-pass inference (#10) -- formulations (a) and (b).

Question
--------
Could a "pass 2" boundary refinement step improve over the current
postprocess-derived boundaries?

We can't directly test formulation (a) [less-smoothed proba] without
retraining or recomputing features. But we CAN test the spirit of both
(a) and (b) by deriving alternative boundaries from the existing pass-1
proba trajectory, using different boundary policies, and comparing them
to (1) the current post-postprocess boundaries and (2) the GT boundaries.

Alternative boundary policies tested
------------------------------------
Within each algo reach span [s_algo, e_algo] (after all production
postprocess), compute:

  - core_start: first frame inside [s_algo, e_algo] with proba >= 0.80
  - core_end:   last frame inside [s_algo, e_algo] with proba >= 0.80
  - first_cross_05: first frame in a windowed region around s_algo
                    where proba transitions from <0.5 to >=0.5
  - last_cross_05:  last frame in a windowed region around e_algo where
                    proba transitions from >=0.5 to <0.5

For each policy, compute:
  - boundary_shift vs current algo (delta in frames)
  - resulting start_delta and span_delta vs GT
  - would TP/TOL classification change?

Target populations
------------------
1. TOL_pair events (algo overlaps GT but matcher rejected) -- could
   refinement bring them into tolerance?
2. TPs with |start_delta| >= 3 OR |span_delta| >= 5 (near tolerance
   edges) -- could refinement improve precision?
3. Random sample of clean TPs -- does refinement hurt precision?

NOT a ship experiment. Diagnostic only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8 import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MERGE_GAP, DEFAULT_MIN_SPAN,
    DEFAULT_TRIM_LK_THRESHOLD, DEFAULT_TRIM_SUSTAIN_N,
    DEFAULT_TRAILING_TRIM_LK_THRESHOLD, DEFAULT_TRAILING_TRIM_SUSTAIN_N,
    DEFAULT_APEX_SPLIT_PROMINENCE, DEFAULT_APEX_SPLIT_DEPTH_MIN,
    DEFAULT_APEX_SPLIT_PEAK2_REL_MAX, DEFAULT_APEX_SPLIT_MIN_DISTANCE,
)
from mousereach.reach.v8.features import extract_features, load_dlc_h5
from mousereach.reach.v8.postprocess import (
    probabilities_to_reaches, trim_leading_sustained_lk,
    trim_trailing_sustained_lk,
    apex_split_at_trough, compute_paw_mean_lk,
    compute_hand_to_boxl_norm_pos,
)


HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.4_dev_multi_pass_boundary_diagnostic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

CORE_THRESHOLD = 0.80   # high threshold for "core" boundary
LOOSE_THRESHOLD = 0.50  # threshold for first/last cross
WINDOW_AROUND_BOUNDARY = 10  # frames to look around algo boundary


def load_live_gt(video_id):
    gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    rlist = data.get("reaches", {}).get("reaches", [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def overlap_exists(a_s, a_e, g_s, g_e):
    return not (a_e < g_s or a_s > g_e)


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
                candidates.append((abs(sd), ai, gi))
    candidates.sort()
    matched = set()
    used_a, used_g = set(), set()
    for _, ai, gi in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        matched.add((ai, gi))
    return matched


def apply_pipeline(proba, paw_lk, norm_pos):
    spans = probabilities_to_reaches(
        proba, threshold=0.5, merge_gap=DEFAULT_MERGE_GAP,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=DEFAULT_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_trailing_sustained_lk(
        spans, paw_lk,
        threshold=DEFAULT_TRAILING_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRAILING_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN)
    spans = apex_split_at_trough(
        spans, norm_pos,
        prominence=DEFAULT_APEX_SPLIT_PROMINENCE,
        depth_min=DEFAULT_APEX_SPLIT_DEPTH_MIN,
        peak2_rel_max=DEFAULT_APEX_SPLIT_PEAK2_REL_MAX,
        min_distance=DEFAULT_APEX_SPLIT_MIN_DISTANCE,
        min_span=DEFAULT_MIN_SPAN)
    return sorted({(int(r.start_frame), int(r.end_frame)) for r in spans})


def refine_boundaries(s_algo, e_algo, proba, n_frames):
    """Compute multiple alternative boundary policies."""
    e_use = min(e_algo, n_frames - 1)
    if e_use < s_algo:
        return None
    span_proba = proba[s_algo:e_use + 1]

    # Core threshold (proba >= CORE_THRESHOLD inside the span)
    core_mask = span_proba >= CORE_THRESHOLD
    if np.any(core_mask):
        core_idxs = np.where(core_mask)[0]
        core_start = s_algo + int(core_idxs[0])
        core_end = s_algo + int(core_idxs[-1])
    else:
        core_start = None
        core_end = None

    # First-cross-0.5 in a window around algo_start (looking before s_algo for
    # earlier crossings, or after s_algo if the trim chewed past).
    lo = max(0, s_algo - WINDOW_AROUND_BOUNDARY)
    hi = min(n_frames - 1, s_algo + WINDOW_AROUND_BOUNDARY)
    win = proba[lo:hi + 1]
    cross_up = None
    for i in range(1, len(win)):
        if win[i] >= LOOSE_THRESHOLD and win[i - 1] < LOOSE_THRESHOLD:
            cross_up = lo + i
            break

    # Last-cross-0.5 in a window around algo_end
    lo_e = max(0, e_algo - WINDOW_AROUND_BOUNDARY)
    hi_e = min(n_frames - 1, e_algo + WINDOW_AROUND_BOUNDARY)
    win_e = proba[lo_e:hi_e + 1]
    cross_down = None
    for i in range(len(win_e) - 1, 0, -1):
        if win_e[i] < LOOSE_THRESHOLD and win_e[i - 1] >= LOOSE_THRESHOLD:
            cross_down = lo_e + (i - 1)
            break

    return {
        "core_start": core_start,
        "core_end": core_end,
        "first_cross_up": cross_up,
        "last_cross_down": cross_down,
        "proba_at_algo_start": float(proba[s_algo]) if s_algo < n_frames else None,
        "proba_at_algo_end": float(proba[e_algo]) if e_algo < n_frames else None,
    }


def in_strict_tolerance(start_delta, span_delta, gt_span):
    sp_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
    return (-STRICT_START_TOL_EARLY <= start_delta <= STRICT_START_TOL_LATE
            and abs(span_delta) <= sp_tol)


def main():
    print("=" * 70)
    print("MULTI-PASS BOUNDARY REFINEMENT DIAGNOSTIC (HOLDOUT)")
    print("=" * 70)
    print()
    print("Loading production model...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    print("Processing holdout videos + characterizing boundaries...", flush=True)

    tol_records = []
    edge_tp_records = []
    clean_tp_records = []

    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        proba = model.predict_proba(X)[:, 1]
        n_frames = len(proba)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt(vid)
        if not gts:
            continue

        algos = apply_pipeline(proba, paw_lk, norm_pos)
        matched = greedy_match(algos, gts)
        matched_gt = {gi for _, gi in matched}
        matched_algo = {ai for ai, _ in matched}

        # TOL_pair events: algo overlaps GT but matcher rejected on tolerance
        # Identify via connected components: 1:1 unmatched algo+GT overlap.
        for ai, a in enumerate(algos):
            if ai in matched_algo:
                continue
            # Find GTs overlapping this algo
            ovl_gts = [(gi, g) for gi, g in enumerate(gts)
                       if overlap_exists(*a, *g) and gi not in matched_gt]
            if len(ovl_gts) != 1:
                continue
            gi, g = ovl_gts[0]
            # Make sure this GT only overlaps this algo
            ovl_algos = [(ai2, a2) for ai2, a2 in enumerate(algos)
                         if overlap_exists(*a2, *g)]
            if len(ovl_algos) != 1:
                continue
            # 1:1 unmatched pair = TOL_pair
            sd = a[0] - g[0]
            algo_span = a[1] - a[0] + 1
            gt_span = g[1] - g[0] + 1
            pd_ = algo_span - gt_span
            refined = refine_boundaries(a[0], a[1], proba, n_frames)
            tol_records.append({
                "video": vid, "ai": ai, "gi": gi,
                "algo_start": a[0], "algo_end": a[1], "algo_span": algo_span,
                "gt_start": g[0], "gt_end": g[1], "gt_span": gt_span,
                "start_delta": sd, "span_delta": pd_,
                "in_tol_currently": False,
                "refined": refined,
            })

        # TPs: characterize tolerance edge / clean
        for ai, gi in matched:
            a = algos[ai]; g = gts[gi]
            sd = a[0] - g[0]
            algo_span = a[1] - a[0] + 1
            gt_span = g[1] - g[0] + 1
            pd_ = algo_span - gt_span
            refined = refine_boundaries(a[0], a[1], proba, n_frames)
            rec = {
                "video": vid, "ai": ai, "gi": gi,
                "algo_start": a[0], "algo_end": a[1], "algo_span": algo_span,
                "gt_start": g[0], "gt_end": g[1], "gt_span": gt_span,
                "start_delta": sd, "span_delta": pd_,
                "in_tol_currently": True,
                "refined": refined,
            }
            if abs(sd) >= 3 or abs(pd_) >= 5:
                edge_tp_records.append(rec)
            else:
                clean_tp_records.append(rec)

    # ============= TOL_pair analysis =============
    print(f"\n=== TOL_pair events: {len(tol_records)} ===\n")
    can_refine_to_tp = 0
    no_refinement_avail = 0
    refinement_worse = 0
    for rec in tol_records:
        r = rec["refined"]
        if r is None:
            continue
        # Try each refined boundary policy
        # For start: pick the closest of {core_start, first_cross_up, algo_start} to GT_start
        # For end: pick the closest of {core_end, last_cross_down, algo_end} to GT_end
        candidates_start = [rec["algo_start"]]
        if r["core_start"] is not None: candidates_start.append(r["core_start"])
        if r["first_cross_up"] is not None: candidates_start.append(r["first_cross_up"])
        candidates_end = [rec["algo_end"]]
        if r["core_end"] is not None: candidates_end.append(r["core_end"])
        if r["last_cross_down"] is not None: candidates_end.append(r["last_cross_down"])
        # Find best combination
        best_sd = None; best_pd = None; best_in_tol = False
        for ns in candidates_start:
            for ne in candidates_end:
                if ne < ns: continue
                new_span = ne - ns + 1
                if new_span < 3: continue
                new_sd = ns - rec["gt_start"]
                new_pd = new_span - rec["gt_span"]
                in_t = in_strict_tolerance(new_sd, new_pd, rec["gt_span"])
                if in_t and (best_sd is None or abs(new_sd) + abs(new_pd) < abs(best_sd) + abs(best_pd)):
                    best_sd = new_sd; best_pd = new_pd; best_in_tol = True
        if best_in_tol:
            can_refine_to_tp += 1
        elif any([r["core_start"] is not None, r["first_cross_up"] is not None]):
            refinement_worse += 1
        else:
            no_refinement_avail += 1

    print(f"TOL_pairs convertible to TP via refinement:        {can_refine_to_tp}")
    print(f"TOL_pairs where refinement available but still TOL: {refinement_worse}")
    print(f"TOL_pairs with no refinement signal:                {no_refinement_avail}")
    print()

    # Detail listing
    print("--- TOL_pair detail ---")
    print(f"{'video':<22} {'algo':>13} {'gt':>13} {'sd':>4} {'pd':>4} "
          f"{'core':>13} {'cross_up':>9} {'best_sd':>8} {'best_pd':>8} {'refineable'}")
    for rec in tol_records:
        r = rec["refined"]
        if r is None: continue
        core_str = (f"{r['core_start']}-{r['core_end']}"
                    if r['core_start'] is not None else "-")
        cross_up = r['first_cross_up'] if r['first_cross_up'] is not None else "-"
        # Recompute best for display
        best_sd_d = ""; best_pd_d = ""; refineable = "no"
        candidates_start = [rec["algo_start"]]
        if r["core_start"] is not None: candidates_start.append(r["core_start"])
        if r["first_cross_up"] is not None: candidates_start.append(r["first_cross_up"])
        candidates_end = [rec["algo_end"]]
        if r["core_end"] is not None: candidates_end.append(r["core_end"])
        if r["last_cross_down"] is not None: candidates_end.append(r["last_cross_down"])
        best = None
        for ns in candidates_start:
            for ne in candidates_end:
                if ne < ns: continue
                new_span = ne - ns + 1
                if new_span < 3: continue
                new_sd = ns - rec["gt_start"]
                new_pd = new_span - rec["gt_span"]
                if in_strict_tolerance(new_sd, new_pd, rec["gt_span"]):
                    if best is None or abs(new_sd) + abs(new_pd) < abs(best[0]) + abs(best[1]):
                        best = (new_sd, new_pd)
        if best is not None:
            best_sd_d = f"{best[0]}"
            best_pd_d = f"{best[1]}"
            refineable = "YES"
        algo_str = f"{rec['algo_start']}-{rec['algo_end']}"
        gt_str = f"{rec['gt_start']}-{rec['gt_end']}"
        print(f"{rec['video']:<22} {algo_str:>13} {gt_str:>13} "
              f"{rec['start_delta']:>4} {rec['span_delta']:>4} "
              f"{core_str:>13} {str(cross_up):>9} {best_sd_d:>8} {best_pd_d:>8}  {refineable}")
    print()

    # ============= Edge TP analysis =============
    print(f"\n=== Edge TPs (|sd|>=3 or |pd|>=5): {len(edge_tp_records)} ===")
    # How many would BECOME tol if we apply refinement?
    n_edge_made_better = 0
    n_edge_made_worse = 0
    for rec in edge_tp_records:
        r = rec["refined"]
        if r is None: continue
        cur_score = abs(rec["start_delta"]) + abs(rec["span_delta"])
        # Use core boundaries if available
        ns = r["core_start"] if r["core_start"] is not None else rec["algo_start"]
        ne = r["core_end"] if r["core_end"] is not None else rec["algo_end"]
        if ne < ns: continue
        new_span = ne - ns + 1
        if new_span < 3: continue
        new_sd = ns - rec["gt_start"]
        new_pd = new_span - rec["gt_span"]
        new_score = abs(new_sd) + abs(new_pd)
        if in_strict_tolerance(new_sd, new_pd, rec["gt_span"]):
            if new_score < cur_score:
                n_edge_made_better += 1
            elif new_score > cur_score:
                n_edge_made_worse += 1
        else:
            n_edge_made_worse += 1  # core-refinement broke the TP
    print(f"  Edge TPs where core-refinement improves boundary: {n_edge_made_better}")
    print(f"  Edge TPs where core-refinement worsens or breaks: {n_edge_made_worse}")
    print()

    # ============= Clean TP analysis =============
    print(f"\n=== Clean TPs (|sd|<3 and |pd|<5): {len(clean_tp_records)} ===")
    n_clean_made_worse = 0
    n_clean_held = 0
    for rec in clean_tp_records:
        r = rec["refined"]
        if r is None: continue
        ns = r["core_start"] if r["core_start"] is not None else rec["algo_start"]
        ne = r["core_end"] if r["core_end"] is not None else rec["algo_end"]
        if ne < ns: continue
        new_span = ne - ns + 1
        if new_span < 3: continue
        new_sd = ns - rec["gt_start"]
        new_pd = new_span - rec["gt_span"]
        if in_strict_tolerance(new_sd, new_pd, rec["gt_span"]):
            n_clean_held += 1
        else:
            n_clean_made_worse += 1
    print(f"  Clean TPs that core-refinement preserves as TP:  {n_clean_held}")
    print(f"  Clean TPs that core-refinement breaks (TP->TOL): {n_clean_made_worse}")
    print()

    out = {
        "tol_records": [
            {k: v for k, v in r.items() if k != "refined"}
            | {"refined": r["refined"]}
            for r in tol_records
        ],
        "summary": {
            "tol_pair_n": len(tol_records),
            "tol_refineable_to_tp": can_refine_to_tp,
            "edge_tp_n": len(edge_tp_records),
            "edge_tp_improved_by_core": n_edge_made_better,
            "edge_tp_worsened_by_core": n_edge_made_worse,
            "clean_tp_n": len(clean_tp_records),
            "clean_tp_preserved_by_core": n_clean_held,
            "clean_tp_broken_by_core": n_clean_made_worse,
        },
    }
    (OUT_DIR / "metrics" / "diagnostic.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'diagnostic.json'}")


if __name__ == "__main__":
    main()
