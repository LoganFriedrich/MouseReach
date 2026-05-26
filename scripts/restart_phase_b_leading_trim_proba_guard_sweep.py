"""v8.0.x experiment: leading-trim proba-guard sweep (holdout).

Targets the 7 LEADING_TRIM_STRANDED holdout FNs surfaced by the 2026-05-26
stranded-FN failure-mode diagnostic. Those FNs have peak proba >= 0.87
inside the GT span, but leading-trim chewed past the GT entirely because
the paw_lk-low run extended through the GT region.

Mechanism: add a `proba_guard` cap to leading-trim. Walk the start frame
forward as today, but STOP advancing once the candidate frame has
proba >= proba_guard. This prevents the trim from removing the
GT-overlapping high-confidence interior while still trimming low-
confidence leading slop.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26 evening, master at 7ecb582):
   - Production v8.0.4 stack: BSW b=1/w=0.8 + mg=0 + leading-trim
     T=0.60/N=3 + trailing-trim T=0.60/N=3 + apex-split
     prom=0.12/depth=0.5/peak2<0.85.
   - Asymmetric strict matcher -2/+5 on live GT.
   - Matcher-aware topology classifier (no COMPLEX).
   - Comparison baseline: holdout 19 with proba_guard=infinity (no guard,
     identical to production).

2. Existing-code-modification check: NO.
   - New `trim_leading_with_proba_guard` implemented inline.
   - postprocess.py untouched.
   - If accepted, integrate into postprocess.py on the ship branch.

3. Unverified hypotheses:
   - Recovers most/all of the 7 LEADING_TRIM_STRANDED FNs.
   - Does not break existing TPs (the leading-trim's original purpose
     was solving a real boundary problem; the proba guard preserves
     trim for the "GBM-emits-too-early" case while suppressing the
     "trim-overchews-the-GT-interior" case).
   - Does not introduce new FRAGMENTED events (guarded trim should not
     create new boundaries; if anything it preserves existing boundaries).

4. FN-direction-reporting: lead with FN direction. Topology paired
   with legacy. Both deltas (vs cumulative best v8.0.4 AND vs pure
   baseline v8.0.0 mg=2).

5. Framework: output to
   Improvement_Snapshots/reach_detection/v8.0.4_dev_leading_trim_proba_guard_sweep/

6. Branch + tag:
   - Pre-experiment tag: pre-leading-trim-proba-guard-2026-05-26 (at 7ecb582)
   - Feature branch: feature/v8-leading-trim-proba-guard (off 7ecb582)

7. Decision rule (applied per proba_guard config):
   ACCEPT if (vs cumulative best v8.0.4):
     - TP non-decreasing on holdout (delta_TP >= 0)
     - FN strictly decreasing on holdout (delta_FN < 0)
     - MERGED non-increasing on holdout
     - FRAGMENTED non-increasing on holdout
     - Cardinal Rule: start_delta abs_median = 0 AND span_delta abs_median = 0
   REJECT if:
     - TP drops on holdout
     - Either Cardinal Rule axis fails
     - FRAGMENTED rises
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
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
    ReachSpan, probabilities_to_reaches,
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
    r"\reach_detection\v8.0.4_dev_leading_trim_proba_guard_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

# Sweep configs: proba_guard cap value
# Baseline = "no guard" (effectively guard = infinity = 999, never blocks)
GUARD_CONFIGS = [
    ("base",  None),    # No guard (production v8.0.4)
    ("g.50",  0.50),    # Stop trimming at the same threshold as the GBM
    ("g.60",  0.60),
    ("g.70",  0.70),
    ("g.80",  0.80),
    ("g.90",  0.90),
]


# -----------------------------------------------------------------
# New leading-trim variant with proba guard (this experiment)
# -----------------------------------------------------------------

def trim_leading_with_proba_guard(reaches, paw_lk, proba,
                                    lk_threshold=0.60, sustain_n=3,
                                    min_span=3, proba_guard=None):
    """Like trim_leading_sustained_lk, but additionally stop trimming
    once the candidate frame has proba >= proba_guard.

    Args:
      reaches: list of ReachSpan.
      paw_lk: per-frame paw_mean_lk array.
      proba: per-frame GBM proba array.
      lk_threshold: paw_lk threshold for trim (production: 0.60).
      sustain_n: required consecutive low-lk frames (production: 3).
      min_span: minimum reach span after trimming (production: 3).
      proba_guard: per-frame proba cap. If proba[new_s] >= guard, stop
                    trimming. None disables the guard (= production behavior).
    """
    if not reaches:
        return []
    n_frames = len(paw_lk)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        new_s = s
        while new_s <= e:
            # Proba guard: if current candidate frame has high proba,
            # don't trim past it.
            if proba_guard is not None and new_s < len(proba):
                if proba[new_s] >= proba_guard:
                    break
            window_end = new_s + sustain_n - 1
            if window_end > e:
                break
            if window_end >= n_frames:
                break
            window = paw_lk[new_s:window_end + 1]
            if np.any(np.isnan(window)):
                break
            if np.any(window >= lk_threshold):
                break
            new_s += 1
        if e - new_s + 1 >= min_span:
            out.append(ReachSpan(start_frame=new_s, end_frame=e))
    return out


# ---- Matcher + topology ----

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


def apply_pipeline(proba, paw_lk, norm_pos, proba_guard):
    """v8.0.4 pipeline with the leading-trim swapped for the proba-guard
    variant. If proba_guard is None, behaves identically to production."""
    spans = probabilities_to_reaches(
        proba, threshold=0.5, merge_gap=DEFAULT_MERGE_GAP,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_leading_with_proba_guard(
        spans, paw_lk, proba,
        lk_threshold=DEFAULT_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN,
        proba_guard=proba_guard)
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


def score_corpus(video_data, proba_guard):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    topo = defaultdict(int)
    start_deltas = []
    span_deltas = []
    for vid, vd in video_data.items():
        algos = apply_pipeline(vd["proba"], vd["paw_lk"], vd["norm_pos"], proba_guard)
        gts = vd["gts"]
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
    }


def main():
    print("=" * 70)
    print("LEADING-TRIM PROBA-GUARD SWEEP (HOLDOUT)")
    print(f"Configs: {[c[0] for c in GUARD_CONFIGS]}")
    print("=" * 70)
    print()

    print("Loading production model...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    print("Computing per-frame proba + DLC features per holdout video...",
          flush=True)
    video_data = {}
    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        proba = model.predict_proba(X)[:, 1]
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt(vid)
        video_data[vid] = {
            "proba": proba, "paw_lk": paw_lk, "norm_pos": norm_pos,
            "gts": gts,
        }
        print(f"  {vid}: GTs={len(gts)}", flush=True)
    print()

    results = {"configs": {}}
    base_r = None
    for label, guard in GUARD_CONFIGS:
        if guard is None:
            print(f"--- {label} (no guard, v8.0.4 production) ---", flush=True)
        else:
            print(f"--- {label} (proba_guard={guard}) ---", flush=True)
        r = score_corpus(video_data, guard)
        results["configs"][label] = {"proba_guard": guard, **r}
        if label == "base":
            base_r = r
            t = r["totals"]
            print(f"  Hol: TP={t['tp']} FP={t['fp']} FN={t['fn']}  "
                  f"abs_med start={r['start_delta_abs_median']} "
                  f"span={r['span_delta_abs_median']}")
            print(f"  Hol topology: {r['topology']}")
        else:
            bt = base_r["totals"]; rt = r["totals"]
            print(f"  Hol: TP={rt['tp']} ({rt['tp']-bt['tp']:+}) "
                  f"FP={rt['fp']} ({rt['fp']-bt['fp']:+}) "
                  f"FN={rt['fn']} ({rt['fn']-bt['fn']:+})  "
                  f"abs_med start={r['start_delta_abs_median']} "
                  f"span={r['span_delta_abs_median']}")
            b_topo = base_r["topology"]; t_topo = r["topology"]
            deltas = []
            for k in ("TP","TOLERANCE_ERROR_pairs","MERGED","FRAGMENTED",
                      "FALSE_POSITIVE","FALSE_NEGATIVE"):
                d = t_topo.get(k, 0) - b_topo.get(k, 0)
                deltas.append(f"{k}={t_topo.get(k,0)}({d:+})")
            print(f"  Hol topology: {' '.join(deltas)}")
        print()

    (OUT_DIR / "metrics" / "sweep_results.json").write_text(json.dumps({
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "configs": [{"label": c[0], "proba_guard": c[1]} for c in GUARD_CONFIGS],
        "results": results,
    }, indent=2, default=int), encoding="utf-8")

    # Summary
    print()
    print("=" * 110)
    print("SUMMARY (legacy matcher counts; topology in detail above)")
    print("=" * 110)
    print(f"{'label':<10}  {'Hol_TP':>6} {'dTP':>4} {'FP':>4} {'dFP':>4} {'FN':>4} {'dFN':>4}  "
          f"{'start':>5} {'span':>5}")
    print("-" * 110)
    hb = base_r["totals"]
    print(f"{'BASE':<10}  {hb['tp']:>6} {'':>4} {hb['fp']:>4} {'':>4} {hb['fn']:>4} {'':>4}  "
          f"{base_r['start_delta_abs_median']:>5} {base_r['span_delta_abs_median']:>5}")
    for label, _ in GUARD_CONFIGS:
        if label == "base":
            continue
        r = results["configs"][label]
        rt = r["totals"]
        print(f"{label:<10}  {rt['tp']:>6} {rt['tp']-hb['tp']:>+4} "
              f"{rt['fp']:>4} {rt['fp']-hb['fp']:>+4} {rt['fn']:>4} {rt['fn']-hb['fn']:>+4}  "
              f"{r['start_delta_abs_median']:>5} {r['span_delta_abs_median']:>5}")
    print()
    print(f"Wrote: {OUT_DIR / 'metrics' / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
