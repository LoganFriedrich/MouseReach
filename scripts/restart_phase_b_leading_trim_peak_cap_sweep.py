"""v8.0.x experiment: leading-trim peak-cap sweep (holdout).

Alternative to the rejected proba_guard variant. Cap leading-trim
advance at the proba peak frame inside the original algo span. The
trim can advance freely as today (advancing through low-paw-lk frames
toward where paw_lk recovers), but it stops BEFORE eating past the
GBM's strongest evidence.

Mechanism (vs failed proba_guard):
- The proba_guard failed because it blocked trim from advancing
  through high-proba frames at all. But the trim's primary purpose is
  to advance through high-proba-low-paw_lk frames to satisfy strict
  asymmetric matcher start-tolerance. Blocking that breaks 12-72 TPs.
- The peak-cap is different. It allows trim to advance through
  high-proba frames freely, just stops BEFORE eating PAST the proba
  argmax. So normal trim operation (advance 1-5 frames into the rise)
  is unaffected, because the peak is past the rise. Only over-extended
  trim (advancing 10-30 frames past the peak through a sustained
  low-paw_lk run that engulfs the GT) gets clipped.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-26 evening, master at 7ecb582):
   - Production v8.0.4 stack (BSW + mg=0 + leading-trim + trailing-trim
     + apex-split, default thresholds).
   - Asymmetric strict matcher -2/+5, matcher-aware topology classifier.
   - Comparison baseline: holdout 19 with no peak-cap (identical to
     production).

2. Existing-code-modification check: NO. Variant implemented inline.

3. Unverified hypotheses:
   - Recovers most/all of the 7 LEADING_TRIM_STRANDED FNs by capping
     the over-extended trim before it eats the GT-overlapping interior.
   - Does not break currently-working trim (peak is typically past
     the algo start by 3-6 frames; normal trim of 1-3 frames stops
     naturally on paw_lk recovery long before the cap).
   - Buffer variants (cap a few frames BEFORE the peak) trade
     aggressiveness for safety; tighter buffer (cap = peak) is more
     aggressive on advancing the start; wider buffer (cap = peak - k)
     keeps more algo content.

4. FN-direction-reporting: lead with FN. Topology paired with legacy.

5. Framework: snapshot dir
   v8.0.4_dev_leading_trim_peak_cap_sweep/

6. Branch + tag:
   - Pre-experiment tag: pre-leading-trim-proba-guard-2026-05-26
     (reused; this is the second variant on the same lever)
   - Feature branch: feature/v8-leading-trim-proba-guard (reused)

7. Decision rule per config:
   ACCEPT if (vs v8.0.4 baseline):
     - TP non-decreasing on holdout
     - FN strictly decreasing on holdout
     - MERGED/FRAGMENTED non-increasing on holdout
     - Cardinal Rule both axes preserved
   REJECT if:
     - TP drops on holdout
     - Cardinal Rule fails
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
    r"\reach_detection\v8.0.4_dev_leading_trim_peak_cap_sweep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5

# Buffer = how many frames BEFORE the peak the cap sits.
# buffer=None means no cap (production behavior).
# buffer=0 means cap exactly at the peak frame.
# buffer=k means cap k frames before the peak.
CAP_CONFIGS = [
    # (label, mode, parameter)
    # mode = "none"      => no cap, baseline
    # mode = "argmax"    => cap at argmax(proba over span) - buffer
    # mode = "first_ge"  => cap at first frame with proba >= threshold
    ("base",         "none",     None),
    ("first_ge_70",  "first_ge", 0.70),
    ("first_ge_80",  "first_ge", 0.80),
    ("first_ge_90",  "first_ge", 0.90),
    ("first_ge_95",  "first_ge", 0.95),
]


# -----------------------------------------------------------------
# New leading-trim variant: peak cap
# -----------------------------------------------------------------

def trim_leading_with_peak_cap(reaches, paw_lk, proba,
                                lk_threshold=0.60, sustain_n=3,
                                min_span=3, cap_mode="none", cap_param=None):
    """Leading-trim variant that caps how far the trim can advance.

    cap_mode = "none":      no cap (production behavior).
    cap_mode = "first_ge":  cap = first frame in [s, e] with proba >= cap_param.
                            If no such frame exists, no cap (trim runs as today).
    """
    if not reaches:
        return []
    n_frames = len(paw_lk)
    out = []
    for r in reaches:
        s, e = r.start_frame, r.end_frame
        # Compute the cap
        cap = e + 1  # default: no cap (trim can theoretically eat the whole reach)
        if cap_mode == "first_ge":
            e_use = min(e, len(proba) - 1)
            if e_use >= s:
                local_proba = proba[s:e_use + 1]
                above = local_proba >= cap_param
                if np.any(above):
                    first_offset = int(np.argmax(above))  # argmax of bool returns first True
                    cap = s + first_offset
                # else: no cap (trim runs normally)
        new_s = s
        while new_s <= e:
            if new_s >= cap:
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


def apply_pipeline(proba, paw_lk, norm_pos, cap_mode, cap_param):
    spans = probabilities_to_reaches(
        proba, threshold=0.5, merge_gap=DEFAULT_MERGE_GAP,
        min_span=DEFAULT_MIN_SPAN)
    spans = trim_leading_with_peak_cap(
        spans, paw_lk, proba,
        lk_threshold=DEFAULT_TRIM_LK_THRESHOLD,
        sustain_n=DEFAULT_TRIM_SUSTAIN_N,
        min_span=DEFAULT_MIN_SPAN,
        cap_mode=cap_mode, cap_param=cap_param)
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


def score_corpus(video_data, cap_mode, cap_param):
    totals = {"tp": 0, "fp": 0, "fn": 0}
    topo = defaultdict(int)
    start_deltas = []
    span_deltas = []
    for vid, vd in video_data.items():
        algos = apply_pipeline(vd["proba"], vd["paw_lk"], vd["norm_pos"], cap_mode, cap_param)
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
        "totals": totals, "topology": dict(topo),
        "start_delta_abs_median": s_abs, "span_delta_abs_median": p_abs,
    }


def main():
    print("=" * 70)
    print("LEADING-TRIM PEAK-CAP SWEEP (HOLDOUT)")
    print(f"Configs: {[c[0] for c in CAP_CONFIGS]}")
    print("=" * 70)
    print()

    print("Loading production model...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    print("Computing per-frame proba per holdout video...", flush=True)
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
            "proba": proba, "paw_lk": paw_lk, "norm_pos": norm_pos, "gts": gts,
        }
        print(f"  {vid}: GTs={len(gts)}", flush=True)
    print()

    results = {"configs": {}}
    base_r = None
    for label, mode, param in CAP_CONFIGS:
        if mode == "none":
            print(f"--- {label} (no cap, v8.0.4 production) ---", flush=True)
        else:
            print(f"--- {label} (mode={mode}, param={param}) ---", flush=True)
        r = score_corpus(video_data, mode, param)
        results["configs"][label] = {"mode": mode, "param": param, **r}
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
        "configs": [{"label": c[0], "mode": c[1], "param": c[2]} for c in CAP_CONFIGS],
        "results": results,
    }, indent=2, default=int), encoding="utf-8")

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
    for label, _, _ in CAP_CONFIGS:
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
