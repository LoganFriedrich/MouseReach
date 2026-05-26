"""Diagnostic: classify stranded FNs by failure mode.

Context
-------
The per-video threshold diagnostic (2026-05-26) found that 38 of 47
stranded holdout FNs have peak proba >= 0.45 INSIDE the GT span. The
model fires inside these GT regions but the algo emission either doesn't
overlap, doesn't sustain min_span, or gets removed by postprocess.

This diagnostic classifies each stranded FN by which step in the pipeline
failed:

1. BELOW_THRESHOLD: peak proba inside GT < 0.50. The model doesn't see it.
2. SUB_MIN_SPAN: peak proba >= 0.50 but no sustained >=3-frame run of
   >= 0.50 inside the GT.
3. PRE_TRIM_EMITTED: GBM emits an algo span that overlaps the GT
   (after probabilities_to_reaches at 0.5 + min_span=3), but the
   leading-trim or trailing-trim removes the overlap.
4. APEX_SPLIT_STRANDED: algo emits and trim preserves overlap, but
   apex-split moves the pieces off the GT.
5. BOUNDARY_SHIFT: algo emits adjacent to or partially overlapping the
   GT, but no algo span actually overlaps after all postprocess. Or
   algo overlaps but the matcher rejects on tolerance and the topology
   classifier flags as TOLERANCE not stranded.
6. NON_GT_OVERLAP: some algo span(s) DO overlap the GT but were
   already assigned to other GTs by the matcher.

For each stranded FN, record which case applies.
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
    r"\reach_detection\v8.0.4_dev_stranded_fn_failure_modes"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


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


def overlap_len(a_s, a_e, g_s, g_e):
    s = max(a_s, g_s); e = min(a_e, g_e)
    return max(0, e - s + 1)


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


def longest_run_above(values, t):
    if values.size == 0:
        return 0
    above = values >= t
    if not np.any(above):
        return 0
    longest = cur = 0
    for b in above:
        if b:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return longest


def to_spans(reaches):
    return [(int(r.start_frame), int(r.end_frame)) for r in reaches]


def classify_stranded_fn(gt_span, proba, paw_lk, norm_pos,
                          spans_raw, spans_after_lt, spans_after_tt,
                          spans_after_apex, all_algos_final, matched_gt_idx,
                          gi):
    """Walk the pipeline and identify which stage stranded this GT."""
    g_s, g_e = gt_span
    e_use = min(g_e, len(proba) - 1)
    span_proba = proba[g_s:e_use + 1] if e_use >= g_s else np.array([])
    if span_proba.size == 0:
        return {"category": "EMPTY", "detail": ""}
    peak = float(np.max(span_proba))

    if peak < 0.50:
        return {"category": "BELOW_THRESHOLD",
                "detail": f"peak={peak:.3f}"}

    longest_above = longest_run_above(span_proba, 0.50)
    if longest_above < DEFAULT_MIN_SPAN:
        return {"category": "SUB_MIN_SPAN",
                "detail": f"peak={peak:.3f} longest_run>=.5={longest_above}"}

    # Check raw (pre-trim) emissions
    raw_overlap = any(overlap_exists(a_s, a_e, g_s, g_e) for a_s, a_e in spans_raw)
    lt_overlap = any(overlap_exists(a_s, a_e, g_s, g_e) for a_s, a_e in spans_after_lt)
    tt_overlap = any(overlap_exists(a_s, a_e, g_s, g_e) for a_s, a_e in spans_after_tt)
    apex_overlap = any(overlap_exists(a_s, a_e, g_s, g_e) for a_s, a_e in spans_after_apex)
    final_overlaps = [(ai, a) for ai, a in enumerate(all_algos_final)
                       if overlap_exists(*a, g_s, g_e)]

    if not raw_overlap:
        return {"category": "RAW_NO_OVERLAP",
                "detail": f"peak={peak:.3f} longest_run>=.5={longest_above} "
                          f"(GBM emits but emission doesn't overlap GT)"}
    if raw_overlap and not lt_overlap:
        return {"category": "LEADING_TRIM_STRANDED",
                "detail": f"peak={peak:.3f} (raw emission overlapped, "
                          f"leading-trim removed overlap)"}
    if lt_overlap and not tt_overlap:
        return {"category": "TRAILING_TRIM_STRANDED",
                "detail": f"peak={peak:.3f} (raw+leading-trim overlapped, "
                          f"trailing-trim removed overlap)"}
    if tt_overlap and not apex_overlap:
        return {"category": "APEX_SPLIT_STRANDED",
                "detail": f"peak={peak:.3f} (apex-split removed overlap)"}
    # Apex_overlap exists in spans_after_apex. Why is GT stranded in final?
    if apex_overlap and not final_overlaps:
        return {"category": "FINAL_NO_OVERLAP",
                "detail": f"peak={peak:.3f} (overlap present after apex but "
                          f"final algos don't overlap -- pipeline mismatch?)"}
    # Final overlap exists -- so the GT IS in a component with algos.
    # But still stranded -- means another GT in same component absorbed
    # the algo via matching, leaving this GT stranded.
    overlapping_algos_matched_to_others = []
    for ai, a in final_overlaps:
        is_matched_elsewhere = any((ai_, gi_) for ai_, gi_ in matched_gt_idx
                                    if ai_ == ai and gi_ != gi)
        overlapping_algos_matched_to_others.append((ai, a, is_matched_elsewhere))
    return {"category": "ALGO_TAKEN_BY_OTHER_GT",
            "detail": f"peak={peak:.3f} ({len(final_overlaps)} algo(s) overlap "
                      f"but matched elsewhere or not at all): "
                      f"{overlapping_algos_matched_to_others}"}


def main():
    print("=" * 70)
    print("STRANDED-FN FAILURE-MODE CLASSIFIER (HOLDOUT 19)")
    print("=" * 70)
    print()

    print("Loading production model...", flush=True)
    bundle = joblib.load(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feat_cols = bundle["feature_columns"]

    summary = defaultdict(int)
    per_video_summary = defaultdict(lambda: defaultdict(int))
    per_fn_records = []

    print("Processing holdout videos...", flush=True)
    for dlc_path in sorted(HOLDOUT_DLC_DIR.glob(f"*{DLC_SUFFIX}.h5")):
        vid = dlc_path.stem.replace(DLC_SUFFIX, "")
        dlc = load_dlc_h5(dlc_path)
        feats = extract_features(dlc)
        X = feats[feat_cols].to_numpy(dtype="float32")
        proba = model.predict_proba(X)[:, 1]
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_hand_to_boxl_norm_pos(dlc)
        gts = load_live_gt(vid)
        if not gts:
            continue

        # Walk pipeline stages
        spans_raw = probabilities_to_reaches(
            proba, threshold=0.5, merge_gap=DEFAULT_MERGE_GAP,
            min_span=DEFAULT_MIN_SPAN)
        spans_after_lt = trim_leading_sustained_lk(
            spans_raw, paw_lk,
            threshold=DEFAULT_TRIM_LK_THRESHOLD,
            sustain_n=DEFAULT_TRIM_SUSTAIN_N,
            min_span=DEFAULT_MIN_SPAN)
        spans_after_tt = trim_trailing_sustained_lk(
            spans_after_lt, paw_lk,
            threshold=DEFAULT_TRAILING_TRIM_LK_THRESHOLD,
            sustain_n=DEFAULT_TRAILING_TRIM_SUSTAIN_N,
            min_span=DEFAULT_MIN_SPAN)
        spans_after_apex = apex_split_at_trough(
            spans_after_tt, norm_pos,
            prominence=DEFAULT_APEX_SPLIT_PROMINENCE,
            depth_min=DEFAULT_APEX_SPLIT_DEPTH_MIN,
            peak2_rel_max=DEFAULT_APEX_SPLIT_PEAK2_REL_MAX,
            min_distance=DEFAULT_APEX_SPLIT_MIN_DISTANCE,
            min_span=DEFAULT_MIN_SPAN)

        raw_sp = to_spans(spans_raw)
        lt_sp = to_spans(spans_after_lt)
        tt_sp = to_spans(spans_after_tt)
        apex_sp = to_spans(spans_after_apex)
        # Final algo set used by matcher (sorted+deduped, matches production)
        all_algos = sorted({(s, e) for s, e in apex_sp})
        matched = greedy_match(all_algos, gts)
        matched_gt_idx = {gi for _, gi in matched}

        # Stranded FNs: GTs with NO algo overlap
        stranded_gis = []
        for gi, g in enumerate(gts):
            if gi in matched_gt_idx:
                continue
            has_overlap = any(overlap_exists(*a, *g) for a in all_algos)
            if not has_overlap:
                stranded_gis.append(gi)

        for gi in stranded_gis:
            g_s, g_e = gts[gi]
            result = classify_stranded_fn(
                gts[gi], proba, paw_lk, norm_pos,
                raw_sp, lt_sp, tt_sp, apex_sp, all_algos, matched, gi)
            summary[result["category"]] += 1
            per_video_summary[vid][result["category"]] += 1
            per_fn_records.append({
                "video": vid, "gt_start": g_s, "gt_end": g_e,
                "gt_span": g_e - g_s + 1,
                "category": result["category"],
                "detail": result["detail"],
            })

    # Print per-video summary
    print("\n=== Per-video stranded FN counts by category ===")
    all_cats = sorted(summary.keys())
    print(f"{'video':<22} " + " ".join(f"{c[:18]:>19}" for c in all_cats) + "  tot")
    print("-" * (22 + 20 * len(all_cats) + 6))
    for vid in sorted(per_video_summary.keys()):
        row = per_video_summary[vid]
        tot = sum(row.values())
        print(f"{vid:<22} " + " ".join(f"{row.get(c,0):>19}" for c in all_cats)
              + f"  {tot:>3}")
    print()

    # Corpus summary
    print("\n=== Corpus-wide stranded FN failure-mode tally ===")
    total = sum(summary.values())
    for c in sorted(summary.keys(), key=lambda k: -summary[k]):
        print(f"  {c:<25}  {summary[c]:>3}  ({100.0*summary[c]/total:.1f}%)")
    print(f"  {'TOTAL':<25}  {total:>3}")
    print()

    # Per-FN detail (top by category)
    print("\n=== Per-FN detail ===")
    by_cat = defaultdict(list)
    for r in per_fn_records:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat.keys()):
        recs = by_cat[cat]
        print(f"\n--- {cat} (n={len(recs)}) ---")
        for r in recs:
            print(f"  {r['video']:<22} gt={r['gt_start']}-{r['gt_end']} "
                  f"span={r['gt_span']}  {r['detail']}")

    # Save
    out = {
        "corpus_summary": dict(summary),
        "per_video_summary": {k: dict(v) for k, v in per_video_summary.items()},
        "per_fn_records": per_fn_records,
    }
    (OUT_DIR / "metrics" / "diagnostic.json").write_text(
        json.dumps(out, indent=2, default=int), encoding="utf-8")
    print(f"\nWrote: {OUT_DIR / 'metrics' / 'diagnostic.json'}")


if __name__ == "__main__":
    main()
