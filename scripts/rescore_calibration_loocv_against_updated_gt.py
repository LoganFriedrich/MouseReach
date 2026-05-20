"""
Pragmatic re-score of v8 calibration LOOCV against updated GT.

Loads the per-event records from the v8.0.0_dev_merge_gap_0_candidate LOOCV
snapshot, extracts the algo reaches per video (frozen since the v8 model
bundle has not changed), loads the current calibration GT from
validation_runs/DLC_2026_03_27/gt/, re-matches the algo reaches against the
updated GT using the same strict matching rule, and writes new
loocv_aggregate.json + loocv_per_fold.json into a NEW snapshot dir.

This is NOT a model retrain. Model outputs are frozen at the last LOOCV
result; only the matching against updated GT changes. Substantially cheaper
than rebuilding train_pool.parquet + retraining 16 folds.

Why a new snapshot dir (not overwrite-in-place):
  The source snapshot (v8.0.0_dev_merge_gap_0_candidate) is the ACCEPTANCE
  snapshot that the v8.0.1 production ship decision was based on. Preserving
  it intact maintains provenance for that decision. The re-score gets its own
  dated snapshot dir so both can be inspected.

INPUT
=====
  Source snapshot (read-only):
    Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_0_candidate/
      metrics/loocv_aggregate.json   -- per-event records with algo + GT frames
      metrics/loocv_per_fold.json    -- fold definitions (val_video_ids)

  Updated GT (read fresh):
    validation_runs/DLC_2026_03_27/gt/<video_id>_unified_ground_truth.json

OUTPUT
======
  Improvement_Snapshots/reach_detection/v8.0.0_dev_merge_gap_0_candidate_rescored_2026-05-20/
    metrics/loocv_aggregate.json     -- same schema as source, new TP/FP/FN
    metrics/loocv_per_fold.json      -- same schema as source
    rescore_manifest.json            -- source snapshot + GT mtimes
    RESULTS.md                       -- what changed and why

After this runs, update CAL_SOURCE in generate_fpfn_review_manifests_v8_0_1.py
to point at the new dir, then re-run that script to refresh the calibration
manifests.

OPERATIONAL NOTES
=================
- Algo reaches are reconstructed per video by reading every TP and FP record's
  (algo_start_frame, algo_end_frame). Deduped (set) and sorted before Reach
  objects are built. This is identical to what reconstruct_per_video in the
  manifest generator does.
- The strict matching parameters (start_tol=2, span_tol_rel=0.5, span_tol_abs=5)
  are the production eval criterion and match what produced the source snapshot.
- merge_gap, boundary_buffer, boundary_weight metadata in the aggregate are
  preserved verbatim from the source (they describe the model + postprocess
  that produced these algo reaches; the re-score does not change them).
- Per-fold summaries are recomputed since the per-fold TP/FP/FN may shift
  with the updated GT.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    Reach, match_reaches, _load_gt_reaches, _find_gt_file,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_merge_gap_0_candidate"
)
SOURCE_AGGREGATE = SOURCE_SNAPSHOT / "metrics" / "loocv_aggregate.json"
SOURCE_PER_FOLD = SOURCE_SNAPSHOT / "metrics" / "loocv_per_fold.json"

GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\gt"
)

RESCORE_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection"
) / f"v8.0.0_dev_merge_gap_0_candidate_rescored_{RESCORE_DATE}"

# Strict matching params (production eval, same as source)
STRICT_START_TOL = 2
STRICT_SPAN_TOL_REL = 0.5
STRICT_SPAN_TOL_ABS = 5


# ---------------------------------------------------------------------------
# Algo-reach reconstruction
# ---------------------------------------------------------------------------

def reconstruct_algo_reaches(records: List[Dict[str, Any]]
                             ) -> Dict[str, List[Reach]]:
    """Per video, extract algo Reach objects from TP+FP records.

    TP records have both algo_start_frame and gt_start_frame. FP records have
    only algo. FN records have only GT (skipped here -- we want algo reaches).
    Deduped by (start, end) and sorted by start.
    """
    algo_by_vid: Dict[str, set] = defaultdict(set)
    for r in records:
        if r["status"] in ("tp", "fp"):
            a_s = int(r["algo_start_frame"])
            a_e = int(r["algo_end_frame"])
            if a_s >= 0:
                algo_by_vid[r["video_id"]].add((a_s, a_e))
    out: Dict[str, List[Reach]] = {}
    for vid, tuples in algo_by_vid.items():
        sorted_tuples = sorted(tuples)
        out[vid] = [Reach(start_frame=s, end_frame=e, index=i)
                    for i, (s, e) in enumerate(sorted_tuples)]
    return out


# ---------------------------------------------------------------------------
# Per-video rescore
# ---------------------------------------------------------------------------

def rescore_video(video_id: str,
                  algo_reaches: List[Reach]
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-match algo reaches against current GT for one video.

    Returns (records, summary) where records are per-event in the same schema
    as loocv_aggregate.json's raw_results, and summary has n_tp/n_fp/n_fn +
    tp_start_delta / tp_span_delta distributions.
    """
    gt_path = _find_gt_file(GT_DIR, video_id)
    if gt_path is None:
        # _find_gt_file returns None for unified GT with exhaustive=True but
        # zero reaches (e.g. videos with no successful reach attempts in the
        # exhaustive window). For LOOCV purposes that just means GT is empty
        # for this video; don't error. If the algo also has 0 reaches the
        # summary is 0/0/0. If the algo has reaches they become FPs.
        gt_reaches: List[Reach] = []
    else:
        gt_reaches = _load_gt_reaches(gt_path)

    results = match_reaches(
        algo_reaches, gt_reaches,
        strict=True,
        strict_start_tol=STRICT_START_TOL,
        strict_span_tol_rel=STRICT_SPAN_TOL_REL,
        strict_span_tol_abs=STRICT_SPAN_TOL_ABS,
    )

    records: List[Dict[str, Any]] = []
    tp_start_deltas: List[int] = []
    tp_span_deltas: List[int] = []
    n_tp = n_fp = n_fn = 0
    for r in results:
        if r.status == "matched":
            n_tp += 1
            span_delta = (r.algo_end - r.algo_start + 1) - (r.gt_end - r.gt_start + 1)
            tp_start_deltas.append(int(r.start_delta))
            tp_span_deltas.append(int(span_delta))
            records.append({
                "status": "tp",
                "video_id": video_id,
                "gt_index": r.gt_reach_index,
                "algo_index": r.algo_reach_index,
                "start_delta": int(r.start_delta),
                "span_delta": int(span_delta),
                "algo_start_frame": int(r.algo_start),
                "algo_end_frame": int(r.algo_end),
                "gt_start_frame": int(r.gt_start),
                "gt_end_frame": int(r.gt_end),
            })
        elif r.status == "fp":
            n_fp += 1
            records.append({
                "status": "fp",
                "video_id": video_id,
                "gt_index": None,
                "algo_index": r.algo_reach_index,
                "start_delta": None,
                "span_delta": None,
                "algo_start_frame": int(r.algo_start),
                "algo_end_frame": int(r.algo_end),
                "gt_start_frame": -1,
                "gt_end_frame": -1,
            })
        elif r.status == "fn":
            n_fn += 1
            records.append({
                "status": "fn",
                "video_id": video_id,
                "gt_index": r.gt_reach_index,
                "algo_index": None,
                "start_delta": None,
                "span_delta": None,
                "algo_start_frame": -1,
                "algo_end_frame": -1,
                "gt_start_frame": int(r.gt_start),
                "gt_end_frame": int(r.gt_end),
            })

    summary = {
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "tp_start_delta": _delta_stats(tp_start_deltas),
        "tp_span_delta": _delta_stats(tp_span_deltas),
    }
    return records, summary


def _delta_stats(values: List[int]) -> Dict[str, Any]:
    """Compute the stat block for a delta distribution, schema-compatible
    with the source snapshot's tp_start_delta / tp_span_delta blocks."""
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "median": None, "abs_median": None, "p10": None, "p90": None}
    arr = np.asarray(values)
    return {
        "n": int(len(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": int(np.median(arr)),
        "abs_median": int(np.median(np.abs(arr))),
        "p10": int(np.percentile(arr, 10)),
        "p90": int(np.percentile(arr, 90)),
    }


def _agg_summary(per_video_summaries: List[Dict[str, Any]],
                 per_video_records: Dict[str, List[Dict[str, Any]]]
                 ) -> Dict[str, Any]:
    total_tp = sum(s["n_tp"] for s in per_video_summaries)
    total_fp = sum(s["n_fp"] for s in per_video_summaries)
    total_fn = sum(s["n_fn"] for s in per_video_summaries)
    all_start = []
    all_span = []
    for recs in per_video_records.values():
        for r in recs:
            if r["status"] == "tp":
                all_start.append(r["start_delta"])
                all_span.append(r["span_delta"])
    return {
        "n_tp": total_tp,
        "n_fp": total_fp,
        "n_fn": total_fn,
        "tp_start_delta": _delta_stats(all_start),
        "tp_span_delta": _delta_stats(all_span),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print(f"Rescore v8 calibration LOOCV against updated GT ({RESCORE_DATE})")
    print(f"  Source: {SOURCE_SNAPSHOT.name}")
    print(f"  GT dir: {GT_DIR}")
    print(f"  Output: {OUTPUT_SNAPSHOT.name}")
    print("=" * 70)
    print()

    if not SOURCE_AGGREGATE.exists():
        raise FileNotFoundError(f"Source aggregate not found: {SOURCE_AGGREGATE}")
    src_agg = json.loads(SOURCE_AGGREGATE.read_text(encoding="utf-8"))
    src_per_fold = json.loads(SOURCE_PER_FOLD.read_text(encoding="utf-8"))

    src_records = src_agg["raw_results"]
    print(f"Loaded {len(src_records)} source records across "
          f"{len(set(r['video_id'] for r in src_records))} videos")

    algo_by_vid = reconstruct_algo_reaches(src_records)
    print(f"Reconstructed algo reaches for {len(algo_by_vid)} videos")
    print()

    # Walk folds in source order (preserves fold identity / val_video_ids)
    new_folds = []
    all_new_records: List[Dict[str, Any]] = []
    per_video_records: Dict[str, List[Dict[str, Any]]] = {}
    per_video_summaries: List[Dict[str, Any]] = []

    print("Per-fold re-score:")
    for fold in src_per_fold:
        val_vids = fold["val_video_ids"]
        # Each fold has one val video for LOOCV
        fold_records: List[Dict[str, Any]] = []
        fold_summaries: List[Dict[str, Any]] = []
        for vid in val_vids:
            algos = algo_by_vid.get(vid, [])
            if not algos:
                # No algo records for this video in the source -- still need
                # to score (could be all FN). Use empty algo list.
                algos = []
            recs, summary = rescore_video(vid, algos)
            fold_records.extend(recs)
            fold_summaries.append(summary)
            all_new_records.extend(recs)
            per_video_records[vid] = recs
            per_video_summaries.append(summary)
            print(f"  {vid:35} TP={summary['n_tp']:>4} "
                  f"FP={summary['n_fp']:>4} FN={summary['n_fn']:>4}")
        # Single-video fold summary == that video's summary; combine if
        # multi-video folds ever appear (LOOCV is 1-val, but be safe).
        if len(fold_summaries) == 1:
            new_folds.append({
                "val_video_ids": val_vids,
                "summary": fold_summaries[0],
            })
        else:
            combined = {
                "n_tp": sum(s["n_tp"] for s in fold_summaries),
                "n_fp": sum(s["n_fp"] for s in fold_summaries),
                "n_fn": sum(s["n_fn"] for s in fold_summaries),
                # Stats over per-video aggregations would need raw deltas;
                # skip for now since LOOCV is 1-val in practice.
                "tp_start_delta": None,
                "tp_span_delta": None,
            }
            new_folds.append({
                "val_video_ids": val_vids,
                "summary": combined,
            })
    print()

    agg_summary = _agg_summary(per_video_summaries, per_video_records)

    new_aggregate = {
        "n_folds": src_agg.get("n_folds"),
        "summary": agg_summary,
        "raw_results": all_new_records,
        # Preserve postprocess + training metadata from source (the model
        # bundle that produced these algo reaches has not changed)
        "merge_gap": src_agg.get("merge_gap"),
        "boundary_buffer": src_agg.get("boundary_buffer"),
        "boundary_weight": src_agg.get("boundary_weight"),
        "schema_version": src_agg.get("schema_version"),
        # New metadata for the rescore
        "rescored_from": SOURCE_SNAPSHOT.name,
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
        "rescore_reason": "GT files in validation_runs/DLC_2026_03_27/gt/ "
                          "were edited via the FP/FN review widget; "
                          "re-match algo reaches against updated GT.",
    }

    # Write outputs
    (OUTPUT_SNAPSHOT / "metrics").mkdir(parents=True, exist_ok=True)
    out_agg = OUTPUT_SNAPSHOT / "metrics" / "loocv_aggregate.json"
    out_per_fold = OUTPUT_SNAPSHOT / "metrics" / "loocv_per_fold.json"

    out_agg.write_text(json.dumps(new_aggregate, indent=2), encoding="utf-8")
    out_per_fold.write_text(json.dumps(new_folds, indent=2), encoding="utf-8")

    # Rescore manifest with provenance
    rescore_manifest = {
        "rescored_from": str(SOURCE_SNAPSHOT),
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
        "gt_dir": str(GT_DIR),
        "n_videos": len(per_video_summaries),
        "n_records_before": len(src_records),
        "n_records_after": len(all_new_records),
        "source_summary": src_agg["summary"],
        "rescored_summary": agg_summary,
    }
    (OUTPUT_SNAPSHOT / "rescore_manifest.json").write_text(
        json.dumps(rescore_manifest, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"AGGREGATE -- after re-score against updated GT")
    print("=" * 70)
    print(f"  Total TP={agg_summary['n_tp']}  FP={agg_summary['n_fp']}  "
          f"FN={agg_summary['n_fn']}")
    print(f"  vs source: TP={src_agg['summary']['n_tp']}  "
          f"FP={src_agg['summary']['n_fp']}  "
          f"FN={src_agg['summary']['n_fn']}")
    print(f"  delta:    TP={agg_summary['n_tp'] - src_agg['summary']['n_tp']:+d}  "
          f"FP={agg_summary['n_fp'] - src_agg['summary']['n_fp']:+d}  "
          f"FN={agg_summary['n_fn'] - src_agg['summary']['n_fn']:+d}")
    print()
    print(f"Boundary precision (matched reaches, n={agg_summary['n_tp']}):")
    sd = agg_summary["tp_start_delta"]
    pd = agg_summary["tp_span_delta"]
    print(f"  start_delta: median={sd['median']}f  "
          f"abs_median={sd['abs_median']}f  mean={sd['mean']:+.3f}f")
    print(f"  span_delta:  median={pd['median']}f  "
          f"abs_median={pd['abs_median']}f  mean={pd['mean']:+.3f}f")
    print()
    print(f"Wrote: {out_agg}")
    print(f"Wrote: {out_per_fold}")
    print(f"Wrote: {OUTPUT_SNAPSHOT / 'rescore_manifest.json'}")


if __name__ == "__main__":
    main()
