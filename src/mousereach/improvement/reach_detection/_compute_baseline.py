"""
Compute three-point baseline numbers for reach detection triangulation.

Produces:
  1. Pre-DLC metrics (DB reach_data with old DLC model + reach_detector v5.3.0)
  2. Best post-DLC metrics (outputs_reach_v7.1.0 with new DLC + reach_detector v7.1.0)

For both, compute:
  - Standard reach matching metrics (FP/FN/start_delta/end_delta)
  - Kinematic completeness metrics
  - Subsetted by exhaustive vs non-exhaustive

NOT a production script -- run once to establish baselines, then delete.
"""
import csv
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mousereach.improvement.reach_detection.metrics import (
    Reach,
    ReachMatchResult,
    KinematicCompletenessResult,
    KinematicCompletenessAggregates,
    match_reaches,
    compute_kinematic_completeness,
)

GT_DIR = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\gt")
ALGO_DIR_POST = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\outputs_reach_v7.1.0")
DB_PATH = Path(r"Y:\2_Connectome\Databases\connectome.db")
SNAPSHOT_PRE = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\baseline_pre_dlc")
SNAPSHOT_POST = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\reach_detection\reach_v7.1.0_visibility_direction_reversal")

WINDOW = 10


def load_gt():
    """Load all unified GT files with reaches."""
    gt_files = sorted(GT_DIR.glob("*_unified_ground_truth.json"))
    gt_data = {}
    for gf in gt_files:
        vid = gf.stem.replace("_unified_ground_truth", "")
        with open(gf, encoding="utf-8") as f:
            data = json.load(f)
        rd = data.get("reaches", {})
        exhaustive = rd.get("exhaustive", False)
        reaches = [
            r for r in rd.get("reaches", [])
            if not r.get("exclude_from_analysis", False)
        ]
        if not reaches:
            continue
        reaches_sorted = sorted(reaches, key=lambda x: x["start_frame"])
        gt_data[vid] = {
            "exhaustive": exhaustive,
            "reaches": reaches_sorted,
            "reach_objs": [
                Reach(int(r["start_frame"]), int(r["end_frame"]), i)
                for i, r in enumerate(reaches_sorted)
            ],
        }
    return gt_data


def load_pre_dlc_from_db(gt_data):
    """Load pre-DLC reaches from connectome.db reach_data table."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    pre_dlc = {}
    for vid in gt_data:
        cur.execute(
            "SELECT start_frame, apex_frame, end_frame "
            "FROM reach_data WHERE video_name = ? ORDER BY start_frame",
            (vid,),
        )
        rows = cur.fetchall()
        if rows:
            pre_dlc[vid] = [
                Reach(int(r["start_frame"]), int(r["end_frame"]), i)
                for i, r in enumerate(rows)
            ]
    conn.close()
    return pre_dlc


def load_post_dlc_from_algo(gt_data):
    """Load post-DLC reaches from algo _reaches.json files."""
    post_dlc = {}
    for vid in gt_data:
        algo_file = ALGO_DIR_POST / f"{vid}_reaches.json"
        if not algo_file.exists():
            continue
        with open(algo_file, encoding="utf-8") as f:
            data = json.load(f)
        all_reaches = []
        for seg in data.get("segments", []):
            for r in seg.get("reaches", []):
                if not r.get("exclude_from_analysis", False):
                    all_reaches.append(r)
        all_reaches.sort(key=lambda x: x["start_frame"])
        post_dlc[vid] = [
            Reach(int(r["start_frame"]), int(r["end_frame"]), i)
            for i, r in enumerate(all_reaches)
        ]
    return post_dlc


def agg_standard(results_list):
    """Aggregate standard matching metrics."""
    if not results_list:
        return {}
    n_matched = sum(1 for r in results_list if r.status == "matched")
    n_fp = sum(1 for r in results_list if r.status == "fp")
    n_fn = sum(1 for r in results_list if r.status == "fn")
    n_gt = n_matched + n_fn
    start_ds = [r.start_delta for r in results_list if r.status == "matched"]
    end_ds = [r.end_delta for r in results_list if r.status == "matched"]

    recall = n_matched / n_gt if n_gt > 0 else 0

    return {
        "n_gt": n_gt,
        "n_algo": n_matched + n_fp,
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "recall": round(recall, 4),
        "fn_rate_pct": round(100 * n_fn / n_gt, 2) if n_gt > 0 else 0,
        "mean_abs_start_delta": round(float(np.mean([abs(d) for d in start_ds])), 3) if start_ds else None,
        "mean_abs_end_delta": round(float(np.mean([abs(d) for d in end_ds])), 3) if end_ds else None,
        "start_exact_0_pct": round(100 * sum(1 for d in start_ds if d == 0) / len(start_ds), 1) if start_ds else None,
        "end_exact_0_pct": round(100 * sum(1 for d in end_ds if d == 0) / len(end_ds), 1) if end_ds else None,
    }


def agg_completeness(comp_list):
    """Aggregate kinematic completeness metrics."""
    if not comp_list:
        return {}
    matched = [c for c in comp_list if c.status == "matched"]
    n_matched = len(matched)
    n_fn = len([c for c in comp_list if c.status == "fn"])

    if n_matched == 0:
        return {"n_total": len(comp_list), "n_matched": 0, "n_fn": n_fn}

    coverages = [c.coverage for c in matched]
    apex_checks = [c.apex_included for c in matched if c.apex_included is not None]

    return {
        "n_total": len(comp_list),
        "n_matched": n_matched,
        "n_fn": n_fn,
        "median_coverage": round(float(np.median(coverages)), 4),
        "mean_coverage": round(float(np.mean(coverages)), 4),
        "frac_coverage_gte_1": round(sum(1 for c in coverages if c >= 1.0) / n_matched, 4),
        "frac_apex_included": round(sum(apex_checks) / len(apex_checks), 4) if apex_checks else None,
        "frac_anchor_start_ok": round(sum(1 for c in matched if c.anchor_at_start_ok) / n_matched, 4),
        "frac_anchor_end_ok": round(sum(1 for c in matched if c.anchor_at_end_ok) / n_matched, 4),
        "frac_both_anchors_ok": round(
            sum(1 for c in matched if c.anchor_at_start_ok and c.anchor_at_end_ok) / n_matched, 4
        ),
    }


def compute_all_metrics(gt_data_dict, algo_reaches_dict, label):
    """Compute both standard matching and kinematic completeness metrics."""
    all_results = {"all": [], "exhaustive": [], "non_exhaustive": []}
    all_completeness = {"all": [], "exhaustive": [], "non_exhaustive": []}

    match_rows = []
    video_rows = []

    for vid in sorted(gt_data_dict.keys()):
        gt_info = gt_data_dict[vid]
        if vid not in algo_reaches_dict:
            continue

        gt_reach_objs = gt_info["reach_objs"]
        algo = algo_reaches_dict[vid]
        exhaustive = gt_info["exhaustive"]
        gt_raw = gt_info["reaches"]

        # Standard matching
        results = match_reaches(algo, gt_reach_objs, window=WINDOW)

        n_gt = len(gt_reach_objs)
        n_algo = len(algo)
        n_matched = sum(1 for r in results if r.status == "matched")
        n_fp = sum(1 for r in results if r.status == "fp")
        n_fn = sum(1 for r in results if r.status == "fn")

        video_rows.append({
            "video_id": vid,
            "exhaustive": exhaustive,
            "n_gt": n_gt,
            "n_algo": n_algo,
            "n_matched": n_matched,
            "n_fp": n_fp,
            "n_fn": n_fn,
        })

        for r in results:
            row = {
                "video_id": vid,
                "exhaustive": exhaustive,
                "status": r.status,
                "gt_start": r.gt_start,
                "gt_end": r.gt_end,
                "algo_start": r.algo_start,
                "algo_end": r.algo_end,
                "start_delta": r.start_delta,
                "end_delta": r.end_delta,
            }
            match_rows.append(row)
            all_results["all"].append(r)
            if exhaustive:
                all_results["exhaustive"].append(r)
            else:
                all_results["non_exhaustive"].append(r)

        # Kinematic completeness
        comp_results, _ = compute_kinematic_completeness(
            gt_raw, algo, anchor_frames=2, window=WINDOW
        )
        for cr in comp_results:
            all_completeness["all"].append(cr)
            if exhaustive:
                all_completeness["exhaustive"].append(cr)
            else:
                all_completeness["non_exhaustive"].append(cr)

    output = {}
    for subset in ["all", "exhaustive", "non_exhaustive"]:
        output[subset] = {
            "standard": agg_standard(all_results[subset]),
            "completeness": agg_completeness(all_completeness[subset]),
        }

    n_videos_total = len(video_rows)
    n_videos_exhaustive = sum(1 for v in video_rows if v["exhaustive"])
    output["n_videos"] = n_videos_total
    output["n_videos_exhaustive"] = n_videos_exhaustive
    output["n_videos_non_exhaustive"] = n_videos_total - n_videos_exhaustive

    return output, match_rows, video_rows


def main():
    print("Loading GT data...")
    gt_data = load_gt()
    print(f"  {len(gt_data)} videos with GT reaches")

    print("Loading pre-DLC reaches from DB...")
    pre_dlc = load_pre_dlc_from_db(gt_data)
    print(f"  {len(pre_dlc)} / {len(gt_data)} GT videos have DB data")

    print("Loading post-DLC reaches from algo files...")
    post_dlc = load_post_dlc_from_algo(gt_data)
    print(f"  {len(post_dlc)} / {len(gt_data)} GT videos have algo data")

    print("\nComputing pre-DLC baseline metrics...")
    pre_metrics, pre_match_rows, pre_video_rows = compute_all_metrics(
        gt_data, pre_dlc, "pre_dlc"
    )

    print("Computing best post-DLC metrics...")
    post_metrics, post_match_rows, post_video_rows = compute_all_metrics(
        gt_data, post_dlc, "post_dlc"
    )

    # ---- Save pre-DLC snapshot ----
    SNAPSHOT_PRE.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_PRE / "metrics").mkdir(exist_ok=True)

    manifest_pre = {
        "version_id": "pre_dlc_baseline",
        "tag": "pre_dlc_production_from_connectome_db",
        "timestamp": "2026-04-24T00:00:00-05:00",
        "pipeline_versions": {
            "mousereach": "2.3.0",
            "dlc_scorer": "DLC_resnet50_MPSAOct27shuffle1_100000",
            "segmenter": "2.1.0",
            "reach_detector": "5.3.0",
            "outcome_detector": "2.4.4",
            "extractor": "1.0.0",
        },
        "inputs": [
            "Reach data from connectome.db (reach_data table, 357824 reaches total across 57 subjects)",
            "Subset matched to GT corpus: {} videos with reach data".format(pre_metrics["n_videos"]),
            "GT dir: Y:\\2_Connectome\\Validation_Runs\\DLC_2026_03_27\\gt",
            "Source: production pipeline with OLD DLC model (DLC_resnet50_MPSAOct27shuffle1_100000)",
        ],
        "metrics_summary": pre_metrics,
        "description": (
            "Pre-DLC-update baseline. Reaches extracted from connectome.db reach_data table, "
            "which was populated by the production pipeline using the original DLC model "
            "(DLC_resnet50_MPSAOct27shuffle1_100000) with reach_detector v5.3.0 and "
            "segmenter v2.1.0. These are the trusted production numbers that the "
            "post-DLC recalibration effort aims to recover and exceed."
        ),
    }

    with open(SNAPSHOT_PRE / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_pre, f, indent=2, ensure_ascii=True)

    with open(SNAPSHOT_PRE / "metrics" / "scalars.json", "w", encoding="utf-8") as f:
        json.dump(pre_metrics, f, indent=2, ensure_ascii=True)

    if pre_match_rows:
        keys = list(pre_match_rows[0].keys())
        with open(SNAPSHOT_PRE / "metrics" / "reach_matches.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(pre_match_rows)

    if pre_video_rows:
        keys = list(pre_video_rows[0].keys())
        with open(SNAPSHOT_PRE / "metrics" / "per_video.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(pre_video_rows)

    print(f"\nPre-DLC snapshot written to: {SNAPSHOT_PRE}")

    # ---- Augment post-DLC snapshot with completeness metrics ----
    completeness_file = SNAPSHOT_POST / "metrics" / "completeness_scalars.json"
    with open(completeness_file, "w", encoding="utf-8") as f:
        json.dump(post_metrics, f, indent=2, ensure_ascii=True)

    print(f"Post-DLC completeness metrics written to: {completeness_file}")

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("THREE-POINT BASELINE NUMBERS")
    print("=" * 80)

    for subset in ["exhaustive", "all", "non_exhaustive"]:
        pre_s = pre_metrics[subset]["standard"]
        post_s = post_metrics[subset]["standard"]
        pre_c = pre_metrics[subset]["completeness"]
        post_c = post_metrics[subset]["completeness"]

        if not pre_s and not post_s:
            continue

        n_pre_vid = pre_metrics.get(f"n_videos_{subset}", pre_metrics.get("n_videos", "?"))
        n_post_vid = post_metrics.get(f"n_videos_{subset}", post_metrics.get("n_videos", "?"))

        print(f"\n--- Subset: {subset} ---")
        print(f"  {'Metric':<30} {'Pre-DLC':>12} {'Post-DLC':>12} {'Delta':>12}")
        print(f"  {'-' * 66}")

        # Standard metrics
        for key in ["n_gt", "n_algo", "n_matched", "n_fp", "n_fn", "recall",
                     "fn_rate_pct", "mean_abs_start_delta", "mean_abs_end_delta",
                     "start_exact_0_pct", "end_exact_0_pct"]:
            pre_v = pre_s.get(key)
            post_v = post_s.get(key)
            pre_str = str(pre_v) if pre_v is not None else "N/A"
            post_str = str(post_v) if post_v is not None else "N/A"
            if pre_v is not None and post_v is not None:
                delta = post_v - pre_v
                delta_str = f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+d}"
            else:
                delta_str = ""
            print(f"  {key:<30} {pre_str:>12} {post_str:>12} {delta_str:>12}")

        print()
        # Completeness metrics
        for key in ["median_coverage", "mean_coverage", "frac_coverage_gte_1",
                     "frac_apex_included", "frac_anchor_start_ok", "frac_anchor_end_ok",
                     "frac_both_anchors_ok"]:
            pre_v = pre_c.get(key)
            post_v = post_c.get(key)
            pre_str = str(pre_v) if pre_v is not None else "N/A"
            post_str = str(post_v) if post_v is not None else "N/A"
            if pre_v is not None and post_v is not None:
                delta = post_v - pre_v
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = ""
            print(f"  {key:<30} {pre_str:>12} {post_str:>12} {delta_str:>12}")

    # ---- GT corpus overlap table ----
    print("\n" + "=" * 80)
    print("GT CORPUS OVERLAP TABLE")
    print("=" * 80)
    print(f"  {'video_id':<30} {'pre_dlc':>8} {'post_dlc':>9} {'n_pre':>6} {'n_post':>7} {'n_gt':>5} {'exhaust':>8}")
    print(f"  {'-' * 78}")

    for vid in sorted(gt_data.keys()):
        has_pre = vid in pre_dlc
        has_post = vid in post_dlc
        n_pre = len(pre_dlc[vid]) if has_pre else 0
        n_post = len(post_dlc[vid]) if has_post else 0
        n_gt = len(gt_data[vid]["reaches"])
        exh = gt_data[vid]["exhaustive"]
        print(f"  {vid:<30} {str(has_pre):>8} {str(has_post):>9} {n_pre:>6} {n_post:>7} {n_gt:>5} {str(exh):>8}")

    print(f"\n  Total GT videos: {len(gt_data)}")
    print(f"  Videos with pre-DLC data: {len(pre_dlc)}")
    print(f"  Videos with post-DLC data: {len(post_dlc)}")
    print(f"  Videos with BOTH: {len(set(pre_dlc.keys()) & set(post_dlc.keys()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
