"""Pull current state of the data from v8.0.4 manifests, broken down by
video (mouse). Reports filtered topology counts per video for both
calibration LOOCV and holdout corpora.

Filters applied (per the production reporting convention):
  - span >= MIN_REPORTED_SPAN=4
  - NOT outside_gt_segmentation
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

MANIFEST_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.0.3"
)
CORPORA = ["calibration_loocv", "holdout_2026_05_11"]


def main():
    for corpus in CORPORA:
        per_video = defaultdict(lambda: defaultdict(int))
        per_video_gt_count = defaultdict(int)
        for f in sorted((MANIFEST_ROOT / corpus).glob("*.json")):
            d = json.load(open(f))
            vid = d["video_id"]
            seen_comp = set()
            for ev in d.get("events", []):
                if ev.get("kinematically_excluded") or ev.get("outside_gt_segmentation"):
                    continue
                topo = ev.get("topology")
                if topo is None:
                    continue
                if topo in ("MERGED", "FRAGMENTED"):
                    cid = ev.get("component_id")
                    if cid in seen_comp:
                        continue
                    seen_comp.add(cid)
                    per_video[vid][topo] += 1
                elif topo == "TP":
                    per_video[vid]["TP"] += 1
                elif topo == "TOLERANCE_ERROR":
                    if ev.get("kind") == "FN":
                        per_video[vid]["TOL_pair"] += 1
                elif topo == "FALSE_POSITIVE":
                    per_video[vid]["FP"] += 1
                elif topo == "FALSE_NEGATIVE":
                    per_video[vid]["FN"] += 1
            # Count GTs from events
            for ev in d.get("events", []):
                if ev.get("kind") in ("TP", "FN") and ev.get("gt"):
                    per_video_gt_count[vid] += 1
        # Totals
        total_gt = 0
        total = defaultdict(int)
        rows = []
        for vid in sorted(per_video.keys()):
            r = per_video[vid]
            n_gt = per_video_gt_count[vid]
            total_errors = (r.get("TOL_pair", 0) + r.get("MERGED", 0) +
                            r.get("FRAGMENTED", 0) + r.get("FP", 0) +
                            r.get("FN", 0))
            rows.append({
                "vid": vid, "n_gt": n_gt,
                "TP": r.get("TP", 0),
                "TOL": r.get("TOL_pair", 0),
                "MERGED": r.get("MERGED", 0),
                "FRAG": r.get("FRAGMENTED", 0),
                "FP": r.get("FP", 0),
                "FN": r.get("FN", 0),
                "errors": total_errors,
            })
            total_gt += n_gt
            for k in ("TP", "TOL", "MERGED", "FRAG", "FP", "FN", "errors"):
                total[k] += rows[-1][k]
        # Sort by total errors descending
        rows.sort(key=lambda x: -x["errors"])

        print(f"\n=== {corpus.upper()} (v8.0.4 filtered topology, sorted by total errors desc) ===")
        print(f"{'video':<22} {'n_GT':>5} {'TP':>5} {'TOL':>4} {'MGD':>4} {'FRG':>4} "
              f"{'FP':>4} {'FN':>4} {'errors':>7}")
        print("-" * 75)
        for r in rows:
            print(f"{r['vid']:<22} {r['n_gt']:>5} {r['TP']:>5} {r['TOL']:>4} "
                  f"{r['MERGED']:>4} {r['FRAG']:>4} {r['FP']:>4} {r['FN']:>4} "
                  f"{r['errors']:>7}")
        print("-" * 75)
        print(f"{'TOTAL':<22} {total_gt:>5} {total['TP']:>5} {total['TOL']:>4} "
              f"{total['MERGED']:>4} {total['FRAG']:>4} {total['FP']:>4} "
              f"{total['FN']:>4} {total['errors']:>7}")


if __name__ == "__main__":
    main()
