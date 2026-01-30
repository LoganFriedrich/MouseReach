#!/usr/bin/env python3
"""Comprehensive comparison: Algorithm vs Human Ground Truth

Compares:
- Segment boundaries (frame accuracy)
- Outcome classification (accuracy, confusion matrix)
- Interaction frame detection (frame error)
- Reach count per segment
- Reach timing (start/apex/end frame errors) when GT available
"""

import json
from pathlib import Path
from collections import defaultdict
import csv

from mousereach.config import Paths

# Frame tolerance for "match"
FRAME_TOLERANCE = 5  # frames


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    PROCESSING = Paths.PROCESSING
    OUTPUT_DIR = PROCESSING.parent

    print("=" * 70)
    print("COMPREHENSIVE ALGORITHM vs HUMAN COMPARISON")
    print("=" * 70)

    # =========================================================================
    # 1. OUTCOME COMPARISON
    # =========================================================================
    outcome_rows = []
    outcome_confusion = defaultdict(lambda: defaultdict(int))
    outcome_total = 0
    outcome_correct = 0

    # Interaction frame errors
    interaction_errors = []

    gt_outcome_files = list(PROCESSING.glob("*_outcome_ground_truth.json"))
    print(f"\nFound {len(gt_outcome_files)} outcome ground truth files")

    for gt_file in sorted(gt_outcome_files):
        video = gt_file.stem.replace("_outcome_ground_truth", "")
        algo_file = PROCESSING / f"{video}_pellet_outcomes.json"

        gt_data = load_json(gt_file)
        algo_data = load_json(algo_file)

        if not gt_data or not algo_data:
            continue

        gt_segs = {s["segment_num"]: s for s in gt_data.get("segments", [])}
        algo_segs = {s["segment_num"]: s for s in algo_data.get("segments", [])}

        for seg_num in sorted(set(gt_segs.keys()) & set(algo_segs.keys())):
            gt_seg = gt_segs[seg_num]
            algo_seg = algo_segs[seg_num]

            gt_outcome = gt_seg.get("outcome", "unknown")
            algo_outcome = algo_seg.get("outcome", "unknown")
            match = gt_outcome == algo_outcome

            outcome_total += 1
            if match:
                outcome_correct += 1
            outcome_confusion[gt_outcome][algo_outcome] += 1

            # Interaction frame comparison
            gt_int = gt_seg.get("interaction_frame")
            algo_int = algo_seg.get("interaction_frame")
            int_error = None
            int_match = "N/A"
            if gt_int is not None and algo_int is not None:
                int_error = abs(algo_int - gt_int)
                int_match = "YES" if int_error <= FRAME_TOLERANCE else "NO"
                interaction_errors.append(int_error)
            elif gt_int is None and algo_int is None:
                int_match = "BOTH_NONE"

            outcome_rows.append({
                "video": video,
                "segment": seg_num,
                "human_outcome": gt_outcome,
                "algo_outcome": algo_outcome,
                "outcome_match": "YES" if match else "NO",
                "human_interaction_frame": gt_int if gt_int else "",
                "algo_interaction_frame": algo_int if algo_int else "",
                "interaction_frame_error": int_error if int_error is not None else "",
                "interaction_match": int_match,
            })

    # Write outcome CSV
    if outcome_rows:
        out_path = OUTPUT_DIR / "comparison_outcomes.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=outcome_rows[0].keys())
            writer.writeheader()
            writer.writerows(outcome_rows)
        print(f"Wrote: {out_path}")

    # =========================================================================
    # 2. SEGMENT BOUNDARY COMPARISON
    # =========================================================================
    seg_rows = []
    boundary_errors = []

    gt_seg_files = list(PROCESSING.glob("*_seg_ground_truth.json"))
    print(f"Found {len(gt_seg_files)} segmentation ground truth files")

    for gt_file in sorted(gt_seg_files):
        video = gt_file.stem.replace("_seg_ground_truth", "")

        # Try to find algo segmentation
        algo_file = PROCESSING / f"{video}_segmentation.json"
        if not algo_file.exists():
            algo_file = PROCESSING / f"{video}_segments.json"

        gt_data = load_json(gt_file)
        algo_data = load_json(algo_file)

        if not gt_data or not algo_data:
            continue

        gt_bounds = gt_data.get("boundaries", [])
        # Algo segmentation stores boundaries differently
        algo_bounds = algo_data.get("weirdness_frames", [])
        if not algo_bounds:
            # Try pellet_segments
            segs = algo_data.get("pellet_segments", [])
            if segs:
                algo_bounds = sorted(set(
                    [s.get("start_frame", 0) for s in segs] +
                    [s.get("end_frame", 0) for s in segs]
                ))

        # Compare boundary counts
        seg_rows.append({
            "video": video,
            "human_n_boundaries": len(gt_bounds),
            "algo_n_boundaries": len(algo_bounds),
            "boundary_count_match": "YES" if len(gt_bounds) == len(algo_bounds) else "NO",
            "human_boundaries": str(gt_bounds[:5]) + "..." if len(gt_bounds) > 5 else str(gt_bounds),
            "algo_boundaries": str(algo_bounds[:5]) + "..." if len(algo_bounds) > 5 else str(algo_bounds),
        })

        # Calculate boundary errors (for matching boundaries)
        for i, gt_b in enumerate(gt_bounds):
            if i < len(algo_bounds):
                err = abs(algo_bounds[i] - gt_b)
                boundary_errors.append(err)

    if seg_rows:
        out_path = OUTPUT_DIR / "comparison_segments.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=seg_rows[0].keys())
            writer.writeheader()
            writer.writerows(seg_rows)
        print(f"Wrote: {out_path}")

    # =========================================================================
    # 3. REACH COMPARISON
    # =========================================================================
    reach_rows = []
    reach_count_errors = []
    apex_errors = []
    start_errors = []
    end_errors = []

    gt_reach_files = list(PROCESSING.glob("*_reach_ground_truth.json"))
    print(f"Found {len(gt_reach_files)} reach ground truth files")

    for gt_file in sorted(gt_reach_files):
        video = gt_file.stem.replace("_reach_ground_truth", "")
        algo_file = PROCESSING / f"{video}_reaches.json"

        gt_data = load_json(gt_file)
        algo_data = load_json(algo_file)

        if not gt_data or not algo_data:
            continue

        gt_segs = {s["segment_num"]: s for s in gt_data.get("segments", [])}
        algo_segs = {s["segment_num"]: s for s in algo_data.get("segments", [])}

        for seg_num in sorted(set(gt_segs.keys()) & set(algo_segs.keys())):
            gt_seg = gt_segs[seg_num]
            algo_seg = algo_segs[seg_num]

            gt_reaches = gt_seg.get("reaches", [])
            algo_reaches = algo_seg.get("reaches", [])

            gt_n = len(gt_reaches)
            algo_n = len(algo_reaches)
            reach_count_errors.append(abs(gt_n - algo_n))

            reach_rows.append({
                "video": video,
                "segment": seg_num,
                "human_reach_count": gt_n,
                "algo_reach_count": algo_n,
                "reach_count_diff": algo_n - gt_n,
                "count_match": "YES" if gt_n == algo_n else "NO",
            })

            # Compare individual reaches by apex frame proximity
            for gt_r in gt_reaches:
                gt_apex = gt_r.get("apex_frame")
                if gt_apex is None:
                    continue

                # Find closest algo reach
                best_match = None
                best_dist = float('inf')
                for algo_r in algo_reaches:
                    algo_apex = algo_r.get("apex_frame")
                    if algo_apex is None:
                        continue
                    dist = abs(algo_apex - gt_apex)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = algo_r

                if best_match and best_dist <= 30:  # within 30 frames = likely same reach
                    apex_errors.append(best_dist)
                    if gt_r.get("start_frame") and best_match.get("start_frame"):
                        start_errors.append(abs(best_match["start_frame"] - gt_r["start_frame"]))
                    if gt_r.get("end_frame") and best_match.get("end_frame"):
                        end_errors.append(abs(best_match["end_frame"] - gt_r["end_frame"]))

    if reach_rows:
        out_path = OUTPUT_DIR / "comparison_reaches.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=reach_rows[0].keys())
            writer.writeheader()
            writer.writerows(reach_rows)
        print(f"Wrote: {out_path}")

    # =========================================================================
    # 4. SUMMARY STATISTICS
    # =========================================================================
    summary_rows = []

    # Outcome accuracy
    if outcome_total > 0:
        accuracy = outcome_correct / outcome_total * 100
        summary_rows.append({"metric": "Outcome Accuracy", "value": f"{accuracy:.1f}%", "n": outcome_total})

        # Per-class accuracy
        all_outcomes = sorted(set(outcome_confusion.keys()) |
                             {k for d in outcome_confusion.values() for k in d.keys()})
        for outcome in all_outcomes:
            total = sum(outcome_confusion[outcome].values())
            correct = outcome_confusion[outcome].get(outcome, 0)
            if total > 0:
                acc = correct / total * 100
                summary_rows.append({"metric": f"  {outcome} accuracy", "value": f"{acc:.1f}%", "n": total})

    # Interaction frame accuracy
    if interaction_errors:
        mean_err = sum(interaction_errors) / len(interaction_errors)
        within_tol = sum(1 for e in interaction_errors if e <= FRAME_TOLERANCE)
        summary_rows.append({"metric": "Interaction Frame Mean Error", "value": f"{mean_err:.1f} frames", "n": len(interaction_errors)})
        summary_rows.append({"metric": f"Interaction Within {FRAME_TOLERANCE} Frames", "value": f"{within_tol}/{len(interaction_errors)}", "n": len(interaction_errors)})

    # Segment boundary accuracy
    if boundary_errors:
        mean_err = sum(boundary_errors) / len(boundary_errors)
        summary_rows.append({"metric": "Boundary Mean Error", "value": f"{mean_err:.1f} frames", "n": len(boundary_errors)})

    # Reach count accuracy
    if reach_count_errors:
        exact_match = sum(1 for e in reach_count_errors if e == 0)
        summary_rows.append({"metric": "Reach Count Exact Match", "value": f"{exact_match}/{len(reach_count_errors)}", "n": len(reach_count_errors)})

    # Reach timing accuracy
    if apex_errors:
        mean_apex = sum(apex_errors) / len(apex_errors)
        summary_rows.append({"metric": "Reach Apex Mean Error", "value": f"{mean_apex:.1f} frames", "n": len(apex_errors)})
    if start_errors:
        mean_start = sum(start_errors) / len(start_errors)
        summary_rows.append({"metric": "Reach Start Mean Error", "value": f"{mean_start:.1f} frames", "n": len(start_errors)})
    if end_errors:
        mean_end = sum(end_errors) / len(end_errors)
        summary_rows.append({"metric": "Reach End Mean Error", "value": f"{mean_end:.1f} frames", "n": len(end_errors)})

    # Write summary
    if summary_rows:
        out_path = OUTPUT_DIR / "comparison_summary.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value", "n"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote: {out_path}")

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for row in summary_rows:
        print(f"{row['metric']:40s} {row['value']:>15s}  (n={row['n']})")

    # Confusion matrix
    if outcome_confusion:
        print("\n" + "=" * 70)
        print("OUTCOME CONFUSION MATRIX (rows=human, cols=algorithm)")
        print("=" * 70)
        all_outcomes = sorted(set(outcome_confusion.keys()) |
                             {k for d in outcome_confusion.values() for k in d.keys()})

        # Header
        header = f"{'Human':20s}"
        for o in all_outcomes:
            header += f" {o[:12]:>12s}"
        print(header)
        print("-" * (20 + 13 * len(all_outcomes)))

        for human in all_outcomes:
            row = f"{human:20s}"
            for algo in all_outcomes:
                count = outcome_confusion[human].get(algo, 0)
                row += f" {count:12d}"
            print(row)

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("  comparison_outcomes.csv  - Per-segment outcome comparison")
    print("  comparison_segments.csv  - Segment boundary comparison")
    print("  comparison_reaches.csv   - Reach count per segment")
    print("  comparison_summary.csv   - Overall accuracy metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
