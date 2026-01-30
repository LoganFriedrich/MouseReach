"""Quick direct evaluation - evaluate GT files against algorithm output."""
import json
import traceback
from pathlib import Path

from mousereach.config import require_processing_root


def main():
    """Run direct evaluation of ground truth files."""
    try:
        processing_dir = require_processing_root() / "Processing"

        # Find unified GT file
        unified_gt_files = list(processing_dir.glob('*_unified_ground_truth.json'))
        reach_gt_files = list(processing_dir.glob('*_reach_ground_truth.json'))

        print("="*70)
        print("GT FILES FOUND")
        print("="*70)
        print(f"Unified GT: {len(unified_gt_files)}")
        for f in unified_gt_files:
            print(f"  - {f.name}")
        print(f"Reach GT: {len(reach_gt_files)}")
        for f in reach_gt_files:
            print(f"  - {f.name}")
        print()

        # Evaluate the unified GT file - ALL FEATURES
        for gt_file in unified_gt_files:
            print("="*70)
            print(f"EVALUATING: {gt_file.name}")
            print("="*70)

            with open(gt_file) as f:
                gt_data = json.load(f)

            video_name = gt_data['video_name']

            # Get algorithm reaches file
            algo_file = processing_dir / f"{video_name}_reaches.json"
            if not algo_file.exists():
                print(f"  WARNING: No algorithm file found: {algo_file.name}")
                continue

            with open(algo_file) as f:
                algo_data = json.load(f)

            # ================================================================
            # SEGMENT BOUNDARIES
            # ================================================================
            print("-"*70)
            print("SEGMENT BOUNDARIES")
            print("-"*70)

            # GT boundaries = the truth (file only contains human-interacted items)
            gt_boundaries = gt_data.get('segmentation', {}).get('boundaries', [])
            gt_boundary_frames = [b['frame'] for b in gt_boundaries]

            # Get algorithm segments file
            seg_file = processing_dir / f"{video_name}_segments.json"
            if seg_file.exists():
                with open(seg_file) as f:
                    seg_data = json.load(f)
                algo_boundary_frames = seg_data.get('boundaries', [])

                print(f"  GT boundaries: {len(gt_boundary_frames)}")
                print(f"  Algorithm boundaries: {len(algo_boundary_frames)}")

                # Match boundaries within tolerance
                BOUNDARY_TOLERANCE = 5  # frames
                boundary_matches = 0
                boundary_errors = []

                for gt_frame in gt_boundary_frames:
                    for algo_frame in algo_boundary_frames:
                        if abs(gt_frame - algo_frame) <= BOUNDARY_TOLERANCE:
                            boundary_matches += 1
                            boundary_errors.append(algo_frame - gt_frame)
                            break

                if gt_boundary_frames:
                    boundary_recall = boundary_matches / len(gt_boundary_frames)
                    print(f"  Matched: {boundary_matches}/{len(gt_boundary_frames)} ({boundary_recall:.1%})")

                    if boundary_errors:
                        exact = sum(1 for e in boundary_errors if e == 0)
                        within_1 = sum(1 for e in boundary_errors if abs(e) <= 1)
                        within_2 = sum(1 for e in boundary_errors if abs(e) <= 2)
                        print(f"  Timing: {exact}/{len(boundary_errors)} exact, {within_1} ±1fr, {within_2} ±2fr")
                        avg_err = sum(boundary_errors) / len(boundary_errors)
                        print(f"  Average error: {avg_err:+.1f} frames")
            else:
                print(f"  No segments file found: {seg_file.name}")
            print()

            # ================================================================
            # OUTCOMES
            # ================================================================
            print("-"*70)
            print("OUTCOMES")
            print("-"*70)

            # GT outcomes = the truth
            gt_outcomes = gt_data.get('outcomes', {}).get('segments', [])

            # Get algorithm outcomes file
            outcome_file = processing_dir / f"{video_name}_pellet_outcomes.json"
            if outcome_file.exists():
                with open(outcome_file) as f:
                    outcome_data = json.load(f)
                algo_outcomes = outcome_data.get('segments', [])

                print(f"  GT outcomes: {len(gt_outcomes)}")
                print(f"  Algorithm outcomes: {len(algo_outcomes)}")

                # Compare outcomes by segment index
                outcome_matches = 0
                outcome_mismatches = []

                for gt_o in gt_outcomes:
                    # Support both unified format (segment_num) and old format (segment_index)
                    gt_seg = gt_o.get('segment_num') or gt_o.get('segment_index')
                    gt_class = gt_o.get('outcome')

                    # Find matching algo outcome
                    for algo_o in algo_outcomes:
                        algo_seg = algo_o.get('segment_num') or algo_o.get('segment_index')
                        if algo_seg == gt_seg:
                            algo_class = algo_o.get('outcome')
                            if gt_class == algo_class:
                                outcome_matches += 1
                            else:
                                outcome_mismatches.append({
                                    'segment': gt_seg,
                                    'gt': gt_class,
                                    'algo': algo_class
                                })
                            break

                if gt_outcomes:
                    outcome_accuracy = outcome_matches / len(gt_outcomes)
                    print(f"  Correct: {outcome_matches}/{len(gt_outcomes)} ({outcome_accuracy:.1%})")

                    if outcome_mismatches:
                        print(f"  Misclassified ({len(outcome_mismatches)}):")
                        for m in outcome_mismatches[:10]:
                            print(f"    Segment {m['segment']}: GT={m['gt']}, Algo={m['algo']}")
            else:
                print(f"  No outcomes file found: {outcome_file.name}")
            print()

            # ================================================================
            # REACHES
            # ================================================================
            print("-"*70)
            print("REACHES")
            print("-"*70)

            # GT reaches = the truth (the file content IS ground truth)
            gt_reaches = gt_data.get('reaches', {}).get('reaches', [])

            print(f"  GT reaches: {len(gt_reaches)}")

            # Count algorithm reaches
            algo_reaches = []
            for seg in algo_data.get('segments', []):
                algo_reaches.extend(seg.get('reaches', []))

            print(f"  Algorithm reaches: {len(algo_reaches)}")
            print()

            # Match GT to algorithm (within 10 frame tolerance)
            TOLERANCE = 10

            matches = 0
            gt_matched = set()
            algo_matched = set()

            for i, gt_r in enumerate(gt_reaches):
                gt_start = gt_r['start_frame']
                gt_end = gt_r['end_frame']

                for j, algo_r in enumerate(algo_reaches):
                    if j in algo_matched:
                        continue

                    algo_start = algo_r['start_frame']
                    algo_end = algo_r['end_frame']

                    # Check overlap / proximity
                    start_diff = abs(gt_start - algo_start)
                    end_diff = abs(gt_end - algo_end)

                    if start_diff <= TOLERANCE and end_diff <= TOLERANCE:
                        matches += 1
                        gt_matched.add(i)
                        algo_matched.add(j)
                        break

            # Calculate metrics
            tp = matches
            fn = len(gt_reaches) - matches  # GT reaches not matched
            fp = len(algo_reaches) - matches  # Algo reaches not matched to GT

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  RESULTS (tolerance={TOLERANCE} frames):")
            print(f"    True Positives (matches): {tp}")
            print(f"    False Negatives (missed): {fn}")
            print(f"    False Positives (extra): {fp}")
            print()
            print(f"    Precision: {precision:.1%}")
            print(f"    Recall:    {recall:.1%}")
            print(f"    F1 Score:  {f1:.2f}")
            print()

            # Analyze timing errors for matches
            if matches > 0:
                start_errors = []
                end_errors = []

                for i, gt_r in enumerate(gt_reaches):
                    if i not in gt_matched:
                        continue
                    gt_start = gt_r['start_frame']
                    gt_end = gt_r['end_frame']

                    # Find the matched algo reach
                    for j, algo_r in enumerate(algo_reaches):
                        algo_start = algo_r['start_frame']
                        algo_end = algo_r['end_frame']

                        start_diff = abs(gt_start - algo_start)
                        end_diff = abs(gt_end - algo_end)

                        if start_diff <= TOLERANCE and end_diff <= TOLERANCE:
                            start_errors.append(algo_start - gt_start)
                            end_errors.append(algo_end - gt_end)
                            break

                print("  TIMING ACCURACY:")
                # Start timing
                exact_start = sum(1 for e in start_errors if e == 0)
                within_1_start = sum(1 for e in start_errors if abs(e) <= 1)
                within_2_start = sum(1 for e in start_errors if abs(e) <= 2)
                within_5_start = sum(1 for e in start_errors if abs(e) <= 5)

                print(f"    Start frame accuracy:")
                print(f"      Exact:    {exact_start}/{len(start_errors)} ({exact_start/len(start_errors)*100:.1f}%)")
                print(f"      ±1 frame: {within_1_start}/{len(start_errors)} ({within_1_start/len(start_errors)*100:.1f}%)")
                print(f"      ±2 frame: {within_2_start}/{len(start_errors)} ({within_2_start/len(start_errors)*100:.1f}%)")
                print(f"      ±5 frame: {within_5_start}/{len(start_errors)} ({within_5_start/len(start_errors)*100:.1f}%)")

                # End timing
                exact_end = sum(1 for e in end_errors if e == 0)
                within_1_end = sum(1 for e in end_errors if abs(e) <= 1)
                within_2_end = sum(1 for e in end_errors if abs(e) <= 2)
                within_5_end = sum(1 for e in end_errors if abs(e) <= 5)

                print(f"    End frame accuracy:")
                print(f"      Exact:    {exact_end}/{len(end_errors)} ({exact_end/len(end_errors)*100:.1f}%)")
                print(f"      ±1 frame: {within_1_end}/{len(end_errors)} ({within_1_end/len(end_errors)*100:.1f}%)")
                print(f"      ±2 frame: {within_2_end}/{len(end_errors)} ({within_2_end/len(end_errors)*100:.1f}%)")
                print(f"      ±5 frame: {within_5_end}/{len(end_errors)} ({within_5_end/len(end_errors)*100:.1f}%)")

                # Average errors
                avg_start_err = sum(start_errors) / len(start_errors)
                avg_end_err = sum(end_errors) / len(end_errors)
                print(f"    Average start error: {avg_start_err:+.1f} frames")
                print(f"    Average end error:   {avg_end_err:+.1f} frames")

            print()

            # Show missed reaches (False Negatives)
            if fn > 0:
                print("  MISSED REACHES (algorithm didn't detect):")
                missed_count = 0
                missed_extents = []
                missed_durations = []
                for i, gt_r in enumerate(gt_reaches):
                    if i not in gt_matched:
                        missed_count += 1
                        seg_idx = gt_r.get('segment_index', '?')
                        reach_id = gt_r.get('id', gt_r.get('reach_id', i))
                        extent = gt_r.get('max_extent_pixels', '?')
                        duration = gt_r['end_frame'] - gt_r['start_frame']
                        if isinstance(extent, (int, float)):
                            missed_extents.append(extent)
                        missed_durations.append(duration)
                        if missed_count <= 10:  # Show first 10
                            print(f"    Reach {reach_id}: frames {gt_r['start_frame']}-{gt_r['end_frame']} (duration={duration}, extent={extent})")
                if missed_count > 10:
                    print(f"    ... and {missed_count - 10} more")

                # Analyze patterns in missed reaches
                if missed_extents:
                    avg_extent = sum(missed_extents) / len(missed_extents)
                    min_extent = min(missed_extents)
                    max_extent = max(missed_extents)
                    print(f"    Missed reach extents: avg={avg_extent:.1f}, min={min_extent:.1f}, max={max_extent:.1f}")
                if missed_durations:
                    avg_dur = sum(missed_durations) / len(missed_durations)
                    min_dur = min(missed_durations)
                    max_dur = max(missed_durations)
                    print(f"    Missed reach durations: avg={avg_dur:.1f}, min={min_dur}, max={max_dur}")
                print()

            # Show false positives
            if fp > 0:
                print("  FALSE POSITIVES (algorithm detected, not in GT):")
                fp_count = 0
                for j, algo_r in enumerate(algo_reaches):
                    if j not in algo_matched:
                        fp_count += 1
                        if fp_count <= 10:
                            print(f"    Reach {algo_r.get('reach_id', j)}: frames {algo_r['start_frame']}-{algo_r['end_frame']}")
                if fp_count > 10:
                    print(f"    ... and {fp_count - 10} more")
                print()

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
