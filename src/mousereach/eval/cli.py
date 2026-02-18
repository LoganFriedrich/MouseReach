"""
MouseReach Algorithm Evaluation CLI

Usage:
    mousereach-eval --seg [path]      Evaluate segmentation algorithm
    mousereach-eval --reach [path]    Evaluate reach detection algorithm
    mousereach-eval --outcome [path]  Evaluate outcome classification
    mousereach-eval --all [path]      Evaluate all algorithms

Examples:
    mousereach-eval --seg dev_SampleData/
    mousereach-eval --all --tolerance 10
    mousereach-eval --reach --output report.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from mousereach.config import Paths


def find_evaluation_dirs(base_path: Optional[Path] = None):
    """
    Find GT and algo directories for evaluation.

    If base_path is provided, look for GT/algo files there.
    Otherwise, use the processing root.
    """
    if base_path:
        base_path = Path(base_path)
        if not base_path.exists():
            print(f"Error: Path does not exist: {base_path}")
            sys.exit(1)
        return base_path, base_path

    # Default to processing root
    processing_root = Paths.PROCESSING_ROOT
    if not processing_root.exists():
        print(f"Error: Processing root does not exist: {processing_root}")
        print("Set MouseReach_PROCESSING_ROOT or provide a path.")
        sys.exit(1)

    return processing_root, processing_root


def evaluate_segmentation(gt_dir: Path, algo_dir: Path, tolerance: int = 5, output: Optional[Path] = None):
    """Run segmentation evaluation."""
    from mousereach.eval import SegmentationEvaluator

    print("=" * 60)
    print("SEGMENTATION EVALUATION")
    print("=" * 60)
    print(f"GT directory:   {gt_dir}")
    print(f"Algo directory: {algo_dir}")
    print(f"Tolerance:      +/-{tolerance} frames")
    print()

    evaluator = SegmentationEvaluator(gt_dir, algo_dir, tolerance)
    results = evaluator.evaluate_all()

    if not results:
        print("No videos found with both GT and algorithm output.")
        return

    report = evaluator.generate_report()
    print(report)

    if output:
        with open(output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output}")


def evaluate_reaches(gt_dir: Path, algo_dir: Path, tolerance: int = 10, output: Optional[Path] = None):
    """Run reach detection evaluation."""
    from mousereach.eval import ReachEvaluator

    print("=" * 60)
    print("REACH DETECTION EVALUATION")
    print("=" * 60)
    print(f"GT directory:   {gt_dir}")
    print(f"Algo directory: {algo_dir}")
    print(f"Tolerance:      +/-{tolerance} frames")
    print()

    evaluator = ReachEvaluator(gt_dir, algo_dir, tolerance)
    results = evaluator.evaluate_all()

    if not results:
        print("No videos found with both GT and algorithm output.")
        return

    report = evaluator.generate_report()
    print(report)

    if output:
        with open(output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output}")


def evaluate_outcomes(gt_dir: Path, algo_dir: Path, tolerance: int = 15, output: Optional[Path] = None):
    """Run outcome classification evaluation."""
    from mousereach.eval import OutcomeEvaluator

    print("=" * 60)
    print("OUTCOME CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"GT directory:   {gt_dir}")
    print(f"Algo directory: {algo_dir}")
    print(f"Frame tolerance: +/-{tolerance} frames (for interaction timing)")
    print()

    evaluator = OutcomeEvaluator(gt_dir, algo_dir, tolerance)
    results = evaluator.evaluate_all()

    if not results:
        print("No videos found with both GT and algorithm output.")
        return

    report = evaluator.generate_report()
    print(report)

    if output:
        with open(output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MouseReach Algorithm Evaluation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mousereach-eval --seg                     # Evaluate segmentation in processing root
    mousereach-eval --seg dev_SampleData/     # Evaluate segmentation in specific folder
    mousereach-eval --all --tolerance 10      # Evaluate all with custom tolerance
    mousereach-eval --reach -o report.txt     # Save report to file

The evaluator looks for GT files (*_ground_truth.json) and algorithm output
files (*_segments.json, *_reaches.json, *_pellet_outcomes.json) in the
specified directory or the MouseReach processing root.
        """
    )

    # Algorithm selection (mutually exclusive)
    algo_group = parser.add_mutually_exclusive_group(required=True)
    algo_group.add_argument(
        "--seg", "-s",
        action="store_true",
        help="Evaluate segmentation algorithm"
    )
    algo_group.add_argument(
        "--reach", "-r",
        action="store_true",
        help="Evaluate reach detection algorithm"
    )
    algo_group.add_argument(
        "--outcome", "-o",
        action="store_true",
        help="Evaluate outcome classification algorithm"
    )
    algo_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Evaluate all algorithms"
    )

    # Optional arguments
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Directory containing GT and algorithm files (default: processing root)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=int,
        default=None,
        help="Frame tolerance for matching (default: varies by algorithm)"
    )
    parser.add_argument(
        "--output", "--save",
        type=Path,
        default=None,
        help="Save report to file"
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Separate GT directory (if different from algo dir)"
    )
    parser.add_argument(
        "--algo-dir",
        type=Path,
        default=None,
        help="Separate algorithm output directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Determine directories
    if args.gt_dir and args.algo_dir:
        gt_dir = args.gt_dir
        algo_dir = args.algo_dir
    else:
        gt_dir, algo_dir = find_evaluation_dirs(args.path)

    # Run evaluations
    if args.seg:
        tolerance = args.tolerance if args.tolerance else 5
        evaluate_segmentation(gt_dir, algo_dir, tolerance, args.output)

    elif args.reach:
        tolerance = args.tolerance if args.tolerance else 10
        evaluate_reaches(gt_dir, algo_dir, tolerance, args.output)

    elif args.outcome:
        tolerance = args.tolerance if args.tolerance else 15
        evaluate_outcomes(gt_dir, algo_dir, tolerance, args.output)

    elif args.all:
        # Run all evaluations
        print("\n" + "=" * 60)
        print("FULL ALGORITHM EVALUATION")
        print("=" * 60)

        seg_tol = args.tolerance if args.tolerance else 5
        reach_tol = args.tolerance if args.tolerance else 10
        outcome_tol = args.tolerance if args.tolerance else 15

        print("\n[1/3] Segmentation...")
        evaluate_segmentation(gt_dir, algo_dir, seg_tol)

        print("\n[2/3] Reach Detection...")
        evaluate_reaches(gt_dir, algo_dir, reach_tol)

        print("\n[3/3] Outcome Classification...")
        evaluate_outcomes(gt_dir, algo_dir, outcome_tol)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
