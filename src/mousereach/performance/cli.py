"""
CLI commands for performance tracking.

Commands:
    mousereach-perf           View performance summary
    mousereach-perf-eval      Run batch evaluation against ground truth
    mousereach-perf-report    Generate scientific report
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .logger import PerformanceLogger


def main_view():
    """View performance summary (mousereach-perf command)."""
    parser = argparse.ArgumentParser(
        description="View algorithm performance summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mousereach-perf                    Show all algorithms
  mousereach-perf --algo reach       Show reach detection only
  mousereach-perf --since 2026-01-01 Show entries since date
  mousereach-perf --json             Output as JSON
        """
    )
    parser.add_argument(
        "--algo", "-a",
        choices=["segmentation", "reach", "outcome", "all"],
        default="all",
        help="Algorithm to show (default: all)"
    )
    parser.add_argument(
        "--since", "-s",
        help="Show entries since date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show per-video details"
    )

    args = parser.parse_args()

    logger = PerformanceLogger()

    if args.algo == "all":
        algos = ["segmentation", "reach", "outcome"]
    else:
        algos = [args.algo]

    if args.json:
        output = {}
        for algo in algos:
            summary = logger.get_summary(algo)
            entries = logger.get_entries(algo, since=args.since)
            output[algo] = {
                "summary": summary.get(algo if algo != "reach" else "reach_detection", {}),
                "n_entries": len(entries)
            }
            if args.detailed:
                output[algo]["entries"] = entries
        print(json.dumps(output, indent=2, default=str))
        return

    # Text output
    print()
    print("=" * 60)
    print("MouseReach Algorithm Performance Summary")
    print("=" * 60)
    print()

    for algo in algos:
        summary_key = "reach_detection" if algo == "reach" else algo
        summary_key = "outcome_classification" if algo == "outcome" else summary_key
        summary = logger.get_summary(algo).get(summary_key, {})

        if not summary or summary.get("n_videos", 0) == 0:
            print(f"{algo.upper().replace('_', ' ')}")
            print("  No performance data yet.")
            print()
            continue

        print(f"{algo.upper().replace('_', ' ')}" +
              (f" (v{summary.get('algorithm_version', '?')})" if summary.get('algorithm_version') else ""))
        print("-" * 40)
        print(f"  Videos validated: {summary.get('n_videos', 0)}")

        if algo == "segmentation":
            print(f"  Mean accuracy: {summary.get('mean_accuracy', 0):.1%}")
            print(f"  Mean error: {summary.get('mean_error_frames', 0):.1f} frames")
            if summary.get('total_missed', 0) > 0:
                print(f"  Total missed boundaries: {summary.get('total_missed', 0)}")
            if summary.get('total_extra', 0) > 0:
                print(f"  Total extra boundaries: {summary.get('total_extra', 0)}")

        elif algo == "reach":
            print(f"  Mean F1: {summary.get('mean_f1', 0):.2f} (+/- {summary.get('std_f1', 0):.2f})")
            print(f"  Mean precision: {summary.get('mean_precision', 0):.2f}")
            print(f"  Mean recall: {summary.get('mean_recall', 0):.2f}")
            print()
            print("  Error breakdown:")
            print(f"    Missed reaches (FN): {summary.get('total_missed', 0)}")
            print(f"    Extra reaches (FP): {summary.get('total_extra', 0)}")
            print(f"    Timing corrections: {summary.get('total_corrected', 0)}")

        elif algo == "outcome":
            print(f"  Mean accuracy: {summary.get('mean_accuracy', 0):.1%}")
            print(f"  Total correct: {summary.get('total_correct', 0)}")
            print(f"  Total incorrect: {summary.get('total_incorrect', 0)}")

        if summary.get('last_updated'):
            print(f"  Last updated: {summary.get('last_updated', '')[:19]}")

        print()

        # Show detailed entries if requested
        if args.detailed:
            entries = logger.get_entries(algo, since=args.since)
            if entries:
                print("  Recent entries:")
                for e in entries[-5:]:
                    video = e.get('video_id', '?')
                    metrics = e.get('metrics', {})
                    if algo == "segmentation":
                        print(f"    {video}: {metrics.get('accuracy', 0):.0%} acc")
                    elif algo == "reach":
                        print(f"    {video}: F1={metrics.get('f1', 0):.2f}")
                    elif algo == "outcome":
                        print(f"    {video}: {metrics.get('accuracy', 0):.0%} acc")
                print()


def main_eval():
    """Run batch evaluation (mousereach-perf-eval command)."""
    parser = argparse.ArgumentParser(
        description="Run batch evaluation against ground truth files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mousereach-perf-eval --algo reach
  mousereach-perf-eval --algo all --gt-dir Processing/
        """
    )
    parser.add_argument(
        "--algo", "-a",
        choices=["segmentation", "reach", "outcome", "all"],
        default="all",
        help="Algorithm to evaluate (default: all)"
    )
    parser.add_argument(
        "--gt-dir", "-g",
        help="Directory containing ground truth files"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=int,
        default=5,
        help="Frame tolerance for matching (default: 5)"
    )

    args = parser.parse_args()

    from mousereach.config import Paths

    gt_dir = Path(args.gt_dir) if args.gt_dir else Paths.PROCESSING

    if args.algo == "all":
        algos = ["segmentation", "reach", "outcome"]
    else:
        algos = [args.algo]

    print()
    print("Running batch evaluation...")
    print(f"Ground truth directory: {gt_dir}")
    print()

    for algo in algos:
        print(f"Evaluating {algo}...")

        # Import the appropriate evaluator
        if algo == "segmentation":
            from mousereach.eval.seg_evaluator import SegmentationEvaluator
            evaluator = SegmentationEvaluator(gt_dir=gt_dir, tolerance=args.tolerance)
        elif algo == "reach":
            from mousereach.eval.reach_evaluator import ReachEvaluator
            evaluator = ReachEvaluator(gt_dir=gt_dir, tolerance=args.tolerance)
        elif algo == "outcome":
            from mousereach.eval.outcome_evaluator import OutcomeEvaluator
            evaluator = OutcomeEvaluator(gt_dir=gt_dir)
        else:
            print(f"  Unknown algorithm: {algo}")
            continue

        results = evaluator.evaluate_all()

        if not results:
            print(f"  No ground truth files found for {algo}")
            continue

        # Print report
        report = evaluator.generate_report()
        print(report)
        print()


def main_report():
    """Generate scientific report (mousereach-perf-report command)."""
    parser = argparse.ArgumentParser(
        description="Generate scientific report from performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mousereach-perf-report
  mousereach-perf-report --output report.md
  mousereach-perf-report --format methods
        """
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: print to stdout)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "methods", "table"],
        default="markdown",
        help="Output format (default: markdown)"
    )

    args = parser.parse_args()

    from .report import ScientificReportGenerator

    generator = ScientificReportGenerator()
    report = generator.generate(format=args.format)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main_view()
