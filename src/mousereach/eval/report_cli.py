"""CLI for generating MouseReach eval report with matplotlib plots."""

import argparse
from datetime import datetime
from pathlib import Path


def main():
    """Generate eval report with plots from unified GT files."""
    parser = argparse.ArgumentParser(
        description="Generate MouseReach eval report with plots"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory for plots (default: eval_reports/YYYY-MM-DD_HHMMSS/)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation, print text summary only"
    )
    parser.add_argument(
        "path", nargs="?", type=Path, default=None,
        help="Processing directory (default: auto-detect from config)"
    )
    args = parser.parse_args()

    from mousereach.config import require_processing_root
    from mousereach.eval.collect_results import collect_all
    from mousereach.eval.plot_results import generate_all_plots

    if args.path:
        processing_dir = Path(args.path)
    else:
        processing_dir = require_processing_root() / "Processing"

    if not processing_dir.exists():
        print(f"Error: Directory not found: {processing_dir}")
        return

    # Collect results
    print(f"Scanning {processing_dir} for unified GT files...")
    results = collect_all(processing_dir)

    # Text summary
    n_seg = len(results.seg_results)
    n_reach = len(results.reach_results)
    n_skip = len(results.skipped_reach)
    n_out = len(results.outcome_results)

    print(f"\nCorpus: {n_seg} seg, {n_reach} reach (excl {n_skip} without GT), {n_out} outcome")

    if results.seg_results:
        import numpy as np
        mean_seg = np.mean([r.recall * 100 for r in results.seg_results])
        print(f"  Segmentation mean recall: {mean_seg:.1f}%")

    if results.reach_results:
        import numpy as np
        mean_p = np.mean([r.precision * 100 for r in results.reach_results])
        mean_r = np.mean([r.recall * 100 for r in results.reach_results])
        mean_f1 = np.mean([r.f1 * 100 for r in results.reach_results])
        print(f"  Reach mean P/R/F1: {mean_p:.1f}% / {mean_r:.1f}% / {mean_f1:.1f}%")

    if results.outcome_results:
        import numpy as np
        mean_acc = np.mean([r.accuracy * 100 for r in results.outcome_results])
        print(f"  Outcome mean accuracy: {mean_acc:.1f}%")

    if results.skipped_reach:
        print(f"  Skipped (no reach GT): {', '.join(results.skipped_reach)}")

    # Generate plots
    if not args.no_plots:
        if args.output_dir is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            project_root = require_processing_root()
            output_dir = project_root / "eval_reports" / ts
        else:
            output_dir = args.output_dir

        print(f"\nGenerating plots to {output_dir}...")
        paths = generate_all_plots(results, output_dir)
        print(f"Saved {len(paths)} figures:")
        for p in paths:
            print(f"  {p.name}")

        # Open the summary dashboard and the output folder
        import os
        dashboard = output_dir / "summary_dashboard.png"
        if dashboard.exists():
            os.startfile(str(dashboard))
        os.startfile(str(output_dir))
    else:
        print("\nPlot generation skipped (--no-plots)")


if __name__ == "__main__":
    main()
