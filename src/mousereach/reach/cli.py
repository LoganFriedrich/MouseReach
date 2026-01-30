#!/usr/bin/env python3
"""
CLI Entry Points for MouseReach Reach Detection (Step 3)
=========================================================

This module provides command-line tools for detecting reaches, triaging
results, advancing validated videos, and launching the review tool.

INSTALLED COMMANDS
------------------
These commands are installed when you `pip install mousereach`:

    mousereach-detect-reaches   - Detect reaching movements in videos
    mousereach-triage-reaches   - Flag anomalous results for review
    mousereach-advance-reaches  - Move validated videos to next stage
    mousereach-review-reaches   - Launch napari review tool

TYPICAL WORKFLOW (v2.3+ Single-Folder Architecture)
----------------------------------------------------
1. Run reach detection on videos with validated segments:
   $ mousereach-detect-reaches -i Processing/

2. Results are saved with validation_status in JSON:
   - Normal patterns   → validation_status: "auto_approved"
   - Unusual patterns  → validation_status: "needs_review"

3. Review flagged results:
   $ mousereach-review-reaches --reaches video_reaches.json

4. After review, status updates to "validated" - no folder movement needed

PREREQUISITES
-------------
Reach detection requires:
- DeepLabCut tracking file (*DLC*.h5)
- Validated segmentation (*_segments.json with validation_status="validated")

FILE FLOW
---------
    Processing/                 All videos and results live here
         ↓ mousereach-detect-reaches
    *_reaches.json created      With validation_status field
         ↓ mousereach-triage-reaches (auto)
    validation_status updated   "auto_approved" or "needs_review"
         ↓ human review (if needed)
    validation_status: "validated"  Ready for outcome detection

NOTE: v2.3+ keeps all files in Processing/. Status is tracked in JSON
metadata (validation_status field), not by folder location.
"""

import argparse
from pathlib import Path


# =============================================================================
# mousereach-detect-reaches: Batch reach detection
# =============================================================================
def main_batch():
    """
    Detect reaching movements in multiple videos.

    This command:
    1. Finds videos with validated segments (*_segments.json)
    2. Loads DLC tracking data
    3. Runs reach detection algorithm (hand visibility + nose engagement)
    4. Saves *_reaches.json with reach timing per segment

    Examples:
        mousereach-detect-reaches -i Processing/
        mousereach-detect-reaches -i Processing/ --skip-validation-check
    """
    from mousereach.reach.core import process_batch

    parser = argparse.ArgumentParser(
        description="Detect reaching movements in DLC tracking data",
        epilog="""
Examples:
  mousereach-detect-reaches -i Processing/
  mousereach-detect-reaches -i Processing/ -s '*_reach_ground_truth.json'
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory with validated segment files")
    parser.add_argument('-o', '--output', type=Path,
                        help="Output directory (default: same as input)")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Minimal output (no per-file progress)")
    parser.add_argument('--skip-validation-check', action='store_true',
                        help="Process even without validated segments (not recommended)")
    parser.add_argument('-s', '--skip-if-exists', nargs='+',
                        help="Skip videos with files matching patterns (e.g., '*_reaches.json')")
    args = parser.parse_args()

    results = process_batch(
        args.input,
        output_dir=args.output,
        verbose=not args.quiet,
        skip_validation_check=args.skip_validation_check,
        skip_if_exists=args.skip_if_exists
    )


# =============================================================================
# mousereach-triage-reaches: Flag anomalous results
# =============================================================================
def main_triage():
    """
    Flag reach detection results that need human review.

    Checks for anomalies like:
    - Unusual reach counts (too many or too few)
    - Very short or long reach durations
    - Tracking quality issues

    Examples:
        mousereach-triage-reaches -i Processing/
    """
    from mousereach.reach.core import triage_results

    parser = argparse.ArgumentParser(
        description="Flag anomalous reach detection results for review",
        epilog="""
Checks for unusual patterns that may indicate algorithm errors.
Videos with anomalies are flagged for human review.
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory with *_reaches.json files")
    args = parser.parse_args()

    results = triage_results(args.input)

    print(f"\nTriage complete:")
    print(f"  Auto-review (normal): {results.get('auto_review', 0)}")
    print(f"  Needs review (anomalies): {results.get('needs_review', 0)}")
    print(f"  Failed: {results.get('failed', 0)}")


# =============================================================================
# mousereach-advance-reaches: Move validated videos
# =============================================================================
def main_advance():
    """
    Mark reach detection as ready for outcome detection (v2.3+ architecture).

    In the single-folder architecture, this command updates the reach
    validation_status to indicate readiness for the next pipeline stage.

    Only processes videos with validation_status == "validated".
    Use --force to process all videos regardless of status.

    Examples:
        mousereach-advance-reaches -i Processing/
    """
    from mousereach.reach.core import advance_videos

    parser = argparse.ArgumentParser(
        description="Mark validated reach detection results as ready for outcome detection",
        epilog="""
Only processes videos with validation_status = "validated".
Use --force to process regardless of status (not recommended).

NOTE: In v2.3+ single-folder architecture, files stay in Processing/.
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Source directory (typically Processing/)")
    parser.add_argument('--force', action='store_true',
                        help="Process all videos even without validation flag")
    args = parser.parse_args()

    moved = advance_videos(args.input, require_validation=not args.force)
    print(f"\nAdvanced {moved} video(s) - ready for outcome detection")


# =============================================================================
# mousereach-review-reaches: Launch review tool
# =============================================================================
def main_review():
    """
    Launch interactive reach review tool.

    Opens a GUI/TUI to:
    1. View each detected reach with video playback
    2. Adjust reach start/end frames if incorrect
    3. Add missing reaches or delete false positives
    4. Save corrected results

    Examples:
        mousereach-review-reaches --reaches video_reaches.json
        mousereach-review-reaches --dir Processing/
    """
    from mousereach.reach._review import interactive_review

    parser = argparse.ArgumentParser(
        description="Review and correct reach detection results",
        epilog="""
Examples:
  mousereach-review-reaches --reaches video_reaches.json  # Review single file
  mousereach-review-reaches --dir Processing/            # Review all in dir

Keyboard shortcuts (in review tool):
  N / P       Next / Previous reach
  S / E       Set reach Start / End to current frame
  A           Add new reach
  DEL         Delete current reach
  Space       Play/pause video
        """
    )
    parser.add_argument('--reaches', type=Path,
                        help="Single *_reaches.json file to review")
    parser.add_argument('--dir', type=Path,
                        help="Directory with multiple reach files to review")

    args = parser.parse_args()

    if args.reaches:
        # Review single file
        interactive_review(args.reaches)
    elif args.dir:
        # Review all files in directory
        reach_files = list(args.dir.glob("*_reaches.json"))
        print(f"Found {len(reach_files)} files to review")

        for i, rf in enumerate(reach_files, 1):
            print(f"\n[{i}/{len(reach_files)}] {rf.name}")
            interactive_review(rf)

            if i < len(reach_files):
                cont = input("\nContinue to next? (y/n) ")
                if cont.lower() != 'y':
                    break
    else:
        parser.print_help()
        print("\nExamples:")
        print("  mousereach-review-reaches --reaches video_reaches.json")
        print("  mousereach-review-reaches --dir Processing/")


if __name__ == "__main__":
    main_review()
