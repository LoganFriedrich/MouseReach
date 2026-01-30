#!/usr/bin/env python3
"""
CLI Entry Points for MouseReach Pellet Outcomes (Step 4)
=========================================================

This module provides command-line tools for detecting pellet outcomes,
triaging results, advancing validated videos, and launching the review tool.

INSTALLED COMMANDS
------------------
These commands are installed when you `pip install mousereach`:

    mousereach-detect-outcomes         - Classify pellet outcomes
    mousereach-triage-outcomes         - Flag low-confidence results
    mousereach-advance-outcomes        - Move validated videos to next stage
    mousereach-review-pellet-outcomes  - Launch napari review tool

OUTCOME CATEGORIES
------------------
Each pellet can have one of these outcomes:
    retrieved      (R) - Mouse successfully grasped and ate the pellet
    displaced_sa   (D) - Pellet knocked but still in scoring area
    displaced_outside (O) - Pellet knocked outside scoring area
    untouched      (U) - Pellet not touched, remains on pillar

TYPICAL WORKFLOW (v2.3+ Single-Folder Architecture)
----------------------------------------------------
1. Run outcome detection on videos with reach data:
   $ mousereach-detect-outcomes -i Processing/

2. Results are saved with validation_status in JSON:
   - High confidence → validation_status: "auto_approved"
   - Low confidence  → validation_status: "needs_review"

3. Review flagged results:
   $ mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json

4. After review, status updates to "validated" - ready for export

PREREQUISITES
-------------
Outcome detection requires:
- DeepLabCut tracking file (*DLC*.h5)
- Validated segmentation (*_segments.json)
- Reach detection results (*_reaches.json)

FILE FLOW
---------
    Processing/                   All videos and results live here
         ↓ mousereach-detect-outcomes
    *_pellet_outcomes.json        With validation_status field
         ↓ mousereach-triage-outcomes (auto)
    validation_status updated     "auto_approved" or "needs_review"
         ↓ human review (if needed)
    validation_status: "validated"  Ready for export

NOTE: v2.3+ keeps all files in Processing/. Status is tracked in JSON
metadata (validation_status field), not by folder location.
"""

import argparse
from pathlib import Path


# =============================================================================
# mousereach-detect-outcomes: Batch outcome detection
# =============================================================================
def main_batch():
    """
    Classify pellet outcomes for multiple videos.

    This command:
    1. Finds videos with reach detection results
    2. Tracks pellet position throughout each segment
    3. Classifies outcome (retrieved, displaced, untouched)
    4. Saves *_pellet_outcomes.json with per-segment classifications

    Examples:
        mousereach-detect-outcomes -i Processing/
    """
    from mousereach.outcomes.core import process_batch

    parser = argparse.ArgumentParser(
        description="Classify pellet outcomes (retrieved/displaced/untouched)",
        epilog="""
Examples:
  mousereach-detect-outcomes -i Processing/
  mousereach-detect-outcomes -i Processing/ -s '*_pellet_outcomes.json'
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory with reach detection results")
    parser.add_argument('-o', '--output', type=Path,
                        help="Output directory (default: same as input)")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Minimal output (no per-file progress)")
    parser.add_argument('-s', '--skip-if-exists', nargs='+',
                        help="Skip videos with files matching patterns")
    args = parser.parse_args()

    results = process_batch(
        args.input,
        output_dir=args.output,
        verbose=not args.quiet,
        skip_if_exists=args.skip_if_exists
    )

    print(f"\nComplete: {results['success']}/{results['total']} succeeded")


# =============================================================================
# mousereach-triage-outcomes: Flag low-confidence results
# =============================================================================
def main_triage():
    """
    Flag outcome classifications that need human review.

    Checks for:
    - Low-confidence classifications
    - Ambiguous pellet tracking
    - Unusual patterns

    Examples:
        mousereach-triage-outcomes -i Processing/
    """
    from mousereach.outcomes.core import triage_results

    parser = argparse.ArgumentParser(
        description="Flag low-confidence outcome classifications for review",
        epilog="""
Checks for ambiguous or low-confidence outcome classifications.
Videos with issues are flagged for human review.
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory with *_pellet_outcomes.json files")
    args = parser.parse_args()

    results = triage_results(args.input)

    print(f"\nTriage complete:")
    print(f"  Auto-review (high confidence): {results.get('auto_review', 0)}")
    print(f"  Needs review (low confidence): {results.get('needs_review', 0)}")
    print(f"  Failed: {results.get('failed', 0)}")


# =============================================================================
# mousereach-advance-outcomes: Move validated videos
# =============================================================================
def main_advance():
    """
    Mark outcome results as ready for export (v2.3+ architecture).

    In the single-folder architecture, this command updates the outcome
    validation_status to indicate readiness for the export stage.

    Only processes videos with validation_status == "validated".
    Use --force to process all videos regardless of status.

    Examples:
        mousereach-advance-outcomes -i Processing/
    """
    from mousereach.outcomes.core import advance_videos

    parser = argparse.ArgumentParser(
        description="Mark validated outcome results as ready for export",
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
    print(f"\nAdvanced {moved} video(s) - ready for export")


# =============================================================================
# mousereach-review-pellet-outcomes: Launch review tool
# =============================================================================
def main_review():
    """
    Launch interactive outcome review tool.

    Opens a GUI/TUI to:
    1. View each segment's outcome classification
    2. Verify or correct the outcome (R/D/O/U)
    3. Set interaction frame (when pellet was touched)
    4. Save corrected results

    Examples:
        mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json
        mousereach-review-pellet-outcomes --dir Processing/
    """
    from mousereach.outcomes._review import interactive_review

    parser = argparse.ArgumentParser(
        description="Review and correct pellet outcome classifications",
        epilog="""
Examples:
  mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json
  mousereach-review-pellet-outcomes --dir Processing/

Keyboard shortcuts (in review tool):
  N / P       Next / Previous segment
  R           Set outcome to Retrieved
  D           Set outcome to Displaced (scoring area)
  O           Set outcome to Displaced (outside)
  U           Set outcome to Untouched
  I           Set interaction frame to current frame
  Space       Play/pause video
        """
    )
    parser.add_argument('--outcomes', type=Path,
                        help="Single *_pellet_outcomes.json file to review")
    parser.add_argument('--dir', type=Path,
                        help="Directory with multiple outcome files to review")

    args = parser.parse_args()

    if args.outcomes:
        # Review single file
        interactive_review(args.outcomes)
    elif args.dir:
        # Review all files in directory
        outcome_files = list(args.dir.glob("*_pellet_outcomes.json"))
        print(f"Found {len(outcome_files)} files to review")

        for i, of in enumerate(outcome_files, 1):
            print(f"\n[{i}/{len(outcome_files)}] {of.name}")
            interactive_review(of)

            if i < len(outcome_files):
                cont = input("\nContinue to next? (y/n) ")
                if cont.lower() != 'y':
                    break
    else:
        parser.print_help()
        print("\nExamples:")
        print("  mousereach-review-pellet-outcomes --outcomes video_pellet_outcomes.json")
        print("  mousereach-review-pellet-outcomes --dir Processing/")


if __name__ == "__main__":
    main_review()
