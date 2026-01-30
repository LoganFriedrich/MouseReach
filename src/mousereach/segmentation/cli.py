#!/usr/bin/env python3
"""
CLI Entry Points for MouseReach Segmentation (Step 2)
======================================================

This module provides command-line tools for running segmentation, triaging
results, advancing validated videos, and launching the review tool.

INSTALLED COMMANDS
------------------
These commands are installed when you `pip install mousereach`:

    mousereach-segment        - Detect trial boundaries in DLC files
    mousereach-triage         - Sort results by confidence (high/low)
    mousereach-advance        - Move validated videos to next stage
    mousereach-segment-review - Launch napari review tool

TYPICAL WORKFLOW (v2.3+ Single-Folder Architecture)
----------------------------------------------------
1. Run segmentation on all DLC-complete videos:
   $ mousereach-segment -i Processing/

2. Results are saved with validation_status in JSON:
   - High confidence → validation_status: "auto_approved"
   - Low confidence  → validation_status: "needs_review"

3. Review low-confidence results:
   $ mousereach-segment-review video.mp4

4. After review, status updates to "validated" - no folder movement needed

FILE FLOW
---------
    Processing/                 All videos and results live here
         ↓ mousereach-segment
    *_segments.json created     With validation_status field
         ↓ mousereach-triage (auto)
    validation_status updated   "auto_approved" or "needs_review"
         ↓ human review (if needed)
    validation_status: "validated"  Ready for reach detection

NOTE: v2.3+ keeps all files in Processing/. Status is tracked in JSON
metadata (validation_status field), not by folder location.
"""

import argparse
from pathlib import Path


# =============================================================================
# mousereach-segment: Batch segmentation
# =============================================================================
def main_batch():
    """
    Run segmentation algorithm on multiple videos.

    This is the main batch processing command. It:
    1. Finds all DLC .h5 files in the input directory
    2. Runs the segmentation algorithm on each
    3. Saves *_segments.json files with boundary frames
    4. Auto-triages results (unless --no-triage)

    Examples:
        mousereach-segment -i Processing/
        mousereach-segment -i Processing/ --no-triage
    """
    from mousereach.segmentation.core import process_batch, find_dlc_files

    parser = argparse.ArgumentParser(
        description="Detect trial boundaries in DLC tracking files",
        epilog="""
Examples:
  mousereach-segment -i Processing/           # Process all DLC files
  mousereach-segment -i Processing/ --no-triage  # Don't auto-triage results
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory containing DLC .h5 files")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Minimal output (no per-file progress)")
    parser.add_argument('--no-triage', action='store_true',
                        help="Don't auto-update validation_status in JSON files")
    args = parser.parse_args()

    # Find DLC files and report count
    dlc_files = find_dlc_files(args.input)
    print(f"Found {len(dlc_files)} DLC files to process")

    if not dlc_files:
        print("No DLC files found. Looking for *DLC*.h5 files.")
        return

    # Run batch processing
    results = process_batch(
        args.input,
        auto_triage=not args.no_triage,
        verbose=not args.quiet
    )


# =============================================================================
# mousereach-triage: Sort results by confidence
# =============================================================================
def main_triage():
    """
    Sort segmentation results by confidence score.

    Updates validation_status in JSON based on algorithm confidence:
    - High confidence (>0.85) → validation_status: "auto_approved"
    - Low confidence  (<0.85) → validation_status: "needs_review"
    - Algorithm failure       → moved to Failed/

    Examples:
        mousereach-triage -i Processing/
    """
    from mousereach.segmentation.core import triage_results

    parser = argparse.ArgumentParser(
        description="Sort segmentation results by confidence (updates validation_status in JSON)",
        epilog="""
This is usually run automatically by mousereach-segment.
Only use manually if you need to re-triage existing results.
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Directory with *_segments.json files to triage")
    args = parser.parse_args()

    results = triage_results(args.input)

    print(f"\nTriage complete:")
    print(f"  Auto-review (high confidence): {results.get('auto_review', 0)}")
    print(f"  Needs review (low confidence): {results.get('needs_review', 0)}")
    print(f"  Failed (algorithm error):      {results.get('failed', 0)}")


# =============================================================================
# mousereach-advance: Move validated videos to next stage
# =============================================================================
def main_advance():
    """
    Mark segmentation as ready for reach detection (v2.3+ architecture).

    In the single-folder architecture, this command updates the segment
    validation_status to indicate readiness for the next pipeline stage.

    Only processes videos where validation_status == "validated" (human-reviewed)
    or validation_status == "auto_approved" (high confidence, no review needed).

    Use --force to process all videos regardless of validation status.

    Examples:
        mousereach-advance -i Processing/
        mousereach-advance -i Processing/ --force
    """
    from mousereach.segmentation.core import advance_videos

    parser = argparse.ArgumentParser(
        description="Mark validated segmentation results as ready for reach detection",
        epilog="""
Only processes videos with validation_status = "validated" or "auto_approved".
Use --force to process regardless of status (not recommended for production).

NOTE: In v2.3+ single-folder architecture, files stay in Processing/.
        """
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help="Source directory (typically Processing/)")
    parser.add_argument('--force', action='store_true',
                        help="Process all videos even without validation flag")
    args = parser.parse_args()

    moved = advance_videos(args.input, require_validation=not args.force)
    print(f"\nAdvanced {moved} video(s) - ready for reach detection")


# =============================================================================
# mousereach-segment-review: Launch napari review tool
# =============================================================================
def main_review():
    """
    Launch the napari-based boundary review tool.

    This opens a GUI where you can:
    1. Load a video and its detected boundaries
    2. Scrub through frames to verify each boundary
    3. Adjust boundaries that are off (set to current frame)
    4. Save the validated result

    Examples:
        mousereach-segment-review                  # Launch empty
        mousereach-segment-review video.mp4       # Launch with video
    """
    import napari
    from mousereach.segmentation.review_widget import BoundaryReviewWidget

    parser = argparse.ArgumentParser(
        description="Launch napari boundary review tool for manual verification",
        epilog="""
Keyboard shortcuts:
  SPACE           Set current boundary to this frame
  N / P           Next / Previous boundary
  S               Save validated boundaries
  Left / Right    Step 1 frame
  Shift+Arrows    Step 10 frames
        """
    )
    parser.add_argument('video', nargs='?', type=Path,
                        help="Video file to auto-load (optional)")
    args = parser.parse_args()

    # Create napari viewer and add review widget
    viewer = napari.Viewer(title="MouseReach Boundary Review Tool")
    widget = BoundaryReviewWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Review", area="right")

    # Auto-load video if provided
    if args.video:
        if args.video.exists():
            print(f"Loading video: {args.video}")
            widget._load_video_from_path(args.video)
        else:
            print(f"Warning: Video file not found: {args.video}")

    # Print keyboard shortcuts for reference
    print("\nKeyboard shortcuts:")
    print("  SPACE     - Set current boundary to this frame")
    print("  N         - Next boundary")
    print("  P         - Previous boundary")
    print("  S         - Save validated (for pipeline)")
    print("  Left/Right - Move 1 frame")
    print("  Shift+Left/Right - Move 10 frames")

    napari.run()


if __name__ == "__main__":
    main_review()
