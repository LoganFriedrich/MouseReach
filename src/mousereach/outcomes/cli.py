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
import json
from pathlib import Path


# =============================================================================
# mousereach-detect-outcomes: Batch outcome detection
# =============================================================================
def main_batch():
    """
    Classify pellet outcomes for multiple videos.

    Default detector: v6 cascade (30 physics-grounded stages,
    first-to-commit wins, triage on uncertain). Use --legacy to fall
    back to the v2.4.4 detector.

    This command:
    1. Finds videos with reach detection results
    2. Tracks pellet position throughout each segment
    3. Classifies outcome (retrieved, displaced, untouched)
    4. Saves *_pellet_outcomes.json with per-segment classifications

    Examples:
        mousereach-detect-outcomes -i Processing/
        mousereach-detect-outcomes -i Processing/ --legacy
    """
    parser = argparse.ArgumentParser(
        description="Classify pellet outcomes (retrieved/displaced/untouched)",
        epilog="""
Examples:
  mousereach-detect-outcomes -i Processing/
  mousereach-detect-outcomes -i Processing/ --legacy
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
    parser.add_argument('--legacy', action='store_true',
                        help="Use legacy v2.4.4 detector instead of v6 cascade")
    args = parser.parse_args()

    if args.legacy:
        from mousereach.outcomes.core import process_batch

        if not args.quiet:
            print("[outcome detector] Using legacy v2.4.4 detector")
        results = process_batch(
            args.input,
            output_dir=args.output,
            verbose=not args.quiet,
            skip_if_exists=args.skip_if_exists
        )
        print(f"\nComplete: {results['success']}/{results['total']} succeeded")
    else:
        _main_batch_v6(args)


def _main_batch_v6(args):
    """v6 cascade batch processing (default detector)."""
    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade, VERSION
    from mousereach.outcomes.core.batch import find_file_sets
    from mousereach.reach.v8.features import load_dlc_h5

    input_dir = args.input
    output_dir = args.output or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if verbose:
        print(f"[outcome detector] v6 cascade (VERSION {VERSION})")

    file_sets = find_file_sets(input_dir, skip_if_exists=args.skip_if_exists)

    if not file_sets:
        if verbose:
            print(f"No file sets found in {input_dir}")
        return

    if verbose:
        print(f"Found {len(file_sets)} video(s) to process")
        print("-" * 70)

    n_success = 0
    n_failed = 0

    for i, fs in enumerate(file_sets, 1):
        video_name = fs["video_name"]
        if verbose:
            print(f"[{i}/{len(file_sets)}] {video_name}...", end=" ", flush=True)

        try:
            # Load DLC
            dlc_df = load_dlc_h5(fs["dlc_file"])

            # Load segments
            seg_data = json.loads(fs["seg_file"].read_text(encoding="utf-8"))
            boundaries = _extract_boundaries(seg_data)
            segments = [
                (boundaries[j], boundaries[j + 1] - 1)
                for j in range(len(boundaries) - 1)
            ]

            # Load reaches (v6 cascade uses reach windows)
            reaches = []
            if fs.get("reach_file") and fs["reach_file"].exists():
                reach_data = json.loads(
                    fs["reach_file"].read_text(encoding="utf-8"))
                reaches = _extract_reaches(reach_data)

            # Detect video dir for Stage 98 CV checks
            video_dir = _find_video_dir(input_dir, video_name)

            # Run cascade
            result = detect_outcomes_v6_cascade(
                dlc_df=dlc_df,
                segments=segments,
                reaches=reaches,
                video_id=video_name,
                video_dir=video_dir,
            )

            # Write output
            out_path = output_dir / f"{video_name}_pellet_outcomes.json"
            out_path.write_text(
                json.dumps(result, indent=2), encoding="utf-8")

            # Summary counts
            counts = {}
            for s in result["segments"]:
                o = s["outcome"]
                counts[o] = counts.get(o, 0) + 1

            n_success += 1
            if verbose:
                parts = []
                for key in ("retrieved", "displaced_sa", "untouched", "triaged"):
                    if counts.get(key, 0) > 0:
                        tag = key[0].upper()
                        parts.append(f"{tag}={counts[key]}")
                print(f"OK ({'/'.join(parts)})")

        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"FAILED: {e}")

    if verbose:
        print("-" * 70)
    print(f"\nComplete: {n_success}/{len(file_sets)} succeeded"
          + (f", {n_failed} failed" if n_failed else ""))


def _extract_boundaries(seg_data):
    """Extract frame boundaries from various segment JSON formats."""
    # Format 1: {"segmentation": {"boundaries": [{"frame": N}, ...]}}
    if "segmentation" in seg_data:
        return [int(b["frame"])
                for b in seg_data["segmentation"]["boundaries"]]
    # Format 2: {"boundaries": [N, N, ...]}
    if "boundaries" in seg_data:
        return [int(b) for b in seg_data["boundaries"]]
    # Format 3: {"segments": [{"start": N, "end": N}, ...]}
    if "segments" in seg_data and isinstance(seg_data["segments"], list):
        bounds = set()
        for s in seg_data["segments"]:
            if "start" in s:
                bounds.add(int(s["start"]))
            if "end" in s:
                bounds.add(int(s["end"]))
        return sorted(bounds)
    raise ValueError(
        "Cannot parse segment boundaries from JSON "
        f"(keys: {list(seg_data.keys())})")


def _extract_reaches(reach_data):
    """Extract (start, end) tuples from various reach JSON formats."""
    reaches = []
    # Format 1: {"reaches": [{"start_frame": N, "end_frame": N}, ...]}
    if "reaches" in reach_data and isinstance(reach_data["reaches"], list):
        for r in reach_data["reaches"]:
            s = r.get("start_frame") or r.get("start")
            e = r.get("end_frame") or r.get("end")
            if s is not None and e is not None:
                reaches.append((int(s), int(e)))
    return reaches


def _find_video_dir(input_dir, video_name):
    """Try to find the directory containing the source video file."""
    for ext in (".avi", ".mp4"):
        candidate = input_dir / f"{video_name}{ext}"
        if candidate.exists():
            return input_dir
    # Check parent and siblings
    for sibling in ("videos", "Videos", "raw"):
        d = input_dir / sibling
        if d.is_dir():
            for ext in (".avi", ".mp4"):
                if (d / f"{video_name}{ext}").exists():
                    return d
    return None


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
