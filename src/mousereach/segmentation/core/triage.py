"""
triage.py - Triage segmentation results into review queues

Extracted from 2_triage_results.py

Paths are now configured via environment variables.
Set MouseReach_PROCESSING_ROOT to customize pipeline location.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from mousereach.config import Paths, parse_tray_type, is_supported_tray_type, FilePatterns


# Default staging directories - derived from configurable environment variables
DEST_AUTO_REVIEW = Paths.SEG_AUTO_REVIEW
DEST_NEEDS_REVIEW = Paths.SEG_NEEDS_REVIEW
DEST_FAILED = Paths.FAILED


def get_associated_files(segments_path: Path) -> Dict:
    """
    Find all files associated with a video.
    """
    folder = segments_path.parent
    video_id = segments_path.stem.replace("_segments", "")
    
    associated = {
        "video_id": video_id,
        "segments": segments_path,
        "video": None,
        "dlc_h5": None,
        "dlc_csv": None,
        "dlc_pickle": None,
    }
    
    # Find video
    for ext in [".mp4", ".avi", ".mkv"]:
        video_path = folder / f"{video_id}{ext}"
        if video_path.exists():
            associated["video"] = video_path
            break
    
    # Find DLC files
    for f in folder.glob(f"{video_id}DLC*"):
        if f.suffix == ".h5":
            associated["dlc_h5"] = f
        elif f.suffix == ".csv":
            associated["dlc_csv"] = f
        elif f.suffix == ".pickle":
            associated["dlc_pickle"] = f
    
    return associated


def check_tray_type_supported(video_path: Path) -> Tuple[bool, str, dict]:
    """
    Check if video has a supported tray type (P=Pillar only).

    E (Easy) and F (Flat) tray types require different algorithms and
    should not be processed by the standard MouseReach pipeline.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_supported, reason, parsed_info)
    """
    parsed = parse_tray_type(video_path.name)

    if parsed['is_supported']:
        return True, "Supported tray type (Pillar)", parsed

    tray_type = parsed.get('tray_type')
    tray_name = parsed.get('tray_name', 'Unknown')

    if tray_type in FilePatterns.UNSUPPORTED_TRAY_TYPES:
        reason = (
            f"UNSUPPORTED TRAY TYPE: {tray_type} ({tray_name}). "
            f"Only Pillar (P) trays are supported. "
            f"Easy (E) and Flat (F) trays require different algorithms."
        )
        return False, reason, parsed

    if tray_type is None:
        reason = f"Could not parse tray type from filename: {video_path.name}"
        return False, reason, parsed

    return False, f"Unknown tray type: {tray_type}", parsed


def reject_unsupported_tray_type(
    video_path: Path,
    dest_folder: Path = None,
    log_file: Path = None,
    dry_run: bool = False,
    verbose: bool = True
) -> Tuple[bool, str, List[str]]:
    """
    Move unsupported tray type videos back to NAS.

    Args:
        video_path: Path to video file
        dest_folder: Destination folder (default: Paths.UNSUPPORTED_TRAY_RETURN)
        log_file: Log file path
        dry_run: If True, only report what would happen
        verbose: Print progress

    Returns:
        Tuple of (was_rejected, reason, files_moved)
    """
    if dest_folder is None:
        dest_folder = Paths.UNSUPPORTED_TRAY_RETURN

    is_supported, reason, parsed = check_tray_type_supported(video_path)

    if is_supported:
        return False, "Tray type is supported", []

    video_id = parsed['video_id']
    tray_type = parsed.get('tray_type', '?')

    if verbose:
        print(f"  [!] REJECTING {video_id}: {tray_type}-type tray not supported")
        print(f"      Reason: {reason}")

    if dry_run:
        if verbose:
            print(f"      [DRY RUN] Would move to: {dest_folder}")
        return True, reason, []

    # Create destination folder if needed
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Find all associated files
    folder = video_path.parent
    files_to_move = []

    # Video file
    if video_path.exists():
        files_to_move.append(video_path)

    # Find associated DLC and JSON files
    for f in folder.glob(f"{video_id}*"):
        if f.is_file() and f not in files_to_move:
            files_to_move.append(f)

    moved_files = []
    for src_file in files_to_move:
        dst_file = dest_folder / src_file.name
        if dst_file.exists():
            # Add timestamp suffix to avoid overwrite
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst_file = dest_folder / f"{src_file.stem}_{timestamp}{src_file.suffix}"

        shutil.move(str(src_file), str(dst_file))
        moved_files.append(src_file.name)

        if verbose:
            print(f"      Moved: {src_file.name}")

    # Log the rejection
    if log_file is None:
        log_file = Paths.PROCESSING_ROOT / "unsupported_tray_log.txt"

    try:
        user = os.getlogin()
    except (OSError, AttributeError):
        user = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

    log_entry = (
        f"{datetime.now().isoformat()}\t{user}\t{video_id}\t"
        f"{tray_type}\t{dest_folder.name}\t{moved_files}\n"
    )

    with open(log_file, "a") as f:
        f.write(log_entry)

    if verbose:
        print(f"      -> Moved {len(moved_files)} files to {dest_folder.name}/")

    return True, reason, moved_files


def classify_segments(segments_path: Path, check_gt: bool = True) -> Tuple[str, str]:
    """
    Read segments.json and determine triage category.

    Returns: (category, reason)

    Classification logic:
    - 'good' = auto-approve (high confidence, no significant issues)
    - 'warning' = needs review (some concerns)
    - 'failed' = critical issues (cannot use without major fixes)

    INFO-level anomalies (like "Used lower velocity threshold") do NOT
    prevent auto-approval since they indicate normal algorithm adaptation.

    If check_gt=True and a GT file exists with human-verified boundaries,
    the video is automatically approved (GT overrides algorithm concerns).
    """
    import numpy as np

    # Check for GT file first - if GT exists with verified boundaries, auto-approve
    if check_gt:
        video_id = segments_path.stem.replace("_segments", "")
        gt_path = segments_path.parent / f"{video_id}_seg_ground_truth.json"
        if gt_path.exists():
            try:
                with open(gt_path) as f:
                    gt_data = json.load(f)
                # Check if GT has the expected number of boundaries
                gt_boundaries = len(gt_data.get('boundaries', []))
                # GT file is considered human-verified if it has human_verified flag
                # OR if it has the correct number of boundaries (21)
                # (legacy GT files don't have verification flags)
                is_verified = gt_data.get('human_verified', False) or gt_boundaries == 21
                if is_verified:
                    return "good", f"GT file exists with {gt_boundaries} boundaries (human verified)"
            except Exception:
                pass  # GT file exists but couldn't be read, continue with normal triage

    try:
        with open(segments_path) as f:
            data = json.load(f)
    except Exception as e:
        return "failed", f"Could not read segments file: {e}"

    if "boundaries" not in data:
        return "failed", "Missing boundaries in segments file"

    n_boundaries = len(data["boundaries"])
    cv = extract_cv(data)

    # Get detection info
    detection = data.get('detection', {})
    confidences = detection.get('confidences', [])
    methods = detection.get('methods', [])
    n_primary = detection.get('n_primary', n_boundaries)
    mean_confidence = np.mean(confidences) if confidences else 0.9
    n_interpolated = sum(1 for m in methods if 'interpolated' in str(m).lower())

    # Get anomalies and classify their severity
    anomalies = data.get("anomalies", [])
    if "diagnostics" in data:
        anomalies = data["diagnostics"].get("anomalies", anomalies)

    # Build context for anomaly classification
    context = {
        'n_boundaries': n_boundaries,
        'n_interpolated': n_interpolated,
        'cv': cv,
        'mean_confidence': mean_confidence,
        'n_primary': n_primary
    }

    # Count anomalies by severity
    severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
    for anom in anomalies:
        sev, _, _ = classify_anomaly_severity(anom, context)
        severity_counts[sev.lower()] += 1

    # FAILED: Cannot determine segments reliably
    if n_boundaries < 19:
        return "failed", f"Only {n_boundaries} boundaries detected"

    if severity_counts['critical'] > 0:
        return "failed", f"Critical anomalies: {anomalies}"

    if n_primary < 10:
        return "failed", f"Primary detection failed (only {n_primary}/21 from primary method)"

    # GOOD (auto-approve): High confidence segmentation
    # INFO anomalies are OK - they just note algorithm adaptations
    if (n_boundaries == 21 and
        cv < 0.10 and
        mean_confidence >= 0.85 and
        n_primary >= 18 and
        severity_counts['warning'] == 0):

        if severity_counts['info'] > 0:
            return "good", f"High confidence ({severity_counts['info']} info notes)"
        else:
            return "good", "High confidence"

    # WARNING (needs review): Some concerns but potentially usable
    reasons = []
    if n_boundaries != 21:
        reasons.append(f"{n_boundaries} boundaries")
    if cv >= 0.10:
        reasons.append(f"CV={cv:.4f}")
    if mean_confidence < 0.85:
        reasons.append(f"confidence={mean_confidence:.2f}")
    if severity_counts['warning'] > 0:
        reasons.append(f"{severity_counts['warning']} warnings")
    if n_interpolated > 0:
        reasons.append(f"{n_interpolated} interpolated")

    return "warning", "; ".join(reasons) if reasons else "Review recommended"


def extract_cv(data: dict) -> float:
    """Extract CV from various JSON formats."""
    if 'intervals' in data and 'cv' in data['intervals']:
        return data['intervals']['cv']
    if 'diagnostics' in data:
        diag = data['diagnostics']
        if 'interval_cv' in diag:
            return diag['interval_cv']
        if 'intervals' in diag and 'cv' in diag['intervals']:
            return diag['intervals']['cv']
    if 'interval_cv' in data:
        return data['interval_cv']
    return 999  # Unknown - will trigger review


def classify_anomaly_severity(anomaly_text: str, context: dict) -> tuple:
    """
    Classify anomaly severity (copied from segmenter_robust.py for standalone use).

    Returns: (severity_str, explanation_str, boundaries_affected_list)
    """
    import re

    # CRITICAL patterns
    if "No candidates found" in anomaly_text:
        return ('CRITICAL', 'Segmentation algorithm failed completely', [])
    if "Primary method unavailable" in anomaly_text:
        return ('CRITICAL', 'Cannot use main detection method', [])
    if "fallback motion detection" in anomaly_text:
        return ('CRITICAL', 'Unreliable detection method used', [])

    # Context-based critical conditions
    if context.get('n_boundaries', 21) < 19:
        return ('CRITICAL', f"Only {context['n_boundaries']} boundaries detected", [])
    if context.get('n_interpolated', 0) >= 5:
        return ('CRITICAL', 'Too many boundaries interpolated', [])
    if context.get('mean_confidence', 1.0) < 0.50:
        return ('CRITICAL', 'Very low average confidence', [])

    # INFO patterns - exception handlers
    if "lower velocity threshold" in anomaly_text:
        return ('INFO', 'Algorithm adapted to slower tray movement', [1])
    if "Detected crossing events before" in anomaly_text:
        return ('INFO', 'Pre-trial movements detected', [])

    # WARNING patterns
    if "Late-start" in anomaly_text or "Estimated B1" in anomaly_text:
        frame_match = re.search(r'frame (\d+)', anomaly_text)
        frame = int(frame_match.group(1)) if frame_match else 0
        if frame > 2000:
            return ('WARNING', 'Video started late in session', [1])
        else:
            return ('INFO', 'Normal start time variation', [1])

    if "Interpolated" in anomaly_text or "interpolated" in anomaly_text.lower():
        n_interp = context.get('n_interpolated', 0)
        if n_interp >= 5:
            return ('CRITICAL', 'Many boundaries interpolated', [])
        elif n_interp >= 2:
            return ('WARNING', f'{n_interp} boundaries needed interpolation', [])
        else:
            return ('INFO', 'Single boundary interpolated', [])

    if "Removed" in anomaly_text and "candidates" in anomaly_text:
        return ('WARNING', 'Had to filter out ambiguous detections', [])
    if "Very short interval" in anomaly_text or "Very long interval" in anomaly_text:
        return ('WARNING', 'Timing variation detected', [])
    if "Interval drift" in anomaly_text:
        return ('WARNING', 'Session timing changed', [])
    if "stuck tray" in anomaly_text:
        return ('WARNING', 'Possible operator issue', [])

    return ('INFO', 'Noted for reference', [])


def classify_segments_graduated(segments_path: Path) -> Tuple[str, str, dict]:
    """
    Graduated triage classification with segment-ID confidence as core criterion.

    Returns: (category, reason, details_dict)
        category: 'auto_approved', 'minor_review', 'major_review', 'failed'
        reason: Human-readable explanation
        details_dict: Structured details for logging
    """
    from ...config import Thresholds
    import numpy as np

    try:
        with open(segments_path) as f:
            data = json.load(f)
    except Exception as e:
        return "failed", f"Could not read segments file: {e}", {}

    # Extract fields (handle different JSON formats)
    n_boundaries = len(data.get('boundaries', []))
    cv = extract_cv(data)

    detection = data.get('detection', {})
    n_primary = detection.get('n_primary', 21)  # Assume all primary if missing
    confidences = detection.get('confidences', [])
    methods = detection.get('methods', [])
    mean_confidence = np.mean(confidences) if confidences else 0.5

    n_interpolated = sum(1 for m in methods if 'interpolated' in str(m))

    # Get anomaly severity breakdown (if available, else classify now)
    if 'anomaly_summary' in data:
        severity_counts = data['anomaly_summary']
    else:
        # Classify on the fly for old format
        anomalies = data.get('anomalies', [])
        severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
        context = {
            'n_boundaries': n_boundaries,
            'n_interpolated': n_interpolated,
            'cv': cv,
            'mean_confidence': mean_confidence,
            'n_primary': n_primary
        }
        for anom in anomalies:
            sev, _, _ = classify_anomaly_severity(anom, context)
            severity_counts[sev.lower()] += 1

    # DECISION TREE

    # FAILED: Cannot determine segments reliably
    if n_boundaries < Thresholds.SEG_ACCEPTABLE_MIN:
        return "failed", f"Only {n_boundaries} boundaries detected (need {Thresholds.SEG_ACCEPTABLE_MIN}+)", {
            'n_boundaries': n_boundaries,
            'reason': 'insufficient_boundaries'
        }

    if severity_counts['critical'] > 0:
        return "failed", "Critical anomalies prevent reliable segmentation", {
            'n_boundaries': n_boundaries,
            'critical_count': severity_counts['critical'],
            'reason': 'critical_anomalies'
        }

    if n_primary < 10:  # Less than half from primary
        return "failed", "Primary detection method failed", {
            'n_boundaries': n_boundaries,
            'n_primary': n_primary,
            'reason': 'primary_method_failed'
        }

    # AUTO_APPROVED: High confidence, segment ID certain
    if (n_boundaries == Thresholds.SEG_PERFECT_COUNT and
        cv < Thresholds.SEG_GOOD_CV and
        n_primary >= Thresholds.SEG_PRIMARY_MIN and
        mean_confidence >= Thresholds.SEG_HIGH_CONFIDENCE and
        severity_counts['warning'] == 0):

        if severity_counts['info'] == 0:
            reason = "Perfect segmentation - no issues detected"
        else:
            info_notes = data.get('anomaly_details', [])
            info_text = [a['text'][:40] for a in info_notes if a.get('severity') == 'INFO'][:2]
            reason = f"High confidence ({severity_counts['info']} info note{'s' if severity_counts['info'] > 1 else ''}: {', '.join(info_text)})"

        return "auto_approved", reason, {
            'n_boundaries': n_boundaries,
            'cv': cv,
            'mean_confidence': mean_confidence,
            'segment_id_certain': True,
            'anomalies': severity_counts
        }

    # MINOR_REVIEW: Good but worth quick check
    if (n_boundaries >= 20 and
        cv < Thresholds.SEG_ACCEPTABLE_CV and
        n_primary >= Thresholds.SEG_PRIMARY_MIN - 2 and
        mean_confidence >= 0.75 and
        severity_counts['warning'] <= 2):

        concerns = []
        if n_boundaries < 21:
            concerns.append(f"{n_boundaries}/21 boundaries")
        if cv >= Thresholds.SEG_GOOD_CV:
            concerns.append(f"CV={cv:.3f}")
        if severity_counts['warning'] > 0:
            concerns.append(f"{severity_counts['warning']} warning{'s' if severity_counts['warning'] > 1 else ''}")
        if n_interpolated > 0:
            concerns.append(f"{n_interpolated} interpolated")

        reason = "Minor issues (quick check recommended): " + ", ".join(concerns)

        return "minor_review", reason, {
            'n_boundaries': n_boundaries,
            'cv': cv,
            'mean_confidence': mean_confidence,
            'segment_id_certain': True,
            'anomalies': severity_counts,
            'concerns': concerns
        }

    # MAJOR_REVIEW: Significant uncertainty
    concerns = []
    if n_boundaries < 21:
        concerns.append(f"{n_boundaries}/21 boundaries")
    if cv >= Thresholds.SEG_ACCEPTABLE_CV:
        concerns.append(f"High CV ({cv:.3f})")
    if n_interpolated >= 3:
        concerns.append(f"{n_interpolated} interpolated")
    if mean_confidence < 0.75:
        concerns.append(f"Low confidence ({mean_confidence:.2f})")
    if severity_counts['warning'] > 2:
        concerns.append(f"{severity_counts['warning']} warnings")

    reason = "Requires careful review: " + ", ".join(concerns)

    return "major_review", reason, {
        'n_boundaries': n_boundaries,
        'cv': cv,
        'mean_confidence': mean_confidence,
        'segment_id_certain': False,
        'anomalies': severity_counts,
        'concerns': concerns
    }


def move_file_bundle(associated_files: Dict, destination: Path, log_file: Path,
                     source_stage: str = None, update_index: bool = True) -> List[str]:
    """Move all associated files to destination folder.

    Args:
        associated_files: Dict with video_id and file paths
        destination: Target folder
        log_file: Path to triage log file
        source_stage: Source stage name (for index update)
        update_index: Whether to update the pipeline index

    Returns:
        List of moved filenames
    """
    video_id = associated_files["video_id"]
    moved_files = []

    for key, src_path in associated_files.items():
        if key == "video_id" or src_path is None:
            continue

        dst_path = destination / src_path.name

        if dst_path.exists():
            continue

        shutil.move(str(src_path), str(dst_path))
        moved_files.append(src_path.name)

    # Log the move
    timestamp = datetime.now().isoformat()
    try:
        user = os.getlogin()
    except (OSError, AttributeError):
        user = os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))

    log_entry = f"{timestamp}\t{user}\t{video_id}\t{destination.name}\t{moved_files}\n"

    with open(log_file, "a") as f:
        f.write(log_entry)

    # Update pipeline index if requested
    if update_index and moved_files:
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            dest_stage = destination.name
            index.record_files_moved(video_id, source_stage or "unknown", dest_stage, moved_files)
            index.save()
        except Exception as e:
            # Don't fail the triage if index update fails
            print(f"[Triage] Warning: Could not update index: {e}")

    return moved_files


def triage_results(
    source_dir: Path,
    dest_auto: Path = None,
    dest_needs: Path = None,
    dest_failed: Path = None,
    reject_unsupported: bool = True,
    verbose: bool = True
) -> Dict:
    """
    [DEPRECATED] Triage all segmentation results in a directory by moving files.

    NOTE: This function is deprecated. The unified pipeline now keeps all
    files in Processing/ and tracks status via validation_status in JSON.
    Use the mousereach dashboard or UnifiedPipelineProcessor instead.

    Args:
        source_dir: Source directory containing segments files
        dest_auto: Destination for auto-approved files
        dest_needs: Destination for files needing review
        dest_failed: Destination for failed files
        reject_unsupported: If True, reject E/F tray types and move to NAS
        verbose: Print progress
    """
    import warnings
    warnings.warn(
        "triage_results() is deprecated. Use the mousereach dashboard or "
        "UnifiedPipelineProcessor instead. Files now stay in Processing/ with "
        "status tracked in JSON metadata.",
        DeprecationWarning,
        stacklevel=2
    )
    if dest_auto is None:
        dest_auto = DEST_AUTO_REVIEW
    if dest_needs is None:
        dest_needs = DEST_NEEDS_REVIEW
    if dest_failed is None:
        dest_failed = DEST_FAILED

    # Check destinations exist
    for dest in [dest_auto, dest_needs, dest_failed]:
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)

    # Find segments files
    segments_files = list(source_dir.glob("*_segments.json"))

    if not segments_files:
        if verbose:
            print(f"No *_segments.json files found in {source_dir}")
        return {'total': 0}

    if verbose:
        print(f"Found {len(segments_files)} segmented videos")
        print("-" * 60)

    log_file = source_dir.parent / "triage_log.txt"
    counts = {"good": 0, "warning": 0, "failed": 0, "unsupported_tray": 0}

    for seg_file in sorted(segments_files):
        video_id = seg_file.stem.replace("_segments", "")

        # Check for unsupported tray type FIRST
        if reject_unsupported:
            is_supported, tray_reason, parsed = check_tray_type_supported(seg_file)
            if not is_supported:
                tray_type = parsed.get('tray_type', '?')
                if verbose:
                    print(f"  {video_id}: UNSUPPORTED TRAY TYPE ({tray_type})")

                # Find video file for rejection
                video_path = None
                for ext in [".mp4", ".avi", ".mkv"]:
                    vp = seg_file.parent / f"{video_id}{ext}"
                    if vp.exists():
                        video_path = vp
                        break

                if video_path:
                    was_rejected, reason, files_moved = reject_unsupported_tray_type(
                        video_path, verbose=verbose
                    )
                    if was_rejected:
                        counts["unsupported_tray"] += 1
                        continue
                else:
                    if verbose:
                        print(f"      Cannot find video file, rejecting segments file only")
                    # Still count it but can't move the video
                    counts["unsupported_tray"] += 1
                    continue

        category, reason = classify_segments(seg_file)
        counts[category] += 1

        if category == "good":
            dest = dest_auto
        elif category == "warning":
            dest = dest_needs
        else:
            dest = dest_failed

        associated = get_associated_files(seg_file)

        if associated["dlc_h5"] is None:
            if verbose:
                print(f"  {video_id}: WARNING - No DLC .h5 file, skipping")
            continue

        moved = move_file_bundle(associated, dest, log_file, source_stage=source_dir.name)

        if verbose:
            print(f"  {video_id} -> {dest.name} ({category}: {reason})")

    if verbose:
        print("-" * 60)
        print(f"AutoReview: {counts['good']}, NeedsReview: {counts['warning']}, Failed: {counts['failed']}")
        if counts['unsupported_tray'] > 0:
            print(f"Unsupported Tray (rejected): {counts['unsupported_tray']}")

    return counts


def scan_and_reject_unsupported_trays(
    source_dir: Path = None,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Scan a folder for unsupported tray types and reject them.

    Args:
        source_dir: Folder to scan (default: Processing)
        dry_run: If True, only report what would happen
        verbose: Print progress

    Returns:
        Dict with counts of files found and rejected
    """
    if source_dir is None:
        source_dir = Paths.PROCESSING

    if not source_dir.exists():
        if verbose:
            print(f"[!] Folder does not exist: {source_dir}")
        return {"found": 0, "rejected": 0, "files_moved": 0}

    if verbose:
        mode = "[DRY RUN]" if dry_run else ""
        print("=" * 60)
        print(f"Scanning for Unsupported Tray Types (E/F) {mode}")
        print(f"Source: {source_dir}")
        print("=" * 60)

    # Find all video files
    video_files = []
    for ext in [".mp4", ".avi", ".mkv"]:
        video_files.extend(source_dir.glob(f"*{ext}"))

    # Filter to unique video IDs (exclude preview files)
    seen_ids = set()
    unique_videos = []
    for vf in video_files:
        if "_preview" in vf.name:
            continue
        video_id = vf.stem
        if video_id not in seen_ids:
            seen_ids.add(video_id)
            unique_videos.append(vf)

    if verbose:
        print(f"\nFound {len(unique_videos)} videos to check")

    unsupported = []
    for video_path in sorted(unique_videos):
        is_supported, reason, parsed = check_tray_type_supported(video_path)
        if not is_supported:
            tray_type = parsed.get('tray_type')
            if tray_type in FilePatterns.UNSUPPORTED_TRAY_TYPES:
                unsupported.append((video_path, parsed))

    if verbose:
        print(f"Found {len(unsupported)} unsupported tray type videos\n")

    files_moved = 0
    for video_path, parsed in unsupported:
        video_id = parsed['video_id']
        tray_type = parsed.get('tray_type', '?')

        if verbose:
            print(f"  {video_id} ({tray_type}-type)")

        was_rejected, reason, moved = reject_unsupported_tray_type(
            video_path,
            dry_run=dry_run,
            verbose=verbose
        )
        files_moved += len(moved)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Summary:")
        print(f"  Videos checked:    {len(unique_videos)}")
        print(f"  Unsupported found: {len(unsupported)}")
        print(f"  Files moved:       {files_moved}")
        if dry_run:
            print("\nThis was a DRY RUN. No files were moved.")
            print("Run without --dry-run to execute.")
        print("=" * 60)

    return {
        "found": len(unsupported),
        "rejected": len(unsupported) if not dry_run else 0,
        "files_moved": files_moved if not dry_run else 0
    }


def cli_reject_unsupported():
    """CLI entry point for rejecting unsupported tray types."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan for and reject unsupported tray types (E/F)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mousereach-reject-tray --dry-run  # Preview what would be rejected
    mousereach-reject-tray            # Execute rejection

Supported tray types: P (Pillar)
Unsupported tray types: E (Easy), F (Flat)

Unsupported videos are moved to:
    {NAS_DRIVE}/Unanalyzed/Unsupported_Tray_Type/
"""
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without moving files"
    )

    parser.add_argument(
        "--folder", "-f",
        type=Path,
        default=None,
        help="Folder to scan (default: Processing)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    scan_and_reject_unsupported_trays(
        source_dir=args.folder,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )
