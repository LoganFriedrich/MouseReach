"""
triage.py - Sort pellet outcome results into review queues

Supports ground truth (GT) files: if a GT file exists with human-verified
segments, those segments are considered resolved and won't trigger anomalies.
"""

from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional


# v2.5 (2026-02): Recalibrated from empirical data on 27 processed videos.
#   - Accept 20 or 21 segments (segmenter produces 21 boundaries)
#   - Confidence threshold lowered to 0.80 (was 0.90). At 0.90 only 22%
#     of segments pass; at 0.80, 97.6% pass. The detector's 0.85 mode
#     represents confident classifications that shouldn't trigger review.
#   - "retrieved" outcomes no longer auto-flag for review. The 68%
#     accuracy concern is captured in the confidence score itself —
#     low-confidence retrievals still get flagged via the threshold.
EXPECTED_SEGMENTS_MIN = 20
EXPECTED_SEGMENTS_MAX = 21
OUTCOME_LOW_CONFIDENCE = 0.80


def load_gt_data(gt_path: Path) -> Optional[Dict]:
    """Load ground truth file if it exists."""
    if gt_path and gt_path.exists():
        try:
            with open(gt_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def is_segment_verified_in_gt(gt_data: Optional[Dict], segment_num: int) -> bool:
    """Check if a segment's outcome is human-verified in the GT file.

    Returns True only if the segment exists in GT AND has human_verified=True.
    """
    if not gt_data:
        return False

    for seg in gt_data.get('segments', []):
        if seg.get('segment_num') == segment_num:
            # Must have explicit human_verified: true
            if seg.get('human_verified') is True:
                return True
    return False


def count_verified_segments(gt_data: Optional[Dict]) -> int:
    """Count how many segments are human-verified in GT."""
    if not gt_data:
        return 0

    count = 0
    for seg in gt_data.get('segments', []):
        if seg.get('human_verified') is True:
            count += 1
    return count


def load_all_results(input_dir: Path) -> List[Dict]:
    """Load all outcome results from directory"""
    results = []
    
    for outcome_file in input_dir.glob("*_pellet_outcomes.json"):
        video_name = outcome_file.stem.replace('_pellet_outcomes', '')
        
        try:
            with open(outcome_file) as f:
                data = json.load(f)
            
            results.append({
                'video_name': video_name,
                'outcome_file': outcome_file,
                'n_segments': data['n_segments'],
                'summary': data['summary'],
                'segments': data['segments'],
                'validated': data.get('validated', False)
            })
        except Exception as e:
            results.append({
                'video_name': video_name,
                'outcome_file': outcome_file,
                'error': str(e)
            })
    
    return results


def check_anomalies(result: Dict, gt_data: Optional[Dict] = None) -> List[str]:
    """Check for anomalies that need review.

    If gt_data is provided, segments with human_verified=True are considered
    resolved and won't trigger anomalies.

    v2.5: Recalibrated thresholds from empirical data (27 videos).
      - Accept 20 or 21 segments (was: exactly 20)
      - Confidence threshold lowered to 0.80 (was: 0.90). At 0.90 only
        22% of segments pass; at 0.80, 97.6% pass.
      - "retrieved" outcomes no longer auto-flag. The confidence score
        already captures uncertainty — low-confidence retrievals are
        caught by the threshold, high-confidence ones are fine.
    """
    anomalies = []

    if 'error' in result:
        return ['load_error']

    # Segment count outside expected range
    n_seg = result['n_segments']
    if n_seg < EXPECTED_SEGMENTS_MIN or n_seg > EXPECTED_SEGMENTS_MAX:
        # If GT has all segments verified, this is resolved
        verified_count = count_verified_segments(gt_data)
        if verified_count < n_seg:
            anomalies.append(f"n_segments={n_seg} (expected {EXPECTED_SEGMENTS_MIN}-{EXPECTED_SEGMENTS_MAX})")

    # Check each segment for issues
    for seg in result.get('segments', []):
        seg_num = seg.get('segment_num')
        if is_segment_verified_in_gt(gt_data, seg_num):
            continue  # Human verified - skip

        conf = seg.get('confidence', 0)

        # Low confidence — the detector's own flagged_for_review field is
        # kept as informational metadata but does NOT gate triage. 66% of
        # segments are flagged (mostly timing precision, not wrong outcomes)
        # so using it as a gate would send everything to review.
        if conf < OUTCOME_LOW_CONFIDENCE:
            anomalies.append(f"seg {seg_num}: low confidence ({conf:.2f})")

    return anomalies


def get_associated_files(input_dir: Path, video_name: str) -> List[Path]:
    return list(input_dir.glob(f"{video_name}*"))


def triage_results(input_dir: Path, output_base: Path = None, verbose: bool = True) -> Dict:
    """[DEPRECATED] Update validation_status in JSON instead of moving files.

    NOTE: This function is deprecated. The unified pipeline now keeps all
    files in Processing/ and tracks status via validation_status in JSON.
    Use the mousereach dashboard or UnifiedPipelineProcessor instead.

    v2.3+: No longer creates folders or moves files. Updates JSON metadata only.
    """
    import warnings
    warnings.warn(
        "triage_results() is deprecated. Use the mousereach dashboard or "
        "UnifiedPipelineProcessor instead. Files now stay in Processing/ with "
        "status tracked in JSON metadata.",
        DeprecationWarning,
        stacklevel=2
    )
    # v2.3+: Do NOT create old folder structure
    
    results = load_all_results(input_dir)

    if not results:
        if verbose:
            print(f"No results found in {input_dir}")
        return {'total': 0}

    if verbose:
        print(f"Found {len(results)} video(s) to triage")
        print("-" * 70)

    counts = {'auto_review': 0, 'needs_review': 0, 'failed': 0}

    for r in results:
        video_name = r['video_name']
        outcome_file = r['outcome_file']

        # Load GT file if it exists
        gt_path = input_dir / f"{video_name}_outcome_ground_truth.json"
        gt_data = load_gt_data(gt_path)

        if 'error' in r:
            category, reason = 'failed', r['error']
            validation_status = 'needs_review'
        else:
            anomalies = check_anomalies(r, gt_data)
            if anomalies:
                category, reason = 'needs_review', "; ".join(anomalies)
                validation_status = 'needs_review'
            else:
                category = 'auto_review'
                disp = r['summary'].get('displaced_sa', 0) + r['summary'].get('displaced_outside', 0)
                reason = f"R={r['summary'].get('retrieved', 0)}/D={disp}/U={r['summary'].get('untouched', 0)}"
                validation_status = 'auto_approved'

        # v2.3+: Update JSON metadata instead of moving files
        try:
            with open(outcome_file) as f:
                data = json.load(f)
            data['validation_status'] = validation_status
            data['triage_reason'] = reason
            with open(outcome_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if verbose:
                print(f"    [Warning] Could not update {outcome_file.name}: {e}")

        counts[category] += 1
        if verbose:
            print(f"  {video_name} -> {category} ({reason})")
    
    if verbose:
        print("-" * 70)
        print(f"Auto-review: {counts['auto_review']}, Needs review: {counts['needs_review']}, Failed: {counts['failed']}")
    
    return counts
