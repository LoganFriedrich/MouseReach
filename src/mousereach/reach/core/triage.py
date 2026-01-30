"""
triage.py - Sort results into review queues based on anomalies

Now supports GT-based anomaly resolution: if a GT file exists with
human_verified items, those specific anomalies are considered resolved.
"""

from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional


# Anomaly thresholds
MIN_REACHES_PER_SEGMENT = 3    # Suspiciously low
MAX_REACHES_PER_SEGMENT = 100  # Suspiciously high
EXPECTED_SEGMENTS = 20


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
    """
    Check if a segment is human-verified in the GT file.

    Returns True if:
    - GT file exists
    - Segment exists in GT
    - Segment has human_verified=True OR all reaches in segment have human_verified=True
    """
    if not gt_data:
        return False

    for seg in gt_data.get('segments', []):
        if seg.get('segment_num') == segment_num:
            # Check segment-level verification
            if seg.get('human_verified') is True:
                return True

            # Check if all reaches in segment are verified
            reaches = seg.get('reaches', [])
            if reaches:
                # Accept both 'human_verified' and legacy 'human_corrected' as true
                # (human_corrected means human touched it, which implies verification)
                all_verified = all(
                    r.get('human_verified') is True or r.get('human_corrected') is True
                    for r in reaches
                )
                if all_verified:
                    return True

    return False


def load_all_results(input_dir: Path) -> List[Dict]:
    """Load all reach detection results from directory"""
    results = []
    
    for reach_file in input_dir.glob("*_reaches.json"):
        video_name = reach_file.stem.replace('_reaches', '')
        
        try:
            with open(reach_file) as f:
                data = json.load(f)
            
            results.append({
                'video_name': video_name,
                'reach_file': reach_file,
                'total_reaches': data['summary']['total_reaches'],
                'n_segments': data['n_segments'],
                'segments': data['segments'],
                'validated': data.get('validated', False)
            })
        except Exception as e:
            results.append({
                'video_name': video_name,
                'reach_file': reach_file,
                'error': str(e)
            })
    
    return results


def check_anomalies(result: Dict, gt_data: Optional[Dict] = None) -> List[str]:
    """
    Check for anomalies that need review.

    If gt_data is provided, anomalies for segments that are human_verified
    in the GT will be skipped (the human already confirmed the answer).

    v2.4: Added confidence-based anomaly detection.
    Reaches with confidence < 0.85 are flagged for review.
    """
    anomalies = []

    if 'error' in result:
        return ['load_error']

    # Wrong segment count - this is a global issue, can't be resolved per-segment
    if result['n_segments'] != EXPECTED_SEGMENTS:
        anomalies.append(f"n_segments={result['n_segments']} (expected {EXPECTED_SEGMENTS})")

    # Check each segment for issues
    for s in result['segments']:
        seg_num = s.get('segment_num')
        if is_segment_verified_in_gt(gt_data, seg_num):
            continue  # Human verified - skip

        # Reach count anomalies
        if s['n_reaches'] < MIN_REACHES_PER_SEGMENT:
            anomalies.append(f"seg {seg_num}: <{MIN_REACHES_PER_SEGMENT} reaches")
        elif s['n_reaches'] > MAX_REACHES_PER_SEGMENT:
            anomalies.append(f"seg {seg_num}: >{MAX_REACHES_PER_SEGMENT} reaches")

        # v2.4: Check reach confidence (threshold 0.85)
        for r in s.get('reaches', []):
            conf = r.get('confidence', 0)
            if conf < 0.85:
                anomalies.append(f"reach {r.get('reach_id')}: low confidence ({conf:.2f})")

    return anomalies


def get_associated_files(input_dir: Path, video_name: str) -> List[Path]:
    """Get all files associated with a video"""
    return list(input_dir.glob(f"{video_name}*"))


def triage_results(
    input_dir: Path,
    output_base: Path = None,
    verbose: bool = True
) -> Dict:
    """
    [DEPRECATED] Update validation_status in JSON instead of moving files.

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
        reach_file = r['reach_file']

        if 'error' in r:
            category = 'failed'
            reason = r['error']
            validation_status = 'needs_review'
        else:
            # Check for GT file to resolve anomalies
            gt_path = input_dir / f"{video_name}_reach_ground_truth.json"
            gt_data = load_gt_data(gt_path)

            anomalies = check_anomalies(r, gt_data=gt_data)
            if anomalies:
                category = 'needs_review'
                reason = "; ".join(anomalies)
                if gt_data:
                    reason += " (GT exists but not all items verified)"
                validation_status = 'needs_review'
            else:
                category = 'auto_review'
                reason = f"{r['total_reaches']} reaches OK"
                if gt_data:
                    reason += " (resolved by GT)"
                validation_status = 'auto_approved'

        # v2.3+: Update JSON metadata instead of moving files
        try:
            with open(reach_file) as f:
                data = json.load(f)
            data['validation_status'] = validation_status
            data['triage_reason'] = reason
            with open(reach_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if verbose:
                print(f"    [Warning] Could not update {reach_file.name}: {e}")

        counts[category] += 1
        if verbose:
            print(f"  {video_name} → {category} ({reason})")
    
    if verbose:
        print("-" * 70)
        print(f"Auto-review: {counts['auto_review']}, Needs review: {counts['needs_review']}, Failed: {counts['failed']}")
    
    return counts
