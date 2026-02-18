"""
batch.py - Batch processing logic for reach detection

VALIDATION GATE: Only processes files with validation_status of
"validated" or "auto_approved". Files with "needs_review" are blocked.
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from .reach_detector import ReachDetector, VideoReaches


def get_associated_files(input_dir: Path, video_name: str) -> List[Path]:
    """Get ALL files associated with a video (everything with video_name prefix)."""
    files = []
    for f in input_dir.iterdir():
        if f.is_file() and f.name.startswith(video_name):
            files.append(f)
    return files


def move_to_folder(files: List[Path], dest_folder: Path, verbose: bool = True):
    """Move files to destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = dest_folder / f.name
        if f.exists() and f != dest:
            shutil.move(str(f), str(dest))


def check_validation_status(seg_path: Path) -> Tuple[bool, str]:
    """
    Check if a segment file is validated and can proceed.
    
    Returns: (can_proceed: bool, status: str)
    """
    try:
        with open(seg_path, 'r') as f:
            data = json.load(f)
        
        status = data.get('validation_status', None)
        
        # Old files without validation_status are assumed OK (backwards compat)
        if status is None:
            # Check for old-style 'validated' flag
            if data.get('validated', False):
                return True, 'legacy_validated'
            # No status at all - treat as needs review to be safe
            return False, 'no_status'
        
        if status in ('validated', 'auto_approved'):
            return True, status
        else:
            return False, status
            
    except Exception as e:
        return False, f'error: {e}'


def find_file_pairs(
    input_dir: Path,
    check_validation: bool = True,
    skip_if_exists: Optional[List[str]] = None
) -> Tuple[List[Tuple[Path, Path, str]], List[Dict]]:
    """
    Find matching DLC .h5 and segment JSON files.

    Args:
        input_dir: Directory to search
        check_validation: If True, filter out unvalidated files
        skip_if_exists: List of file patterns - skip videos that have matching files.
                       Any glob pattern (e.g., "*reach_ground_truth.json").
                       Extracts video names from matched files and skips those videos.

    Returns:
        - List of valid (dlc_path, seg_path, video_name) tuples
        - List of skipped files with reasons
    """
    # Find videos to skip based on glob patterns
    skip_video_names = set()
    if skip_if_exists:
        for pattern in skip_if_exists:
            for matched_file in input_dir.glob(pattern):
                # Extract video name from matched file
                stem = matched_file.stem
                # Remove common suffixes to get video name
                for suffix in ['_reach_ground_truth', '_seg_ground_truth', '_reaches',
                              '_pellet_outcomes', '_segments_v2', '_segments', '_seg_validation']:
                    if stem.endswith(suffix):
                        video_name = stem[:-len(suffix)]
                        skip_video_names.add(video_name)
                        break

    valid_pairs = []
    skipped = []

    for h5_file in input_dir.glob("*DLC*.h5"):
        video_name = h5_file.stem.split('DLC')[0]
        if video_name.endswith('_'):
            video_name = video_name[:-1]

        # Skip if this video name is in the skip list
        if video_name in skip_video_names:
            skipped.append({
                'video_name': video_name,
                'reason': 'matched_skip_pattern'
            })
            continue

        seg_file = None
        for pattern in [
            f"{video_name}_segments.json",
            f"{video_name}_seg_validation.json",
            f"{video_name}_segments_v2.json",
            f"{video_name}_seg_ground_truth.json",
        ]:
            candidate = input_dir / pattern
            if candidate.exists():
                seg_file = candidate
                break

        if not seg_file:
            skipped.append({
                'video_name': video_name,
                'reason': 'no_segments_file'
            })
            continue

        # Check validation status
        if check_validation:
            can_proceed, status = check_validation_status(seg_file)
            if not can_proceed:
                skipped.append({
                    'video_name': video_name,
                    'reason': f'validation_status: {status}'
                })
                continue

        valid_pairs.append((h5_file, seg_file, video_name))

    return valid_pairs, skipped


def process_single(
    dlc_path: Path,
    seg_path: Path,
    output_dir: Optional[Path] = None
) -> Dict:
    """Process a single video. Returns summary dict."""
    if output_dir is None:
        output_dir = dlc_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = ReachDetector()
    results = detector.detect(dlc_path, seg_path)
    
    output_path = output_dir / f"{results.video_name}_reaches.json"
    detector.save_results(results, output_path)
    
    return {
        'video_name': results.video_name,
        'total_reaches': results.summary['total_reaches'],
        'n_segments': results.n_segments,
        'output_file': str(output_path)
    }


def process_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    copy_sources: bool = True,
    skip_validation_check: bool = False,
    verbose: bool = True,
    skip_if_exists: Optional[List[str]] = None
) -> Dict:
    """
    Process all videos in a directory.

    Args:
        input_dir: Directory containing validated segment files
        output_dir: Output directory (default: same as input)
        copy_sources: Copy source files to output dir
        skip_validation_check: If True, process even unvalidated files (not recommended)
        verbose: Print progress
        skip_if_exists: List of glob patterns - skip videos with matching files
                       (e.g., ["*reach_ground_truth.json"])
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, skipped = find_file_pairs(
        input_dir,
        check_validation=not skip_validation_check,
        skip_if_exists=skip_if_exists
    )
    
    if verbose and skipped:
        print(f"[!] Skipped {len(skipped)} file(s) - not validated:")
        for s in skipped:
            print(f"   {s['video_name']}: {s['reason']}")
        print()
        if not pairs:
            print("No validated files to process.")
            print("Run Step 2 review (napari) to validate files first.")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': skipped, 'videos': []}
    
    if not pairs:
        if verbose:
            print(f"No validated file pairs found in {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': skipped, 'videos': []}
    
    if verbose:
        print(f"Found {len(pairs)} validated video(s) to process")
        print("-" * 60)
    
    results = {
        'total': len(pairs),
        'success': 0,
        'failed': 0,
        'skipped': skipped,
        'videos': [],
        'processed_at': datetime.now().isoformat()
    }
    
    for i, (dlc_file, seg_file, video_name) in enumerate(pairs, 1):
        if verbose:
            print(f"[{i}/{len(pairs)}] {video_name}...", end=" ")
        
        try:
            video_result = process_single(dlc_file, seg_file, output_dir)
            
            if copy_sources and output_dir != input_dir:
                shutil.copy2(dlc_file, output_dir / dlc_file.name)
                shutil.copy2(seg_file, output_dir / seg_file.name)
            
            results['success'] += 1
            results['videos'].append({'status': 'success', **video_result})
            
            if verbose:
                print(f"OK ({video_result['total_reaches']} reaches)")
                
        except Exception as e:
            results['failed'] += 1
            results['videos'].append({
                'video_name': video_name,
                'status': 'failed',
                'error': str(e)
            })
            if verbose:
                print(f"FAILED: {e}")
    
    if verbose:
        print("-" * 60)
        print(f"Complete: {results['success']}/{results['total']} succeeded")
    
    # Save batch summary
    summary_path = output_dir / f"batch_reaches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
