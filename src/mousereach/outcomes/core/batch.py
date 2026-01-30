"""
batch.py - Batch processing logic for pellet outcome detection
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional

from .pellet_outcome import PelletOutcomeDetector


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


def find_file_sets(input_dir: Path, skip_if_exists: Optional[List[str]] = None) -> List[Dict]:
    """
    Find matching DLC, segment, and reach files.

    Args:
        input_dir: Directory to search
        skip_if_exists: List of file patterns - skip videos that have matching files.
                       Any glob pattern (e.g., "*outcome_ground_truth.json").
                       Extracts video names from matched files and skips those videos.
    """
    # Find videos to skip based on glob patterns
    skip_video_names = set()
    if skip_if_exists:
        for pattern in skip_if_exists:
            for matched_file in input_dir.glob(pattern):
                # Extract video name from matched file
                # Handle various naming conventions
                stem = matched_file.stem
                # Remove common suffixes to get video name
                for suffix in ['_outcome_ground_truth', '_seg_ground_truth', '_pellet_outcomes',
                              '_reaches', '_segments_v2', '_segments', '_seg_validation']:
                    if stem.endswith(suffix):
                        video_name = stem[:-len(suffix)]
                        skip_video_names.add(video_name)
                        break

    file_sets = []

    for h5_file in input_dir.glob("*DLC_*.h5"):
        video_name = h5_file.stem.split('DLC_')[0]

        # Skip if this video name is in the skip list
        if video_name in skip_video_names:
            continue

        seg_file = None
        for pattern in [f"{video_name}_seg_validation.json", f"{video_name}_segments_v2.json",
                       f"{video_name}_segments.json", f"{video_name}_seg_ground_truth.json"]:
            candidate = input_dir / pattern
            if candidate.exists():
                seg_file = candidate
                break

        reach_file = input_dir / f"{video_name}_reaches.json"
        if not reach_file.exists():
            reach_file = None

        if seg_file:
            file_sets.append({
                'video_name': video_name,
                'dlc_file': h5_file,
                'seg_file': seg_file,
                'reach_file': reach_file
            })

    return file_sets


def process_single(
    dlc_path: Path,
    seg_path: Path,
    reach_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """Process a single video."""
    if output_dir is None:
        output_dir = dlc_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = PelletOutcomeDetector()
    results = detector.detect(dlc_path, seg_path, reach_path)
    
    output_path = output_dir / f"{results.video_name}_pellet_outcomes.json"
    detector.save_results(results, output_path)
    
    return {
        'video_name': results.video_name,
        'n_segments': results.n_segments,
        **results.summary,
        'output_file': str(output_path)
    }


def process_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    copy_sources: bool = True,
    verbose: bool = True,
    skip_if_exists: Optional[List[str]] = None
) -> Dict:
    """
    Process all videos in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory (default: same as input)
        copy_sources: Copy source files to output
        verbose: Print progress
        skip_if_exists: List of glob patterns - skip videos with matching files
                       (e.g., ["*outcome_ground_truth.json"])
    """
    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    file_sets = find_file_sets(input_dir, skip_if_exists)
    
    if not file_sets:
        if verbose:
            print(f"No file sets found in {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0, 'videos': []}
    
    if verbose:
        print(f"Found {len(file_sets)} video(s) to process")
        print("-" * 70)
    
    results = {
        'total': len(file_sets),
        'success': 0,
        'failed': 0,
        'videos': [],
        'processed_at': datetime.now().isoformat()
    }
    
    for i, fs in enumerate(file_sets, 1):
        video_name = fs['video_name']
        
        if verbose:
            print(f"[{i}/{len(file_sets)}] {video_name}...", end=" ")
        
        try:
            video_result = process_single(
                fs['dlc_file'], fs['seg_file'], fs['reach_file'], output_dir
            )
            
            if copy_sources and output_dir != input_dir:
                shutil.copy2(fs['dlc_file'], output_dir / fs['dlc_file'].name)
                shutil.copy2(fs['seg_file'], output_dir / fs['seg_file'].name)
                if fs['reach_file']:
                    shutil.copy2(fs['reach_file'], output_dir / fs['reach_file'].name)
            
            results['success'] += 1
            results['videos'].append({'status': 'success', **video_result})
            
            if verbose:
                s = video_result
                disp = s.get('displaced_sa', 0) + s.get('displaced_outside', 0)
                print(f"OK (R={s.get('retrieved', 0)}/D={disp}/U={s.get('untouched', 0)})")
                
        except Exception as e:
            results['failed'] += 1
            results['videos'].append({'video_name': video_name, 'status': 'failed', 'error': str(e)})
            if verbose:
                print(f"FAILED: {e}")
    
    if verbose:
        print("-" * 70)
        print(f"Complete: {results['success']}/{results['total']} succeeded")
    
    summary_path = output_dir / f"batch_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
