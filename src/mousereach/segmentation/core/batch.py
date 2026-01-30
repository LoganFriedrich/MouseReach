"""
batch.py - Batch segmentation logic with validation status

Architecture (v2.3+):
    - All files stay in single Processing/ folder
    - validation_status stored in JSON metadata
    - No folder-based triage (status determines review queue)

Validation statuses:
    - "auto_approved" - High confidence, 21 boundaries, low CV
    - "needs_review" - Needs human verification
    - "validated" - Human reviewed and approved
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
import shutil

from .segmenter_robust import segment_video_robust, save_segmentation, SEGMENTER_VERSION


def find_dlc_files(input_dir: Path) -> List[Path]:
    """Find all DLC .h5 files in directory"""
    return list(input_dir.glob("*DLC*.h5"))


def add_validation_status(json_path: Path, status: str):
    """Add or update validation_status in a segment JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    data['validation_status'] = status
    data['validation_timestamp'] = datetime.now().isoformat()

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_associated_files(dlc_path: Path, video_id: str) -> List[Path]:
    """Get ALL files associated with a video (everything with video_id prefix)."""
    parent = dlc_path.parent
    files = []

    # Grab everything that starts with the video_id
    for f in parent.iterdir():
        if f.is_file() and f.name.startswith(video_id):
            files.append(f)

    return files


def move_to_folder(files: List[Path], dest_folder: Path, verbose: bool = True):
    """Move files to destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = dest_folder / f.name
        if f.exists() and f != dest:
            shutil.move(str(f), str(dest))


def process_single(dlc_path: Path, output_dir: Optional[Path] = None) -> Dict:
    """
    Process a single DLC file.
    
    Returns dict with video_name, status, and diagnostics.
    """
    if output_dir is None:
        output_dir = dlc_path.parent
    
    # Extract video ID
    video_id = dlc_path.stem.split("DLC")[0]
    output_path = output_dir / f"{video_id}_segments.json"
    
    try:
        boundaries, diag = segment_video_robust(dlc_path)
        save_segmentation(boundaries, diag, output_path)
        
        # Categorize based on what actually matters:
        # - Did we find exactly 21 boundaries?
        # - Are the intervals reasonably consistent?
        has_21 = len(boundaries) == 21
        reasonable_cv = diag.interval_cv < 0.3  # ~500 frame variance OK
        
        if has_21 and reasonable_cv:
            status = "good"
        elif has_21:
            # Got 21 but high CV means timing is off somewhere
            status = "warning"
        else:
            # Wrong number of boundaries - needs human review
            status = "failed"
        
        return {
            'video_name': video_id,
            'status': status,
            'n_boundaries': len(boundaries),
            'interval_cv': diag.interval_cv,
            'anomalies': diag.anomalies,
            'output_file': str(output_path),
            'dlc_path': dlc_path,
            'success': True
        }
        
    except Exception as e:
        return {
            'video_name': video_id,
            'status': 'failed',
            'error': str(e),
            'dlc_path': dlc_path,
            'success': False
        }


def process_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    auto_triage: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process all DLC files in a directory.

    New architecture (v2.3+):
        - All files go to Processing/ folder (single folder)
        - validation_status set in JSON metadata
        - No folder-based triage

    Args:
        input_dir: Directory containing DLC files
        output_dir: Output directory (default: Processing/ under PROCESSING_ROOT)
        auto_triage: Set validation_status based on confidence (default: True)
        verbose: Print progress

    Returns:
        Summary dict with counts and per-video results.
    """
    from mousereach.config import Paths

    input_dir = Path(input_dir)

    # Default output is Processing folder
    if output_dir is None:
        output_dir = Paths.PROCESSING
    else:
        output_dir = Path(output_dir)

    # Ensure Processing folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    dlc_files = find_dlc_files(input_dir)

    if not dlc_files:
        if verbose:
            print(f"No DLC .h5 files found in {input_dir}")
        return {'total': 0, 'good': 0, 'warning': 0, 'failed': 0, 'videos': []}

    if verbose:
        print(f"Segmenter version: {SEGMENTER_VERSION}")
        print(f"Output: {output_dir}")
        print("-" * 60)

    results = {
        'total': len(dlc_files),
        'good': 0,
        'warning': 0,
        'failed': 0,
        'videos': [],
        'processed_at': datetime.now().isoformat()
    }

    for i, dlc_file in enumerate(sorted(dlc_files), 1):
        if verbose:
            print(f"[{i}/{len(dlc_files)}] {dlc_file.name}...", end=" ")

        video_result = process_single(dlc_file, output_dir)
        results[video_result['status']] += 1
        results['videos'].append(video_result)

        if verbose:
            status = video_result['status'].upper()
            if 'error' in video_result:
                print(f"FAILED: {video_result['error']}")
            else:
                print(f"{status} (n={video_result['n_boundaries']}, CV={video_result['interval_cv']:.4f})")

        # Set validation_status in JSON
        if auto_triage and video_result.get('success'):
            json_path = Path(video_result['output_file'])
            video_id = video_result['video_name']

            if video_result['status'] == 'good':
                add_validation_status(json_path, 'auto_approved')
            else:
                add_validation_status(json_path, 'needs_review')

            # Move all associated files to Processing folder (if not already there)
            if input_dir != output_dir:
                files = get_associated_files(dlc_file, video_id)
                move_to_folder(files, output_dir, verbose=False)

        # Update pipeline index
        try:
            from mousereach.index import PipelineIndex
            index = PipelineIndex()
            index.load()
            index.record_file_created(
                Path(video_result['output_file']),
                metadata={
                    "seg_validation": "auto_approved" if video_result['status'] == 'good' else "needs_review",
                    "seg_boundaries": video_result.get('n_boundaries', 0),
                    "seg_confidence": video_result.get('interval_cv', 1.0),
                }
            )
            index.save()
        except Exception:
            pass  # Silent fail - index will catch up on rebuild

    if verbose:
        print("-" * 60)
        print(f"Good: {results['good']}, Warning: {results['warning']}, Failed: {results['failed']}")

        if auto_triage:
            print()
            auto_approved = results['good']
            needs_review = results['warning'] + results['failed']
            print(f"All files output to: {output_dir}")
            print()
            if auto_approved > 0:
                print(f"  {auto_approved} auto-approved (validation_status: auto_approved)")
            if needs_review > 0:
                print(f"  {needs_review} need review (validation_status: needs_review)")
                print()
                print("Next step: Review in Napari")
                print("  mousereach --reviews    # or mousereach-segment-review")

    return results
