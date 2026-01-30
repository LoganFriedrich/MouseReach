"""
advance.py - Move validated results to next pipeline stage
"""

from pathlib import Path
import json
import shutil
from datetime import datetime
import os
from typing import List, Dict

from .reach_detector import ReachDetector
from mousereach.utils import get_username


def get_associated_files(input_dir: Path, video_name: str) -> List[Path]:
    """Get all files associated with a video"""
    return list(input_dir.glob(f"{video_name}*"))


def mark_as_validated(reach_file: Path, notes: str = "") -> bool:
    """
    Mark a reaches file as validated.
    Updates the file in place with validation info.
    """
    try:
        with open(reach_file) as f:
            data = json.load(f)
        
        data['validated'] = True
        data['validated_by'] = get_username()
        data['validated_at'] = datetime.now().isoformat()
        if notes:
            data['validation_notes'] = notes
        
        with open(reach_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error marking {reach_file} as validated: {e}")
        return False


def advance_videos(
    input_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict:
    """
    Move validated videos to next pipeline stage.
    
    Files are marked as validated before moving.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find videos by reach files
    video_names = set()
    for f in input_dir.glob("*_reaches.json"):
        video_name = f.stem.replace('_reaches', '')
        video_names.add(video_name)
    
    if not video_names:
        if verbose:
            print(f"No videos found in {input_dir}")
        return {'total': 0, 'advanced': 0}
    
    if verbose:
        print(f"Found {len(video_names)} video(s) to advance")
        print("-" * 60)
    
    advanced = 0
    for video_name in sorted(video_names):
        if verbose:
            print(f"  {video_name}...", end=" ")
        
        try:
            # Mark reaches file as validated
            reach_file = input_dir / f"{video_name}_reaches.json"
            if reach_file.exists():
                mark_as_validated(reach_file)

            # Move all associated files
            moved_files = []
            for f in get_associated_files(input_dir, video_name):
                shutil.move(str(f), str(output_dir / f.name))
                moved_files.append(f.name)

            # Update pipeline index
            if moved_files:
                try:
                    from mousereach.index import PipelineIndex
                    index = PipelineIndex()
                    index.load()
                    index.record_files_moved(video_name, input_dir.name, output_dir.name, moved_files)
                    index.record_validation_changed(video_name, "reach", "validated")
                    index.save()
                except Exception as idx_e:
                    if verbose:
                        print(f"(index warning: {idx_e})", end=" ")

            advanced += 1
            if verbose:
                print("OK")
                
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
    
    if verbose:
        print("-" * 60)
        print(f"Advanced {advanced}/{len(video_names)} videos to {output_dir}")
    
    return {'total': len(video_names), 'advanced': advanced}
