#!/usr/bin/env python3
"""
Video cropping - Split 8-camera collage into single-animal videos

The collage layout is 2x4:
    1  2  3  4     (top row)
    5  6  7  8     (bottom row)

Input: {NAS_DRIVE}/Unanalyzed/Multi-Animal/
       20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv

Output: {NAS_DRIVE}/Unanalyzed/Single_Animal/
        20250704_CNT0101_P1.mp4, 20250704_CNT0205_P1.mp4, etc.

Cohort "00" means skip that position (blank/unused).
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from mousereach.config import Paths, AnimalID


# =============================================================================
# CONFIGURATION - Derived from environment variables
# =============================================================================

# Default paths - now configured via MouseReach_NAS_DRIVE and MouseReach_PROCESSING_ROOT
DEFAULT_MULTI_ANIMAL = Paths.MULTI_ANIMAL_SOURCE
DEFAULT_SINGLE_ANIMAL = Paths.SINGLE_ANIMAL_OUTPUT
DEFAULT_DLC_QUEUE = Paths.DLC_QUEUE
DEFAULT_ARCHIVE = Paths.ANALYZED_OUTPUT / "Multi-Animal"

# Crop coordinates: (width, height, x_offset, y_offset)
# Collage is 1920x1080, grid is 4x2, each cell is 480x540
CROP_COORDS = [
    (480, 540, 0, 0),      # Position 1: Top-left
    (480, 540, 480, 0),    # Position 2: Top-center-left
    (480, 540, 960, 0),    # Position 3: Top-center-right
    (480, 540, 1440, 0),   # Position 4: Top-right
    (480, 540, 0, 540),    # Position 5: Bottom-left
    (480, 540, 480, 540),  # Position 6: Bottom-center-left
    (480, 540, 960, 540),  # Position 7: Bottom-center-right
    (480, 540, 1440, 540)  # Position 8: Bottom-right
]


# =============================================================================
# PARSING
# =============================================================================

def is_blank_animal(animal_id: str) -> bool:
    """Check if animal ID is blank (cohort 00)."""
    # Format: {letters}{cohort:2d}{subject:2d}
    # e.g., CNT0001 has cohort "00" at positions 3:5
    if len(animal_id) >= 5:
        return animal_id[3:5] == "00"
    return False


def get_experiment_code(animal_id: str) -> str:
    """Extract experiment code from animal ID."""
    # Find where letters end
    for i, c in enumerate(animal_id):
        if c.isdigit():
            return animal_id[:i]
    return animal_id


def parse_collage_filename(filename: str) -> dict:
    """
    Parse multi-animal collage filename.
    
    Example: 20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Cannot parse filename: {filename}")
    
    date = parts[0]
    animal_ids_str = parts[1]
    last_part = '_'.join(parts[2:])  # e.g., "P1" or "E1_extra"
    
    animal_ids = animal_ids_str.split(',')
    if len(animal_ids) != 8:
        raise ValueError(f"Expected 8 animal IDs, got {len(animal_ids)}")
    
    return {
        'date': date,
        'animal_ids': animal_ids,
        'last_part': last_part,
        'original': filename
    }


# =============================================================================
# CROPPING
# =============================================================================

def crop_collage(
    input_path: Path,
    output_dir: Path,
    verbose: bool = True
) -> List[dict]:
    """
    Crop 8-camera collage into single-animal videos using ffmpeg.
    
    Args:
        input_path: Path to collage video (.mkv)
        output_dir: Directory for output videos
        verbose: Print progress
        
    Returns:
        List of result dicts
    """
    info = parse_collage_filename(input_path.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Processing: {input_path.name}")
    
    results = []
    
    for i, (animal_id, coords) in enumerate(zip(info['animal_ids'], CROP_COORDS)):
        pos = i + 1
        
        # Skip blank positions
        if is_blank_animal(animal_id):
            if verbose:
                print(f"  [{pos}/8] {animal_id} - SKIPPED (blank)")
            results.append({
                'position': pos,
                'animal_id': animal_id,
                'status': 'skipped',
                'reason': 'blank_cohort_00'
            })
            continue
        
        width, height, x_off, y_off = coords
        output_name = f"{info['date']}_{animal_id}_{info['last_part']}.mp4"
        output_path = output_dir / output_name
        
        # ffmpeg command (matches original script)
        cmd = (
            f'ffmpeg -y -i "{input_path}" '
            f'-filter:v "crop={width}:{height}:{x_off}:{y_off}" '
            f'-c:a copy "{output_path}" -loglevel error'
        )
        
        if verbose:
            print(f"  [{pos}/8] {animal_id} -> {output_name}")
        
        ret = os.system(cmd)
        
        if ret == 0 and output_path.exists():
            results.append({
                'position': pos,
                'animal_id': animal_id,
                'output_path': str(output_path),
                'status': 'success'
            })
        else:
            results.append({
                'position': pos,
                'animal_id': animal_id,
                'status': 'failed',
                'error': f'ffmpeg returned {ret}'
            })
    
    return results


def crop_all(
    input_dir: Path = None,
    output_dir: Path = None,
    verbose: bool = True
) -> dict:
    """Crop all collages in a directory."""
    input_dir = Path(input_dir or DEFAULT_MULTI_ANIMAL)
    output_dir = Path(output_dir or DEFAULT_SINGLE_ANIMAL)
    
    mkv_files = sorted(input_dir.glob("*.mkv"))
    
    if verbose:
        print(f"Found {len(mkv_files)} collages in {input_dir}")
        print(f"Output: {output_dir}\n")
    
    all_results = []
    
    for mkv in mkv_files:
        try:
            results = crop_collage(mkv, output_dir, verbose)
            all_results.extend(results)
        except Exception as e:
            print(f"ERROR: {mkv.name} - {e}")
            all_results.append({'file': mkv.name, 'status': 'error', 'error': str(e)})
        if verbose:
            print()
    
    success = sum(1 for r in all_results if r.get('status') == 'success')
    skipped = sum(1 for r in all_results if r.get('status') == 'skipped')
    failed = sum(1 for r in all_results if r.get('status') in ('failed', 'error'))
    
    return {
        'total_collages': len(mkv_files),
        'success': success,
        'skipped': skipped,
        'failed': failed,
        'results': all_results
    }


# =============================================================================
# FILE MOVEMENT
# =============================================================================

def copy_to_dlc_queue(
    source_dir: Path = None,
    dest_dir: Path = None,
    move: bool = False,
    verbose: bool = True
) -> int:
    """Copy/move cropped videos to DLC queue."""
    source_dir = Path(source_dir or DEFAULT_SINGLE_ANIMAL)
    dest_dir = Path(dest_dir or DEFAULT_DLC_QUEUE)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    mp4_files = list(source_dir.glob("*.mp4"))
    action = "Moving" if move else "Copying"
    
    if verbose:
        print(f"{action} {len(mp4_files)} videos to {dest_dir}")
    
    count = 0
    for f in mp4_files:
        try:
            if move:
                shutil.move(str(f), str(dest_dir / f.name))
            else:
                shutil.copy2(str(f), str(dest_dir / f.name))
            count += 1
            if verbose:
                print(f"  {f.name}")
        except Exception as e:
            print(f"  ERROR: {f.name} - {e}")
    
    return count


def archive_collages(
    source_dir: Path = None,
    dest_dir: Path = None,
    verbose: bool = True
) -> int:
    """Move processed collages to archive."""
    source_dir = Path(source_dir or DEFAULT_MULTI_ANIMAL)
    dest_dir = Path(dest_dir or DEFAULT_ARCHIVE)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    mkv_files = list(source_dir.glob("*.mkv"))
    
    if verbose:
        print(f"Archiving {len(mkv_files)} collages to {dest_dir}")
    
    count = 0
    for f in mkv_files:
        try:
            shutil.move(str(f), str(dest_dir / f.name))
            count += 1
            if verbose:
                print(f"  {f.name}")
        except Exception as e:
            print(f"  ERROR: {f.name} - {e}")
    
    return count


def sort_to_experiment_folders(
    source_dir: Path,
    dest_base: Path = None,
    move: bool = True,
    verbose: bool = True
) -> dict:
    """Sort videos into experiment folders (CNT, ENCR, OPT, etc.)"""
    dest_base = Path(dest_base or Paths.ANALYZED_OUTPUT)
    
    # Find all relevant files
    patterns = ['*.mp4', '*.h5', '*.json']
    files = []
    for pattern in patterns:
        files.extend(source_dir.glob(pattern))
    
    if verbose:
        print(f"Sorting {len(files)} files by experiment")
    
    counts = {}
    for f in files:
        # Extract animal ID from filename (e.g., CNT0101 from 20250704_CNT0101_P1.mp4)
        parts = f.stem.split('_')
        if len(parts) >= 2:
            animal_id = parts[1].split(',')[0]  # Handle comma-separated multi-animal
            project, cohort = AnimalID.get_project_and_cohort(animal_id)
            label = f"{project}/{cohort}"
        else:
            label = 'UNKNOWN'
            project, cohort = 'UNKNOWN', 'UNKNOWN'

        dest_dir = dest_base / project / cohort
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if move:
                shutil.move(str(f), str(dest_dir / f.name))
            else:
                shutil.copy2(str(f), str(dest_dir / f.name))
            
            counts[label] = counts.get(label, 0) + 1
            if verbose:
                print(f"  {f.name} -> {label}/")
        except Exception as e:
            print(f"  ERROR: {f.name} - {e}")
    
    return counts


# =============================================================================
# MKV TO MP4 CONVERSION
# =============================================================================

def convert_mkv_to_mp4(input_path: Path, output_path: Path = None) -> Path:
    """Convert single MKV to MP4."""
    input_path = Path(input_path)
    output_path = Path(output_path or input_path.with_suffix('.mp4'))
    
    cmd = f'ffmpeg -y -i "{input_path}" -c copy "{output_path}" -loglevel error'
    os.system(cmd)
    
    return output_path
