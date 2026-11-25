#!/usr/bin/env python3
"""
Video cropping - Split 8-camera collage into single-animal videos

The collage layout is 2x4:
    1  2  3  4     (top row)
    5  6  7  8     (bottom row)

Input: {NAS_ROOT}/Unanalyzed/Multi-Animal/   (collages waiting to be cut)
       20250704_CNT0101,CNT0205,CNT0305,CNT0306,CNT0102,CNT0605,CNT0309,CNT0906_P1.mkv

Output: {NAS_ROOT}/Unanalyzed/Single_Animal/   (singles waiting for a pose)
        20250704_CNT0101_P1.mp4, 20250704_CNT0205_P1.mp4, etc.

Cohort "00" means skip that position (blank/unused).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from mousereach.config import Paths, AnimalID, ConfigurationError


# =============================================================================
# CONFIGURATION - default folders, read from the machine's config WHEN USED
# =============================================================================

# Which configured folder each default comes from: (Paths attribute, subfolder).
# WHY these are looked up at call time and not as module constants: the archive
# default built at import (Paths.ANALYZED_OUTPUT / "Multi-Animal")
# is None / str on a machine without a MouseReach config, so merely
# IMPORTING the cropper -- and with it mousereach.video_prep.core, the prep
# window and every test that touches either -- would crash with a TypeError that
# said nothing about setup. Looking the folder up when a function actually
# needs it keeps behaviour identical on a configured machine, lets an
# unconfigured one import the module (and pass explicit folders), and turns a
# missing default into a message naming the command that fixes it.
_DEFAULT_DIRS = {
    "multi_animal": ("MULTI_ANIMAL_SOURCE", None),
    "single_animal": ("SINGLE_ANIMAL_OUTPUT", None),
    "dlc_queue": ("DLC_QUEUE", None),
    "archive": ("ANALYZED_OUTPUT", "Multi-Animal"),
}


def default_dir(which: str) -> Path:
    """The configured default folder `which` (see _DEFAULT_DIRS), read now.

    Raises ConfigurationError naming mousereach-setup when the setting it
    hangs off (nas_root or processing_root) is not configured.
    """
    attr, sub = _DEFAULT_DIRS[which]
    base = getattr(Paths, attr)
    if base is None:
        raise ConfigurationError(
            f"No default folder for '{which}': MouseReach is not configured "
            f"on this machine (Paths.{attr} is unset).\n\n"
            "Run: mousereach-setup\n\n"
            "or pass the folder explicitly."
        )
    return Path(base) / sub if sub else Path(base)

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
    """Is this grid position an empty box (cohort 00)?

    Delegates to AnimalID, which finds where the letters end rather than
    assuming a three-character prefix. The old ``animal_id[3:5] == "00"``
    matched only 3-letter prefixes: ASPA0001 read 'A0' and ENCR0001 read
    'R0', so empty boxes were cropped, posed and analysed as animals. The
    first 4-letter-prefix import produced 52 such phantoms out of 256
    singles, and no collage carrying one ever satisfied retirement (a
    phantom counts as expected offspring but gets retired, so all_complete
    could never become true). The validator and AnimalID always agreed on
    the right answer; this function was the odd one out.
    """
    return AnimalID.is_blank(animal_id)


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
        
        # subprocess, not os.system: ffmpeg's own error text is the only clue to
        # why a slot failed, and os.system sent it to a console nobody reads --
        # all eight slots of one collage can fail in the same instant with
        # the log saying only "ffmpeg returned 1" (cause never known).
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              errors='replace')
        ret = proc.returncode

        if ret == 0 and output_path.exists():
            results.append({
                'position': pos,
                'animal_id': animal_id,
                'output_path': str(output_path),
                'status': 'success'
            })
        else:
            said = ' | '.join(line.strip() for line in (proc.stderr or '').splitlines()
                              if line.strip())[-500:]
            if ret == 0:
                said = said or 'exit 0 but no output file was written'
            # A failed run can leave a partial file under the single's name;
            # nothing may take it for a cut video.
            try:
                output_path.unlink()
            except OSError:
                pass
            results.append({
                'position': pos,
                'animal_id': animal_id,
                'status': 'failed',
                'error': f'ffmpeg returned {ret}' + (f': {said}' if said else '')
            })

    # Provenance: record what this crop did to the collage's offspring, next to
    # the collage, so "what happened to my children" lives with the collage.
    # (local import avoids a circular dependency with collage_provenance)
    try:
        from .collage_provenance import write_crop_manifest
        write_crop_manifest(input_path, results, output_dir)
    except Exception as _e:
        print(f"  [!] crop manifest not written: {_e}")

    return results


def crop_all(
    input_dir: Path = None,
    output_dir: Path = None,
    verbose: bool = True
) -> dict:
    """Crop all collages in a directory."""
    input_dir = Path(input_dir) if input_dir else default_dir("multi_animal")
    output_dir = Path(output_dir) if output_dir else default_dir("single_animal")
    
    mkv_files = sorted(input_dir.glob("*.mkv"))
    
    if verbose:
        print(f"Found {len(mkv_files)} collages in {input_dir}")
        print(f"Output: {output_dir}\n")
    
    all_results = []
    
    for mkv in mkv_files:
        try:
            # A failed recording (empty, a fraction of a second, unreadable) is
            # never cut: the same check the watcher's intake applies
            # (watcher/recording_issue.py). Raises RecordingIssueError.
            from mousereach.watcher.recording_issue import refuse_if_issue
            refuse_if_issue(mkv)
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

def hand_to_pose(output_paths, verbose: bool = True) -> dict:
    """Hand cut single-animal videos to the pose step, the watcher's way.

    ``output_paths`` are the videos THIS crop made (each crop result's
    ``output_path``); nothing else in their folder is touched. Each one is put
    in the shared singles folder (``Paths.SINGLE_ANIMAL_OUTPUT``), unclaimed,
    where a graphics-card PC's watcher claims it (a rename into
    ``.inflight/<host>/``) and poses it -- exactly one machine per video
    (watcher/single_claim.py). A video already in that folder is left where it
    is: it is already waiting there. Nothing is copied into a DLC queue.

    WHY: the hand crop tools ("Copy outputs to DLC Queue" on the crop tab,
    ``mousereach-crop --queue``, ``mousereach-prep``) copied EVERY mp4 of the
    output folder straight into this machine's DLC queue, with no claim. The
    watcher then adopted the copies as untracked queue files while the same
    videos still waited, unclaimed, in the shared folder -- so another machine
    could claim and pose them too (two poses of one video), and a video that
    was already posed, filed or held was queued again.

    Never overwrites: a video of the same name already in the shared folder,
    or one some machine already holds for pose, is not handed again
    (single_claim.publish_to_front_door). Returns counts and one line per
    video: ``handed`` (put in the shared folder), ``waiting`` (already there),
    ``not_handed`` (refused; see the line), ``lines``.
    """
    from mousereach.watcher import single_claim
    out = {"handed": 0, "waiting": 0, "not_handed": 0, "lines": []}
    door = single_claim.front_door()
    if door is None:
        raise ConfigurationError(
            "No shared singles folder: MouseReach is not configured on this machine "
            "(Paths.SINGLE_ANIMAL_OUTPUT is unset).\n\nRun: mousereach-setup")
    for raw in output_paths or []:
        if not raw:
            continue
        f = Path(raw)
        if single_claim.in_front_door(f):
            out["waiting"] += 1
            line = f"{f.name}: waiting in the shared singles folder for a pose"
        elif single_claim.publish_to_front_door(f) is not None:
            out["handed"] += 1
            line = f"{f.name}: put in the shared singles folder for a pose"
        else:
            out["not_handed"] += 1
            line = (f"{f.name}: NOT handed on -- a video of that name is already in "
                    f"{door} or held for pose by a machine (see the log)")
        out["lines"].append(line)
        if verbose:
            print(f"  {line}")
    return out


def copy_to_dlc_queue(
    source_dir: Path = None,
    dest_dir: Path = None,
    move: bool = False,
    verbose: bool = True
) -> int:
    """Hand the mp4s of ``source_dir`` to the pose step (``hand_to_pose``).

    Kept under its old name for scripts that call it. It no longer copies or
    moves anything into a DLC queue (``dest_dir`` and ``move`` are ignored):
    the only way into a DLC queue is a watcher's claim (``hand_to_pose`` says
    why). Returns how many videos are now waiting for a pose in the shared
    singles folder (handed on, or already there)."""
    source_dir = Path(source_dir) if source_dir else default_dir("single_animal")
    res = hand_to_pose(sorted(source_dir.glob("*.mp4")), verbose=verbose)
    return res["handed"] + res["waiting"]


def archive_collages(
    source_dir: Path = None,
    dest_dir: Path = None,
    verbose: bool = True
) -> int:
    """Move collages whose single-animal videos are ALL done to the archive.

    The watcher's own retirement rule (collage_provenance.
    retire_completed_collages, run by the processing machine every few hours):
    a collage moves only when every one of its single-animal videos is filed
    with the current versions and has no review pending; a collage with a
    child still in the pipeline, held in a review queue or not yet cut stays
    where it is. Returns how many collages moved.

    WHY: ``mousereach-prep --archive`` must not move EVERY collage of the folder
    after cropping, whatever its children's state: a collage whose children
    had failed, were held in a review queue, or had not been cut at all would leave
    the intake folder, and nothing could cut it again from there."""
    from .collage_provenance import (retire_completed_collages, build_downstream_index,
                                     build_complete_stems)
    source_dir = Path(source_dir) if source_dir else default_dir("multi_animal")
    dest_dir = Path(dest_dir) if dest_dir else default_dir("archive")
    summ = retire_completed_collages(source_dir, build_downstream_index(),
                                     complete_stems=build_complete_stems(),
                                     dest_dir=dest_dir, dry_run=False)
    if verbose:
        print(f"Collages in {source_dir}: {summ.get('scanned', 0)}")
        print(f"  moved to {dest_dir} (every child done): {summ.get('retired', 0)}")
        for name in summ.get("moved", []):
            print(f"    {name}")
        print(f"  kept (a child not done yet): {summ.get('not_complete', 0)}")
        print(f"  kept (not cut yet): {summ.get('uncropped', 0)}")
        if summ.get("errors"):
            print(f"  [!] could not be checked or moved: {summ['errors']} (see the log)")
    return int(summ.get("retired", 0))


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
