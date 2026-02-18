#!/usr/bin/env python3
"""
DLC batch processing - Wrapper for DeepLabCut batch inference

Pipeline paths are now configured via environment variables.
Set MouseReach_PROCESSING_ROOT to customize the location.
"""

from pathlib import Path
from typing import List, Optional
import shutil
from mousereach.config import Paths


# Default paths - derived from configurable environment variables
DEFAULT_DLC_QUEUE = Paths.DLC_QUEUE
DEFAULT_DLC_COMPLETE = Paths.DLC_COMPLETE
# Note: DLC config path is user-selected via file dialog, not hardcoded


def fix_config_yaml(config_path: Path) -> Path:
    """
    Fix common YAML issues in DLC config files.
    
    For analysis, we don't need the video_sets section (which often has
    broken multi-line paths). We strip it out entirely.
    
    Returns path to a cleaned temp config file in the same directory
    (to preserve relative model paths).
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_video_sets = False
    
    for line in lines:
        # Detect start of video_sets section
        if line.strip().startswith('video_sets:'):
            in_video_sets = True
            # Keep the video_sets key but make it empty
            fixed_lines.append('video_sets: {}\n')
            continue
        
        # Detect end of video_sets section (next non-indented, non-comment line)
        if in_video_sets:
            # If line is not indented and not empty/comment, we're out of video_sets
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                in_video_sets = False
                fixed_lines.append(line)
            # Skip all video_sets content
            continue
        
        fixed_lines.append(line)
    
    # Write to same directory as original to preserve relative model paths
    temp_config = config_path.parent / f"config_mousereach_temp.yaml"
    
    with open(temp_config, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    return temp_config


def find_videos_for_dlc(
    input_dir: Path = None,
    extensions: List[str] = None,
    reject_unsupported_tray: bool = True
) -> List[Path]:
    """
    Find videos that need DLC processing.

    Returns videos that don't have a corresponding DLC output file.
    Optionally filters out unsupported tray types (E, F).
    """
    from mousereach.config import is_supported_tray_type

    input_dir = Path(input_dir or DEFAULT_DLC_QUEUE)
    extensions = extensions or ['.mp4', '.avi']

    videos = []
    for ext in extensions:
        videos.extend(input_dir.glob(f'*{ext}'))

    # Filter out those that already have DLC output
    needs_processing = []
    skipped_tray = []
    for video in videos:
        # Check for unsupported tray types (E, F)
        if reject_unsupported_tray and not is_supported_tray_type(video.name):
            skipped_tray.append(video.name)
            continue

        h5_files = list(video.parent.glob(f"{video.stem}DLC*.h5"))
        if not h5_files:
            needs_processing.append(video)

    # Report skipped files
    if skipped_tray:
        print(f"[!] Skipped {len(skipped_tray)} unsupported tray type videos (E/F):")
        for name in skipped_tray[:5]:  # Show first 5
            print(f"    - {name}")
        if len(skipped_tray) > 5:
            print(f"    ... and {len(skipped_tray) - 5} more")
        print("    Use 'mousereach-reject-tray' to move these to the unsupported folder.")

    return sorted(needs_processing)


def run_dlc_batch(
    video_paths: List[Path],
    config_path: Path,
    output_dir: Path = None,
    gpu: int = 0,
    save_as_csv: bool = True
) -> List[dict]:
    """
    Run DeepLabCut batch inference.

    NOTE: Requires DeepLabCut to be installed.
    NOTE: You must train your own DLC model for your specific camera/animal setup.

    Args:
        video_paths: List of video paths to process
        config_path: Path to your trained DLC model's config.yaml (required)
        output_dir: Output directory (default: same as video)
        gpu: GPU device number (None for CPU)
        save_as_csv: Also save CSV output

    Returns:
        List of result dicts
    """
    config_path = Path(config_path)
    
    try:
        import deeplabcut
    except ImportError:
        raise ImportError(
            "DeepLabCut not installed.\n"
            "Install with: pip install deeplabcut\n"
            "Or use conda: conda install -c conda-forge deeplabcut"
        )
    
    if not config_path.exists():
        raise FileNotFoundError(f"DLC config not found: {config_path}")
    
    # Fix any YAML issues in config
    try:
        fixed_config = fix_config_yaml(config_path)
        print(f"Using cleaned config: {fixed_config}")
    except Exception as e:
        print(f"Warning: Could not preprocess config ({e}), using original")
        fixed_config = config_path
    
    results = []
    
    for video_path in video_paths:
        try:
            dest = str(output_dir) if output_dir else str(video_path.parent)
            
            print(f"Processing: {video_path.name}")
            
            deeplabcut.analyze_videos(
                str(fixed_config),
                [str(video_path)],
                destfolder=dest,
                save_as_csv=save_as_csv,
                gputouse=gpu
            )
            
            results.append({
                'video': str(video_path),
                'status': 'success'
            })
            print(f"  [OK] Complete")

        except Exception as e:
            results.append({
                'video': str(video_path),
                'status': 'failed',
                'error': str(e)
            })
            print(f"  [FAIL] Failed: {e}")
    
    return results


def move_completed_to_output(
    source_dir: Path = None,
    dest_dir: Path = None,
    verbose: bool = True,
    create_previews: bool = True
) -> int:
    """Move videos with DLC output to Processing folder.

    Also creates compressed preview videos for review widgets.
    """
    source_dir = Path(source_dir or DEFAULT_DLC_QUEUE)
    # v2.3+: Move to Processing/ instead of DLC_Complete/
    dest_dir = Path(dest_dir or Paths.PROCESSING)
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for mp4 in source_dir.glob("*.mp4"):
        # Skip preview files
        if '_preview' in mp4.stem:
            continue

        # Check if DLC output exists
        h5_files = list(source_dir.glob(f"{mp4.stem}DLC*.h5"))
        if h5_files:
            # Create preview video before moving (saves memory in review widgets)
            if create_previews:
                try:
                    from mousereach.video_prep.compress import create_preview, get_preview_path
                    preview_path = get_preview_path(mp4)
                    if not preview_path.exists():
                        if verbose:
                            print(f"  Creating preview: {mp4.stem}_preview.mp4")
                        create_preview(mp4)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not create preview: {e}")

            # Move video and all DLC outputs
            files_to_move = [mp4] + h5_files
            csv_files = list(source_dir.glob(f"{mp4.stem}DLC*.csv"))
            files_to_move.extend(csv_files)

            # Also move preview if it exists
            preview_path = source_dir / f"{mp4.stem}_preview.mp4"
            if preview_path.exists():
                files_to_move.append(preview_path)

            for f in files_to_move:
                shutil.move(str(f), str(dest_dir / f.name))
                if verbose:
                    print(f"  Moved: {f.name}")
            moved += 1

    return moved


def run_dlc_workflow(
    input_dir: Path = None,
    config_path: Path = None,
    gpu: int = 0,
    move_when_done: bool = True,
    verbose: bool = True
) -> dict:
    """
    Full DLC processing workflow:
    1. Find videos needing processing
    2. Run DLC on each
    3. Move completed to Processing/
    """
    input_dir = Path(input_dir or DEFAULT_DLC_QUEUE)
    config_path = Path(config_path or DEFAULT_DLC_CONFIG)
    
    if verbose:
        print("=" * 60)
        print("MouseReach DLC Processing")
        print("=" * 60)
        print(f"Input:  {input_dir}")
        print(f"Config: {config_path}")
        print()
    
    # Find videos
    videos = find_videos_for_dlc(input_dir)
    
    if verbose:
        print(f"Found {len(videos)} videos to process\n")
    
    if not videos:
        return {'total': 0, 'success': 0, 'failed': 0}
    
    # Run DLC
    results = run_dlc_batch(videos, config_path, gpu=gpu)
    
    success = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - success
    
    if verbose:
        print(f"\nDLC complete: {success}/{len(results)} successful")
    
    # Move completed
    if move_when_done and success > 0:
        if verbose:
            print("\nMoving completed files...")
        moved = move_completed_to_output(input_dir, verbose=verbose)
        if verbose:
            print(f"Moved {moved} video sets to Processing/")
    
    return {
        'total': len(videos),
        'success': success,
        'failed': failed,
        'results': results
    }
