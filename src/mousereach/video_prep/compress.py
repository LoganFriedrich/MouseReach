"""
MouseReach Video Compression Utility
================================

Creates 75% resolution copies of videos for review widgets.
Original videos are kept for DLC analysis; compressed versions used for UI.

Naming convention:
    Original: 20250704_CNT0101_P1.mp4
    Compressed: 20250704_CNT0101_P1_preview.mp4
"""

import subprocess
from pathlib import Path
from typing import Optional, List
import shutil


def get_preview_path(video_path: Path) -> Path:
    """Get the path for the compressed preview version of a video."""
    return video_path.parent / f"{video_path.stem}_preview.mp4"


def has_preview(video_path: Path) -> bool:
    """Check if a compressed preview exists for this video."""
    return get_preview_path(video_path).exists()


def create_preview(
    video_path: Path,
    scale: float = 0.75,
    crf: int = 28,
    overwrite: bool = False,
    max_size_mb: int = 500
) -> Optional[Path]:
    """
    Create a compressed preview version of a video.

    Args:
        video_path: Path to original video
        scale: Resolution scale factor (0.75 = 75% resolution)
        crf: FFmpeg CRF value (18-28, higher = smaller file, lower quality)
        overwrite: If True, overwrite existing preview
        max_size_mb: Target maximum file size in MB (will adjust compression)

    Returns:
        Path to preview file, or None if failed
    """
    preview_path = get_preview_path(video_path)

    if preview_path.exists() and not overwrite:
        return preview_path

    # Check ffmpeg is available
    if not shutil.which('ffmpeg'):
        print("ERROR: ffmpeg not found in PATH")
        return None

    # Get video info to estimate output size
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Estimate memory needed for in-memory loading (frames * h * w * 3 bytes)
        scaled_w = int(width * scale)
        scaled_h = int(height * scale)
        estimated_mb = (n_frames * scaled_h * scaled_w * 3) / (1024 * 1024)

        # If estimated memory > max_size_mb * 50 (rough heuristic), increase compression
        if estimated_mb > max_size_mb * 50:
            # Aggressively compress large videos
            crf = min(35, crf + 10)
            scale = min(scale, 0.5)
            print(f"Large video detected ({estimated_mb:.0f}MB estimated), using aggressive compression (scale={scale}, crf={crf})")

    # Build ffmpeg command
    # -vf scale: resize, ensuring even dimensions (H.264 requirement)
    # trunc(iw*scale/2)*2 rounds down to nearest even number
    # -c:v libx264: H.264 codec (widely compatible)
    # -crf: quality (18=high, 35=very low, 23=default)
    # -preset fast: encoding speed vs compression tradeoff
    # -an: no audio (these videos don't have audio anyway)
    scale_filter = f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"
    cmd = [
        'ffmpeg',
        '-y' if overwrite else '-n',  # overwrite or skip
        '-i', str(video_path),
        '-vf', scale_filter,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', 'fast',
        '-an',  # no audio
        str(preview_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and preview_path.exists():
            # Verify the file is valid (has non-zero size)
            if preview_path.stat().st_size > 1000:
                orig_size = video_path.stat().st_size / (1024 * 1024)
                new_size = preview_path.stat().st_size / (1024 * 1024)
                print(f"Created preview: {preview_path.name} ({orig_size:.1f}MB -> {new_size:.1f}MB, {new_size/orig_size*100:.0f}%)")
                return preview_path
            else:
                # Corrupt/empty file
                preview_path.unlink()
                print(f"FFmpeg created invalid file, removed")
                return None
        else:
            # Clean up any partial file
            if preview_path.exists():
                preview_path.unlink()
            print(f"FFmpeg failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout creating preview for {video_path.name}")
        if preview_path.exists():
            preview_path.unlink()
        return None
    except Exception as e:
        print(f"Error creating preview: {e}")
        if preview_path.exists():
            preview_path.unlink()
        return None


def create_previews_batch(
    folder: Path,
    pattern: str = "*.mp4",
    scale: float = 0.75,
    crf: int = 28,
    overwrite: bool = False
) -> dict:
    """
    Create preview versions for all videos in a folder.

    Args:
        folder: Directory containing videos
        pattern: Glob pattern for video files
        scale: Resolution scale factor
        crf: FFmpeg CRF value
        overwrite: If True, overwrite existing previews

    Returns:
        Dict with 'created', 'skipped', 'failed' counts
    """
    results = {'created': 0, 'skipped': 0, 'failed': 0}

    videos = list(folder.glob(pattern))
    # Filter out preview files themselves
    videos = [v for v in videos if '_preview' not in v.stem]

    print(f"Found {len(videos)} videos in {folder}")

    for i, video in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {video.name}...")

        if has_preview(video) and not overwrite:
            print(f"  Skipped (preview exists)")
            results['skipped'] += 1
            continue

        preview = create_preview(video, scale, crf, overwrite)
        if preview:
            results['created'] += 1
        else:
            results['failed'] += 1

    return results


def main():
    """CLI entry point for batch preview creation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create compressed preview versions of videos for MouseReach review widgets"
    )
    parser.add_argument(
        'folder',
        type=Path,
        help="Folder containing videos to compress"
    )
    parser.add_argument(
        '--pattern', '-p',
        default="*.mp4",
        help="Glob pattern for video files (default: *.mp4)"
    )
    parser.add_argument(
        '--scale', '-s',
        type=float,
        default=0.75,
        help="Resolution scale factor (default: 0.75)"
    )
    parser.add_argument(
        '--crf', '-q',
        type=int,
        default=28,
        help="FFmpeg CRF quality (18-28, higher=smaller, default: 28)"
    )
    parser.add_argument(
        '--overwrite', '-f',
        action='store_true',
        help="Overwrite existing preview files"
    )

    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"ERROR: {args.folder} is not a directory")
        return 1

    results = create_previews_batch(
        args.folder,
        args.pattern,
        args.scale,
        args.crf,
        args.overwrite
    )

    print(f"\nDone: {results['created']} created, {results['skipped']} skipped, {results['failed']} failed")
    return 0


if __name__ == "__main__":
    exit(main())
