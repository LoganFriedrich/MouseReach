"""
mousereach.aspa.feed - Feed old single-animal mp4s into DLC queue for reprocessing.

Old mp4s live in:
    Analyzed/{cohort}/Single_Animal/   (on NAS, e.g. X:/! DLC Output/Analyzed/H/Single_Animal/)

Copies (does NOT move) mp4s to local DLC_Queue or a staging area.
Default batch size is 50 to avoid overwhelming the queue.

CLI:
    mousereach-aspa-feed [--cohort H] [--all] [--dry-run] [--batch-size 50]
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_single_animal_videos(
    nas_root: Path,
    cohort: Optional[str] = None,
) -> Iterator[Tuple[str, Path]]:
    """Yield (cohort, mp4_path) for all mp4s under nas_root/Analyzed/{cohort}/Single_Animal/.

    Args:
        nas_root: Root of NAS output tree (contains Analyzed/ subfolder).
        cohort:   If given, restrict to that cohort only.
    """
    analyzed = nas_root / "Analyzed"
    if not analyzed.exists():
        print(f"[!] Analyzed directory not found: {analyzed}")
        return

    if cohort:
        cohort_dirs = [analyzed / cohort]
    else:
        cohort_dirs = sorted(
            d for d in analyzed.iterdir() if d.is_dir()
        )

    for cohort_dir in cohort_dirs:
        sa_dir = cohort_dir / "Single_Animal"
        if not sa_dir.exists():
            continue
        for mp4 in sorted(sa_dir.rglob("*.mp4")):
            yield cohort_dir.name, mp4


def already_in_queue(mp4_path: Path, queue_dir: Path) -> bool:
    """Check whether the mp4 filename already exists in queue_dir (flat copy)."""
    return (queue_dir / mp4_path.name).exists()


# ---------------------------------------------------------------------------
# Copy helper
# ---------------------------------------------------------------------------

def copy_to_queue(
    mp4_path: Path,
    queue_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Copy a single mp4 to queue_dir.

    Returns:
        True if the file was copied (or would be in dry_run), False if skipped.
    """
    dest = queue_dir / mp4_path.name

    if dest.exists():
        # Already present - skip silently
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would copy: {mp4_path.name}")
        return True

    queue_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(mp4_path), str(dest))
    return True


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy old ASPA single-animal mp4s to the DLC queue for reprocessing.\n"
            "Copies files (does NOT move originals)."
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cohort", metavar="COHORT",
                       help="Feed single cohort (e.g. H)")
    group.add_argument("--all", action="store_true",
                       help="Feed all cohorts found under Analyzed/")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without actually copying")
    parser.add_argument("--batch-size", type=int, default=50, metavar="N",
                        help="Maximum number of files to copy per run (default: 50)")
    parser.add_argument("--queue-dir", metavar="PATH",
                        help="Override DLC queue directory (default: MouseReach_PROCESSING_ROOT/DLC_Queue)")

    args = parser.parse_args()

    # Resolve NAS root
    from mousereach.config import Paths, require_nas_drive, require_processing_root

    try:
        nas_drive = require_nas_drive()
        nas_root  = nas_drive / "! DLC Output"
    except Exception as e:
        print(f"[FAIL] Could not resolve NAS root: {e}")
        print("       Set MouseReach_NAS_DRIVE or run mousereach-setup.")
        sys.exit(1)

    # Resolve queue directory
    if args.queue_dir:
        queue_dir = Path(args.queue_dir)
    else:
        try:
            queue_dir = require_processing_root() / "DLC_Queue"
        except Exception as e:
            print(f"[FAIL] Could not resolve DLC_Queue: {e}")
            print("       Set MouseReach_PROCESSING_ROOT, run mousereach-setup, or use --queue-dir.")
            sys.exit(1)

    print(f"Source NAS root : {nas_root}")
    print(f"Destination     : {queue_dir}")
    print(f"Batch size      : {args.batch_size}")
    if args.dry_run:
        print("Mode            : DRY-RUN (no files will be copied)")
    print()

    cohort_filter = args.cohort if not args.all else None

    copied  = 0
    skipped = 0
    errors  = 0

    for cohort, mp4_path in find_single_animal_videos(nas_root, cohort_filter):
        if copied >= args.batch_size:
            print(f"\n[!] Batch size limit ({args.batch_size}) reached. Stopping.")
            print(f"    Run again to continue with the next batch.")
            break

        try:
            was_copied = copy_to_queue(mp4_path, queue_dir, dry_run=args.dry_run)
            if was_copied:
                if not args.dry_run:
                    print(f"  [OK] {cohort} / {mp4_path.name}")
                copied += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  [FAIL] {mp4_path.name}: {exc}")
            errors += 1

    print(f"\nDone.")
    print(f"  Copied : {copied}")
    print(f"  Skipped (already in queue): {skipped}")
    print(f"  Errors : {errors}")
    if args.dry_run:
        print("  (dry-run: no files were actually copied)")


if __name__ == "__main__":
    main()
