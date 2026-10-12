"""
mousereach.aspa.feed - Feed old single-animal mp4s into DLC queue for reprocessing.

Old mp4s live under the configured NAS output root, one folder per cohort:
    <nas output root>/Analyzed/{cohort}/Single_Animal/

Copies (does NOT move) mp4s to local DLC_Queue or a staging area. The originals
and any retained analyses beside them are never touched: a lab may be required
to keep an old generation of results intact, so everything here works on copies.
Default batch size is 50 to avoid overwhelming the queue.

A cohort is how the archive is FILED, not necessarily what anyone wants to
process. ``--subjects`` / ``--subjects-file`` restrict a run to named animals,
so getting at a handful of animals spread across several cohorts does not drag
in every other animal -- which would cost the same processing hours and land in
the same review queues for a person to work through.

CLI:
    mousereach-aspa-feed --cohort COHORT [--dry-run] [--batch-size 50]
    mousereach-aspa-feed --all --subjects-file my_animals.txt
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

    # WHY: superseded outputs live under Analyzed/Archive/ with their original
    # names. That folder is not a cohort, and walking into it would feed an
    # older generation's videos back into DLC as if they were live.
    from mousereach.pipeline.analyzed_tree import is_superseded_dir, iter_files

    if cohort:
        cohort_dirs = [analyzed / cohort]
    else:
        cohort_dirs = sorted(
            d for d in analyzed.iterdir()
            if d.is_dir() and not is_superseded_dir(d.name)
        )

    for cohort_dir in cohort_dirs:
        sa_dir = cohort_dir / "Single_Animal"
        if not sa_dir.exists():
            continue
        # iter_files, not rglob: same *.mp4 files, but never inside a nested
        # superseded folder (same reason as above).
        for mp4 in sorted(iter_files(sa_dir, "*.mp4")):
            yield cohort_dir.name, mp4


def already_in_queue(mp4_path: Path, queue_dir: Path) -> bool:
    """Check whether the mp4 filename already exists in queue_dir (flat copy)."""
    return (queue_dir / mp4_path.name).exists()


# ---------------------------------------------------------------------------
# Choosing which animals to feed
# ---------------------------------------------------------------------------
#
# WHY A SUBJECT FILTER AND NOT JUST --cohort
# A cohort is the unit the archive is FILED in; it is not necessarily the unit
# anyone wants to process. A study is usually built from some of the animals in
# several cohorts, so feeding whole cohorts to get at them pulls in every other
# animal too -- and every one of those then costs the same processing hours, and
# lands in the same review queues for a person to work through. Filtering here,
# at the point of entry, is the only place that cost is avoidable.
#
# WHICH animals a lab wants is a lab fact and is never written down in this
# repository: it is supplied per run, inline or in a file kept wherever the lab
# keeps such things.

def subject_of(mp4_name) -> str:
    """The animal id in a single-animal filename: {date}_{animal}_{tray}{n}.mp4

    Returns "" when the name does not carry one, which the caller must treat as
    "cannot tell" rather than "does not match" -- silently dropping a file whose
    name we failed to parse would lose data with no trace.
    """
    stem = Path(str(mp4_name)).stem
    parts = stem.split("_")
    return parts[1].strip().upper() if len(parts) >= 3 else ""


def normalise_subject(animal_id: str) -> str:
    """One spelling for comparison, whichever spelling was supplied.

    Archive filenames carry the lab's own id (the short lettered form); the
    pipeline stores the encoded form. A person writing a list should not have to
    know or care which, so both are accepted and reduced to the lab form -- the
    one the filenames being matched actually use.
    """
    from mousereach.aspa.identity import decode_animal, is_encoded

    text = str(animal_id).strip().upper()
    if not text:
        return ""
    if is_encoded(text):
        try:
            return decode_animal(text).upper()
        except (ValueError, KeyError):
            return text
    return text


def load_subjects(inline=None, path=None) -> set:
    """The set of animals to feed, from an inline list and/or a file.

    Accepts commas, whitespace or one-per-line; ignores blank lines and
    anything after a '#'. Both sources may be given and are combined, because a
    lab's standing list plus a couple of extras is a normal thing to want.

    Returns an empty set when neither is supplied, which means "no filter".
    """
    raw = []
    if inline:
        raw.append(str(inline))
    if path:
        raw.append(Path(path).read_text(encoding="utf-8"))

    subjects = set()
    for blob in raw:
        for line in blob.splitlines():
            line = line.split("#", 1)[0]
            for token in line.replace(",", " ").split():
                normalised = normalise_subject(token)
                if normalised:
                    subjects.add(normalised)
    return subjects


def select_subjects(pairs, subjects, seen=None, unparsed=None):
    """Pass through only (cohort, mp4) pairs for the wanted animals.

    ``subjects`` empty means pass everything: a filter nobody asked for must
    never be applied.

    ``seen`` and ``unparsed``, if given, are sets the caller owns and this
    fills in -- so afterwards it can say which requested animals produced no
    videos at all, and which files could not be read. Both are silent failures
    otherwise: a typo in a list looks exactly like an animal with no data.
    """
    for cohort, mp4_path in pairs:
        if not subjects:
            yield cohort, mp4_path
            continue
        animal = subject_of(mp4_path.name)
        if not animal:
            if unparsed is not None:
                unparsed.add(mp4_path.name)
            continue
        if normalise_subject(animal) in subjects:
            if seen is not None:
                seen.add(normalise_subject(animal))
            yield cohort, mp4_path


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

    parser.add_argument("--subjects", metavar="LIST",
                        help="Feed only these animals, e.g. 'A01,A02,A07'. "
                             "Either the lab's own ids or the encoded ids; both "
                             "work. Without this, every animal in the chosen "
                             "cohorts is fed.")
    parser.add_argument("--subjects-file", metavar="PATH",
                        help="Same, read from a file: commas, spaces or one per "
                             "line, '#' starts a comment. Use this for a list "
                             "worth keeping -- and keep it with your own records, "
                             "not in this repository.")
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

    try:
        subjects = load_subjects(args.subjects, args.subjects_file)
    except OSError as e:
        print(f"[FAIL] Could not read the subject list: {e}")
        sys.exit(1)

    print(f"Source NAS root : {nas_root}")
    print(f"Destination     : {queue_dir}")
    print(f"Batch size      : {args.batch_size}")
    if subjects:
        print(f"Animals         : {len(subjects)} named "
              f"(others in these cohorts are left alone)")
    else:
        print("Animals         : every animal in the chosen cohorts")
    if args.dry_run:
        print("Mode            : DRY-RUN (no files will be copied)")
    print()

    cohort_filter = args.cohort if not args.all else None

    copied  = 0
    skipped = 0
    errors  = 0
    seen = set()
    unparsed = set()

    found = find_single_animal_videos(nas_root, cohort_filter)
    for cohort, mp4_path in select_subjects(found, subjects, seen, unparsed):
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

    # A list is a claim about what exists. Checking it costs nothing here and
    # the alternative is a typo that looks exactly like an animal with no data.
    if subjects:
        missing = sorted(subjects - seen)
        if missing:
            print(f"\n  [!] {len(missing)} of the {len(subjects)} animals you "
                  f"named have no single-animal videos in what was searched:")
            print("      %s" % ", ".join(missing))
            print("      Either they are filed under a cohort this run did not "
                  "look at (try --all), or the id is not spelled as the "
                  "filenames spell it.")
        else:
            print(f"\n  All {len(subjects)} named animals were found.")
    if unparsed:
        print(f"\n  [!] {len(unparsed)} file(s) have no readable animal id in "
              f"their name and were left alone, e.g. {sorted(unparsed)[0]}")


if __name__ == "__main__":
    main()
