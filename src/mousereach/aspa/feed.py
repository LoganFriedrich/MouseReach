"""
mousereach.aspa.feed - Feed ASPA's own single-animal cuts into the pipeline for posing.

ASPA's cuts live in the historical archive, one folder per cohort:
    <NAS root>/Archive/historical/ASPA/{cohort}/Single_Animal/

Copies (does NOT move) mp4s into the shared singles folder
(``Unanalyzed/Single_Animal``), where any GPU node takes them, RENAMED into
pipeline form (``20220602_H09_P1`` -> ``20220602_ASPA0809_P1``, see
:mod:`mousereach.aspa.identity`). Only the videos are copied; ASPA's old pose
files are from a model the pipeline does not use. The originals and any
retained analyses beside them are never touched: a lab may be required to keep
an old generation of results intact, so everything here works on copies.
Default batch size is 50 to avoid overwhelming the queue.

WHY these defaults (changed 2026-10-21): the tool used to read
``<NAS drive>/! DLC Output/Analyzed/{cohort}/Single_Animal`` -- a pipeline root
retired in February -- copy into ONE node's private DLC queue, where no other
node could see or take the videos, and keep ASPA's own names, which the intake
does not accept. ``--source`` and ``--dest`` still point it anywhere.

``--sessions-file <list>`` feeds only the listed sessions (one
``{date}_{animal}_{tray}{position}`` per line, either naming) and names every
listed session it could not find. A video the pipeline already has -- in the
shared folder, claimed by a node, waiting posed, or in its Analyzed folder --
is never fed again.

A cohort is how the archive is FILED, not necessarily what anyone wants to
process. ``--subjects`` / ``--subjects-file`` restrict a run to named animals,
so getting at a handful of animals spread across several cohorts does not drag
in every other animal -- which would cost the same processing hours and land in
the same review queues for a person to work through.

CLI:
    mousereach-aspa-feed --cohort COHORT [--dry-run] [--batch-size 50]
    mousereach-aspa-feed --all --subjects-file my_animals.txt
    mousereach-aspa-feed --sessions-file my_sessions.txt [--dry-run]
"""

import argparse
import os
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
    """Yield (cohort, mp4) under the OLD layout ``nas_root/Analyzed/{cohort}/Single_Animal``.

    Kept for a root still laid out that way; the default source is now the
    historical archive (iter_cohort_singles). Never enters Analyzed/Archive/:
    superseded outputs keep their original names there, and feeding them would
    put an older generation's videos back into DLC as if they were live.
    """
    yield from iter_cohort_singles(Path(nas_root) / "Analyzed", cohort)


def iter_cohort_singles(source_root: Path, cohort: Optional[str] = None
                        ) -> Iterator[Tuple[str, Path]]:
    """Yield (cohort, mp4) for every mp4 under ``source_root/{cohort}/Single_Animal``
    -- the layout of the historical ASPA archive. Never enters a superseded
    folder (same reason as find_single_animal_videos)."""
    from mousereach.pipeline.analyzed_tree import is_superseded_dir, iter_files

    source_root = Path(source_root)
    if not source_root.exists():
        print(f"[!] Source not found: {source_root}")
        return
    dirs = [source_root / cohort] if cohort else sorted(
        d for d in source_root.iterdir() if d.is_dir() and not is_superseded_dir(d.name))
    for cohort_dir in dirs:
        sa_dir = cohort_dir / "Single_Animal"
        if not sa_dir.exists():
            continue
        for mp4 in sorted(iter_files(sa_dir, "*.mp4")):
            yield cohort_dir.name, mp4


def pipeline_name(mp4_name) -> Optional[str]:
    """The pipeline's file name for an ASPA cut (``20220602_H09_P1.mp4`` ->
    ``20220602_ASPA0809_P1.mp4``); None when the name is not a session."""
    from mousereach.aspa.import_collages import normalise_session

    s = normalise_session(Path(str(mp4_name)).stem)
    return f"{s}.mp4" if s else None


def already_in_pipeline(name: str) -> Optional[str]:
    """Where the pipeline already has the video ``name`` (pipeline form), or
    None. WHY: feeding it again would pose and analyse it twice."""
    from mousereach.config import Paths
    from mousereach.watcher import single_claim

    stem = Path(name).stem
    try:
        door = Paths.SINGLE_ANIMAL_OUTPUT
        if door and (Path(door) / name).exists():
            return "waiting in the shared singles folder"
        holder = single_claim.inflight_ids().get(stem)
        if holder:
            return f"claimed by {holder} for posing"
        if Paths.DLC_STAGING and (Path(Paths.DLC_STAGING) / name).exists():
            return "posed, waiting for analysis"
        if Paths.ANALYZED_OUTPUT:
            from mousereach.archive.core import get_archive_destination
            if (get_archive_destination(stem) / name).exists():
                return "analysed"
    except Exception as e:
        return f"could not check ({e}); not fed"
    return None


def feed_one(mp4_path: Path, dest_dir: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Copy one ASPA cut into ``dest_dir`` under its pipeline name, through a
    temporary name, never overwriting. Returns (copied, what happened)."""
    name = pipeline_name(mp4_path.name)
    if name is None:
        return False, "name is not {date}_{animal}_{tray}{position}"
    why = already_in_pipeline(name)
    if why:
        return False, why
    dest = Path(dest_dir) / name
    if dry_run:
        return True, f"would copy as {name}"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    temp = Path(dest_dir) / (name + ".part")
    shutil.copy2(str(mp4_path), str(temp))
    os.utime(temp, None)          # a fresh arrival, not a week-old file
    os.rename(temp, dest)         # refuses an existing destination on Windows
    return True, f"copied as {name}"


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
# CLI main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mousereach-aspa-feed",
        description=(
            "Copy ASPA's own single-animal cuts from the historical archive into the "
            "shared singles folder, renamed into pipeline form, for any GPU node to pose. "
            "Copies files (does NOT move originals)."
        )
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--cohort", metavar="COHORT",
                       help="Feed single cohort (e.g. H)")
    group.add_argument("--all", action="store_true",
                       help="Feed every cohort found in the source")

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
    parser.add_argument("--sessions-file", metavar="PATH",
                        help="Feed only these sessions, one {date}_{animal}_{tray}{position} "
                             "per line (lab or pipeline naming); every listed session not "
                             "found is named. Searches every cohort unless --cohort is given.")
    parser.add_argument("--source", metavar="PATH",
                        help="Folder holding {cohort}/Single_Animal (default: the NAS root's "
                             "Archive/historical/ASPA).")
    parser.add_argument("--dest", "--queue-dir", dest="dest", metavar="PATH",
                        help="Where the copies go (default: the shared singles folder, "
                             "Unanalyzed/Single_Animal, where any GPU node takes them).")

    args = parser.parse_args(argv)
    if not (args.cohort or args.all or args.sessions_file):
        parser.error("one of --cohort, --all or --sessions-file is required")

    from mousereach.aspa.import_collages import default_source_root, read_sessions_file
    from mousereach.config import Paths

    source_root = Path(args.source) if args.source else default_source_root()
    dest_dir = Path(args.dest) if args.dest else (
        Path(Paths.SINGLE_ANIMAL_OUTPUT) if Paths.SINGLE_ANIMAL_OUTPUT else None)
    if source_root is None or dest_dir is None:
        print("[FAIL] Source or destination not configured: run mousereach-setup, "
              "or pass --source and --dest.")
        return 1

    try:
        subjects = load_subjects(args.subjects, args.subjects_file)
        sessions, bad = (read_sessions_file(Path(args.sessions_file))
                         if args.sessions_file else ([], []))
    except OSError as e:
        print(f"[FAIL] Could not read a list: {e}")
        return 1
    wanted = set(sessions)

    print(f"Source      : {source_root}")
    print(f"Destination : {dest_dir}")
    print(f"Batch size  : {args.batch_size}")
    if sessions:
        print(f"Sessions    : {len(sessions)} listed")
        for b in bad[:10]:
            print(f"  [!] not a session name, ignored: {b[:80]}")
    if subjects:
        print(f"Animals     : {len(subjects)} named")
    if args.dry_run:
        print("Mode        : DRY-RUN (nothing is copied)")
    print()

    copied = skipped = errors = 0
    seen, unparsed, found_sessions = set(), set(), set()
    pairs = select_subjects(iter_cohort_singles(source_root, args.cohort), subjects,
                            seen, unparsed)
    for cohort, mp4_path in pairs:
        if wanted:
            name = pipeline_name(mp4_path.name)
            if name is None or Path(name).stem not in wanted:
                continue
            found_sessions.add(Path(name).stem)
        if copied >= args.batch_size:
            print(f"\n[!] Batch size limit ({args.batch_size}) reached. Run again for the next batch.")
            break
        try:
            ok, what = feed_one(mp4_path, dest_dir, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  [FAIL] {mp4_path.name}: {exc}")
            errors += 1
            continue
        print(f"  [{'OK' if ok else 'SKIP'}] {cohort} / {mp4_path.name}: {what}")
        if ok:
            copied += 1
        else:
            skipped += 1

    print(f"\nDone. {'Would copy' if args.dry_run else 'Copied'}: {copied}  "
          f"Skipped: {skipped}  Errors: {errors}")
    if sessions:
        missing = [s for s in sessions if s not in found_sessions]
        if missing:
            print(f"[!] {len(missing)} listed session(s) have no ASPA cut in {source_root}:")
            for s in missing[:20]:
                print(f"    {s}")
        else:
            print(f"[OK] every listed session was found.")
    if subjects:
        missing = sorted(subjects - seen)
        if missing:
            print(f"[!] {len(missing)} named animal(s) have no single-animal videos here: "
                  + ", ".join(missing))
    if unparsed:
        print(f"[!] {len(unparsed)} file name(s) could not be read; not fed.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
