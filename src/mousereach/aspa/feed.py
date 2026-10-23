"""
mousereach.aspa.feed - Bring in ASPA sessions whose recording is lost, by rebuilding it.

The input to MouseReach is the multi-animal recording; the pipeline cuts it
itself, and that cut records which recording each single came from. ASPA's
archive (``<NAS root>/Archive/historical/ASPA/{cohort}/``) holds its recordings
(``Multi-Animal/``) and its own single-animal cuts (``Single_Animal/``). A
session whose recording survives comes in through the recording import
(``mousereach-aspa-import``), never here. For a session whose recording is
lost but whose cuts survive, this tool REBUILDS the recording from the cuts
(video_prep/core/rebuild.py: each cut back in its grid cell, checked to line up
in time) and writes it into the recording intake, renamed into pipeline form
(``H09`` -> ``ASPA0809``, see :mod:`mousereach.aspa.identity`).

WHICH cuts share a rebuilt recording is read from the cuts themselves: every
cut of one recording is exactly as long as the recording, to the frame, so the
lost cuts of one day and tray that have the same frame count (and frame rate)
are the cuts of one recording. WHY not the groups of other days (it used them
until 2026-10-23): groups change from day to day -- on some days the cuts
fall into id-order groups that no surviving recording shows -- and a group
copied from another day stitched together cuts that were never simultaneous,
or left an animal alone that had shared its recording.

For a person, never rebuilt:

- cuts as long as a recording that SURVIVES that day and tray (within
  ``SURVIVOR_TOLERANCE`` frames): they came out of that recording, whose name
  does not list them -- a seat named as empty, or under another animal's id.
  Which animal sat where is an identity question; a rebuild would enter the
  animal a second time from a recording that never existed;
- cuts as long as another cut of that day and tray whose recording survives:
  they came out of the same recording. This catches what the first test cannot
  -- a surviving recording that was trimmed in time after it was cut (or kept
  untrimmed when the cuts were made from a trimmed copy) is no longer as long
  as its cuts, but its cuts are still as long as each other;
- cuts shorter than ``MIN_SESSION_SECONDS``: no session fits in them (a run
  presents twenty pellets one at a time);
- more than eight cuts of one length, or a cut whose length cannot be read.

WHERE each animal sat: the order of a recorded group (same tray kind, the
nearest day) that holds every animal of the rebuilt recording; with none, id
order -- the order the lab filled recordings in. Which of the two applies is
written into the rebuild's record. A position nobody's cut fills is empty, so
the pipeline's cut never produces a video of an animal that is not there.
Every rebuild is recorded under ``<NAS root>/pipeline_records/rebuilt_recordings/``.

WHY NOT FEED THE CUTS (it did until 2026-10-23): a single dropped into the
intake has no link to its recording, and the recording itself never entered
the pipeline. The originals and any retained analyses are never touched: a lab
may be required to keep an old generation of results intact, so everything
here only reads the archive.

``--sessions-file <list>`` brings in only the recordings holding the listed
sessions (one ``{date}_{animal}_{tray}{position}`` per line, either naming)
and names every listed session it could not find. ``--subjects`` /
``--subjects-file`` restrict a run to named animals. A recording whose sessions
the pipeline already has (in any stage, or analysed) is not rebuilt; when it
has only some of them, their seats are left empty in the rebuild, so no
session is ever made twice.

CLI:
    mousereach-aspa-feed --cohort COHORT [--dry-run] [--batch-size 50]
    mousereach-aspa-feed --all --subjects-file my_animals.txt
    mousereach-aspa-feed --sessions-file my_sessions.txt [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


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


_CUT = re.compile(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z]+\d+)\.mp4$")
# A recording's name, with anything after the tray (" uncropped", " (2)", ...)
# allowed: for PLACING animals any surviving copy shows the group and order.
_RECORDING = re.compile(r"^(\d{8})_([A-Za-z0-9]+(?:,[A-Za-z0-9]+){7})_([A-Za-z]+\d+)"
                        r"(.*)\.(?:mkv|mp4)$", re.IGNORECASE)


def cohort_recordings(source_root: Path, cohort: str) -> List[Tuple[str, List[str], str, str]]:
    """``(date, [8 ids], tray, file name)`` for every recording the archive has
    for ``cohort`` (``{cohort}/Multi-Animal``). Names that do not carry exactly
    eight ids are not recordings this tool can place animals by."""
    d = Path(source_root) / cohort / "Multi-Animal"
    out = []
    if d.is_dir():
        for n in sorted(x.name for x in d.iterdir()):
            m = _RECORDING.match(n)
            if m:
                out.append((m.group(1), m.group(2).split(","), m.group(3), n))
    return out


# A cut and the recording it came from can differ by a frame or two (each was
# encoded separately), and a recording's length is read from its header, which
# can be a frame or two off as well.
SURVIVOR_TOLERANCE = 5
# No session fits in less: a run presents twenty pellets one at a time.
MIN_SESSION_SECONDS = 60


def recording_length(path: Path) -> Optional[Tuple[int, float]]:
    """(frames, fps) of a recording from its header, without reading it through
    (a recording is a gigabyte or more); None when it cannot be opened."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        return (n, fps) if n > 0 and fps > 0 else None
    finally:
        cap.release()


def surviving_recordings(source_root: Path, cohort: str, d: str, tray: str) -> List[Path]:
    """Every video in the cohort's recordings folder for that day and tray,
    however its name is spelled (``_P1``, ``,IP1``, `` uncropped``...)."""
    folder = Path(source_root) / cohort / "Multi-Animal"
    if not folder.is_dir():
        return []
    pat = re.compile(rf"^{d}_.*[_,][A-Za-z]*{re.escape(tray)}(?!\d)", re.IGNORECASE)
    return sorted(x for x in folder.iterdir()
                  if x.suffix.lower() in (".mkv", ".mp4") and pat.match(x.name))


def cut_timing(path: Path) -> Tuple[int, float]:
    """(frame count, frame rate) of one cut, exact (rebuild.probe)."""
    from mousereach.video_prep.core import rebuild as rb
    pr = rb.probe(Path(path))
    return pr["frames"], pr["fps"]


def plan_recordings(cuts: List[Tuple[str, Path]], source_root: Path,
                    selected: Optional[set] = None, timing=None):
    """Group the selected cuts into the recordings to rebuild.

    Returns ``(units, skipped)``. Each unit is ``(cohort, date, tray, [8 raw
    ids or None for an empty position], {raw id: cut path}, where the
    positions came from)``; ``skipped`` is ``[(cut name, why)]`` -- above all,
    a cut whose own recording survives in the archive (that recording comes in
    through the recording import).

    ``cuts`` must be EVERY cut of the cohorts concerned: which cuts share a
    recording is judged on all the lost cuts of a day and tray. ``selected``
    (cut paths), when given, keeps only the recordings that hold at least one
    selected cut, and only those days are measured. ``timing(path) -> (frames,
    fps)`` measures a cut (default: cut_timing, which reads the file)."""
    from datetime import date as _date
    from mousereach.video_prep.core.rebuild import RebuildRefused
    timing = timing or cut_timing
    recs: Dict[str, list] = {}
    skipped = []
    # What the recording import will bring in, read the way IT reads names
    # (import_collages.scan/choose), so a recording it imports is never also
    # rebuilt here: {(date, tray, encoded animal id)}.
    from mousereach.aspa.import_collages import choose, scan
    from mousereach.aspa.identity import encode_animal
    by_session, _ = scan(Path(source_root), sorted({c for c, _ in cuts}))
    importable = {}
    for enc_stem, cands in by_session.items():
        pick = choose(cands)
        if pick is None:
            continue
        m = re.match(r"^(\d{8})_(.+)_([A-Za-z]+\d+)$", enc_stem)
        if m:
            for e in m.group(2).split(","):
                importable[(m.group(1), m.group(3), e)] = pick["source_stem"]

    def day(s):
        # None for a name whose date is not a real day (a typo such as 20220631):
        # that file is named and left alone, never allowed to stop the run.
        try:
            return _date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None

    lost: Dict[tuple, Dict[str, Path]] = {}
    # Cuts whose own recording survives, by day and tray, with that recording.
    kept: Dict[tuple, List[Tuple[Path, str]]] = {}
    for cohort, mp4 in cuts:
        m = _CUT.match(mp4.name)
        if not m:
            skipped.append((mp4.name, "name is not {date}_{animal}_{tray}{position}"))
            continue
        d, aid, tray = m.groups()
        if day(d) is None:
            skipped.append((mp4.name, f"{d} is not a real date"))
            continue
        try:
            own = importable.get((d, tray, encode_animal(aid)))
        except ValueError:
            own = None
        if own:
            skipped.append((mp4.name, f"its recording survives ({own}): it comes in "
                                      f"through the recording import, not as cuts"))
            kept.setdefault((cohort, d, tray), []).append((mp4, own))
            continue
        lost.setdefault((cohort, d, tray), {})[aid] = mp4

    def kind(tray):
        return re.match(r"[A-Za-z]+", tray).group(0)

    # The lost cuts of one day and tray that are the same length to the frame
    # are the cuts of one recording (see the module docstring for why lengths
    # and not other days' groups).
    out = []
    for (cohort, d, tray), have in sorted(lost.items()):
        if selected is not None and not any(p in selected for p in have.values()):
            continue
        if cohort not in recs:
            recs[cohort] = cohort_recordings(source_root, cohort)
        by_length: Dict[tuple, List[str]] = {}
        for aid in sorted(have):
            try:
                by_length.setdefault(tuple(timing(have[aid])), []).append(aid)
            except (RebuildRefused, OSError) as e:
                skipped.append((have[aid].name, f"its length cannot be read ({e}): "
                                "which recording it came from is for a person to say"))
        groups = [(r[0], r[1]) for r in recs[cohort]
                  if day(r[0]) is not None and kind(r[2]) == kind(tray)]
        survivors = siblings = None
        for length, members in sorted(by_length.items(), key=lambda kv: kv[1]):
            def person(why):
                for a in members:
                    skipped.append((have[a].name, why))
            if length[0] < MIN_SESSION_SECONDS * length[1]:
                person(f"{length[0]} frames ({length[0] / length[1]:.1f} s): too short to "
                       "be a session; for a person")
                continue
            if survivors is None:
                survivors = [(r, recording_length(r))
                             for r in surviving_recordings(source_root, cohort, d, tray)]
            unread = [r.name for r, n in survivors if n is None]
            same = [r.name for r, n in survivors
                    if n is not None and abs(n[0] - length[0]) <= SURVIVOR_TOLERANCE]
            if same:
                person(f"as long as {same[0]}, which survives and does not list this "
                       "animal: which seat it had is for a person to say")
                continue
            if unread:
                person(f"cannot be compared with {unread[0]} (its length cannot be "
                       "read): for a person")
                continue
            if siblings is None:
                siblings = [(c, rec, recording_length(c))
                            for c, rec in kept.get((cohort, d, tray), [])]
            twin = [(c.name, rec) for c, rec, n in siblings
                    if n is not None and abs(n[0] - length[0]) <= SURVIVOR_TOLERANCE]
            if twin:
                person(f"as long as {twin[0][0]}, which came out of {twin[0][1]}: it "
                       "came out of that recording too, whose name does not list this "
                       "animal -- which seat it had is for a person to say")
                continue
            blind = [c.name for c, _, n in siblings if n is None]
            if blind:
                person(f"cannot be compared with {blind[0]} (its length cannot be "
                       "read): for a person")
                continue
            if len(members) > 8:
                for a in members:
                    skipped.append((have[a].name, f"{len(members)} cuts that day are "
                                    f"{length[0]} frames long, more than one recording "
                                    "holds: which recording it came from is for a person "
                                    "to say"))
                continue
            # Positions: a recorded group holding every one of these animals,
            # fewest other animals first, then the nearest day.
            best = None
            for d2, g in groups:
                named = {a for a in g if not a.isdigit()}      # digits = an empty box
                if set(members) <= named:
                    key = (len(named) - len(members), abs((day(d2) - day(d)).days), d2)
                    if best is None or key < best[0]:
                        best = (key, g, d2)
            if best:
                ids = [a if a in members else None for a in best[1]]
                basis = f"positions as recorded on {best[2]}"
            else:
                ids = members + [None] * (8 - len(members))
                basis = ("positions in id order: no recording of this group is in "
                         "the archive")
            if selected is None or any(have[a] in selected for a in members):
                out.append((cohort, d, tray, ids, {a: have[a] for a in members}, basis))
    return out, skipped


def encoded_recording(d: str, tray: str, ids: List[Optional[str]]) -> Tuple[str, Dict[str, str]]:
    """The pipeline name of a rebuilt recording and ``{encoded id: raw id}``.
    An empty position gets a blank id (``ASPA00{position}``)."""
    from mousereach.aspa.identity import encode_animal
    enc, back = [], {}
    for pos, aid in enumerate(ids, start=1):
        if aid is None:
            enc.append(encode_animal(f"{pos:03d}"))
        else:
            e = encode_animal(aid)
            enc.append(e)
            back[e] = aid
    return f"{d}_{','.join(enc)}_{tray}", back


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
            "Bring in ASPA sessions whose recording is lost: rebuild the recording "
            "from ASPA's own single-animal cuts and write it into the recording intake, "
            "renamed into pipeline form. Only reads the archive."
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
                        help="Check and show what would be rebuilt; write nothing")
    parser.add_argument("--batch-size", type=int, default=50, metavar="N",
                        help="Maximum number of recordings to rebuild per run (default: 50)")
    parser.add_argument("--sessions-file", metavar="PATH",
                        help="Feed only these sessions, one {date}_{animal}_{tray}{position} "
                             "per line (lab or pipeline naming); every listed session not "
                             "found is named. Searches every cohort unless --cohort is given.")
    parser.add_argument("--source", metavar="PATH",
                        help="Folder holding {cohort}/Single_Animal (default: the NAS root's "
                             "Archive/historical/ASPA).")
    parser.add_argument("--dest", "--queue-dir", dest="dest", metavar="PATH",
                        help="Where rebuilt recordings go (default: the recording intake, "
                             "Unanalyzed/Multi-Animal).")

    args = parser.parse_args(argv)
    if not (args.cohort or args.all or args.sessions_file):
        parser.error("one of --cohort, --all or --sessions-file is required")

    from mousereach.aspa.import_collages import default_source_root, read_sessions_file
    from mousereach.config import Paths

    source_root = Path(args.source) if args.source else default_source_root()
    dest_dir = Path(args.dest) if args.dest else (
        Path(Paths.MULTI_ANIMAL_SOURCE) if Paths.MULTI_ANIMAL_SOURCE else None)
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
        print("Mode        : DRY-RUN (nothing is written)")
    print()

    from mousereach.video_prep.core.rebuild import RebuildRefused, plan, rebuild

    built = skipped = errors = 0
    seen, unparsed, found_sessions = set(), set(), set()
    chosen = []
    for cohort, mp4_path in select_subjects(iter_cohort_singles(source_root, args.cohort),
                                            subjects, seen, unparsed):
        name = pipeline_name(mp4_path.name)
        if wanted:
            if name is None or Path(name).stem not in wanted:
                continue
            found_sessions.add(Path(name).stem)
        chosen.append((cohort, mp4_path))
    # Every cut of the cohorts concerned decides whether a group is complete;
    # the selection only decides which recordings to build.
    every = [pair for c in sorted({c for c, _ in chosen})
             for pair in iter_cohort_singles(source_root, c)]
    units, skips = plan_recordings(every, source_root, selected={p for _, p in chosen})
    chosen_names = {p.name for _, p in chosen}
    skips = [(cut, why) for cut, why in skips if cut in chosen_names]
    for cut, why in skips:
        print(f"  [SKIP] {cut}: {why}")
        skipped += 1
    records = (Path(Paths.NAS_ROOT) / "pipeline_records" / "rebuilt_recordings"
               if Paths.NAS_ROOT else None)
    analysed_dir = (Path(Paths.ANALYZED_OUTPUT) / "Multi-Animal") if Paths.ANALYZED_OUTPUT else None
    # A recording already in the intake, cut and filed, or held in quarantine
    # for a person is never built again.
    from mousereach.aspa.import_collages import default_quarantine_dir
    homes = [x for x in (dest_dir, analysed_dir, default_quarantine_dir()) if x]
    for cohort, d, tray, ids, cuts_by_raw, basis in units:
        stem, back = encoded_recording(d, tray, ids)
        if any((x / f"{stem}.mkv").exists() for x in homes):
            print(f"  [SKIP] {stem}: already in the pipeline")
            skipped += 1
            continue
        have = {e: already_in_pipeline(f"{d}_{e}_{tray}.mp4") for e in back}
        if have and all(have.values()):
            print(f"  [SKIP] {stem}: every session is already in the pipeline")
            skipped += 1
            continue
        there = {back[e]: why for e, why in have.items() if why}
        if there:
            # A session the pipeline already has is never made a second time:
            # its seat is left empty, and the rest of the recording comes in.
            ids = [None if a in there else a for a in ids]
            for raw, why in sorted(there.items()):
                print(f"  [SKIP] {d}_{raw}_{tray}: already in the pipeline ({why}); its "
                      f"seat is left empty in the rebuilt recording")
                skipped += 1
            stem, back = encoded_recording(d, tray, ids)
            if any((x / f"{stem}.mkv").exists() for x in homes):
                print(f"  [SKIP] {stem}: already in the pipeline")
                continue
        if built >= args.batch_size:
            print(f"\n[!] Batch size limit ({args.batch_size}) reached. Run again for the next batch.")
            break
        singles = {e: cuts_by_raw[raw] for e, raw in back.items()}
        try:
            p = plan(stem, singles)
            if args.dry_run:
                print(f"  [OK] {cohort}: would rebuild {stem} ({len(singles)} cuts, "
                      f"{8 - len(singles)} empty, {p['frames']} frames; {basis})")
            else:
                path = rebuild(stem, singles, dest_dir, records,
                               f"ASPA {cohort} {d} {tray}: the recording is not in the "
                               f"archive; rebuilt from ASPA's own cuts, the ones that day "
                               f"of exactly this length; {basis}")
                print(f"  [OK] {cohort}: rebuilt {path.name} ({basis})")
            built += 1
        except RebuildRefused as exc:
            print(f"  [FAIL] {stem}: {exc}")
            errors += 1

    print(f"\nDone. {'Would rebuild' if args.dry_run else 'Rebuilt'}: {built} recording(s)  "
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
