"""Build the expected-session denominator for a pipeline census.

WHY THIS EXISTS
---------------
The folder scan in ``mousereach.dashboard.folder_scan`` counts the videos that
EXIST. That makes "never started" unrepresentable: a session with no file on disk
is invisible to it, so the pipeline cannot report work it has not begun.

The denominator has to come from the COLLAGES instead -- the multi-animal
recordings say which single-animal sessions ought to exist -- UNIONED with every
session that has a real artifact. Both halves are required; see `expected_sessions`.

Every rule below was verified against a live corpus. The docstrings say why, because
a rule whose reason is lost cannot safely be kept or removed.

No lab paths, hostnames or project names appear in this module. All roots are
parameters.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

# A collage name is <date>_<id1,id2,...,idN>_<trailer>. The trailer carries the
# tray position (P1..P4, E1.., F1..) and sometimes variant text.
_STEM_RE = re.compile(r"^(?P<date>\d{8})_(?P<ids>[^_]+)_(?P<trailer>.+)$")
# A clean trailer is exactly one tray letter plus a position number. Anything else
# is a variant (a duplicate, a re-export, a split recording) -- see classify_trailer.
_CLEAN_TRAILER_RE = re.compile(r"^(?P<tray>[A-Z])(?P<pos>\d+)$")


def classify_trailer(trailer):
    """Return 'clean', 'duplicate' or 'variant'.

    WHY: several files can describe ONE session, differing only in trailing text --
    "(2)" re-exports, "-proj" project files, an untrimmed original renamed
    "uncropped", and time-split parts. Counting them all inflates the denominator;
    dropping them blindly loses sessions whose only file is a variant. So classify,
    prefer clean, and keep the rest visible.
    """
    t = trailer.strip()
    if _CLEAN_TRAILER_RE.match(t):
        return "clean"
    if "(2)" in t or t.endswith("-proj") or t.lower().endswith(".llc"):
        return "duplicate"
    return "variant"


def session_key(date, ids, trailer):
    """Identity of a SESSION, independent of which file describes it.

    WHY: the clean file and its "uncropped" twin are the same session. Collapsing on
    (date, animal set, tray position) is what makes "1,421 files -> 1,342 sessions"
    correct rather than a guess.

    The position is the clean tray+number PREFIX of the trailer when one exists
    ("P1 uncropped" -> P1, split parts "P1,1"/"P1,2" -> P1 -- one session, two
    files), falling back to a squashed literal only for trailers with no leading
    position at all (a dual-tray "XP3,YP1" stays its own visible oddity).
    """
    t = trailer.strip()
    m = _CLEAN_TRAILER_RE.match(t) or re.match(r"^([A-Z]\d+)", t)
    pos = m.group(0) if m else re.sub(r"[^A-Za-z0-9]", "", t)[:4]
    return (date, ids, pos)


def tray_of(trailer_or_stem):
    """Tray letter from a trailer or a full single-animal stem, else None.

    WHY: trays are not interchangeable assays. A raised-pillar tray and a level-divot
    tray differ in what the animal must do, so counts that mix them are meaningless.
    """
    s = trailer_or_stem.strip()
    m = _CLEAN_TRAILER_RE.match(s)
    if m:
        return m.group("tray")
    m = re.search(r"_([A-Z])\d+$", s)
    return m.group(1) if m else None


def collect_collages(roots, exts=(".mkv", ".mp4")):
    """Every collage-shaped file under the given roots, recursively.

    WHY recursive: at least one collage directory on a real system contained a
    SUBFOLDER of further collages, invisible to a non-recursive listing.
    """
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and "," in p.stem:
                out.append(p)
    return out


def select_sources(collages):
    """Collapse files to one per session, preferring the clean file.

    Returns (kept_paths, dropped_counts_by_reason).
    """
    by_session = {}
    unparsed = []
    for p in collages:
        m = _STEM_RE.match(p.stem)
        if not m:
            unparsed.append(p)
            continue
        k = session_key(m.group("date"), m.group("ids"), m.group("trailer"))
        by_session.setdefault(k, []).append((classify_trailer(m.group("trailer")), p))

    rank = {"clean": 0, "variant": 1, "duplicate": 2}
    kept = []
    dropped = {"unparsed_name": len(unparsed)}
    for entries in by_session.values():
        entries.sort(key=lambda e: rank.get(e[0], 9))
        kept.append(entries[0][1])
        for kind, _ in entries[1:]:
            dropped[kind] = dropped.get(kind, 0) + 1
    return kept, dropped


def expected_sessions(collage_roots, offspring_fn, found_sessions=None,
                      encode_stem=None, extra_collages=None):
    """The denominator: {session_id: info}, plus diagnostics that must be displayed.

    ``offspring_fn`` is ``collage_provenance.expected_offspring``.
    ``encode_stem`` optionally rewrites a collage stem before expansion, for
    projects whose archived filenames use a different id scheme than the pipeline.
    ``found_sessions`` is every session id that has a real artifact on disk.
    ``extra_collages`` are collage paths the caller already found elsewhere
    (e.g. retired collages met during its own walk of the final output tree);
    they join the same source selection and expansion as the scanned roots.

    WHY found_sessions is required, not optional:
    a collage is retired from the intake folder once all of its children clear, and
    conversely a cropped collage LINGERS there until they do. So the intake folder is
    not a backlog signal in either direction, and a collage-only denominator both
    misses completed work and overstates pending work. The union is the only correct
    denominator. On one real corpus this recovered 169 analysed sessions that no
    surviving collage accounted for.

    WHY parse failures are returned rather than swallowed:
    the underlying parser requires exactly 8 comma-separated ids and raises otherwise;
    ``expected_offspring`` catches that and returns an empty list. On one real corpus
    47 collages (~330 sessions) vanished silently. A session that cannot be parsed
    must surface as a PROBLEM, never as an absence.
    """
    collages = collect_collages(collage_roots) + [Path(p) for p in (extra_collages or ())]
    kept, dropped = select_sources(collages)

    sessions = {}
    blanks = 0
    parse_failures = []

    for path in kept:
        stem = path.stem
        parse_stem = (encode_stem(stem) if encode_stem else None) or stem
        offspring = offspring_fn(parse_stem)
        if not offspring:
            parse_failures.append(path.name)
            continue
        m = _STEM_RE.match(parse_stem)
        tray = tray_of(m.group("trailer")) if m else None
        for o in offspring:
            # Blank camera boxes are not videos. The underlying parser flags them;
            # honour the flag. On one real corpus ~1,500 blank slots would otherwise
            # have entered the denominator as sessions that can never exist.
            if o.get("blank"):
                blanks += 1
                continue
            sid = o.get("offspring_stem")
            if sid:
                sessions[sid] = {"collage": path.name, "tray": tray,
                                 "source": "collage"}

    orphans = 0
    for sid in (found_sessions or ()):
        if sid not in sessions:
            sessions[sid] = {"collage": None, "tray": tray_of(sid),
                             "source": "artifact"}
            orphans += 1

    return {
        "sessions": sessions,
        "diagnostics": {
            "collage_files_seen": len(collages),
            "collage_sessions_after_selection": len(kept),
            "dropped_by_reason": dropped,
            "blank_camera_slots_excluded": blanks,
            "collages_that_parsed_to_nothing": parse_failures,
            "sessions_with_artifact_but_no_collage": orphans,
        },
    }
