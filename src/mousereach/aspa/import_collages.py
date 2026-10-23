"""Bring ASPA multi-animal collages into the MouseReach pipeline.

What this does
--------------
Copies ASPA collages out of the read-only historical archive into the watcher's
intake (``Unanalyzed/Multi-Animal``), renaming each one into pipeline form on the
way (see :mod:`mousereach.aspa.identity`). Originals and the old ASPA analyses are
never touched -- this only ever READS the archive.

Scope: cohorts D and later
--------------------------
Only ``OptD`` .. ``OptG`` plus ``H`` .. ``M`` are in scope. Excluded:

- ``ABS1/2/3``, ``GI-A,B``, ``OptA/B/C`` -- the earlier generation, whose collages
  are named with animal RANGES (``20210924_ABS2_13-15_f``) rather than the eight
  ids the pipeline needs.
- ``N1``/``N2`` -- despite the later letters these are Dec 2021 / Jan 2022,
  i.e. EARLIER than OptD, and share the older malformed naming.

("Opt" in these directory names is a known misnomer; the cohorts are just D..G.)

Source selection: one file per session
--------------------------------------
A session can have several files on disk that differ only in trailing text::

    20220217_D01,...,D08_P1 uncropped.mkv     1.70 GB   Feb 2022  <- original
    20220217_D01,...,D08_P1.mkv               1.21 GB   Jan 2025  <- trimmed
    20220217_D01,...,D08_P1-proj.llc          347 B     Jan 2025  <- LosslessCut project

Those encode to the SAME name, so exactly one must be chosen. "uncropped" here
means *not trimmed in time*: someone opened the original in LosslessCut and cut it
down to the actual session. That matters -- cameras left running past the end of a
session produce junk frames that wreck DLC reference coverage and segmentation --
so the trimmed file is the one we want.

Preference: the clean ``_P{n}.mkv`` wins. If a session has no trimmed version, the
original is the only candidate and is used. LosslessCut ``-seg`` exports, ``(2)``
duplicates and ``.llc`` project files are always skipped.

Names written slightly wrong
----------------------------
Two kinds of name are read although they are not in the standard form
(read_recording_stem), because what they say is not in doubt:

- a separator typo between ids -- ``G25.G26,...`` or ``H40_001,...``: read with
  commas, but only when that gives exactly eight ids and only one way of reading
  the name works;
- one recording holding animals of two cohorts on different runs of the day,
  ``..._JP3,IP1``: the J animals' run 3 and the I animals' run 1. A pipeline name
  carries one run, so the recording is copied in once per run, each copy naming
  only that run's animals (the others' positions empty) -- every animal is cut
  once, under its own run.

Anything else that does not read is reported, never guessed at. Every name read
this way is printed by the dry run and written into the mapping file with the
original name.

Bringing in only the recordings some sessions need
---------------------------------------------------
``--sessions-file <list>`` imports only the recordings that hold at least one
listed session, in the order of the list. A session is one animal's tray run,
``{date}_{animal}_{tray}{position}``, written either the lab's way
(``20220811_J11_P3``) or the pipeline's (``20220811_ASPA1011_P3``); one per line,
blank lines and ``#`` comments ignored. WHY: a recording holds eight animals,
and the sessions a question needs are scattered across far fewer recordings than
a whole cohort -- for one study, 346 recordings instead of 959. The dry run names
every listed session that has no recording, so nothing a list asks for goes
missing without being said.

A recording the pipeline already has -- still in the intake folder, cut and
moved to ``Analyzed/Multi-Animal``, or held in quarantine -- is never brought in
again.

Names checked against the archive's own cuts
--------------------------------------------
Before copying, each recording's name is checked against ASPA's own cuts of
that day (:mod:`mousereach.aspa.name_audit`): a recording whose seats the cuts
name for animals it does not list is NOT brought in -- it would file one
animal's video under another's id -- and is named for a person. Only the days
being imported are measured (headers only), so a small import stays quick.

Usage
-----
    python -m mousereach.aspa.import_collages            # dry run (default)
    python -m mousereach.aspa.import_collages --apply    # actually copy
    mousereach-aspa-import-collages --sessions-file <list>            # dry run
    mousereach-aspa-import-collages --sessions-file <list> --apply

Dry run is the default on purpose: it prints exactly what would be copied,
skipped, and why, plus every session where more than one source competed.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .identity import encode_animal, encode_collage_stem, is_encoded

# Cohort directories in scope (see module docstring for the exclusions).
IN_SCOPE_COHORTS = ["OptD", "OptE", "OptF", "OptG", "H", "I", "J", "K", "L", "M"]

VIDEO_EXTENSIONS = {".mkv", ".mp4"}

# Trailing-junk classification -> (kind, rank). Lower rank wins.
_CLEAN, _UNCROPPED, _REJECT = 0, 1, 99


def default_source_root() -> Optional[Path]:
    from ..config import Paths
    root = Paths.NAS_ROOT
    return (Path(root) / "Archive" / "historical" / "ASPA") if root else None


def default_dest() -> Optional[Path]:
    from ..config import Paths
    return Path(Paths.MULTI_ANIMAL_SOURCE) if Paths.MULTI_ANIMAL_SOURCE else None


def classify(trailing: str) -> Tuple[str, int]:
    """Classify a filename's trailing text into (kind, preference rank).

    Everything after ``{date}_{animals}_{tray}{position}`` is trailing text. It
    carries no information the pipeline needs, but it DOES decide which of several
    competing files is the right source for the session.
    """
    t = (trailing or "").strip().lower()
    if t == "":
        return "clean", _CLEAN
    if "uncropped" in t:
        return "uncropped", _UNCROPPED
    if re.search(r"-\d{2}\.\d{2}\.\d{2}", t) or "seg" in t:
        return "losslesscut_segment", _REJECT
    if re.match(r"^\s*\(\d+\)$", t):
        return "duplicate", _REJECT
    return "other", _REJECT


_TRAY = re.compile(r"^[A-Za-z]\d+$")
_COHORT_TRAY = re.compile(r"^([A-Za-z]+)([A-Za-z]\d+)$")
_ANIMAL_TOKEN = re.compile(r"^(?:[A-Za-z]+\d+|\d{1,3})$")
_TRAYS = re.compile(r"^([A-Za-z]+\d+(?:,[A-Za-z]+\d+)*)(.*)$")


def read_recording_stem(stem: str) -> Optional[Tuple[List[Tuple[str, str]], str]]:
    """Read an archive recording name: ``([(session stem, how it was read), ...],
    trailing text)``, or None when it does not read (see "Names written slightly
    wrong" in the module docstring).

    Each session stem is in the standard lab form ``{date}_{8 ids}_{tray}`` and
    "how it was read" is "" for a standard name. Readings are tried at every
    ``_`` after the date; a name that reads more than one way is not read."""
    stem = (stem or "").strip()
    if not re.match(r"^\d{8}_", stem):
        return None
    date, rest = stem[:8], stem[9:]
    readings = []
    for k in [i for i, ch in enumerate(rest) if ch == "_"]:
        block, after = rest[:k], rest[k + 1:]
        m = _TRAYS.match(after)
        if not m:
            continue
        trays, trailing = m.group(1).split(","), m.group(2)
        # A name that reads with commas is read as it always was, whatever its
        # number of ids (a name with seven is a question of its own, not one
        # for this reader). Only a REPAIRED reading must give exactly eight.
        ids = block.split(",")
        note = ""
        if not all(_ANIMAL_TOKEN.match(a) for a in ids):
            ids = re.split(r"[,._]", block)
            if len(ids) != 8 or not all(_ANIMAL_TOKEN.match(a) for a in ids):
                continue
            note = "separator typo between ids read as a comma"
        if len(trays) == 1 and _TRAY.match(trays[0]):
            readings.append(([(f"{date}_{','.join(ids)}_{trays[0]}", note)], trailing))
            continue
        # One run per cohort: every animal's cohort letters must name a run.
        runs = {}
        for tok in trays:
            mm = _COHORT_TRAY.match(tok)
            if not mm or mm.group(1).upper() in runs:
                runs = None
                break
            runs[mm.group(1).upper()] = mm.group(2)
        if not runs:
            continue
        cohort_of = [None if a.isdigit() else re.match(r"[A-Za-z]+", a).group(0).upper()
                     for a in ids]
        if any(c is not None and c not in runs for c in cohort_of):
            continue
        sessions = []
        for c, tray in runs.items():
            kept = [a if who == c else f"{pos:03d}"
                    for pos, (a, who) in enumerate(zip(ids, cohort_of), start=1)]
            why = (f"one recording, one run per cohort: the {c} animals' {tray}, "
                   f"the others' positions empty")
            sessions.append((f"{date}_{','.join(kept)}_{tray}",
                             f"{note}; {why}" if note else why))
        readings.append((sessions, trailing))
    return readings[0] if len(readings) == 1 else None


def scan(source_root: Path, cohorts: List[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Group candidate source files by the session they belong to.

    Returns ``(by_session, unparseable)``. ``by_session`` maps the ENCODED stem to
    its competing candidates; ``unparseable`` holds files whose names do not have
    the expected shape (reported, never guessed at).
    """
    by_session: Dict[str, List[dict]] = defaultdict(list)
    unparseable: List[dict] = []

    for cohort in cohorts:
        multi = source_root / cohort / "Multi-Animal"
        if not multi.is_dir():
            continue
        for p in sorted(multi.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue  # .llc project files and anything non-video
            read = read_recording_stem(p.stem)
            if read is None:
                unparseable.append({"cohort": cohort, "path": p, "reason": "name shape"})
                continue
            sessions, trailing = read
            encoded_all = [(encode_collage_stem(s), s, how) for s, how in sessions]
            if not all(e for e, _, _ in encoded_all):
                unparseable.append({"cohort": cohort, "path": p, "reason": "animal ids"})
                continue
            kind, rank = classify(trailing)
            for encoded, session_stem, how in encoded_all:
                by_session[encoded].append({
                    "cohort": cohort, "path": p, "kind": kind, "rank": rank,
                    "size": p.stat().st_size, "source_stem": p.stem,
                    "read_as": session_stem if how else "", "repair": how,
                })
    return by_session, unparseable


def choose(candidates: List[dict]) -> Optional[dict]:
    """Pick the one source file for a session; None if every candidate is rejected.

    Preference is by rank (clean beats uncropped); ties break on the larger file,
    which is only reachable when two equally-clean files exist.
    """
    usable = [c for c in candidates if c["rank"] != _REJECT]
    if not usable:
        return None
    usable.sort(key=lambda c: (c["rank"], -c["size"]))
    return usable[0]


def build_plan(source_root: Path, dest: Path, cohorts: List[str]) -> dict:
    by_session, unparseable = scan(source_root, cohorts)
    plan, contested, dropped = [], [], []

    for encoded_stem in sorted(by_session):
        candidates = by_session[encoded_stem]
        pick = choose(candidates)
        if pick is None:
            dropped.append({"stem": encoded_stem, "candidates": candidates})
            continue
        if len(candidates) > 1:
            contested.append({"stem": encoded_stem, "chosen": pick, "candidates": candidates})
        plan.append({
            "encoded_stem": encoded_stem,
            "src": pick["path"],
            "dst": dest / f"{encoded_stem}{pick['path'].suffix.lower()}",
            "cohort": pick["cohort"],
            "kind": pick["kind"],
            "size": pick["size"],
            "source_stem": pick["source_stem"],
            "read_as": pick.get("read_as", ""),
            "repair": pick.get("repair", ""),
        })
    return {"plan": plan, "contested": contested, "dropped": dropped,
            "unparseable": unparseable}


_SINGLE = re.compile(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z]\d+)$")


def normalise_session(text: str) -> Optional[str]:
    """One listed session in pipeline form, or None when it is not a session name.

    ``20220811_J11_P3`` and ``20220811_ASPA1011_P3`` both give
    ``20220811_ASPA1011_P3``."""
    m = _SINGLE.match((text or "").strip())
    if not m:
        return None
    date, animal, tray = m.groups()
    if not is_encoded(animal):
        animal = encode_animal(animal)
        if not animal:
            return None
    return f"{date}_{animal}_{tray.upper()}"


def read_sessions_file(path: Path) -> Tuple[List[str], List[str]]:
    """``(sessions in file order, lines that are not a session name)``.
    Duplicates are kept once, at their first position."""
    sessions, bad, seen = [], [], set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        s = normalise_session(line)
        if s is None:
            bad.append(line)
        elif s not in seen:
            seen.add(s)
            sessions.append(s)
    return sessions, bad


def sessions_in(encoded_stem: str) -> List[str]:
    """The single-animal sessions a recording holds: one per animal in its name."""
    m = re.match(r"^(\d{8})_(.+)_([A-Za-z]\d+)$", encoded_stem)
    if not m:
        return []
    date, animals, tray = m.groups()
    return [f"{date}_{a.strip()}_{tray.upper()}" for a in animals.split(",") if a.strip()]


def select_for_sessions(plan: List[dict], sessions: List[str]) -> Tuple[List[dict], List[str]]:
    """The recordings holding at least one listed session, in list order (a
    recording is placed where the first session needing it is listed), and the
    listed sessions no recording holds."""
    holder = {}
    for item in plan:
        for s in sessions_in(item["encoded_stem"]):
            holder.setdefault(s, item)
    chosen, placed, missing = [], set(), []
    for s in sessions:
        item = holder.get(s)
        if item is None:
            missing.append(s)
        elif item["encoded_stem"] not in placed:
            placed.add(item["encoded_stem"])
            chosen.append(item)
    return chosen, missing


def already_in_pipeline(item: dict, retired_dir: Optional[Path],
                        held_dir: Optional[Path] = None) -> Optional[str]:
    """Where the pipeline already has this recording, or None. WHY: a recording
    copied in again after it was cut would sit in the intake as a stray, or be
    cut twice; one held in quarantine is waiting for a person, and a second
    copy in the intake would be cut behind that person's back."""
    if item["dst"].exists():
        return "in the intake folder"
    if retired_dir is not None and (retired_dir / item["dst"].name).exists():
        return "cut, and moved to " + str(retired_dir)
    if held_dir is not None and (held_dir / item["dst"].name).exists():
        return "held in quarantine for a person (" + str(held_dir) + ")"
    return None


def default_quarantine_dir() -> Optional[Path]:
    """The watcher's quarantine folder, or None when it cannot be resolved."""
    try:
        from ..config import WatcherConfig
        return Path(WatcherConfig.load().get_quarantine_dir())
    except Exception:
        return None


def default_retired_dir() -> Optional[Path]:
    from ..config import Paths
    return (Path(Paths.ANALYZED_OUTPUT) / "Multi-Animal") if Paths.ANALYZED_OUTPUT else None


def copy_file(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` (replacing ``dst``), keeping the modified time.

    On Windows this is the system copy (``CopyFileW``). WHY: when both paths
    are on the same network share, the file server copies the file itself and
    no data passes through this machine. Measured on the lab share 2026-10-21:
    a 1 GB recording took 3 s that way against 41 s read and written back
    through this machine -- and the read-and-write copy of a large import
    filled the processing server's network link, slowing its own work tenfold
    for as long as the import ran. Elsewhere, ``shutil.copy2``.
    """
    import os
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.CopyFileW(str(src), str(dst), False):
            return
        raise OSError(ctypes.GetLastError(), f"CopyFileW failed: {src} -> {dst}")
    shutil.copy2(src, dst)


def write_mapping(plan: List[dict], dest_root: Path) -> Path:
    """Write the versioned source -> encoded mapping next to the imported videos.

    The encoding is a rule and reconstructs itself, so nothing depends on this
    file -- it exists so the provenance of every imported video is recorded
    explicitly rather than inferred.
    """
    out = dest_root / "ASPA_import_mapping.json"
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    entries = existing.get("entries", {})
    for item in plan:
        entries[item["encoded_stem"]] = {
            "source_path": str(item["src"]),
            "source_stem": item["source_stem"],
            "cohort_dir": item["cohort"],
            "selected_because": item["kind"],
            "bytes": item["size"],
        }
        if item.get("repair"):
            entries[item["encoded_stem"]]["name_read_as"] = item["read_as"]
            entries[item["encoded_stem"]]["name_repair"] = item["repair"]
    out.write_text(json.dumps({
        "schema_version": "1.0",
        "description": "ASPA historical collages imported into the MouseReach pipeline. "
                       "Encoded ids decode by rule: cohort number = alphabet position.",
        "updated_at": datetime.now().isoformat(),
        "entries": entries,
    }, indent=2), encoding="utf-8")
    return out


def _fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mousereach-aspa-import-collages",
        description=(
            "Copy ASPA historical collages (cohorts D and later) into the watcher's "
            "intake, renamed into pipeline form. The archive is only ever READ -- "
            "originals and old ASPA analyses are never modified. Dry run by default."),
        epilog="Examples:\n"
               "  python -m mousereach.aspa.import_collages\n"
               "  python -m mousereach.aspa.import_collages --apply\n"
               "  mousereach-aspa-import-collages --sessions-file sessions.txt\n"
               "  mousereach-aspa-import-collages --sessions-file sessions.txt --apply\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy. Without this, prints the plan and changes nothing.")
    parser.add_argument("--source", default=None, help="ASPA archive root (default: NAS Archive/historical/ASPA).")
    parser.add_argument("--dest", default=None, help="Intake dir (default: Unanalyzed/Multi-Animal).")
    parser.add_argument("--cohorts", default=None,
                        help="Comma-separated cohort dirs to import (default: %s)." % ",".join(IN_SCOPE_COHORTS))
    parser.add_argument("--limit", type=int, default=0, help="Only act on the first N sessions (testing).")
    parser.add_argument("--sessions-file", default=None,
                        help="Import only the recordings holding at least one session listed in "
                             "this file (one {date}_{animal}_{tray}{position} per line), in list order.")
    args = parser.parse_args(argv)

    source_root = Path(args.source) if args.source else default_source_root()
    dest = Path(args.dest) if args.dest else default_dest()
    cohorts = args.cohorts.split(",") if args.cohorts else IN_SCOPE_COHORTS

    if not source_root or not source_root.is_dir():
        print(f"[FAIL] ASPA archive not found: {source_root}")
        return 1
    if not dest:
        print("[FAIL] Could not resolve the intake dir (is this machine configured?).")
        return 1

    print(f"Source: {source_root}")
    print(f"Dest:   {dest}")
    print(f"Cohorts: {', '.join(cohorts)}")
    print("Scanning...")
    result = build_plan(source_root, dest, cohorts)
    plan = result["plan"]
    missing_sessions: List[str] = []
    if args.sessions_file:
        sessions, bad = read_sessions_file(Path(args.sessions_file))
        print(f"Sessions listed: {len(sessions)}  ({args.sessions_file})")
        if bad:
            print(f"[!] Lines that are not a session name (ignored): {len(bad)}")
            for b in bad[:10]:
                print(f"  {b[:80]}")
        plan, missing_sessions = select_for_sessions(plan, sessions)
    retired = default_retired_dir()
    held_dir = default_quarantine_dir()
    have = [(p, already_in_pipeline(p, retired, held_dir)) for p in plan]
    already = [(p, why) for p, why in have if why]
    plan = [p for p, why in have if not why]
    if args.limit:
        plan = plan[:args.limit]

    # Each recording's name against the archive's own cuts of that day (see
    # "Names checked against the archive's own cuts" in the module docstring).
    contested = []
    if plan:
        from .name_audit import audit
        days = {(p["encoded_stem"][:8], p["encoded_stem"].rsplit("_", 1)[-1].upper())
                for p in plan}
        print(f"Checking {len(plan)} recording name(s) against the archive's own cuts "
              f"({len(days)} day(s) and tray(s), headers only)...")
        res = audit(source_root, cohorts, only={(c, d, t) for c in cohorts for d, t in days})
        if res["unreadable"]:
            print(f"[!] {len(res['unreadable'])} cut(s) of those days could not be measured; "
                  f"the names were checked without them: {', '.join(res['unreadable'][:5])}")
        by_stem = {s: c for c in res["conflicts"] for s in c["pipeline_recordings"]}
        contested = [(p, by_stem[p["encoded_stem"]]) for p in plan if p["encoded_stem"] in by_stem]
        plan = [p for p in plan if p["encoded_stem"] not in by_stem]

    total = sum(p["size"] for p in plan)
    print("")
    print(f"Recordings to import: {len(plan)}  ({_fmt_gb(total)})")
    by_cohort = defaultdict(int)
    for p in plan:
        by_cohort[p["cohort"]] += 1
    for c in cohorts:
        if by_cohort.get(c):
            print(f"    {c:<6} {by_cohort[c]:5d}")

    if result["contested"]:
        print("")
        print(f"Sessions with more than one candidate: {len(result['contested'])}")
        for c in result["contested"][:10]:
            print(f"  {c['stem']}")
            for cand in sorted(c["candidates"], key=lambda x: x["rank"]):
                mark = "->" if cand is c["chosen"] else "  "
                print(f"    {mark} {cand['kind']:<20} {_fmt_gb(cand['size']):>9}  {cand['source_stem'][:60]}")
        if len(result["contested"]) > 10:
            print(f"  ... and {len(result['contested']) - 10} more")

    if result["dropped"]:
        print("")
        print(f"[!] Sessions with NO usable candidate: {len(result['dropped'])}")
        for d in result["dropped"][:5]:
            print(f"  {d['stem']}  ({', '.join(c['kind'] for c in d['candidates'])})")

    repaired = [p for p in result["plan"] if p.get("repair")]
    if repaired:
        print("")
        print(f"Names read although not in the standard form: {len(repaired)}")
        for p in repaired:
            print(f"  {p['source_stem'][:70]}")
            print(f"    -> {p['read_as']}  ({p['repair']})")

    if result["unparseable"]:
        print("")
        print(f"[!] Files whose names could not be parsed: {len(result['unparseable'])}")
        for u in result["unparseable"][:10]:
            print(f"  [{u['reason']}] {u['path'].name[:80]}")
        if len(result["unparseable"]) > 10:
            print(f"  ... and {len(result['unparseable']) - 10} more")

    if contested:
        print("")
        print(f"[!] NOT imported -- the archive's own cuts name a seat for an animal the "
              f"recording's name does not list (which name is right is for a person): "
              f"{len(contested)}")
        for p, c in contested:
            named = ",".join(sorted({x["animal"] for x in c["cuts_named_differently"]}))
            print(f"  {p['source_stem'][:70]}  (cut(s) named {named})")

    if already:
        print("")
        print(f"Already in the pipeline (not imported again): {len(already)}")
        for p, why in already[:5]:
            print(f"  {p['encoded_stem'][:70]}  -- {why}")

    if args.sessions_file:
        print("")
        if missing_sessions:
            print(f"[!] Listed sessions with NO recording in the archive: {len(missing_sessions)}")
            for s in missing_sessions[:20]:
                print(f"  {s}")
            if len(missing_sessions) > 20:
                print(f"  ... and {len(missing_sessions) - 20} more")
        else:
            print("[OK] Every listed session has a recording.")

    if not args.apply:
        print("")
        print("DRY RUN -- nothing copied. Re-run with --apply to perform the import.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = failed = 0
    for n, item in enumerate(plan, 1):
        if already_in_pipeline(item, retired, held_dir):
            skipped += 1
            continue
        try:
            tmp = item["dst"].with_suffix(item["dst"].suffix + ".partial")
            copy_file(item["src"], tmp)
            tmp.rename(item["dst"])
            copied += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {item['encoded_stem']}: {e}")
        if n % 25 == 0 or n == len(plan):
            print(f"  {n}/{len(plan)}  copied={copied} skipped={skipped} failed={failed}")

    mapping = write_mapping(plan, dest)
    print("")
    print(f"[OK] copied={copied} skipped={skipped} failed={failed}")
    print(f"Mapping written: {mapping}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
