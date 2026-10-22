"""Recordings left on old storage: find them, and bring in what is missing.

WHY THIS EXISTS
---------------
A lab that moved its pipeline to a new shared folder can still have recordings
that exist only in the old one -- copied there by hand, cut there by an old
script, or never moved. Nothing in the pipeline looks at the old place, so those
recordings are invisible to every machine that could process them, and the old
storage cannot be retired while they are there. This lists every video under an
old root, says for each whether the pipeline already has it, and, when asked,
copies what is missing into the stage folder where the pipeline picks it up.

    mousereach-legacy-inventory --from <old root>              # report only
    mousereach-legacy-inventory --from <old root> --bring-in   # report, then copy

(Until the command is installed on a machine:
``python -m mousereach.pipeline.legacy_storage --from <old root>``.)

What counts as "the pipeline already has it" -- any one is enough:
  * a video of the same name anywhere under the pipeline root, same size;
  * a cut video whose session the pipeline knows: its video under any name
    extension, its results (a processing manifest), or a row in the shared
    record (watcher_central.db) -- so one being posed on a GPU machine counts;
  * a recording each of whose animal sessions is held that way, or by a
    recording under the pipeline root (blank slots, subject 00, are ignored) --
    so a recording renamed since it was copied is not brought in twice.

What is brought in, and where (``--bring-in``):
  recording, Pillar tray   -> Unanalyzed/Multi-Animal (the intake: cut, then posed)
  recording, Easy/Flat     -> Unanalyzed/Unsupported_Tray_Type/collages
  cut video, Pillar        -> Unanalyzed/Single_Animal (posed by any GPU machine)
  cut video, Easy/Flat     -> Unanalyzed/Unsupported_Tray_Type
Easy and Flat go where the pipeline already keeps those trays waiting. A Pillar
recording whose cut videos ALL come in from the old storage too is already cut:
it goes to Analyzed/Multi-Animal, where cut recordings are kept, not to the
intake, where it would be cut a second time.

Nothing else is copied. Left for a person, and named in the report: a name that
is not a pipeline name (extra text such as "(LONG)" or " uncropped", ranges of
animals), a malformed animal id (a typo such as CNT407), a video the pipeline
holds under the same name at a DIFFERENT size, a recording only some of
whose cut videos are on the old storage, and copies of one name in several old
folders that differ in size. (Copies of one name at one size are one video:
it is brought in once.)

How a copy is made: to ``<name>.part`` in the destination (no pipeline step
reads a .part file), then its size and SHA-256 are compared with the original,
then it is renamed to its name. An existing file is never replaced. Each copy is
recorded, one JSON line per file, in
``<pipeline root>/legacy_inventory/<label>_brought_in.jsonl``. The old storage
is only ever read.

Report: ``<pipeline root>/legacy_inventory/<label>.csv`` (every old video: what
it is, whether the pipeline has it and why, where it would go) and
``<label>.summary.json`` (counts). Re-run any time. "only here" in the summary
is the number of pipeline videos that still exist only on the old storage; at
zero, the old storage is no longer needed for any of them.

ASCII-only console output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov")
# Folders no recording lives in; everything else, dot-folders included, is
# walked (a GPU machine's claim folder, .inflight, holds videos being posed).
_SKIP_DIRS = {"$recycle.bin", "system volume information"}

_ID = re.compile(r"^[A-Za-z]+\d{4}$")
_RECORDING = re.compile(r"^(\d{8})_([^_]+,[^_]+)_([A-Za-z])(\d+)$")
_CUT = re.compile(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z])(\d+)$")

# Report statuses.
SAME_FILE = "on pipeline (same file)"
DIFFERENT_SIZE = "on pipeline under the same name at a DIFFERENT size -- for a person"
HAS_SESSION = "on pipeline (session known)"
HAS_ALL_SESSIONS = "on pipeline (every session known)"
NOT_PIPELINE_NAME = "not a pipeline name -- for a person"
BAD_ID = "malformed animal id -- for a person"
PARTLY_CUT = "recording partly cut on the old storage -- for a person"
COPIES_DIFFER = "copies on the old storage differ in size -- for a person"
SAME_AS_ANOTHER = "only here, as another copy on the old storage that is brought in"
ONLY_HERE = "only here"


def _walk(root: Path) -> Iterator[os.DirEntry]:
    """Every file under ``root`` (DirEntry: its stat comes with the listing)."""
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.lower() not in _SKIP_DIRS:
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        yield e
        except OSError:
            continue


def _blank(animal: str) -> bool:
    """A blank camera slot: subject 00 (e.g. CNT0000, CNT0600)."""
    return animal.endswith("00")


def classify(stem: str) -> dict:
    """What a file stem is: ``{"kind": recording|cut|other, "tray", "sessions",
    "bad_ids"}``. Sessions are the cut-video stems it stands for."""
    m = _RECORDING.match(stem)
    if m:
        date, animals, tray, pos = m.groups()
        ids = [a.strip() for a in animals.split(",") if a.strip()]
        bad = [a for a in ids if not _ID.match(a)]
        sessions = [f"{date}_{a}_{tray.upper()}{pos}" for a in ids
                    if _ID.match(a) and not _blank(a)]
        return {"kind": "recording", "tray": tray.upper(), "sessions": sessions,
                "bad_ids": bad}
    m = _CUT.match(stem)
    if m:
        date, animal, tray, pos = m.groups()
        bad = [] if _ID.match(animal) else [animal]
        return {"kind": "cut", "tray": tray.upper(),
                "sessions": [] if bad else [f"{date}_{animal}_{tray.upper()}{pos}"],
                "bad_ids": bad}
    return {"kind": "other", "tray": None, "sessions": [], "bad_ids": []}


class PipelineIndex:
    """What the pipeline root holds, from one walk of it (plus the shared record)."""

    def __init__(self, nas_root: Path, walk=_walk):
        self.names: Dict[str, List[int]] = {}
        self.stems: set = set()
        self.results: set = set()
        self.sessions_held: set = set()
        self.known: set = set()
        for e in walk(nas_root):
            n = e.name
            low = n.lower()
            if low.endswith(VIDEO_EXT):
                try:
                    size = e.stat().st_size
                except OSError:
                    size = -1
                self.names.setdefault(low, []).append(size)
                stem = os.path.splitext(n)[0]
                self.stems.add(stem.upper())
                info = classify(stem)
                if info["kind"] == "recording":
                    self.sessions_held.update(s.upper() for s in info["sessions"])
            elif low.endswith("_processing_manifest.json"):
                self.results.add(n[: -len("_processing_manifest.json")].upper())
        db = Path(nas_root) / "watcher_central.db"
        if db.is_file():
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
                try:
                    for table in ("videos", "pipeline_videos"):
                        try:
                            self.known.update(str(v).upper() for (v,) in
                                              con.execute(f"SELECT video_id FROM {table}"))
                        except sqlite3.Error:
                            pass
                finally:
                    con.close()
            except sqlite3.Error:
                pass

    def has_session(self, session: str) -> bool:
        s = session.upper()
        return (s in self.stems or s in self.results or s in self.known
                or s in self.sessions_held)


def destination(info: dict, name: str, already_cut: bool = False) -> Optional[Path]:
    """Where a missing pipeline video goes on the pipeline root, or None.

    ``already_cut``: every session of this recording comes in as a cut video
    from the old storage too. A Pillar recording then goes where cut
    recordings are kept (Analyzed/Multi-Animal), never to the intake -- WHY:
    in the intake it would be cut a second time beside its own cut videos.
    Easy/Flat recordings wait in one folder either way."""
    from mousereach.config import FilePatterns, Paths
    unsupported = info["tray"] in FilePatterns.UNSUPPORTED_TRAY_TYPES
    if info["kind"] == "recording":
        if unsupported:
            base = Path(Paths.UNSUPPORTED_TRAY_RETURN) / "collages"
        elif already_cut:
            base = Path(Paths.ANALYZED_OUTPUT) / "Multi-Animal" if Paths.ANALYZED_OUTPUT else None
        else:
            base = Paths.MULTI_ANIMAL_SOURCE
    elif info["kind"] == "cut":
        base = Paths.UNSUPPORTED_TRAY_RETURN if unsupported else Paths.SINGLE_ANIMAL_OUTPUT
    else:
        return None
    return Path(base) / name if base else None


def inventory(old_root: Path, index: PipelineIndex) -> List[dict]:
    """One row per video under ``old_root``."""
    rows = []
    for e in _walk(old_root):
        if not e.name.lower().endswith(VIDEO_EXT):
            continue
        try:
            size = e.stat().st_size
        except OSError:
            size = -1
        stem = os.path.splitext(e.name)[0]
        info = classify(stem)
        same = index.names.get(e.name.lower(), [])
        if same and size in same:
            status = SAME_FILE
        elif same:
            status = DIFFERENT_SIZE
        elif info["kind"] == "other":
            status = NOT_PIPELINE_NAME
        elif info["bad_ids"]:
            status = BAD_ID
        elif info["sessions"] and all(index.has_session(s) for s in info["sessions"]):
            status = HAS_SESSION if info["kind"] == "cut" else HAS_ALL_SESSIONS
        else:
            status = ONLY_HERE
        rel = os.path.relpath(e.path, old_root)
        rows.append({
            "old_path": e.path, "name": e.name,
            "folder": rel.split(os.sep)[0] if os.sep in rel else "",
            "size": size, "kind": info["kind"], "tray": info["tray"] or "",
            "status": status, "destination": "", "bad_ids": " ".join(info["bad_ids"]),
            "_info": info,
        })

    # The same video can sit in more than one old folder. Copies of one name
    # at one size are one video: the first (by path) is brought in, the others
    # are noted as that copy. Copies that differ in size need a person to say
    # which is right. WHY: otherwise the second copy is refused at bring-in
    # ("a file of that name is already there") and "only here" never reaches
    # zero (12 recordings were in two old folders, 2026-10-22).
    from collections import defaultdict
    by_name = defaultdict(list)
    for r in rows:
        if r["status"] == ONLY_HERE:
            by_name[r["name"].lower()].append(r)
    for group in by_name.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r["old_path"])
        if len({r["size"] for r in group}) > 1:
            for r in group:
                r["status"] = COPIES_DIFFER
        else:
            for r in group[1:]:
                r["status"] = SAME_AS_ANOTHER
                r["same_as"] = group[0]["old_path"]

    # Where each missing video goes. A recording is placed by what else comes in
    # with it: cut videos of the same sessions (see destination()).
    cut_here = {s.upper() for r in rows if r["status"] == ONLY_HERE
                and r["kind"] == "cut" for s in r["_info"]["sessions"]}
    for r in rows:
        info = r.pop("_info")
        if r["status"] != ONLY_HERE:
            continue
        already_cut = False
        if r["kind"] == "recording":
            cut = [s for s in info["sessions"] if s.upper() in cut_here]
            if cut and len(cut) < len(info["sessions"]):
                r["status"] = PARTLY_CUT
                continue
            already_cut = bool(cut)
        dest = destination(info, r["name"], already_cut=already_cut)
        r["destination"] = str(dest) if dest else ""
    rows.sort(key=lambda r: r["old_path"])
    return rows


def bring_in(row: dict, ledger: Path) -> str:
    """Copy one 'only here' video to its destination, verified. Returns what
    happened ("copied", or why not). Never replaces a file; never raises."""
    from mousereach.aspa.import_collages import copy_file
    from mousereach.pipeline.fsutil import sha256_file
    src, dst = Path(row["old_path"]), Path(row["destination"])
    part = dst.with_name(dst.name + ".part")
    try:
        if dst.exists():
            return "not copied: a file of that name is already there"
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_file(src, part)
        src_size, part_size = src.stat().st_size, part.stat().st_size
        if part_size != src_size:
            part.unlink()
            return f"not copied: size differs after the copy ({part_size} vs {src_size})"
        digest = sha256_file(src)
        if sha256_file(part) != digest:
            part.unlink()
            return "not copied: content differs after the copy (SHA-256)"
        if dst.exists():
            part.unlink()
            return "not copied: a file of that name appeared during the copy"
        os.rename(part, dst)
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"old_path": str(src), "new_path": str(dst),
                                 "bytes": src_size, "sha256": digest,
                                 "at": datetime.now().isoformat(timespec="seconds")}) + "\n")
        return "copied"
    except Exception as e:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass
        return f"not copied: {e}"


def _label(old_root: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(old_root).name).strip("_") or "old_storage"


def write_report(rows: List[dict], out_dir: Path, label: str, old_root: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["old_path", "folder", "size", "kind", "tray", "status", "destination",
              "same_as", "bad_ids", "brought_in"]
    with open(out_dir / f"{label}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    only = [r for r in rows if r["status"] == ONLY_HERE]
    summary = {
        "old_root": str(old_root),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "videos": len(rows),
        "by_status": dict(Counter(r["status"] for r in rows).most_common()),
        "only_here": len(only),  # one per video: duplicate copies are not counted
        "only_here_bytes": sum(max(r["size"], 0) for r in only),
        "only_here_by_folder": dict(Counter(r["folder"] for r in only).most_common()),
        "for_a_person_by_folder": dict(Counter(
            r["folder"] for r in rows if r["status"].endswith("for a person")).most_common()),
    }
    (out_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mousereach-legacy-inventory",
        description="List every video on an old storage root, say whether the pipeline "
                    "has it, and (with --bring-in) copy what is missing into the stage "
                    "folder that picks it up. The old storage is only read.")
    parser.add_argument("--from", dest="old_root", required=True,
                        help="The old storage root to inventory (only read).")
    parser.add_argument("--bring-in", action="store_true",
                        help="Copy every 'only here' pipeline video to its stage folder, "
                             "verified. Without it, only the report is written.")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --bring-in: copy at most this many files this run.")
    parser.add_argument("--label", default=None,
                        help="Name for the report files (default: from the old root's name).")
    args = parser.parse_args(argv)

    from mousereach.config import Paths
    if not Paths.NAS_ROOT:
        print("[FAIL] nas_root is not configured -- run mousereach-setup.")
        return 1
    # Paths stay as configured (a mapped drive stays a drive letter in the
    # report); they are resolved only to test that the two roots are apart.
    nas_root = Path(Paths.NAS_ROOT)
    old_root = Path(args.old_root)
    if not old_root.is_dir():
        print(f"[FAIL] Not a folder: {old_root}")
        return 1
    real_old, real_nas = old_root.resolve(), nas_root.resolve()
    if real_old == real_nas or real_nas in real_old.parents or real_old in real_nas.parents:
        print(f"[FAIL] {old_root} and the pipeline root {nas_root} overlap; give a "
              f"separate old storage root.")
        return 1
    label = args.label or _label(old_root)
    out_dir = nas_root / "legacy_inventory"

    print(f"Pipeline root: {nas_root}")
    print("Indexing the pipeline root (one walk)...", flush=True)
    index = PipelineIndex(nas_root)
    print(f"Inventorying {old_root}...", flush=True)
    rows = inventory(old_root, index)

    if args.bring_in:
        todo = [r for r in rows if r["status"] == ONLY_HERE and r["destination"]]
        if args.limit is not None:
            todo = todo[: args.limit]
        out_dir.mkdir(parents=True, exist_ok=True)
        ledger = out_dir / f"{label}_brought_in.jsonl"
        done = Counter()
        for r in todo:
            result = bring_in(r, ledger)
            r["brought_in"] = result
            done["copied" if result == "copied" else "not copied"] += 1
            if result != "copied":
                print(f"  [!] {Path(r['old_path']).name}: {result}")
        print(f"Brought in: {done['copied']} copied, {done['not copied']} not copied "
              f"(ledger: {ledger})")
        # The report says what is still only on the old storage after the copies.
        copied = {r["old_path"] for r in rows if r.get("brought_in") == "copied"}
        for r in rows:
            if r["old_path"] in copied or r.get("same_as") in copied:
                r["status"] = SAME_FILE
                r["destination"] = ""

    summary = write_report(rows, out_dir, label, old_root)
    print(f"{summary['videos']} videos on the old storage:")
    for status, n in summary["by_status"].items():
        print(f"  {n:7d}  {status}")
    print(f"Only here: {summary['only_here']} ({summary['only_here_bytes'] / 1e9:.1f} GB)")
    for folder, n in summary["only_here_by_folder"].items():
        print(f"  {n:7d}  in {folder or '(top level)'}")
    if summary["for_a_person_by_folder"]:
        print("For a person (see the report's status column):")
        for folder, n in summary["for_a_person_by_folder"].items():
            print(f"  {n:7d}  in {folder or '(top level)'}")
    print(f"Report: {out_dir / (label + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
