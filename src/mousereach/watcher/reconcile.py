"""Compare every single-animal video on the share against the definition of done.

WHY THIS EXISTS
---------------
A video is done only when BOTH are true (docs/DESIGN_FILESYSTEM_AS_STATE.md):

  1. its analysis is current, or declared compatible with current, with every
     saved human review reflected in it; and
  2. that analysis sits where it SHOULD be -- beside the video in
     Analyzed/<project>/<cohort>/, worked out from the video's name.

This command checks exactly that, from the files alone. It is the instrument
for moving to "the folder is the state": run it before and after every step,
and the set of mismatches must shrink and never grow.

It deliberately does NOT compare against a database. A database and the files
can agree on the wrong folder or on an old version, so agreement proves
nothing; the right place and the current versions are fixed by rule.

Mismatches are listed PER VIDEO. Counts hide errors: two lists can be the same
length while being wrong in both directions.

The currency judgement is not re-implemented here. It is the version scanner's
own (``ReprocessingScanner``): the same compatibility declaration, the same
exemption for human-authored segmentation, the same test for a saved review the
kinematics have not applied. A second copy of those rules would drift from the
first, and then this check and the watcher would disagree about what "current"
means.

READ-ONLY. It never moves, writes or marks anything.

    mousereach-reconcile          # mismatches per video; exit 1 if any
    mousereach-reconcile --all    # also list every video that is fine
    mousereach-reconcile --json   # everything, machine-readable

Exit codes: 0 no mismatches, 1 mismatches found (or the share still has a
retired stage folder, see below), 2 could not run.

SCOPE: single-animal videos. Collages are not judged yet.

WHERE IT LOOKS
--------------
  - the waiting folders, taken from ``Paths`` when the check runs
    (``Paths.SINGLE_ANIMAL_OUTPUT`` = Unanalyzed/Single_Animal, a single waiting
    for a pose; ``Paths.DLC_STAGING`` = Processing/Posed, posed and waiting for
    the algorithms). Never restated here, so this check and the watcher cannot
    disagree about where a single waits;
  - the review queues, Failed and Quarantine;
  - each queue's build folder, <queue>/.incoming/<stem>/. The router assembles
    a bundle there and renames it into the queue in one step, so the files have
    already left Analyzed while the bundle is not yet a queue bundle. Unread,
    a video mid-route -- or one whose route was interrupted -- would vanish
    from the report. It is listed as held for a person, with a note to check
    whether it persists. A video that is in a queue AND has files in a build
    folder keeps its queue verdict, with a note naming the leftover (the return
    path refuses that bundle until a person folds the files in);
  - Analyzed/, minus the folders every walk of it skips;
  - leftovers: every folder at Processing/_leftovers_pending_cleanup_*/*. When
    the folder layout changed, the retired stage folders were set aside there.
    They hold copies of finished work kept until cleanup and are NOT a stage.

The retired folder names themselves (Processing/Single_Animal,
Processing/DLC_Complete, Processing/Review/flagged_for_review) are never listed:
a plain file sits at each so old code fails loudly, and a file where a folder
was expected reads as empty here rather than crashing the check.

They ARE checked, stat only, for still being real FOLDERS. That means the share
was never migrated to the stage layout, so work may be waiting where this check
does not look. It is reported under ``unmigrated_folders``, printed as a warning
before anything else, and makes the exit code 1: a clean "no mismatches" on an
unmigrated share would be a plausible wrong answer.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from mousereach.census.runner import bundles_in, ids_in_dir
from mousereach.pipeline.analyzed_tree import SUPERSEDED_DIR_NAMES
from mousereach.pipeline.pipe_structure import UNMIGRATED_MESSAGE, retired_folders_present

# The folder inside each queue where the router builds a bundle before renaming
# it into place. Taken from the router so the two cannot disagree about where
# a route in progress sits; the literal is only a fallback for a router that
# predates the constant.
try:
    from mousereach.watcher.review_routing import INCOMING_DIR_NAME
except ImportError:
    INCOMING_DIR_NAME = ".incoming"

_MANIFEST = "_processing_manifest.json"
_FEATURES = "_features.json"

# Folders under Analyzed that never hold a finished single's results: pose
# generations and collages sit beside or inside the cohort folders, and the
# template and UNKNOWN folders are not cohorts. Names starting with "." or "_"
# are scratch or retired copies. Descending into any of them would invent
# videos, or count a pose-only folder as an analysis in the wrong place.
# The superseded-output folder (Analyzed/Archive) comes from the one shared
# rule in pipeline.analyzed_tree instead of being restated here: its files keep
# their original names, so reading them would list a current video's older
# generation as a second copy in the wrong place, and every walker of Analyzed
# must agree on which folder that is.
_ANALYZED_SKIP = {"Folder Template", "UNKNOWN", "Multi-Animal"} | set(SUPERSEDED_DIR_NAMES)

# Where leftovers are set aside when the folder layout changes: one dated
# folder per change under Processing/, each holding the retired stage folders.
# What they hold is kept as insurance until cleanup and is NOT a stage: almost
# all of it is a copy of work finished elsewhere. A video found ONLY there is
# still listed, per video, because it may be real waiting work. A pattern, not
# a list of names, so a later layout change needs no code change here.
_LEFTOVERS_PATTERN = "_leftovers_pending_cleanup_*"


def _waiting_folders() -> List[Path]:
    """Folders where a single waits before analysis, read from Paths at call
    time. WHY not constants: the watcher moves singles into exactly these
    folders, so a copy of the names here would silently disagree the day
    either side changed -- and a waiting video would read as a lost one."""
    from mousereach.config import Paths
    return [Path(p) for p in (Paths.SINGLE_ANIMAL_OUTPUT, Paths.DLC_STAGING) if p]


def _leftover_folders(nas) -> List[Path]:
    """Every folder at <nas>/Processing/_leftovers_pending_cleanup_*/*.
    Files anywhere in that walk are skipped: only folders are leftovers, and
    listing a file would raise."""
    processing = Path(nas) / "Processing"
    if not processing.is_dir():
        return []
    found = []
    for parent in sorted(processing.glob(_LEFTOVERS_PATTERN)):
        if parent.is_dir():
            found.extend(sorted(c for c in parent.iterdir() if c.is_dir()))
    return found

# Verdicts that are fine as they stand.
DONE = "done"
HELD = "held_for_person"
NEEDS_PERSON = "needs_person"
WAITING = "waiting_for_machine"
ONLY_IN_OLD_FOLDER = "only_copy_in_old_folder"
UNSUPPORTED_TRAY = "unsupported_tray"
# Mismatches: not done, and not waiting on anything that will make it done.
NOT_CURRENT = "not_current"
WRONG_PLACE = "wrong_place"
STRAY_BUNDLE = "stray_review_bundle"
MISMATCHES = (NOT_CURRENT, WRONG_PLACE, STRAY_BUNDLE)


class ReconcileUnavailable(RuntimeError):
    """The check cannot judge anything (no share, or no version declaration).
    Raised rather than reporting zero videos: a wrong or missing root that
    answers plausibly costs wrong conclusions."""


def _same_dir(a, b) -> bool:
    # Both sides derive from the same configured root, so case and separator
    # are the only differences worth normalising.
    return (os.path.normcase(os.path.normpath(str(a)))
            == os.path.normcase(os.path.normpath(str(b))))


def _rel(p, root) -> str:
    try:
        return Path(p).relative_to(root).as_posix()
    except ValueError:
        return str(p)


def _read_json(p) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def _quarantine_dir():
    from mousereach.config import Paths, WatcherConfig
    try:
        return WatcherConfig.load().get_quarantine_dir()
    except Exception:
        return (Path(Paths.NAS_ROOT) / "Processing" / "Quarantine") if Paths.NAS_ROOT else None


def walk_analyzed(root) -> Dict[str, Dict[str, List[Path]]]:
    """One pruned walk of the final output tree:
    ``{stem: {"manifest": [...], "features": [...], "video": [...]}}``.
    Every copy is kept, not just the first, so a second copy in another folder
    is visible as the mismatch it is."""
    found: Dict[str, Dict[str, List[Path]]] = {}
    root = Path(root) if root else None
    if not root or not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith((".", "_"))
                       and d not in _ANALYZED_SKIP
                       and not d.startswith("DLC Model")]
        for name in filenames:
            if name.endswith(_MANIFEST):
                key, stem = "manifest", name[:-len(_MANIFEST)]
            elif name.endswith(_FEATURES):
                key, stem = "features", name[:-len(_FEATURES)]
            elif (name.lower().endswith(".mp4") and "," not in name
                  and "DLC" not in name):
                key, stem = "video", name[:-4]
            else:
                continue
            found.setdefault(stem, {}).setdefault(key, []).append(Path(dirpath) / name)
    return found


def _why_not_current(comparison: dict, review) -> str:
    parts = []
    stale = [c for c in comparison.get("stale_components") or () if c != "dlc"]
    if comparison.get("needs_full_reprocess"):
        parts.append("pose from an older model")
    if stale:
        parts.append("outdated: " + ", ".join(stale))
    if comparison.get("unrecorded_components"):
        parts.append("version not recorded: " + ", ".join(comparison["unrecorded_components"]))
    if review is not None:
        parts.append("a saved human review is not reflected in it")
    return "; ".join(parts) or "not current"


def judge(stem: str, where: set, files: dict, scanner, current: dict, nas) -> dict:
    """One video's verdict against the definition of done. Pure apart from
    reading the video's own files."""
    from mousereach.archive.core import get_archive_destination
    from mousereach.config import is_supported_tray_type
    from mousereach.pipeline.versions import compare_manifest_to_current

    row = {"video_id": stem, "found_in": sorted(where), "should_be": None,
           "verdict": None, "detail": ""}

    queues = sorted(where & {"triage", "deep_review"})
    incoming = sorted(w.split(":", 1)[1] for w in where if w.startswith("incoming:"))

    def verdict(v, detail=""):
        if incoming and queues:
            # A queue bundle AND files in a build folder: a route did not
            # finish, and the return path refuses such a bundle until a person
            # folds the leftover in. Named here, or nobody learns why it never
            # comes back.
            note = ("files also left in " + ", ".join(
                q + "/" + INCOMING_DIR_NAME for q in incoming)
                + " -- a route did not finish; check them")
            detail = detail + "; " + note if detail else note
        row["verdict"], row["detail"] = v, detail
        return row

    if not is_supported_tray_type(f"{stem}.mp4"):
        return verdict(UNSUPPORTED_TRAY, "tray type this pipeline does not analyse")

    person = sorted(where & {"failed", "quarantine"})
    manifests = files.get("manifest") or []

    # Only in a queue's build folder: the router has taken the files out of
    # Analyzed and not yet renamed the bundle into the queue. Judged before the
    # analysis, because whatever is still in Analyzed is the route's half-moved
    # residue (a manifest without its video reads as wrong_place). A route takes
    # seconds, so one that is still here on a later run was interrupted.
    if incoming and not queues:
        return verdict(HELD, "; ".join(
            "being routed into " + q + ", or an interrupted route -- check if this persists"
            for q in incoming))

    if not manifests:
        if queues:
            return verdict(HELD, "in " + ", ".join(queues))
        if person:
            return verdict(NEEDS_PERSON, "in " + ", ".join(person))
        if "analyzed" in where:
            folders = sorted({_rel(v.parent, nas) for v in files.get("video") or ()})
            return verdict(NOT_CURRENT, "video in " + ", ".join(folders)
                           + " with no processing manifest beside it")
        if any(w.startswith("waiting:") for w in where):
            return verdict(WAITING)
        return verdict(ONLY_IN_OLD_FOLDER,
                       "only copy is in " + ", ".join(w.split(":", 1)[1] for w in sorted(where)))

    should = get_archive_destination(stem)
    row["should_be"] = _rel(should, nas)
    here = [m for m in manifests if _same_dir(m.parent, should)]
    if not here:
        return verdict(WRONG_PLACE, "analysis is in "
                       + ", ".join(sorted({_rel(m.parent, nas) for m in manifests})))

    manifest = _read_json(here[0])
    if manifest is None:
        return verdict(NOT_CURRENT, "processing manifest is unreadable")
    comparison = compare_manifest_to_current(manifest, current)
    comparison = scanner._drop_human_seg_staleness(stem, {stem: here[0]}, comparison)
    feats = [f for f in files.get("features") or () if _same_dir(f.parent, should)]
    feats_mtime = {stem: feats[0].stat().st_mtime} if feats else {}
    review = scanner._pending_review_path(stem, manifest, feats_mtime)

    if not (comparison["is_current"] and review is None):
        why = _why_not_current(comparison, review)
        if queues:
            return verdict(HELD, "in " + ", ".join(queues) + " -- " + why)
        if person:
            return verdict(NEEDS_PERSON, "in " + ", ".join(person) + " -- " + why)
        return verdict(NOT_CURRENT, why)

    # Current. Now it must be the ONLY copy, with its video, and not also held.
    if not any(_same_dir(v.parent, should) for v in files.get("video") or ()):
        return verdict(WRONG_PLACE, "current analysis has no video beside it "
                       "(re-cuttable if its collage still exists)")
    extra = sorted({_rel(m.parent, nas) for m in manifests if m not in here})
    if extra:
        return verdict(WRONG_PLACE, "another copy of the analysis is in " + ", ".join(extra))
    if queues:
        return verdict(STRAY_BUNDLE, "already done, but a bundle is still in " + ", ".join(queues))
    return verdict(DONE)


def reconcile() -> dict:
    """The whole check, as one JSON-able dict. Raises ReconcileUnavailable
    when there is nothing it can honestly judge."""
    from mousereach.config import Paths
    from mousereach.pipeline.versions import get_current_versions
    from mousereach.watcher.reprocessor import ReprocessingScanner

    if not Paths.NAS_ROOT:
        raise ReconcileUnavailable("the share (nas_root) is not configured -- run mousereach-setup")
    nas = Path(Paths.NAS_ROOT)
    current = get_current_versions(nas)
    if not (current or {}).get("versions"):
        raise ReconcileUnavailable(
            "no version declaration at %s -- nothing to judge currency against"
            % (nas / "pipeline_versions.json"))

    # WHY check (stat only) before judging: a retired stage name that is still a
    # real folder means this share was never migrated, and whatever waits there
    # is invisible to every walk below. Reported, never listed, so the check
    # keeps its promise not to read the retired folders.
    unmigrated = retired_folders_present(nas)

    where: Dict[str, set] = {}

    def mark(stems, label):
        for s in stems:
            # A comma means a collage name (several animals). Collages are out
            # of scope here, and judged as singles they would be listed as
            # videos -- quarantine holds mostly collages, which first
            # surfaced as dozens of phantom single videos needing a person.
            if "," in s:
                continue
            where.setdefault(s, set()).add(label)

    for folder in _waiting_folders():
        mark(ids_in_dir(folder), "waiting:" + _rel(folder, nas))
    for folder in _leftover_folders(nas):
        mark(ids_in_dir(folder), "old:" + _rel(folder, nas))
    mark(bundles_in(Paths.TRIAGE_REVIEW), "triage")
    mark(bundles_in(Paths.DEEP_REVIEW), "deep_review")
    # A route in progress (or interrupted) sits in the queue's build folder,
    # which bundles_in above never lists (not date-named). Its own label, so it
    # is never mistaken for a finished queue bundle.
    for queue, root in (("triage", Paths.TRIAGE_REVIEW), ("deep_review", Paths.DEEP_REVIEW)):
        if root:
            mark(bundles_in(Path(root) / INCOMING_DIR_NAME), "incoming:" + queue)
    mark(ids_in_dir(Paths.FAILED), "failed")
    mark(ids_in_dir(_quarantine_dir()), "quarantine")
    analyzed = walk_analyzed(Paths.ANALYZED_OUTPUT)
    mark(analyzed.keys(), "analyzed")

    scanner = ReprocessingScanner(db=None, nas_root=nas)
    rows = [judge(stem, where[stem], analyzed.get(stem) or {}, scanner, current, nas)
            for stem in sorted(where)]
    leftovers = sorted(r["video_id"] for r in rows
                       if r["verdict"] != ONLY_IN_OLD_FOLDER
                       and any(w.startswith("old:") for w in r["found_in"]))
    return {
        "share": str(nas),
        "versions_file": str(nas / "pipeline_versions.json"),
        "checked": len(rows),
        "mismatches": [r for r in rows if r["verdict"] in MISMATCHES],
        "leftover_copies": leftovers,
        "unmigrated_folders": unmigrated,
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mousereach-reconcile",
        description="Check every single-animal video against the definition of "
                    "done, from the files alone. Read-only.")
    ap.add_argument("--json", action="store_true", help="print everything as JSON")
    ap.add_argument("--all", action="store_true",
                    help="also list every video that is fine, and every leftover copy")
    args = ap.parse_args(argv)

    try:
        result = reconcile()
    except ReconcileUnavailable as e:
        print(f"[reconcile] cannot run: {e}")
        return 2

    unmigrated = result.get("unmigrated_folders") or []

    def warn_unmigrated(stream):
        # WHY first and on every output mode: "no mismatches" on an unmigrated
        # share is a plausible wrong answer; the warning must be seen before it.
        for rel in unmigrated:
            print(f"[reconcile] WARNING: {rel}: {UNMIGRATED_MESSAGE}", file=stream)
        if unmigrated:
            print("[reconcile] Work waiting in those folders is NOT checked here. "
                  "Migrate the share to the stage layout, then run this again.",
                  file=stream)

    if args.json:
        # The warning goes to stderr so stdout stays valid JSON; the same list
        # is in the JSON as "unmigrated_folders".
        warn_unmigrated(sys.stderr)
        print(json.dumps(result, indent=2))
        return 1 if (result["mismatches"] or unmigrated) else 0

    print(f"[reconcile] share:    {result['share']}")
    print(f"[reconcile] versions: {result['versions_file']}")
    if unmigrated:
        print()
        warn_unmigrated(sys.stdout)
    print(f"Checked {result['checked']:,} single-animal videos against the definition of done.")
    print()

    def show(rows):
        for r in rows:
            where = r["should_be"] or ", ".join(r["found_in"])
            print(f"  {r['video_id']:<32} {r['verdict']:<24} {where}"
                  + (f" -- {r['detail']}" if r["detail"] else ""))

    mism = result["mismatches"]
    if mism:
        print(f"MISMATCHES: {len(mism):,} videos -- each one is a case to investigate")
        show(mism)
    else:
        print("MISMATCHES: none")
    print()

    only_old = [r for r in result["rows"] if r["verdict"] == ONLY_IN_OLD_FOLDER]
    if only_old:
        print(f"Only copy is in leftovers set aside when the folder layout changed "
              f"({len(only_old):,}) -- not a mismatch, but may be real waiting work:")
        show(only_old)
        print()

    fine = [r for r in result["rows"]
            if r["verdict"] not in MISMATCHES and r["verdict"] != ONLY_IN_OLD_FOLDER]
    print("Fine as they stand:")
    for v in (DONE, HELD, NEEDS_PERSON, WAITING, UNSUPPORTED_TRAY):
        n = sum(1 for r in fine if r["verdict"] == v)
        print(f"  {v:<24} {n:>7,}")
    if args.all:
        show(fine)
    print()
    print(f"Leftover copies pending cleanup: {len(result['leftover_copies']):,}"
          + ("" if args.all else " (list them with --all)"))
    if args.all:
        for s in result["leftover_copies"]:
            print(f"  {s}")
    return 1 if (mism or unmigrated) else 0


if __name__ == "__main__":
    sys.exit(main())
