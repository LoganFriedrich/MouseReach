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

Exit codes: 0 no mismatches, 1 mismatches found, 2 could not run.

SCOPE: single-animal videos. Collages are not judged yet.

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

# Folders where a single waits before analysis, in the target layout.
_WAITING = ("Unanalyzed/Single_Animal", "Processing/Posed")
# Folders from the previous layout. What they hold is kept as insurance until
# cleanup and is NOT a stage: almost all of it is a copy of work finished
# elsewhere. A video found ONLY there is still listed, per video, because it
# may be real waiting work.
_OLD_FOLDERS = ("Processing/Single_Animal", "Processing/DLC_Complete")

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

    def verdict(v, detail=""):
        row["verdict"], row["detail"] = v, detail
        return row

    if not is_supported_tray_type(f"{stem}.mp4"):
        return verdict(UNSUPPORTED_TRAY, "tray type this pipeline does not analyse")

    queues = sorted(where & {"triage", "deep_review"})
    person = sorted(where & {"failed", "quarantine"})
    manifests = files.get("manifest") or []

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

    for rel in _WAITING:
        mark(ids_in_dir(nas / rel), "waiting:" + rel)
    for rel in _OLD_FOLDERS:
        mark(ids_in_dir(nas / rel), "old:" + rel)
    mark(bundles_in(Paths.TRIAGE_REVIEW), "triage")
    mark(bundles_in(Paths.DEEP_REVIEW), "deep_review")
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

    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if result["mismatches"] else 0

    print(f"[reconcile] share:    {result['share']}")
    print(f"[reconcile] versions: {result['versions_file']}")
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
        print(f"Only copy is in a folder from the previous layout ({len(only_old):,}) "
              "-- not a mismatch, but may be real waiting work:")
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
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
