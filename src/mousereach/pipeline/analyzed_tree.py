"""Walk the Analyzed tree without reading superseded outputs as live ones.

WHY THIS EXISTS
---------------
Superseded outputs belong under ``Analyzed/Archive/``, and they keep their
ORIGINAL names: ``{stem}_features.json``, ``{stem}_processing_manifest.json``,
``{stem}DLC_...h5``. That is the TARGET layout of the folder migration, not
yet what the code writes: ``archive.supersede.default_archive_root`` still
returns ``<NAS_ROOT>/Archive``, a sibling of Analyzed (a strict-xfail
contract test in tests/test_analyzed_tree.py flips when the move lands, so
this name and supersede's folder cannot drift apart unnoticed). A walker that
descends the whole Analyzed tree with
``rglob`` therefore finds an older generation's files beside the live ones and
cannot tell them apart by name. Every consequence is silent: the version scan
reads a superseded manifest and marks a current video outdated, a returning
review bundle is re-run on an old-model pose, the bench-disagreement route
writes flags into an archived file and moves an archive folder into a review
queue, and the cohort kinematics export gains rows from old generations.

So every walk of Analyzed goes through here, and none enters a superseded
folder.

``DLC Model <N>/`` folders are deliberately NOT skipped. They are live pose
storage -- a finished video's current pose can sit only there -- and the pose
lookups depend on finding it. A tool that judges RESULTS only may prune more
on top of this rule (``watcher.reconcile`` also skips pose-only and scratch
folders); that is its own choice.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator, Optional

# "Archive": the superseded-output root (see above).
# "_archived": MouseReach's own pre-modification backups -- e.g. the
# backfill-manifest-versions CLI and the segmentation fixer copy originals,
# ORIGINAL names kept, into a dated folder under a "_archived" parent. Run with
# --root Analyzed/<project>, that parent was Analyzed/_archived, two levels
# deep like a live Analyzed/<project>/<cohort>/<file>, so the version scan read
# a pre-backfill manifest as live. watcher.reconcile already pruned it ("_"
# prefix); the shared rule must not be laxer than the tool it replaced.
SUPERSEDED_DIR_NAMES = frozenset({"Archive", "_archived"})


def is_superseded_dir(name: str) -> bool:
    """True for a folder that holds superseded outputs, never live ones."""
    return name in SUPERSEDED_DIR_NAMES


def walk_onerror(error: OSError) -> None:
    """``os.walk`` onerror that keeps ``Path.rglob``'s rule for a folder whose
    listing fails: a PermissionError is skipped quietly, anything else raises.

    WHY: os.walk's default (onerror=None) silently skips EVERY folder it cannot
    list, including a transient NAS error (WinError 59/64). A walk that used to
    raise would then return a partial answer that looks complete -- a pose
    "not found" recorded as a video failure, a backfill or export missing
    rows -- and nothing here may fail silently. Python 3.10's rglob catches
    only PermissionError around its scandir (its ENOENT/ENOTDIR/EBADF/ELOOP
    and winerror 21/123/1921 ignore-list covers the per-entry is_dir probe,
    not the listing), so this is the exact rule the walkers had before."""
    if isinstance(error, PermissionError):
        return
    raise error


def iter_files(root, pattern: str = "*") -> Iterator[Path]:
    """Every file under ``root`` whose name matches ``pattern`` (fnmatch, with
    the platform's case rules, as ``Path.rglob`` has), never descending into a
    superseded folder. ``root`` itself is always walked, whatever its name; a
    missing root yields nothing. The drop-in replacement for
    ``root.rglob(pattern)`` restricted to files: an unlistable folder raises
    as it did under rglob (PermissionError alone is skipped; see
    ``walk_onerror``)."""
    root = Path(root) if root else None
    if root is None or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, onerror=walk_onerror):
        dirnames[:] = [d for d in dirnames if not is_superseded_dir(d)]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield Path(dirpath) / name


def first_file(root, pattern: str) -> Optional[Path]:
    """The first matching file, or None -- the replacement for
    ``next(root.rglob(pattern), None)``."""
    return next(iter_files(root, pattern), None)
