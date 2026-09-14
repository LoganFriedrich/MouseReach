"""Walk the Analyzed tree without reading superseded outputs as live ones.

WHY THIS EXISTS
---------------
Superseded outputs belong under ``Analyzed/Archive/`` (the archive root in
``archive.supersede``), and they keep their ORIGINAL names:
``{stem}_features.json``, ``{stem}_processing_manifest.json``,
``{stem}DLC_...h5``. A walker that descends the whole Analyzed tree with
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

SUPERSEDED_DIR_NAMES = frozenset({"Archive"})


def is_superseded_dir(name: str) -> bool:
    """True for a folder that holds superseded outputs, never live ones."""
    return name in SUPERSEDED_DIR_NAMES


def iter_files(root, pattern: str = "*") -> Iterator[Path]:
    """Every file under ``root`` whose name matches ``pattern`` (fnmatch, with
    the platform's case rules, as ``Path.rglob`` has), never descending into a
    superseded folder. ``root`` itself is always walked, whatever its name; a
    missing root yields nothing. The drop-in replacement for
    ``root.rglob(pattern)`` restricted to files."""
    root = Path(root) if root else None
    if root is None or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_superseded_dir(d)]
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield Path(dirpath) / name


def first_file(root, pattern: str) -> Optional[Path]:
    """The first matching file, or None -- the replacement for
    ``next(root.rglob(pattern), None)``."""
    return next(iter_files(root, pattern), None)
