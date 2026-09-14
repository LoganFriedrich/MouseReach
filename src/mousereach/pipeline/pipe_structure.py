"""
mousereach.pipeline.pipe_structure -- the canonical pipe folder layout + an
idempotent builder that ensures it exists.

The pipe is organized by STATE (the operator's rule). A folder is a place work
RESTS, named for what the work is waiting for:
  * waiting for something to be done to it -> Unanalyzed
  * waiting between algorithm steps / for a person -> Processing
  * everything is done                     -> Analyzed

Target layout (relative to the pipe root = the configured data-pipe drive):

    Unanalyzed/
        Multi-Animal/                 raw collages -- waiting to be cut into
                                      singles (crop+DLC are coupled on the GPU
                                      node); a collage also waits here until all
                                      its offspring finish
        Single_Animal/                cut singles, waiting for a pose
    Processing/
        Posed/                        posed singles, waiting for the MouseReach
                                      algorithms (a node claims them from here)
        Repose_Queue/                 re-pose requests (one json per video) for
                                      any GPU node to pull -- see watcher/repose.py
        Review/
            triage/                   held for a person -> the triage review tool
            deep_review/              held for a person -> causal / ground-truth
                                      deep review
        Quarantine/
        Failed/
    Analyzed/                         everything done (per-cohort below, on demand)
        <project>/<cohort>/
            multi/  single/  historical/     historical = non-CNT / old-tool copies
        Archive/                      superseded old-version outputs; every walk
                                      of Analyzed skips it (analyzed_tree.py)

Retired names (NEVER created here): Processing/Single_Animal,
Processing/DLC_Complete, Processing/Review/flagged_for_review. On a migrated drive
each of those is a plain guard FILE, so any code still aimed at the old layout
raises instead of silently rebuilding a folder nothing else watches. A top-level
Archive/ may exist for read-only source material (Archive/historical/); it is not
part of the working skeleton, so it is not created here.

``ensure_pipe_structure`` creates only what is MISSING -- every step is
"exists? -> skip and report success : create" -- so it is safe to run or test at
any time and never moves, replaces, or deletes anything. Folder names are chosen
to say why the folder exists and hint at what's inside.

Docs: when this layout changes, update docs/PIPELINE_AS_BUILT.md and the other
docs that name pipeline folders.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Fixed skeleton, relative to the pipe root. Per-cohort dirs under Analyzed are
# created on demand by ``ensure_cohort_dirs`` (cohorts aren't known up front).
# config.Paths derives its stage folders from the same names; a test
# (tests/test_stage_layout.py) holds the two together so they cannot drift.
TARGET_DIRS: List[str] = [
    "Unanalyzed/Multi-Animal",
    "Unanalyzed/Single_Animal",
    "Processing/Posed",
    "Processing/Repose_Queue",
    "Processing/Review/triage",
    "Processing/Review/deep_review",
    "Processing/Quarantine",
    "Processing/Failed",
    "Analyzed",
    "Analyzed/Archive",
]

# Folder names of the previous layout. WHY kept as a list: the data migration
# leaves a guard FILE at each of these paths, and the builder refuses to create
# any of them (or anything beneath one) even if a future edit re-adds one to
# TARGET_DIRS -- recreating one would split waiting work across two folders.
RETIRED_DIRS: List[str] = [
    "Processing/Single_Animal",
    "Processing/DLC_Complete",
    "Processing/Review/flagged_for_review",
]

# Per-cohort leaves under Analyzed/<project>/<cohort>/.
COHORT_LEAVES: List[str] = ["multi", "single", "historical"]

# What a person is told when a retired name is still a real FOLDER on the share.
# WHY one shared sentence: the skeleton builder, reconcile and the watcher's
# startup check all say it, and a person searching for the message should find
# every place that raises it.
UNMIGRATED_MESSAGE = ("retired folder still exists as a folder -- this share has "
                      "not been migrated to the stage layout")


def _is_retired(rel: str) -> bool:
    """True when ``rel`` is a retired folder name or lies beneath one."""
    norm = rel.replace("\\", "/").strip("/")
    return any(norm == old or norm.startswith(old + "/") for old in RETIRED_DIRS)


def retired_folders_present(pipe_root) -> List[str]:
    """The ``RETIRED_DIRS`` entries that are still real FOLDERS under ``pipe_root``.

    WHY this exists: current code never lists the retired folders (a migrated
    share has a guard FILE at each). On a share that was never migrated -- another
    lab upgrading, or a node pointed at an old copy -- those names are still real
    folders that may hold waiting work, and every stage walk ignores them. Without
    this check that work drops out of sight with no message anywhere.

    Stat only: it never creates, lists or opens the folder, so it is safe on a
    migrated share (a guard file is not a folder and is not reported) and on an
    unmigrated one. An unreadable path is not reported rather than raising."""
    if not pipe_root:
        return []
    root = Path(pipe_root)
    found: List[str] = []
    for rel in RETIRED_DIRS:
        try:
            if (root / rel).is_dir():
                found.append(rel)
        except OSError:
            continue
    return found


def default_pipe_root() -> Optional[Path]:
    """The configured data-pipe drive root (Paths.NAS_ROOT), or None."""
    from ..config import Paths
    return Path(Paths.NAS_ROOT) if Paths.NAS_ROOT else None


def ensure_pipe_structure(pipe_root=None, *, dry_run: bool = False) -> Dict:
    """Ensure the canonical pipe skeleton exists under ``pipe_root``.

    Idempotent and non-destructive: each folder is created only if missing; an
    existing folder is left exactly as-is and still counts as success. Returns a
    summary ``{root, created, existed, failed, ok}``; ``ok`` is True when every
    target dir is present at the end. ``dry_run`` reports what WOULD be created
    without touching disk. A retired (old-layout) name is never created -- it is
    reported under ``failed`` instead. A plain file sitting where a folder is
    expected is also reported under ``failed`` rather than raising."""
    pipe_root = Path(pipe_root) if pipe_root else default_pipe_root()
    summary: Dict = {"root": str(pipe_root) if pipe_root else None,
                     "created": [], "existed": [], "failed": [], "ok": False,
                     "dry_run": dry_run}
    if pipe_root is None:
        summary["failed"].append("(pipe root / NAS not configured)")
        return summary

    # WHY report (not fix) a retired name that is still a real folder: this share
    # was never migrated, so work may be waiting in a folder no stage walk reads.
    # Moving it is the data migration's job, not this builder's -- it never
    # moves or deletes anything -- but it must not answer "ok" either. A guard
    # FILE at that path is the migrated state and is left alone, unreported.
    for rel in retired_folders_present(pipe_root):
        summary["failed"].append(f"{rel}: {UNMIGRATED_MESSAGE}")
        logger.warning("ensure_pipe_structure: %s: %s", rel, UNMIGRATED_MESSAGE)

    for rel in TARGET_DIRS:
        if _is_retired(rel):
            # WHY refuse rather than skip silently: a retired name in the
            # skeleton is a code defect a person must see, not a folder to build.
            summary["failed"].append(f"{rel}: retired folder name, not created")
            logger.warning("ensure_pipe_structure: refusing retired folder %s", rel)
            continue
        d = pipe_root / rel
        if d.is_dir():
            summary["existed"].append(rel)
            continue
        if dry_run:
            summary["created"].append(rel)  # would create
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            summary["created"].append(rel)
        except OSError as e:
            summary["failed"].append(f"{rel}: {e}")
            logger.warning("ensure_pipe_structure: could not create %s: %s", d, e)

    summary["ok"] = not summary["failed"] and (
        dry_run or all((pipe_root / r).is_dir() for r in TARGET_DIRS))
    return summary


def ensure_cohort_dirs(project: str, cohort: str, pipe_root=None,
                       *, dry_run: bool = False) -> Dict:
    """Ensure ``Analyzed/<project>/<cohort>/{multi,single,historical}`` exists.
    Same idempotent, non-destructive contract as ``ensure_pipe_structure``."""
    pipe_root = Path(pipe_root) if pipe_root else default_pipe_root()
    summary: Dict = {"created": [], "existed": [], "failed": [], "ok": False}
    if pipe_root is None:
        summary["failed"].append("(pipe root not configured)")
        return summary
    base = pipe_root / "Analyzed" / project / cohort
    for leaf in COHORT_LEAVES:
        d = base / leaf
        rel = str(d.relative_to(pipe_root))
        if d.is_dir():
            summary["existed"].append(rel)
            continue
        if dry_run:
            summary["created"].append(rel)
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            summary["created"].append(rel)
        except OSError as e:
            summary["failed"].append(f"{rel}: {e}")
    summary["ok"] = not summary["failed"]
    return summary


def format_summary(summary: Dict) -> str:
    """Human-readable one-block report (for CLI / GUI)."""
    lines = [f"Pipe structure @ {summary.get('root')}"]
    if summary.get("dry_run"):
        lines.append("  (dry run -- nothing was created)")
    lines.append(f"  created: {len(summary.get('created', []))}"
                 f"  |  already present: {len(summary.get('existed', []))}"
                 f"  |  failed: {len(summary.get('failed', []))}")
    for rel in summary.get("created", []):
        lines.append(f"    + {rel}")
    for f in summary.get("failed", []):
        lines.append(f"    ! {f}")
    lines.append(f"  OK: {summary.get('ok')}")
    return "\n".join(lines)
