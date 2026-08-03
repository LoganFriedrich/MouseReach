"""
mousereach.pipeline.pipe_structure -- the canonical pipe folder layout + an
idempotent builder that ensures it exists.

The pipe is organized by STATE (the operator's rule):
  * waiting to have something done to it  -> Unanalyzed
  * something is being done to it         -> Processing
  * everything is done                    -> Analyzed
plus Archive for superseded / moved-aside material.

Target layout (relative to the pipe root = the configured data-pipe drive):

    Unanalyzed/
        Multi-Animal/                 raw collages -- waiting to be claimed for
                                      crop+DLC (coupled on the GPU node); a collage
                                      also waits here until all its offspring finish
    Processing/                       anything being worked (not raw, not done)
        Single_Animal/                cropped singles (cropping counts as processing)
        DLC_Complete/                 posed, waiting for the MouseReach claim
        Review/
            triage/                   -> the triage review tool
            flagged_for_review/       -> deeper causal / ground-truth review
        Quarantine/
        Failed/
    Analyzed/                         everything done (per-cohort below, on demand)
        <project>/<cohort>/
            multi/  single/  historical/     historical = non-CNT / old-tool copies
    Archive/                          superseded old-version outputs + moved-out cruft

``ensure_pipe_structure`` creates only what is MISSING -- every step is
"exists? -> skip and report success : create" -- so it is safe to run or test at
any time and never moves, replaces, or deletes anything. Folder names are chosen
to say why the folder exists and hint at what's inside.

Docs: when this layout changes, update docs/pipeline/SYSTEM_ARCHITECTURE.md
(folder-structure section) and the top-level CLAUDE.md folder map.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Fixed skeleton, relative to the pipe root. Per-cohort dirs under Analyzed are
# created on demand by ``ensure_cohort_dirs`` (cohorts aren't known up front).
TARGET_DIRS: List[str] = [
    "Unanalyzed/Multi-Animal",
    "Processing/Single_Animal",
    "Processing/DLC_Complete",
    "Processing/Review/triage",
    "Processing/Review/flagged_for_review",
    "Processing/Quarantine",
    "Processing/Failed",
    "Analyzed",
    "Archive",
]

# Per-cohort leaves under Analyzed/<project>/<cohort>/.
COHORT_LEAVES: List[str] = ["multi", "single", "historical"]


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
    without touching disk."""
    pipe_root = Path(pipe_root) if pipe_root else default_pipe_root()
    summary: Dict = {"root": str(pipe_root) if pipe_root else None,
                     "created": [], "existed": [], "failed": [], "ok": False,
                     "dry_run": dry_run}
    if pipe_root is None:
        summary["failed"].append("(pipe root / NAS not configured)")
        return summary

    for rel in TARGET_DIRS:
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
