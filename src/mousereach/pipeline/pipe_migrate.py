"""
mousereach.pipeline.pipe_migrate -- one-time migration of the OLD pipe layout into
the canonical layout defined in ``pipe_structure`` (regroup the in-process folders
under ``Processing/``, rename the review root).

Safe by construction:
  * dry_run=True by DEFAULT -- prints the exact old->new move map, touches nothing.
  * NEVER deletes: a directory is MOVED (same-drive rename is instant); if the
    destination already holds that item (a prior partial run), the source items are
    merged in, never overwritten -- an on-disk name clash is versioned.
  * Idempotent: re-running after a completed migration is a no-op (sources gone,
    destinations present) and still reports success.

The config path constants are repointed to the new locations SEPARATELY, in
config.py, once the data is in place -- this tool only moves data.

Docs: reflect any layout change here + in pipe_structure + SYSTEM_ARCHITECTURE.md.
ASCII-only console output.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _quarantine_old() -> Optional[Path]:
    try:
        from ..config import WatcherConfig
        return WatcherConfig.load().get_quarantine_dir()
    except Exception:
        return None


def build_migration_map() -> List[Tuple[str, Path, Path]]:
    """``[(label, old_dir, new_dir), ...]`` for the regroup. Only entries whose
    OLD dir currently exists are returned."""
    from ..config import Paths
    root = Path(Paths.NAS_ROOT) if Paths.NAS_ROOT else None
    if root is None:
        return []
    proc = root / "Processing"
    candidates = [
        ("cropped singles", Paths.SINGLE_ANIMAL_OUTPUT, proc / "Single_Animal"),
        ("post-DLC staging", Paths.DLC_STAGING, proc / "DLC_Complete"),
        ("triage review", Paths.TRIAGE_REVIEW, proc / "Review" / "triage"),
        ("deep review", Paths.DEEP_REVIEW, proc / "Review" / "flagged_for_review"),
        ("quarantine", _quarantine_old(), proc / "Quarantine"),
        ("failed", Paths.FAILED, proc / "Failed"),
    ]
    out = []
    for label, old, new in candidates:
        if old and Path(old).exists() and Path(old).resolve() != Path(new).resolve():
            out.append((label, Path(old), Path(new)))
    return out


def _count(d: Path) -> int:
    try:
        return sum(1 for _ in d.iterdir())
    except Exception:
        return -1


def _merge_move(src: Path, dst: Path) -> Dict:
    """Move everything in ``src`` into ``dst`` (created if missing), never
    overwriting: a same-named dest item gets the source versioned (.1, .2 ...).
    Then remove ``src`` if it ends up empty. Same-drive moves are instant."""
    res = {"moved": 0, "versioned": 0, "errors": []}
    dst.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        target = dst / item.name
        if target.exists():
            i = 1
            while (dst / f"{item.name}.{i}").exists():
                i += 1
            target = dst / f"{item.name}.{i}"
            res["versioned"] += 1
        try:
            shutil.move(str(item), str(target))
            res["moved"] += 1
        except Exception as e:
            res["errors"].append(f"{item.name}: {e}")
    try:
        if not any(src.iterdir()):
            src.rmdir()
    except Exception:
        pass
    return res


def migrate_pipe(*, dry_run: bool = True) -> Dict:
    """Regroup the OLD in-process folders into ``Processing/`` per the canonical
    layout. dry_run=True (default) reports the move map only. Returns a summary."""
    from .pipe_structure import ensure_pipe_structure
    mp = build_migration_map()
    summary: Dict = {"dry_run": dry_run, "moves": [], "created_skeleton": None,
                     "ok": True}
    if not dry_run:
        summary["created_skeleton"] = ensure_pipe_structure()

    for label, old, new in mp:
        entry = {"label": label, "old": str(old), "new": str(new),
                 "old_entries": _count(old)}
        if dry_run:
            entry["action"] = "would move"
        else:
            r = _merge_move(old, new)
            entry["action"] = "moved"
            entry["result"] = r
            if r["errors"]:
                summary["ok"] = False
        summary["moves"].append(entry)
    return summary


def format_summary(summary: Dict) -> str:
    lines = ["Pipe migration " + ("(DRY RUN -- nothing moved)" if summary.get("dry_run") else "(EXECUTED)")]
    for m in summary.get("moves", []):
        lines.append(f"  {m['action']}: {m['label']}")
        lines.append(f"      {m['old']}  ({m['old_entries']} entries)")
        lines.append(f"   -> {m['new']}")
        if m.get("result"):
            r = m["result"]
            lines.append(f"      moved {r['moved']}, versioned {r['versioned']}, errors {len(r['errors'])}")
    if not summary.get("moves"):
        lines.append("  (nothing to migrate -- already in canonical layout)")
    lines.append(f"  OK: {summary.get('ok')}")
    return "\n".join(lines)
