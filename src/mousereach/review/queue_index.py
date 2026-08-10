"""Triage-review queue index -- a push/pop list of the videos currently needing
review, so the review tool READS the next video instead of SCANNING every bundle.

The model (per Logan, 2026-08)
------------------------------
- PRESENCE in the triage folder == needs review. A video's bundle is in the folder
  iff it still has triaged elements a human has not addressed. The tool MOVES a
  fully-reviewed video OUT of the folder, so nothing in the folder is "done".
  => determining the queue needs NO per-bundle content reads (the 4.5-minute scan).
- WHOLE-VIDEO per load: the tool addresses ALL of a video's triaged elements, then
  that video is done -> moved out + popped from the index.
- The index is PUSH/POP, not derived by the tool:
    * watcher / bring-current PUSH a bundle in as it lands in the folder,
    * the review tool POPS a bundle out (index + folder) when fully reviewed.
  The tool never builds the index; it only consumes it.

Store
-----
SQLite (WAL) at ``review_records/triage_queue.db`` -- one row per queued video:
``stem, bundle_path, staged_at``. Every present row needs review by definition, so
there is no per-segment state to track here (the bundle JSONs remain the source of
truth for WHICH elements to show once a video is loaded). WAL makes push (many
concurrent stagers) + pop (the tool) safe. The index is rebuildable from the folder
(``seed_from_folder``) so it is never a single point of failure; callers fall back
to a folder listing if it is unavailable.

ASCII-only console output. Update docs/ + AGENTS.md when wired into staging/review.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional


def default_index_path() -> Optional[Path]:
    """``review_records/triage_queue.db`` next to the causal-review index."""
    try:
        from ..config import Paths
        root = Paths.NAS_ROOT or Paths.PROCESSING_ROOT
        return (Path(root) / "review_records" / "triage_queue.db") if root else None
    except Exception:
        return None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS queue (
               stem         TEXT PRIMARY KEY,
               bundle_path  TEXT NOT NULL,
               staged_at    TEXT
           )""")
    return con


class QueueIndex:
    """Push/pop index of videos needing review. Every row == needs review."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else default_index_path()
        if self.db_path is None:
            raise RuntimeError("queue index path unresolved (no NAS/PROCESSING root)")

    # -- push: watcher / bring-current, as a bundle lands in the folder -----
    def push(self, stem: str, bundle_path, staged_at: Optional[str] = None) -> None:
        con = _connect(self.db_path)
        try:
            con.execute(
                """INSERT INTO queue(stem, bundle_path, staged_at) VALUES(?,?,?)
                   ON CONFLICT(stem) DO UPDATE SET bundle_path=excluded.bundle_path""",
                (stem, str(bundle_path), staged_at))
        finally:
            con.close()

    # -- pop: review tool, when a video is fully reviewed + moved out -------
    def pop(self, stem: str) -> None:
        con = _connect(self.db_path)
        try:
            con.execute("DELETE FROM queue WHERE stem=?", (stem,))
        finally:
            con.close()

    # -- read: the next video(s) to review (no scan, no content reads) ------
    def next_path(self, exclude: Iterable[str] = ()) -> Optional[Path]:
        """Oldest queued video not in ``exclude`` (FIFO by staged_at)."""
        ex = {str(e) for e in exclude}
        con = _connect(self.db_path)
        try:
            rows = con.execute(
                "SELECT bundle_path FROM queue ORDER BY staged_at IS NULL, staged_at, stem"
            ).fetchall()
        finally:
            con.close()
        for (p,) in rows:
            if p not in ex:
                return Path(p)
        return None

    def all_paths(self) -> List[Path]:
        con = _connect(self.db_path)
        try:
            rows = con.execute("SELECT bundle_path FROM queue").fetchall()
        finally:
            con.close()
        return [Path(p) for (p,) in rows]

    def count(self) -> int:
        con = _connect(self.db_path)
        try:
            return con.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
        finally:
            con.close()


def seed_from_folder(pending_dir, db_path: Optional[Path] = None, *,
                     exclude_reviewed: bool = True, progress=None) -> dict:
    """One-time (or self-heal) seed of the index from the folder.

    Going forward the index is maintained by push/pop, but the CURRENT folder still
    holds already-reviewed bundles (the old tool left them in place). With
    ``exclude_reviewed=True`` this does a one-time content check so those are NOT
    seeded as needing review; once the reviewed bundles have been moved out, the
    folder is unreviewed-only and this is a plain presence listing
    (``exclude_reviewed=False``)."""
    import json
    pending_dir = Path(pending_dir)
    idx = QueueIndex(db_path)
    seeded = skipped = 0
    root = pending_dir.parent
    if exclude_reviewed:
        from .triage_status import triaged_segments, resolved_segments, segmentation_failed
        from .causal_review_io import (bundle_manifest_path, load_causal_review,
                                       has_gt, is_session_flagged)
    for b in sorted(pending_dir.iterdir()):
        if not b.is_dir() or b.name.startswith((".", "_")):
            continue
        stem = b.name
        try:
            from .causal_review_io import bundle_manifest_path as _bmp
            if not _bmp(b).exists():
                continue
        except Exception:
            continue
        if exclude_reviewed:
            def _rj(name):
                p = b / name
                try:
                    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
                except Exception:
                    return None
            od = _rj(f"{stem}_pellet_outcomes.json")
            ad = _rj(f"{stem}_reach_assignments.json")
            tri = triaged_segments(od, ad)
            if not tri or segmentation_failed(_rj(f"{stem}_segments.json")) \
                    or is_session_flagged(stem, root) or has_gt(stem, extra_dirs=[b]):
                skipped += 1
                continue
            resolved = set()
            dirs = [b]
            try:
                man = json.loads(bundle_manifest_path(b).read_text(encoding="utf-8"))
                cvp = man.get("canonical_video_path")
                if cvp:
                    dirs.append(Path(cvp).parent)
            except Exception:
                pass
            for d in dirs:
                try:
                    doc, by = load_causal_review(stem, d)
                except Exception:
                    doc, by = None, {}
                if by:
                    resolved |= resolved_segments(doc or {"segments": list(by.values())})
            if tri.issubset(resolved):
                skipped += 1
                continue
        idx.push(stem, b)
        seeded += 1
        if progress and (seeded + skipped) % 100 == 0:
            progress(seeded + skipped)
    return {"seeded": seeded, "skipped_reviewed": skipped, "db": str(idx.db_path)}
