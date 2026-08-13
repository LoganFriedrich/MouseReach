"""Per-video version index -- WHICH algo versions each video was processed with,
recorded at processing time so the dashboard READS instead of SCANNING.

The model (per Logan, 2026-08)
------------------------------
- The writer already knows the answer. ``create_processing_manifest()`` composes
  the exact versions a video was processed with, then writes them to that video's
  ``_processing_manifest.json``. Nothing else learns anything new later.
- So the dashboard should not re-derive it. Before this module it did: an rglob
  across the archive for ~2,657 manifests, then a NAS read+parse of each one
  (~22 ms) to answer "is this video current?". That is thousands of round-trips
  for data that was already known at write time.
- => WRITERS PUSH, READERS READ. Every manifest write upserts one row here; the
  dashboard reads the whole table in a single query.

Store facts, not verdicts
-------------------------
Rows hold the video's OWN versions (``pipeline_versions`` + ``dlc_scorer``), NOT a
"current/outdated" verdict. The verdict is a comparison against the shipped
``pipeline_versions.json``, which changes whenever we ship an algo -- caching the
verdict would mean rewriting every row on each bump. Storing the facts lets
``status_maps()`` derive verdicts in memory at read time (microseconds), so a
version bump is picked up for free on the next read.

This also keeps the comparison in ONE place: rows are reassembled into the two
keys ``compare_manifest_to_current()`` actually reads, and that existing function
is reused unchanged rather than reimplemented here.

Never a single point of failure
-------------------------------
- Writes are best-effort: ``upsert_from_manifest`` swallows its own errors, so an
  unavailable/locked index can NEVER break processing. The manifest JSON on disk
  remains the source of truth.
- The index is fully rebuildable from those manifests (``seed_from_manifests``),
  and readers fall back to the old manifest scan for anything missing, so a stale
  or absent index degrades to the previous behaviour instead of lying.

Store: SQLite (WAL) at ``pipeline_records/version_index.db``. WAL because several
compute nodes write concurrently while the dashboard reads.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def default_index_path() -> Optional[Path]:
    """``pipeline_records/version_index.db`` under the NAS (or processing) root."""
    try:
        from ..config import Paths
        root = Paths.NAS_ROOT or Paths.PROCESSING_ROOT
        return (Path(root) / "pipeline_records" / "version_index.db") if root else None
    except Exception:
        return None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS versions (
               stem              TEXT PRIMARY KEY,
               pipeline_versions TEXT,
               dlc_scorer        TEXT,
               manifest_path     TEXT,
               updated_at        TEXT
           )""")
    return con


class VersionIndex:
    """One row per video: the versions it was processed with."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else default_index_path()
        if self.db_path is None:
            raise RuntimeError("version index path unresolved (no NAS/PROCESSING root)")

    # -- write: every manifest write pushes one row -------------------------
    def upsert(self, stem: str, pipeline_versions: Optional[dict],
               dlc_scorer: Optional[str], manifest_path=None,
               updated_at: Optional[str] = None) -> None:
        con = _connect(self.db_path)
        try:
            con.execute(
                """INSERT INTO versions(stem, pipeline_versions, dlc_scorer,
                                        manifest_path, updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(stem) DO UPDATE SET
                       pipeline_versions=excluded.pipeline_versions,
                       dlc_scorer=excluded.dlc_scorer,
                       manifest_path=excluded.manifest_path,
                       updated_at=excluded.updated_at""",
                (stem,
                 json.dumps(pipeline_versions or {}),
                 dlc_scorer or None,
                 str(manifest_path) if manifest_path else None,
                 updated_at or datetime.now().isoformat()))
        finally:
            con.close()

    def upsert_from_manifest(self, stem: str, manifest: dict, manifest_path=None) -> bool:
        """Push a row from a manifest dict. Best-effort BY DESIGN: the caller is
        the processing pipeline, and a bookkeeping index must never be able to
        fail a video's processing. Returns True if the row landed."""
        try:
            self.upsert(
                stem,
                manifest.get("pipeline_versions") or {},
                (manifest.get("dlc_model") or {}).get("dlc_scorer"),
                manifest_path,
                manifest.get("created_at"),
            )
            return True
        except Exception as e:
            logger.debug(f"version index upsert failed for {stem}: {e}")
            return False

    # -- read: one query, no per-video I/O ----------------------------------
    def read_all(self) -> Dict[str, dict]:
        """``{stem: manifest-shaped dict}`` -- reassembled into exactly the two
        keys ``compare_manifest_to_current()`` reads, so the existing comparison
        works on these rows unchanged."""
        con = _connect(self.db_path)
        try:
            rows = con.execute(
                "SELECT stem, pipeline_versions, dlc_scorer FROM versions").fetchall()
        finally:
            con.close()
        out: Dict[str, dict] = {}
        for stem, pv, scorer in rows:
            try:
                versions = json.loads(pv) if pv else {}
            except Exception:
                versions = {}
            out[stem] = {"pipeline_versions": versions,
                         "dlc_model": {"dlc_scorer": scorer or ""}}
        return out

    def upsert_many(self, rows) -> int:
        """Bulk upsert under ONE connection + transaction.

        The per-row ``upsert`` opens its own connection, which is right for the
        steady state (one video finishing at a time, several nodes concurrent) but
        pathological for a backfill: thousands of connections to a database on the
        NAS. Returns the number of rows written.

        ``rows``: iterables of ``(stem, pipeline_versions dict, dlc_scorer,
        manifest_path, updated_at)``.
        """
        payload = [
            (stem, json.dumps(pv or {}), scorer or None,
             str(mp) if mp else None, updated or datetime.now().isoformat())
            for stem, pv, scorer, mp, updated in rows
        ]
        if not payload:
            return 0
        con = _connect(self.db_path)
        try:
            con.execute("BEGIN")
            con.executemany(
                """INSERT INTO versions(stem, pipeline_versions, dlc_scorer,
                                        manifest_path, updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(stem) DO UPDATE SET
                       pipeline_versions=excluded.pipeline_versions,
                       dlc_scorer=excluded.dlc_scorer,
                       manifest_path=excluded.manifest_path,
                       updated_at=excluded.updated_at""",
                payload)
            con.execute("COMMIT")
        finally:
            con.close()
        return len(payload)

    def count(self) -> int:
        con = _connect(self.db_path)
        try:
            return con.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
        finally:
            con.close()


def status_maps(current_versions: Optional[dict],
                db_path: Optional[Path] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``({stem: 'current'|'outdated'}, {stem: dlc_scorer})`` from ONE index read.

    This is the dashboard's fast path: it replaces an archive-wide rglob plus a
    NAS read per video with a single query, and derives each verdict in memory so
    a shipped-version bump needs no index rewrite. Returns empty maps if the index
    is unavailable, which the caller treats as 'fall back to scanning'.
    """
    try:
        idx = VersionIndex(db_path)
        rows = idx.read_all()
    except Exception as e:
        logger.debug(f"version index unavailable: {e}")
        return {}, {}

    status_map: Dict[str, str] = {}
    dlc_map: Dict[str, str] = {}
    have_versions = bool(current_versions and current_versions.get("versions"))
    if have_versions:
        from .versions import compare_manifest_to_current
    for stem, manifest in rows.items():
        scorer = (manifest.get("dlc_model") or {}).get("dlc_scorer")
        if scorer:
            dlc_map[stem] = scorer
        if have_versions:
            cmp = compare_manifest_to_current(manifest, current_versions)
            status_map[stem] = "current" if cmp.get("is_current") else "outdated"
    return status_map, dlc_map


def seed_from_manifests(roots, db_path: Optional[Path] = None, progress=None) -> dict:
    """Backfill the index from the manifests already on disk.

    Needed once, because ~2,657 videos were processed before this index existed.
    Also serves as the self-heal / rebuild path, which is what lets readers treat
    the index as an optimisation rather than a source of truth.

    Returns a summary dict: ``{scanned, indexed, failed}``.
    """
    from ..dashboard.version_currency import build_manifest_index

    idx = VersionIndex(db_path)
    if progress:
        progress("Finding manifests...")
    manifest_index = build_manifest_index(roots)

    from concurrent.futures import ThreadPoolExecutor

    total = len(manifest_index)
    if progress:
        progress(f"Reading {total} manifests...")

    def read_one(item):
        stem, mp = item
        try:
            m = json.loads(Path(mp).read_text(encoding="utf-8"))
        except Exception:
            return None
        return (stem, m.get("pipeline_versions") or {},
                (m.get("dlc_model") or {}).get("dlc_scorer"),
                mp, m.get("created_at"))

    # The manifest reads are I/O-bound NAS round-trips, so a small pool collapses
    # the wall-clock cost; the writes then go in as ONE transaction.
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(read_one, manifest_index.items()))

    rows = [r for r in results if r is not None]
    failed = len(results) - len(rows)
    if progress:
        progress(f"Writing {len(rows)} rows...")
    indexed = idx.upsert_many(rows)
    return {"scanned": total, "indexed": indexed, "failed": failed}


def _default_roots():
    """The pipeline roots that hold processed videos (working + final output)."""
    from ..config import Paths
    return [Paths.PROCESSING, Paths.ANALYZED_OUTPUT]


def cli_build():
    """``mousereach-version-index-build`` -- backfill/rebuild the version index."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mousereach-version-index-build",
        description=(
            "Build (or rebuild) the per-video version index that the dashboard "
            "reads. Normally you never need this: every processing run pushes its "
            "own row automatically. Run it ONCE to backfill videos processed "
            "before the index existed, or any time you want to rebuild it from "
            "scratch -- it is always safe to re-run, since rows are upserted by "
            "video and rebuilt from each video's manifest on disk."),
        epilog="Example:  mousereach-version-index-build")
    parser.add_argument(
        "--db", default=None,
        help="Index location (default: pipeline_records/version_index.db on the NAS root).")
    args = parser.parse_args()

    db = Path(args.db) if args.db else default_index_path()
    if db is None:
        print("[FAIL] Could not resolve the index location (no NAS/processing root configured).")
        print("       Run 'mousereach-setup' first, or pass --db explicitly.")
        return 1
    print(f"Building version index at: {db}")
    summary = seed_from_manifests(_default_roots(), db, progress=lambda m: print(f"  {m}"))
    print(f"[OK] scanned={summary['scanned']}  indexed={summary['indexed']}  failed={summary['failed']}")
    return 0


def cli_status():
    """``mousereach-version-index-status`` -- what does the index currently hold?"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mousereach-version-index-status",
        description=(
            "Show the per-video version index the dashboard reads: where it "
            "lives, how many videos it covers, and how many of those are current "
            "vs outdated against the shipped algorithm versions."),
        epilog="Example:  mousereach-version-index-status")
    parser.add_argument("--db", default=None, help="Index location (default: the NAS root).")
    args = parser.parse_args()

    db = Path(args.db) if args.db else default_index_path()
    if db is None:
        print("[FAIL] Could not resolve the index location (no NAS/processing root configured).")
        return 1
    if not Path(db).exists():
        print(f"[!] No index yet at: {db}")
        print("    Build it with: mousereach-version-index-build")
        return 0

    from .versions import get_current_versions
    idx = VersionIndex(db)
    print(f"Index: {db}")
    print(f"Videos indexed: {idx.count()}")
    status_map, dlc_map = status_maps(get_current_versions(), db)
    if not status_map:
        print("Shipped versions not declared -- cannot judge current vs outdated.")
        print("  (see pipeline_versions.json / mousereach-version-check)")
        return 0
    current = sum(1 for v in status_map.values() if v == "current")
    outdated = sum(1 for v in status_map.values() if v == "outdated")
    print(f"  current:  {current}")
    print(f"  outdated: {outdated}")
    scorers = {}
    for s in dlc_map.values():
        scorers[s] = scorers.get(s, 0) + 1
    if scorers:
        print("DLC models:")
        for s, n in sorted(scorers.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {s}")
    return 0
