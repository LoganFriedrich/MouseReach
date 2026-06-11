"""
Algo Eval Registry for the MouseReach Improvement Process.

A SQLite-backed, git-recoverable record of how each algo-element version
performed on each evaluation corpus. Its reason for existing is a concrete
failure: an evaluation was run against a STALE runtime (a detector build that
did not match the git commit), and nobody could tell for hours. This registry
makes that impossible to miss by capturing, alongside every recorded number,
a fingerprint of the source that WOULD ACTUALLY RUN if you imported the algo
right now.

The registry stores one row per evaluation run in table `eval_runs`. The
anti-stale-runtime field is `code_fingerprint`: a sha256 over the source text
of the algo's runtime module files, resolved from the IMPORTED package (not a
hard-coded path), so it reflects what the live interpreter would execute. Pair
that with `git_commit` / `git_dirty` (the tool-repo HEAD at record time) and you
can always answer two questions later:

  1. What commit produced this number?  -> git_commit / git_dirty
  2. Does the runtime STILL match what produced this number?  -> verify_run()

Data, not code: the DB itself is DATA and lives under the MouseReach_Improvement
data area (NOT in the git repo):

    Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\algo_eval.db

Override with the `db_path` argument on any function, or the
MOUSEREACH_EVAL_DB environment variable.

Usage (Python)
--------------
    from mousereach.improvement import registry

    registry.init_db()  # idempotent

    run_id = registry.record_eval(
        algo_element="reach_detection",
        version="8.0.4",
        metrics={"tp": 3089, "fp": 79, "fn": 92,
                 "recall": 0.9711, "precision": 0.9751},
        corpus_id="holdout_generalization_2026-05-11",
        video_ids=["20250625_CNT0106_P2", "20250711_CNT0216_P1"],
        grading_standard={"matcher": "analyze._match", "start_tol": 2,
                          "span_tol_rel": 0.5, "span_tol_abs": 5},
        snapshot_path=r"...\\Improvement_Snapshots\\reach_detection\\headtohead_..._holdout",
        notes="v8.0.4 head-to-head, holdout @ threshold 0.50",
    )

    row = registry.get_run(run_id)
    check = registry.verify_run(run_id)   # fingerprint match/mismatch vs current runtime

Usage (CLI)
-----------
    python -m mousereach.improvement.registry list [algo_element]
    python -m mousereach.improvement.registry show <id>
    python -m mousereach.improvement.registry latest <algo_element>
    python -m mousereach.improvement.registry verify <id>
    python -m mousereach.improvement.registry backfill

    # or via the installed console script:
    mousereach-eval-registry list reach_detection

ASCII-only: every string sent to print()/stdout uses ASCII so the Windows
console (cp1252/cp437) cannot crash on it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical DB path. DATA -- lives in the MouseReach_Improvement data area,
#: NOT in the git repo. Overridable per-call via db_path, or globally via the
#: MOUSEREACH_EVAL_DB environment variable.
DEFAULT_DB = Path(
    os.environ.get(
        "MOUSEREACH_EVAL_DB",
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement\algo_eval.db",
    )
)

#: The four algo elements this registry tracks.
ALGO_ELEMENTS = ("segmentation", "reach_detection", "outcome", "assignment")

#: Maps each algo element to the RUNTIME module(s) whose source defines what
#: actually runs. Each entry may be a package (its whole dir is hashed) or a
#: single module (only that .py is hashed). Resolution is by IMPORT, so the
#: fingerprint reflects the live interpreter's view of the code, not a path on
#: disk that might be a stale C: copy.
#:
#: NOTE: span_to_reaches is listed per the registry spec; if a named module is
#: not importable in a given install it is skipped (with an ASCII warning) so a
#: missing optional module never crashes a record. The fingerprint still covers
#: every module that DID resolve, which is the point: it pins the live runtime.
ALGO_MODULES: Dict[str, List[str]] = {
    "reach_detection": [
        "mousereach.reach.v8",
        "mousereach.reach.core.span_to_reaches",
    ],
    "segmentation": ["mousereach.segmentation"],
    "outcome": ["mousereach.outcomes"],
    "assignment": ["mousereach.assignment"],
}

#: Sentinel fingerprint used when backfilling historical snapshots whose
#: runtime source is not recoverable. We do NOT fabricate a hash.
BACKFILL_FINGERPRINT = "backfilled-unknown"

#: Phase dir name -> algo_element. Phases on disk include "features", which has
#: no single algo element; it is skipped during backfill.
PHASE_TO_ELEMENT: Dict[str, str] = {
    "segmentation": "segmentation",
    "reach_detection": "reach_detection",
    "outcome": "outcome",
    "assignment": "assignment",
}

#: Keys whose values can be huge per-event arrays in a snapshot scalars.json.
#: We strip them before storing `metrics` so a row stays small; the scalar
#: summary fields (n_tp, recall, etc.) are kept.
_BULKY_METRIC_KEYS = ("matches", "confusion_rows", "per_reach", "rows")

_DEFAULT_SNAPSHOTS_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
)


# ---------------------------------------------------------------------------
# DB schema / init
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    algo_element      TEXT NOT NULL,
    version           TEXT NOT NULL,
    git_commit        TEXT,
    git_dirty         INTEGER,
    code_fingerprint  TEXT,
    corpus_id         TEXT,
    corpus_video_hash TEXT,
    n_videos          INTEGER,
    grading_standard  TEXT,
    metrics           TEXT,
    snapshot_path     TEXT,
    created_at        TEXT NOT NULL,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_element
    ON eval_runs (algo_element);
CREATE INDEX IF NOT EXISTS idx_eval_runs_element_version
    ON eval_runs (algo_element, version);
CREATE INDEX IF NOT EXISTS idx_eval_runs_corpus
    ON eval_runs (corpus_id);
"""

#: Column order used everywhere we map a sqlite Row -> dict.
_COLUMNS = (
    "id", "algo_element", "version", "git_commit", "git_dirty",
    "code_fingerprint", "corpus_id", "corpus_video_hash", "n_videos",
    "grading_standard", "metrics", "snapshot_path", "created_at", "notes",
)


def _connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open a connection, creating the parent dir and table on first use.

    Robust to a missing DB: the parent directory is created and the schema is
    applied (idempotent) so the very first write works even on a clean machine.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def init_db(db_path: Path | str = DEFAULT_DB) -> None:
    """Create the eval_runs table and indexes if absent. Idempotent."""
    conn = _connect(db_path)
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite Row into a plain dict, decoding the JSON columns."""
    d = {k: row[k] for k in _COLUMNS}
    for json_col in ("grading_standard", "metrics"):
        raw = d.get(json_col)
        if raw:
            try:
                d[json_col] = json.loads(raw)
            except (ValueError, TypeError):
                # Leave the raw string in place if it is not valid JSON.
                pass
    return d


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def _infer_repo_root() -> Path:
    """Infer the tool-repo root, preferring the canonical Y: git working copy.

    Evals normally run from the C: runtime MIRROR, which is NOT a git repo, so
    this file's own location (four parents up) yields a non-git path and a null
    commit. We therefore try, in order, and return the first that is actually a
    git repo:
      1. $CONNECTOME_ROOT/Behavior/MouseReach  (Y:, set by the conda activate
         scripts);
      2. the project's fixed Y: tool-repo path;
      3. four parents up from this file  (correct when imported from Y: itself).

    This records the INTENDED HEAD of the canonical source even when the run
    happened on the C: mirror. code_fingerprint() independently pins what
    ACTUALLY ran, so a stale mirror surfaces as a commit/fingerprint mismatch
    (see verify_run / verify_against_commit) rather than a silent wrong number.
    """
    candidates: List[Path] = []
    croot = os.environ.get("CONNECTOME_ROOT")
    if croot:
        candidates.append(Path(croot) / "Behavior" / "MouseReach")
    candidates.append(Path(r"Y:\2_Connectome\Behavior\MouseReach"))
    candidates.append(Path(__file__).resolve().parents[4])
    for cand in candidates:
        if (cand / ".git").is_dir():
            return cand
    return candidates[-1]


def git_provenance(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Return {git_commit, git_dirty} for the tool repo.

    git_commit : short HEAD hash (str) or None if not a git repo / git missing.
    git_dirty  : 1 if the working tree has uncommitted changes, else 0; None if
                 commit could not be determined.

    Tolerates a non-git checkout, a missing git executable, and a detached repo
    gracefully -- in any failure case git_commit is None.
    """
    root = Path(repo_root) if repo_root else _infer_repo_root()

    def _git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(root)] + args,
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout

    head = _git(["rev-parse", "--short", "HEAD"])
    if head is None:
        return {"git_commit": None, "git_dirty": None}
    commit = head.strip() or None

    status = _git(["status", "--porcelain"])
    if status is None:
        dirty: Optional[int] = None
    else:
        dirty = 1 if status.strip() else 0

    return {"git_commit": commit, "git_dirty": dirty}


# ---------------------------------------------------------------------------
# Code fingerprint -- the anti-stale-runtime field
# ---------------------------------------------------------------------------

def _module_py_files(module) -> List[Path]:
    """Return the *.py files that define an imported module/package.

    - For a package (has __path__): every *.py under the package directory,
      recursively (so submodules count toward the fingerprint).
    - For a plain module: just its own source file.
    """
    pkg_paths = getattr(module, "__path__", None)
    if pkg_paths:
        files: List[Path] = []
        for p in pkg_paths:
            base = Path(p)
            if base.exists():
                files.extend(sorted(base.rglob("*.py")))
        return files

    src = getattr(module, "__file__", None)
    if src and Path(src).suffix == ".py":
        return [Path(src)]
    return []


def code_fingerprint(algo_element: str) -> str:
    """sha256 over the runtime source of an algo element's module files.

    THIS IS THE KEY ANTI-STALE-RUNTIME FIELD. We resolve each module in
    ALGO_MODULES[algo_element] by IMPORT, so the files hashed are exactly the
    ones the live interpreter would execute -- if a stale C: copy is on the
    import path, this hash reflects the stale copy, and verify_run() will later
    surface the mismatch.

    Hashing protocol:
      - Resolve each configured module by import. Modules that fail to import
        (e.g. an optional/placeholder module not present in this install) are
        skipped with an ASCII warning; this never crashes a record.
      - Collect every *.py file, de-duplicate, sort by absolute path for
        determinism, and feed each file's bytes into a single sha256 with a
        path-relative header so file boundaries are unambiguous.

    Raises
    ------
    ValueError
        If algo_element is not a known key in ALGO_MODULES.
    RuntimeError
        If NO configured module could be resolved (nothing to hash) -- this is
        itself a stale/broken-runtime signal and must not be silently hashed as
        an empty string.
    """
    if algo_element not in ALGO_MODULES:
        raise ValueError(
            "Unknown algo_element %r; expected one of %s"
            % (algo_element, ", ".join(sorted(ALGO_MODULES)))
        )

    files: List[Path] = []
    resolved_any = False
    for mod_name in ALGO_MODULES[algo_element]:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - any import failure is tolerated
            sys.stderr.write(
                "[!] code_fingerprint: could not import %s (%s); skipping\n"
                % (mod_name, type(exc).__name__)
            )
            continue
        resolved_any = True
        files.extend(_module_py_files(mod))

    if not resolved_any:
        raise RuntimeError(
            "code_fingerprint(%r): none of the configured runtime modules "
            "could be imported (%s). The runtime is missing or broken."
            % (algo_element, ", ".join(ALGO_MODULES[algo_element]))
        )

    # De-duplicate while preserving determinism: sort by resolved absolute path.
    unique: Dict[str, Path] = {}
    for f in files:
        try:
            key = str(f.resolve())
        except OSError:
            key = str(f)
        unique[key] = f

    h = hashlib.sha256()
    for key in sorted(unique):
        f = unique[key]
        try:
            data = f.read_bytes()
        except OSError as exc:
            sys.stderr.write(
                "[!] code_fingerprint: could not read %s (%s); skipping\n"
                % (key, type(exc).__name__)
            )
            continue
        # Use the file name (not the absolute path) as the boundary header so
        # the same code produces the same hash regardless of which drive it is
        # imported from. Drive location is captured by git/verify, not here.
        h.update(("\n--FILE %s--\n" % Path(key).name).encode("utf-8"))
        h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Corpus hashing
# ---------------------------------------------------------------------------

def corpus_video_hash(video_ids: List[str]) -> str:
    """sha256 of the sorted, comma-joined video ids. Order-independent."""
    joined = ",".join(sorted(str(v) for v in video_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core write / read API
# ---------------------------------------------------------------------------

def record_eval(
    algo_element: str,
    version: str,
    metrics: Dict[str, Any],
    corpus_id: str,
    video_ids: List[str],
    grading_standard: Dict[str, Any],
    snapshot_path: str = "",
    notes: str = "",
    created_at: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB,
) -> int:
    """Record one evaluation run; returns the new row id.

    Auto-fills, so the caller only supplies the numbers and context:
      - git_commit / git_dirty via git_provenance()
      - code_fingerprint via code_fingerprint(algo_element)  [anti-stale field]
      - corpus_video_hash + n_videos from video_ids
      - created_at via the passed value or datetime.now().isoformat()

    The `created_at` parameter is accepted explicitly because some runtimes
    forbid clock access; pass an ISO-8601 string from the calling context if so.
    """
    if algo_element not in ALGO_ELEMENTS:
        sys.stderr.write(
            "[!] record_eval: algo_element %r is not one of %s (recording anyway)\n"
            % (algo_element, ", ".join(ALGO_ELEMENTS))
        )

    prov = git_provenance()

    try:
        fingerprint = code_fingerprint(algo_element)
    except (ValueError, RuntimeError) as exc:
        # A failed fingerprint is itself important signal. Record it verbatim
        # rather than aborting the whole eval record.
        sys.stderr.write(
            "[!] record_eval: code_fingerprint failed: %s\n" % exc
        )
        fingerprint = "fingerprint-error: %s" % type(exc).__name__

    vhash = corpus_video_hash(video_ids)
    n_videos = len(video_ids)
    if created_at is None:
        created_at = datetime.now().isoformat()

    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO eval_runs (
                algo_element, version, git_commit, git_dirty,
                code_fingerprint, corpus_id, corpus_video_hash, n_videos,
                grading_standard, metrics, snapshot_path, created_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                algo_element, version, prov["git_commit"], prov["git_dirty"],
                fingerprint, corpus_id, vhash, n_videos,
                json.dumps(grading_standard), json.dumps(metrics),
                snapshot_path, created_at, notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def query_runs(
    algo_element: Optional[str] = None,
    version: Optional[str] = None,
    corpus_id: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB,
) -> List[Dict[str, Any]]:
    """Return rows (newest first) filtered by any combination of the args."""
    clauses: List[str] = []
    params: List[Any] = []
    if algo_element is not None:
        clauses.append("algo_element = ?")
        params.append(algo_element)
    if version is not None:
        clauses.append("version = ?")
        params.append(version)
    if corpus_id is not None:
        clauses.append("corpus_id = ?")
        params.append(corpus_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = "SELECT * FROM eval_runs %s ORDER BY id DESC" % where

    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_run(run_id: int, db_path: Path | str = DEFAULT_DB) -> Optional[Dict[str, Any]]:
    """Return a single row by id, or None if it does not exist."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def latest(
    algo_element: str,
    corpus_id: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB,
) -> Optional[Dict[str, Any]]:
    """Return the most recently recorded run for an algo element.

    Most-recent is defined by row id (monotonic with insertion order), which is
    robust even when several rows share a created_at timestamp.
    """
    clauses = ["algo_element = ?"]
    params: List[Any] = [algo_element]
    if corpus_id is not None:
        clauses.append("corpus_id = ?")
        params.append(corpus_id)
    sql = (
        "SELECT * FROM eval_runs WHERE %s ORDER BY id DESC LIMIT 1"
        % " AND ".join(clauses)
    )
    conn = _connect(db_path)
    try:
        row = conn.execute(sql, params).fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def verify_run(run_id: int, db_path: Path | str = DEFAULT_DB) -> Dict[str, Any]:
    """Recompute the CURRENT fingerprint for a run's algo and compare.

    Answers: "does the runtime still match what produced this number?" Recompute
    code_fingerprint(algo_element) against the live interpreter and compare it to
    the value stored when the run was recorded.

    Returns a dict:
        {
          "run_id": int,
          "algo_element": str,
          "stored_fingerprint": str|None,
          "current_fingerprint": str|None,
          "match": bool,
          "reason": str,   # ASCII explanation
        }
    """
    row = get_run(run_id, db_path=db_path)
    if row is None:
        return {
            "run_id": run_id, "algo_element": None,
            "stored_fingerprint": None, "current_fingerprint": None,
            "match": False, "reason": "no such run id",
        }

    algo = row["algo_element"]
    stored = row.get("code_fingerprint")

    if stored == BACKFILL_FINGERPRINT:
        return {
            "run_id": run_id, "algo_element": algo,
            "stored_fingerprint": stored, "current_fingerprint": None,
            "match": False,
            "reason": "backfilled run: original runtime source not recorded",
        }

    try:
        current = code_fingerprint(algo)
    except (ValueError, RuntimeError) as exc:
        return {
            "run_id": run_id, "algo_element": algo,
            "stored_fingerprint": stored, "current_fingerprint": None,
            "match": False,
            "reason": "could not compute current fingerprint: %s" % exc,
        }

    match = bool(stored) and stored == current
    reason = (
        "runtime matches recorded fingerprint"
        if match else
        "MISMATCH: current runtime source differs from what produced this run"
    )
    return {
        "run_id": run_id, "algo_element": algo,
        "stored_fingerprint": stored, "current_fingerprint": current,
        "match": match, "reason": reason,
    }


# ---------------------------------------------------------------------------
# Snapshot convenience
# ---------------------------------------------------------------------------

def _strip_bulky(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large per-event arrays so a stored metrics blob stays compact."""
    return {k: v for k, v in metrics.items() if k not in _BULKY_METRIC_KEYS}


def _read_snapshot_metrics(snapshot_dir: Path) -> Dict[str, Any]:
    """Load the scalar metrics from a snapshot's metrics/ directory.

    Prefers metrics/scalars.json, then metrics/reach_detection_scalars.json.
    Bulky arrays (matches, etc.) are stripped so only the canonical scalars are
    kept. Returns {} if no metrics file is found.
    """
    metrics_dir = snapshot_dir / "metrics"
    for name in ("scalars.json", "reach_detection_scalars.json"):
        candidate = metrics_dir / name
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                return _strip_bulky(data)
            return {"value": data}
    return {}


def _read_snapshot_video_ids(snapshot_dir: Path) -> List[str]:
    """Best-effort recovery of the corpus video ids for a snapshot.

    Tries, in order:
      1. manifest.json -> "video_ids"
      2. the gt_dir referenced in manifest.json -> *.json stems
      3. an algo_outputs/ or gt/ subdir of the snapshot -> *.json stems
    Returns [] if nothing is recoverable.
    """
    manifest_path = snapshot_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = {}

    vids = manifest.get("video_ids")
    if isinstance(vids, list) and vids:
        return [str(v) for v in vids]

    gt_dir = manifest.get("gt_dir")
    if isinstance(gt_dir, str) and gt_dir:
        gt_path = Path(gt_dir)
        if gt_path.exists():
            stems = sorted({p.stem for p in gt_path.glob("*.json")})
            if stems:
                return stems

    for sub in ("gt", "algo_outputs"):
        sub_path = snapshot_dir / sub
        if sub_path.exists():
            stems = sorted({p.stem for p in sub_path.glob("*.json")})
            if stems:
                return stems
    return []


def record_from_snapshot(
    snapshot_dir: Path | str,
    algo_element: str,
    version: str,
    corpus_id: str,
    grading_standard: Dict[str, Any],
    notes: str = "",
    db_path: Path | str = DEFAULT_DB,
) -> int:
    """Record an eval run by reading metrics and video ids out of a snapshot.

    Convenience wrapper around record_eval(): reads metrics from the snapshot's
    metrics/scalars.json (bulky arrays stripped) and the corpus video ids from
    manifest.json / gt_dir, then records with snapshot_path set to snapshot_dir.
    The fingerprint and git provenance are captured LIVE (not from the snapshot),
    so this records the runtime as of when you call it.
    """
    snapshot_dir = Path(snapshot_dir)
    metrics = _read_snapshot_metrics(snapshot_dir)
    video_ids = _read_snapshot_video_ids(snapshot_dir)
    return record_eval(
        algo_element=algo_element,
        version=version,
        metrics=metrics,
        corpus_id=corpus_id,
        video_ids=video_ids,
        grading_standard=grading_standard,
        snapshot_path=str(snapshot_dir),
        notes=notes,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def _row_exists(
    conn: sqlite3.Connection, algo_element: str, version: str, snapshot_path: str
) -> bool:
    """True if a row with the same (algo_element, version, snapshot_path) exists."""
    row = conn.execute(
        "SELECT 1 FROM eval_runs WHERE algo_element = ? AND version = ? "
        "AND snapshot_path = ? LIMIT 1",
        (algo_element, version, snapshot_path),
    ).fetchone()
    return row is not None


def _find_metrics_snapshots(phase_dir: Path) -> List[Path]:
    """Find snapshot dirs under a phase dir that have a metrics scalars file.

    A snapshot is any directory containing metrics/scalars.json or
    metrics/reach_detection_scalars.json. Nested layouts (some snapshots have
    model31/metrics/scalars.json) are handled by walking for the metrics file
    and taking its grandparent as the snapshot dir.
    """
    found: List[Path] = []
    seen: set = set()
    if not phase_dir.exists():
        return found
    for name in ("scalars.json", "reach_detection_scalars.json"):
        for metrics_file in phase_dir.rglob("metrics/%s" % name):
            snap = metrics_file.parent.parent  # .../<snap>/metrics/<file>
            key = str(snap.resolve()) if snap.exists() else str(snap)
            if key not in seen:
                seen.add(key)
                found.append(snap)
    return sorted(found, key=lambda p: str(p))


def _parse_manifest_best_effort(snapshot_dir: Path) -> Dict[str, Any]:
    """Pull version / corpus / grading_standard / git_commit from a manifest.

    Manifest shapes vary across the corpus (some have version_id, some only a
    _note and video_ids). All lookups are defensive; missing fields return
    sensible defaults.
    """
    manifest_path = snapshot_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = {}

    version = (
        manifest.get("version_id")
        or manifest.get("version")
        or snapshot_dir.name  # fall back to the directory name
    )
    corpus = (
        manifest.get("corpus_id")
        or manifest.get("corpus")
        or manifest.get("_note")
        or ""
    )
    grading = (
        manifest.get("grading_standard")
        or manifest.get("grading")
        or {}
    )
    if not isinstance(grading, dict):
        grading = {"raw": grading}
    git_commit = manifest.get("git_commit") or manifest.get("code_hash") or None

    return {
        "version": str(version),
        "corpus_id": str(corpus),
        "grading_standard": grading,
        "git_commit": git_commit,
    }


def backfill(
    snapshots_root: Path | str = _DEFAULT_SNAPSHOTS_ROOT,
    db_path: Path | str = DEFAULT_DB,
) -> Dict[str, int]:
    """Walk Improvement_Snapshots and insert a row per metrics-bearing snapshot.

    For each phase in {segmentation, reach_detection, outcome, assignment,
    features}, finds every snapshot with a metrics scalars file and inserts a
    row with:
      - algo_element from the phase dir (features is skipped: no single element)
      - version / corpus_id / grading_standard parsed best-effort from manifest
      - metrics from the scalars file (bulky arrays stripped)
      - snapshot_path set, notes="backfilled"
      - git_commit from manifest if present, else None
      - code_fingerprint set to the literal "backfilled-unknown" (NOT fabricated)

    Idempotent-ish: skips a snapshot if a row with the same
    (algo_element, version, snapshot_path) already exists.

    Returns a summary dict: {"inserted": N, "skipped": M, "scanned": K}.
    """
    snapshots_root = Path(snapshots_root)
    summary = {"inserted": 0, "skipped": 0, "scanned": 0}

    init_db(db_path)
    conn = _connect(db_path)
    try:
        # Iterate the known phase dirs (plus "features" so it is scanned and
        # explicitly skipped rather than silently ignored).
        for phase in list(PHASE_TO_ELEMENT) + ["features"]:
            phase_dir = snapshots_root / phase
            element = PHASE_TO_ELEMENT.get(phase)
            for snap in _find_metrics_snapshots(phase_dir):
                summary["scanned"] += 1
                if element is None:
                    # "features" has no algo element -> skip.
                    summary["skipped"] += 1
                    continue

                info = _parse_manifest_best_effort(snap)
                snap_path = str(snap)

                if _row_exists(conn, element, info["version"], snap_path):
                    summary["skipped"] += 1
                    continue

                metrics = _read_snapshot_metrics(snap)
                video_ids = _read_snapshot_video_ids(snap)
                conn.execute(
                    """
                    INSERT INTO eval_runs (
                        algo_element, version, git_commit, git_dirty,
                        code_fingerprint, corpus_id, corpus_video_hash,
                        n_videos, grading_standard, metrics, snapshot_path,
                        created_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        element,
                        info["version"],
                        info["git_commit"],
                        None,  # git_dirty unknown for historical runs
                        BACKFILL_FINGERPRINT,
                        info["corpus_id"],
                        corpus_video_hash(video_ids) if video_ids else "",
                        len(video_ids),
                        json.dumps(info["grading_standard"]),
                        json.dumps(metrics),
                        snap_path,
                        datetime.now().isoformat(),
                        "backfilled",
                    ),
                )
                summary["inserted"] += 1
        conn.commit()
    finally:
        conn.close()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_metric_summary(metrics: Any) -> str:
    """One-line ASCII summary of the most relevant scalar metrics."""
    if not isinstance(metrics, dict):
        return ""
    parts: List[str] = []
    # Prefer recall/precision, then tp/fp/fn (under either naming).
    def _get(*keys):
        for k in keys:
            if k in metrics and metrics[k] is not None:
                return metrics[k]
        return None

    recall = _get("recall", "Recall")
    precision = _get("precision", "Precision")
    tp = _get("tp", "n_tp")
    fp = _get("fp", "n_fp")
    fn = _get("fn", "n_fn")
    if recall is not None:
        parts.append("recall=%s" % _round(recall))
    if precision is not None:
        parts.append("prec=%s" % _round(precision))
    if tp is not None:
        parts.append("tp=%s" % tp)
    if fp is not None:
        parts.append("fp=%s" % fp)
    if fn is not None:
        parts.append("fn=%s" % fn)
    return " ".join(parts)


def _round(v: Any) -> str:
    try:
        return "%.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    """Print a compact ASCII table of runs to stdout."""
    if not rows:
        print("(no runs)")
        return
    header = (
        "%-4s %-15s %-9s %-9s %-5s %-34s %-28s %s"
        % ("id", "algo_element", "version", "commit", "dirty",
           "corpus_id", "metrics", "created_at")
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        commit = r.get("git_commit") or "-"
        dirty = r.get("git_dirty")
        dirty_s = "?" if dirty is None else ("yes" if dirty else "no")
        corpus = (r.get("corpus_id") or "")[:34]
        metrics_s = _fmt_metric_summary(r.get("metrics"))[:28]
        created = (r.get("created_at") or "")[:19]
        print(
            "%-4s %-15s %-9s %-9s %-5s %-34s %-28s %s"
            % (
                r.get("id"), (r.get("algo_element") or "")[:15],
                (r.get("version") or "")[:9], str(commit)[:9],
                dirty_s, corpus, metrics_s, created,
            )
        )


def _print_run_full(run: Dict[str, Any]) -> None:
    """Print a full row, with the metrics/grading JSON pretty-printed."""
    print("Run #%s" % run.get("id"))
    print("  algo_element     : %s" % run.get("algo_element"))
    print("  version          : %s" % run.get("version"))
    print("  git_commit       : %s" % run.get("git_commit"))
    print("  git_dirty        : %s" % run.get("git_dirty"))
    print("  code_fingerprint : %s" % run.get("code_fingerprint"))
    print("  corpus_id        : %s" % run.get("corpus_id"))
    print("  corpus_video_hash: %s" % run.get("corpus_video_hash"))
    print("  n_videos         : %s" % run.get("n_videos"))
    print("  snapshot_path    : %s" % run.get("snapshot_path"))
    print("  created_at       : %s" % run.get("created_at"))
    print("  notes            : %s" % run.get("notes"))
    print("  grading_standard :")
    print(_indent_json(run.get("grading_standard")))
    print("  metrics          :")
    print(_indent_json(run.get("metrics")))


def _indent_json(obj: Any, indent: str = "    ") -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        text = str(obj)
    return "\n".join(indent + line for line in text.splitlines())


def _cmd_list(args: argparse.Namespace) -> int:
    rows = query_runs(algo_element=args.algo_element, db_path=args.db)
    _print_table(rows)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    run = get_run(args.id, db_path=args.db)
    if run is None:
        print("No run with id %s" % args.id)
        return 1
    _print_run_full(run)
    return 0


def _cmd_latest(args: argparse.Namespace) -> int:
    run = latest(args.algo_element, db_path=args.db)
    if run is None:
        print("No runs recorded for algo_element %r" % args.algo_element)
        return 1
    _print_run_full(run)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_run(args.id, db_path=args.db)
    status = "MATCH" if result["match"] else "MISMATCH"
    print("verify run #%s [%s]" % (result["run_id"], result["algo_element"]))
    print("  result : %s" % status)
    print("  reason : %s" % result["reason"])
    print("  stored : %s" % result["stored_fingerprint"])
    print("  current: %s" % result["current_fingerprint"])
    return 0 if result["match"] else 2


def _cmd_backfill(args: argparse.Namespace) -> int:
    summary = backfill(snapshots_root=args.snapshots_root, db_path=args.db)
    print(
        "backfill complete: inserted=%d skipped=%d scanned=%d"
        % (summary["inserted"], summary["skipped"], summary["scanned"])
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mousereach-eval-registry",
        description=(
            "Algo Eval Registry: a git-recoverable SQLite record of how each "
            "algo-element version performed, with a runtime code fingerprint "
            "that makes stale-runtime evals impossible to miss."
        ),
    )
    p.add_argument(
        "--db", default=str(DEFAULT_DB),
        help="Path to the registry SQLite DB (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List recorded eval runs as a table")
    p_list.add_argument(
        "algo_element", nargs="?", default=None,
        help="Optional filter: segmentation|reach_detection|outcome|assignment",
    )
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show one run with full metrics JSON")
    p_show.add_argument("id", type=int, help="Run id")
    p_show.set_defaults(func=_cmd_show)

    p_latest = sub.add_parser("latest", help="Show the most recent run for an element")
    p_latest.add_argument("algo_element", help="segmentation|reach_detection|outcome|assignment")
    p_latest.set_defaults(func=_cmd_latest)

    p_verify = sub.add_parser(
        "verify", help="Check a run's fingerprint vs the current runtime"
    )
    p_verify.add_argument("id", type=int, help="Run id")
    p_verify.set_defaults(func=_cmd_verify)

    p_backfill = sub.add_parser(
        "backfill", help="Import historical Improvement_Snapshots into the registry"
    )
    p_backfill.add_argument(
        "--snapshots-root", default=str(_DEFAULT_SNAPSHOTS_ROOT),
        help="Improvement_Snapshots root (default: %(default)s)",
    )
    p_backfill.set_defaults(func=_cmd_backfill)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Console entry point for `mousereach-eval-registry` / `python -m`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
