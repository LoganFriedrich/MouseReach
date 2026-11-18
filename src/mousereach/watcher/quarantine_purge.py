"""What ``mousereach-watch-quarantine --purge`` may delete, and the deleting.

THE RULE (why it is this and nothing else)

  * A MULTI-ANIMAL recording (a collage) is the irreplaceable source: it is
    the camera's own recording of a whole tray. Losing one loses that piece
    of the experiment for ever. The purge NEVER deletes a collage, and it
    treats every file it cannot prove to be a single as a collage.
  * A SINGLE-animal video is cut from a collage. It can always be cut again,
    so it is redundant -- but only while the collage it was cut from still
    exists. A single is deleted only when that source collage is found.
  * Byte-for-byte comparison is deliberately NOT the test. Two cuts of the
    same collage differ in their bytes (another cutting run, another
    encoder), so "is there an identical copy?" would keep singles that are
    perfectly replaceable and say nothing about the one thing that matters:
    whether the source recording survives.

WHAT COUNTS AS A SINGLE

A file whose name is a valid single-animal name (YYYYMMDD_ID_TRAYRUN.mp4, one
animal, validator.validate_single_filename) and that is less than half the
size of the collage found for it. The size check catches a whole-tray
recording that was given a one-animal name by mistake: a cut single is a
small fraction of its collage. Anything else -- a name with several animals,
a name that cannot be read, a non-video file -- is kept.

WHERE A SOURCE COLLAGE IS LOOKED FOR (only under the configured pipeline
root, nas_root)

  * the multi-animal intake, ``Unanalyzed/Multi-Animal``;
  * every folder named ``Multi-Animal`` inside ``Analyzed`` and inside
    ``Archive`` (a cut collage is moved to ``Analyzed/.../Multi-Animal``);
  * a collage the watcher database has a row for, at the path that row
    records, when the file is really there AND that path is under the
    pipeline root (a scratch copy on a machine's own disk is not the
    durable source).
A collage matches when its own name is a valid collage name with the same
date, the same tray letter and run, and the single's animal among its eight.
The quarantine folder itself never counts. A backup location
(backup.backup_root) is never read and never counts: a collage that exists
only there is treated as not found, and the single is kept.

A crystallized (publication-locked) video is never touched: the purge keeps
its file and says why (db.crystallized_refusal).

DRY RUN BY DEFAULT

``--purge`` on its own only shows what would be deleted (with the collage it
relies on) and what would be kept (with the reason). Deleting needs
``--purge --delete`` and a typed "y". WHY a default rather than a separate
dry-run step that must come first: a forgotten flag can then never delete
anything, and there is no saved "approved" list that could go stale. The real
run works the list out again from the folders as they are at that moment and
checks each file once more just before deleting it.

WHAT A DELETION DOES, in this order
  1. the shared database's rows saying the video is quarantined are taken
     away (coordination.PipelineCoordinator.retire_quarantined_video);
  2. this machine's row likewise (WatcherDB.retire_quarantined_video), with
     a processing_log line;
  3. a line is appended to the purge log (``watcher_logs/quarantine_purge.jsonl``
     under the pipeline root): time, machine, file, size, SHA-256 of the
     deleted file, and the collage relied on;
  4. the file and its ``.quarantine.json`` note are deleted.
If step 1 or 2 fails the file is kept. Rows go before the file so that a
deletion that fails part-way leaves a file with no row (the next purge
handles it) rather than a row for a file that is gone.
"""

import hashlib
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mousereach.watcher.validator import (
    VIDEO_EXTENSIONS, validate_collage_filename, validate_single_filename)

NOTE_SUFFIX = '.quarantine.json'
PURGE_LOG_NAME = 'quarantine_purge.jsonl'
# The top-level folders of the pipeline root searched for "Multi-Animal"
# folders, and how deep. Deep enough for Analyzed/<grant>/<project>/<cohort>/
# Multi-Animal and the archive's historical trees, shallow enough that the
# walk never lists every per-video folder of the filed tree.
SEARCHED_TREES = ('Analyzed', 'Archive')
MAX_DEPTH = 6


@dataclass
class Decision:
    path: Path
    action: str                 # 'delete' or 'keep'
    reason: str
    kind: str                   # 'single', 'collage', 'unknown'
    video_id: Optional[str] = None
    collage: Optional[Path] = None
    size: int = 0
    mtime_ns: int = 0
    note: Optional[dict] = field(default=None, repr=False)


def _read_note(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.with_name(path.name + NOTE_SUFFIX).read_text())
    except Exception:
        return None


def _same(a, b) -> bool:
    try:
        return os.path.normcase(os.path.abspath(str(a))) == \
            os.path.normcase(os.path.abspath(str(b)))
    except Exception:
        return False


def _under(path, root) -> bool:
    """Is ``path`` inside ``root``? (Case-insensitive on Windows.)"""
    try:
        p = os.path.normcase(os.path.abspath(str(path)))
        r = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([p, r]) == r
    except Exception:
        return False


def _multi_animal_folders(pipeline_root: Path) -> List[Path]:
    """Every folder a cut or waiting collage lives in (module docstring)."""
    found = []
    intake = pipeline_root / 'Unanalyzed' / 'Multi-Animal'
    if intake.is_dir():
        found.append(intake)
    for top in SEARCHED_TREES:
        base = pipeline_root / top
        if not base.is_dir():
            continue
        stack = [(base, 0)]
        while stack:
            folder, depth = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for e in entries:
                if not e.is_dir(follow_symlinks=False) or e.name.startswith('.'):
                    continue
                if e.name == 'Multi-Animal':
                    found.append(Path(e.path))
                elif depth + 1 < MAX_DEPTH:
                    stack.append((Path(e.path), depth + 1))
    return found


class CollageIndex:
    """Every collage file the pipeline knows about, by (date, tray letter,
    tray run): a list of (path, set of animal ids)."""

    def __init__(self, pipeline_root: Path, quarantine_dir: Path, db=None):
        self.by_key: Dict[tuple, List[tuple]] = {}
        self.quarantine_dir = quarantine_dir
        self.pipeline_root = Path(pipeline_root)
        for folder in _multi_animal_folders(Path(pipeline_root)):
            try:
                names = [e for e in os.scandir(folder) if e.is_file()]
            except OSError:
                continue
            for e in names:
                self._add(Path(e.path))
        if db is not None:
            for path in self._db_collage_paths(db):
                if _under(path, self.pipeline_root):
                    self._add(path)

    @staticmethod
    def _db_collage_paths(db) -> Iterable[Path]:
        try:
            with db._lock:
                conn = db._get_connection()
                try:
                    rows = conn.execute(
                        "SELECT source_path, archive_path FROM collages").fetchall()
                finally:
                    conn.close()
        except Exception:
            return []
        out = []
        for r in rows:
            for p in (r['source_path'], r['archive_path']):
                if p:
                    out.append(Path(p))
        return out

    def _add(self, path: Path) -> None:
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        if _same(path.parent, self.quarantine_dir):
            return
        result = validate_collage_filename(path.name)
        if not result.valid or not result.parsed:
            return
        p = result.parsed
        key = (p['date'], p['tray_type'], str(p['tray_run']))
        bucket = self.by_key.setdefault(key, [])
        if any(_same(path, q) for q, _ in bucket):
            return
        bucket.append((path, set(p['animal_ids'])))

    def source_of(self, parsed_single: dict) -> List[Path]:
        key = (parsed_single['date'], parsed_single['tray_type'],
               str(parsed_single['tray_run']))
        animal = parsed_single['animal_id']
        return [p for p, ids in self.by_key.get(key, []) if animal in ids]


def _existing_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def decide(path: Path, index: CollageIndex, db=None,
           analyzed_root=None) -> Decision:
    """The decision for one file in the quarantine folder (module docstring)."""
    from mousereach.config import get_video_id
    try:
        st = path.stat()
        size, mtime_ns = st.st_size, st.st_mtime_ns
    except OSError as e:
        return Decision(path, 'keep', f"cannot be read ({e})", 'unknown')
    note = _read_note(path)

    def keep(reason, kind='unknown', **kw):
        return Decision(path, 'keep', reason, kind, size=size, mtime_ns=mtime_ns,
                        note=note, **kw)

    name = path.name
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return keep("not a video file; left for a person")
    if ',' in name or validate_collage_filename(name).valid:
        return keep("multi-animal recording (collage) -- never deleted: it is the "
                    "irreplaceable source", 'collage')
    single = validate_single_filename(name)
    if not single.valid or not single.parsed:
        return keep(f"name cannot be read as a single ({single.error}); it may be a "
                    f"whole-tray recording, so it is kept like a collage", 'collage')

    video_id = get_video_id(name)
    from mousereach.watcher.db import crystallized_refusal
    refusal = crystallized_refusal(db, video_id, 'purged', analyzed_root=analyzed_root)
    if refusal:
        return keep(refusal, 'single', video_id=video_id)

    sources = index.source_of(single.parsed)
    live = [(p, _existing_size(p)) for p in sources]
    live = [(p, s) for p, s in live if s is not None]
    if not live:
        return keep("only copy -- kept: the multi-animal recording it was cut from "
                    "was not found anywhere in the pipeline", 'single', video_id=video_id)
    big = [(p, s) for p, s in live if s > 2 * size]
    if not big:
        return keep(f"as large as its collage ({live[0][0].name}); it may be a "
                    f"whole-tray recording, so it is kept like a collage",
                    'collage', video_id=video_id)
    return Decision(path, 'delete', "its multi-animal recording exists", 'single',
                    video_id=video_id, collage=big[0][0], size=size,
                    mtime_ns=mtime_ns, note=note)


def plan(quarantine_dir: Path, index: CollageIndex, db=None,
         analyzed_root=None) -> List[Decision]:
    """The decision for every file in the quarantine folder (notes excluded:
    each goes with its file)."""
    quarantine_dir = Path(quarantine_dir)
    if not quarantine_dir.is_dir():
        return []
    return [decide(p, index, db, analyzed_root)
            for p in sorted(quarantine_dir.iterdir())
            if p.is_file() and not p.name.endswith(NOTE_SUFFIX)]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def purge_log_path(pipeline_root: Path) -> Path:
    return Path(pipeline_root) / 'watcher_logs' / PURGE_LOG_NAME


def _append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def delete_one(d: Decision, index: CollageIndex, db, coordinator,
               log_path: Path, analyzed_root=None) -> tuple:
    """Delete one file planned for deletion, after deciding again from the
    folders as they are now. Returns (deleted, message)."""
    again = decide(d.path, index, db, analyzed_root)
    if again.action != 'delete':
        return False, f"kept: {again.reason}"
    if (again.size, again.mtime_ns) != (d.size, d.mtime_ns):
        return False, "kept: the file changed since the list was made"
    note_text = f"purged from quarantine; source collage {again.collage}"
    try:
        if coordinator is not None:
            coordinator.retire_quarantined_video(again.video_id)
        local = db.retire_quarantined_video(again.video_id, note_text) if db else False
    except Exception as e:
        return False, f"kept: its rows could not be retired ({e})"
    try:
        digest = sha256_of(again.path)
        _append_log(log_path, {
            'time': datetime.now().isoformat(timespec='seconds'),
            'machine': socket.gethostname(),
            'action': 'deleted',
            'file': str(again.path),
            'video_id': again.video_id,
            'size': again.size,
            'sha256': digest,
            'source_collage': str(again.collage),
            'local_row_retired': bool(local),
            'quarantine_note': again.note,
        })
    except Exception as e:
        return False, f"kept: the purge log could not be written ({e})"
    try:
        again.path.unlink()
    except OSError as e:
        try:
            _append_log(log_path, {'time': datetime.now().isoformat(timespec='seconds'),
                                   'action': 'delete_failed', 'file': str(again.path),
                                   'error': str(e)})
        except Exception:
            pass
        return False, f"kept: it could not be deleted ({e})"
    note = again.path.with_name(again.path.name + NOTE_SUFFIX)
    try:
        if note.exists():
            note.unlink()
    except OSError:
        pass
    return True, f"deleted (source collage {again.collage})"
