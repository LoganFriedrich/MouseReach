"""The main loop cannot spin: dispatch verdicts, terminal archive failures,
and the manifest cache.

WHY: overnight 2026-09-04 the processing loop spun -- a work item that
failed in milliseconds counted as progress, cycles became free, and the
cycle-counted scan gate fired the 37-minute version scan 15 times
back-to-back (~9.5 of 12 hours scanning, ~6 videos processed). These tests
pin the guards that make that impossible: a dispatch that did nothing says
so, an archive with nothing on disk is terminal rather than silently
retried forever, and a steady-state scan does not re-read unchanged
manifests over the NAS.
"""
import json

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.orchestrator import DLCOrchestrator, ProcessingOrchestrator
from mousereach.watcher.reprocessor import ReprocessingScanner


class RecDB:
    """Records every call; answers nothing."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name,) + a)
        return rec

    def called(self, name):
        return [c for c in self.calls if c[0] == name]


def _bare(cls):
    o = object.__new__(cls)
    o.db = RecDB()
    return o


# --- dispatch verdicts -----------------------------------------------------

def test_processing_dispatch_unknown_type_fails_row_and_declines():
    o = _bare(ProcessingOrchestrator)
    assert o._dispatch_work({"type": "nonsense", "id": "v1"}) is False
    assert o.db.called("mark_failed")


def test_dlc_dispatch_unknown_type_fails_row_and_declines():
    o = _bare(DLCOrchestrator)
    o._backup_local_db = lambda: None
    assert o._dispatch_work({"type": "nonsense", "id": "v1"}) is False
    assert o.db.called("mark_failed")


def test_dispatch_propagates_a_handler_that_declined():
    o = _bare(ProcessingOrchestrator)
    o._archive_to_nas = lambda work: False
    assert o._dispatch_work({"type": "archive", "id": "v1"}) is False


def test_dispatch_counts_legacy_none_as_progress():
    o = _bare(ProcessingOrchestrator)
    o._run_pipeline = lambda work: None
    assert o._dispatch_work({"type": "pipeline", "id": "v1"}) is not False


def test_dispatch_exception_marks_failed_and_declines():
    o = _bare(ProcessingOrchestrator)

    def boom(work):
        raise RuntimeError("handler died")

    o._run_pipeline = boom
    assert o._dispatch_work({"type": "pipeline", "id": "v1"}) is False
    assert o.db.called("mark_failed")


# --- archive failures ------------------------------------------------------

def _archive_result(error=None):
    return {"success": error is None, "error": error, "files_moved": [],
            "destination": None}


def _proc_for_archive(monkeypatch, error):
    o = _bare(ProcessingOrchestrator)
    o.staging_dir = None
    monkeypatch.setattr(orch.Paths, "TRIAGE_REVIEW", None)
    monkeypatch.setattr(orch.Paths, "DEEP_REVIEW", None)
    import mousereach.archive.core as ac
    monkeypatch.setattr(ac, "archive_video",
                        lambda *a, **k: _archive_result(error))
    return o


def test_no_files_on_disk_is_terminal(monkeypatch):
    """'No files found in Processing/' cannot be fixed by waiting; retrying
    it forever under backoff kept a gone-files row invisible for three days."""
    o = _proc_for_archive(monkeypatch, "No files found in Processing/")
    assert o._archive_to_nas({"id": "20240101_ABC0101_P1"}) is False
    assert o.db.called("mark_failed")


def test_other_archive_errors_keep_the_backoff_retry(monkeypatch):
    o = _proc_for_archive(monkeypatch, "destination busy")
    assert o._archive_to_nas({"id": "20240101_ABC0101_P1"}) is False
    assert not o.db.called("mark_failed")          # retriable, not terminal
    assert o._archive_backoff_active("20240101_ABC0101_P1")


# --- manifest cache --------------------------------------------------------

def test_manifest_cache_skips_unchanged_reads(tmp_path, monkeypatch):
    scanner = object.__new__(ReprocessingScanner)
    scanner._manifest_cache = {}
    scanner.archive_dir = tmp_path      # only used by the fallback path
    p = tmp_path / "20240101_ABC0101_P1_processing_manifest.json"
    p.write_text(json.dumps({"v": 1}), encoding="utf-8")
    index = {"20240101_ABC0101_P1": p}

    reads = {"n": 0}
    real_load = json.load

    def counting_load(fh):
        reads["n"] += 1
        return real_load(fh)

    import mousereach.watcher.reprocessor as rp
    monkeypatch.setattr(rp.json, "load", counting_load)

    m1 = scanner._load_manifest_indexed("20240101_ABC0101_P1", index)
    m2 = scanner._load_manifest_indexed("20240101_ABC0101_P1", index)
    assert m1 == m2 == {"v": 1}
    assert reads["n"] == 1              # second call served from the cache

    # A changed manifest (new mtime) is re-read.
    import os
    import time as _t
    p.write_text(json.dumps({"v": 2}), encoding="utf-8")
    os.utime(p, (_t.time() + 10, _t.time() + 10))
    m3 = scanner._load_manifest_indexed("20240101_ABC0101_P1", index)
    assert m3 == {"v": 2}
    assert reads["n"] == 2
