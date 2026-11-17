"""Only the watcher creates its database; every other tool refuses a missing one.

WHY: opening WatcherDB on a path that does not exist CREATES an empty database
there. A tool pointed at the wrong file -- the default watcher.db on a node
whose watcher writes a configured db_path, or a new machine whose watcher never
ran -- then worked silently on an empty decoy: queued videos the watcher never
saw, unlocked nothing, recorded re-pose requests nobody read. Now
mousereach-watch-process-animal (before its y/N question),
mousereach-uncrystallize, mousereach-repose-videos, the archive command's
filing record and the dashboard's database buttons all refuse, and the file is
still not there afterwards.

Everything lives in tmp_path; no machine configuration is read.
"""
import sys

import pytest

from mousereach.watcher import cli


@pytest.fixture
def missing(tmp_path, monkeypatch):
    path = tmp_path / "work" / "watcher.db"
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: path)
    from mousereach.watcher import db_location
    monkeypatch.setattr(db_location, "resolve_watcher_db_path", lambda: path)
    return path


def _run(monkeypatch, func, *argv):
    monkeypatch.setattr(sys, "argv", list(argv))
    try:
        func()
    except SystemExit as e:
        return e.code
    return 0


def test_process_animal_refuses_before_asking_and_queues_nothing(missing, tmp_path, monkeypatch, capsys):
    from mousereach.config import Paths
    single = tmp_path / "share" / "Single_Animal"
    single.mkdir(parents=True)
    (single / "20240101_ABC9902_P1.mp4").write_bytes(b"v")
    queue = tmp_path / "work" / "DLC_Queue"
    for name, d in {"SINGLE_ANIMAL_OUTPUT": single, "DLC_QUEUE": queue,
                    "MULTI_ANIMAL_SOURCE": tmp_path / "share" / "Multi",
                    "PROCESSING": tmp_path / "work" / "Processing",
                    "DLC_STAGING": tmp_path / "share" / "Posed"}.items():
        monkeypatch.setattr(Paths, name, d, raising=False)
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")

    code = _run(monkeypatch, cli.main_process_animal,
                "mousereach-watch-process-animal", "ABC9902")
    assert code == 1
    assert asked == [], "the person must not be asked to confirm a run that cannot be recorded"
    assert "Watcher database not found" in capsys.readouterr().err
    assert not missing.exists()
    assert not (queue / "20240101_ABC9902_P1.mp4").exists()


def test_uncrystallize_refuses_a_missing_database(missing, monkeypatch, capsys):
    code = _run(monkeypatch, cli.main_uncrystallize,
                "mousereach-uncrystallize", "--label", "LABEL_A")
    assert code == 1
    assert "Watcher database not found" in capsys.readouterr().err
    assert not missing.exists()


def test_repose_videos_refuses_a_missing_database(missing, tmp_path, monkeypatch, capsys):
    from mousereach.watcher import repose, repose_videos
    monkeypatch.setattr(repose, "declared_scorer", lambda: "MODEL_A")
    ids = tmp_path / "ids.txt"
    ids.write_text("20990101_TEST0101_P1\n", encoding="utf-8")
    code = repose_videos.main(["--ids", str(ids), "--label", "L", "--reason", "r", "--dry-run"])
    assert code == 2
    assert "nothing done" in capsys.readouterr().err
    assert not missing.exists()


def test_the_dashboard_never_creates_the_database(missing):
    from mousereach.dashboard.widget import PipelineDashboard
    with pytest.raises(RuntimeError, match="no watcher database"):
        PipelineDashboard._watcher_db(object())
    assert not missing.exists()


def test_reprocess_outdated_marks_videos_in_the_live_database(tmp_path, monkeypatch):
    """The dashboard's 'Reprocess outdated' opened <processing_root>/watcher.db.
    On a node whose watcher writes a configured db_path that file is a decoy,
    so videos were marked out of date where the watcher never looks."""
    import types
    from mousereach.config import Paths
    from mousereach.dashboard import widget
    from mousereach.watcher import db_location, reprocessor
    from mousereach.watcher.db import WatcherDB
    work = tmp_path / "work"
    WatcherDB(work / "watcher.db")                     # the decoy, left from before the override
    live = tmp_path / "local" / "watcher_local.db"
    WatcherDB(live)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", work, raising=False)
    monkeypatch.setattr(db_location, "resolve_watcher_db_path", lambda: live)
    monkeypatch.setattr(widget, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(widget, "show_error", lambda *a, **k: None)
    seen = []

    class Scanner:
        def __init__(self, db, nas):
            seen.append(db.db_path)

        def scan(self, mark_outdated):
            return {}

    class Worker:
        def __init__(self):
            self.done = types.SimpleNamespace(connect=lambda f: None, emit=lambda r: None)

    monkeypatch.setattr(reprocessor, "ReprocessingScanner", Scanner)
    monkeypatch.setattr(widget, "_BackfillWorker", Worker)
    dash = types.SimpleNamespace(_on_reprocess_done=lambda r: None)
    dash._watcher_db = lambda: widget.PipelineDashboard._watcher_db(dash)
    widget.PipelineDashboard._reprocess_outdated(dash)
    dash._reproc_thread.join(10)
    assert seen == [live]
