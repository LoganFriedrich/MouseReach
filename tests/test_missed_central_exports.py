"""The processing server re-sends the shared record what it lacks.

WHY (2026-10-22): an export that failed right after archiving (a locked or
unreachable share) was a warning and never retried, so five archived videos
from the August/September outages were missing from watcher_central.db for
good. WHY the check compares THIS machine's rows (2026-10-26): it first
looked for a video with no row from any machine, but a video posed on a GPU
PC already has that PC's row, so a failed export from the server was never
sent again; four were lost that morning while the share was slow.
"""
import sqlite3

import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB


def _central(tmp_path, rows):
    central = tmp_path / "watcher_central.db"
    con = sqlite3.connect(central)
    con.execute("CREATE TABLE videos (video_id TEXT, source_machine TEXT, updated_at TEXT)")
    con.executemany("INSERT INTO videos VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


def _orch(tmp_path, monkeypatch, archived):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    db = WatcherDB(tmp_path / "local.db")
    for v in archived:
        db.register_video(video_id=v, source_path="x")
        db.force_state(v, "archived", reason="test fixture")
    db.register_video(video_id="20250101_CNT0199_P1", source_path="x")   # not archived
    sent = []
    monkeypatch.setattr(db, "export_to_central_db", lambda v, path=None: sent.append(v))
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = db
    return orch, db, sent


def test_a_video_with_only_another_machines_row_is_sent(tmp_path, monkeypatch):
    orch, db, sent = _orch(tmp_path, monkeypatch, ["20250101_CNT0101_P1", "20250101_CNT0102_P1"])
    local = {v["video_id"]: v["updated_at"] for v in db.get_videos_in_state("archived")}
    _central(tmp_path, [
        ("20250101_CNT0101_P1", db._hostname, local["20250101_CNT0101_P1"]),   # current
        ("20250101_CNT0102_P1", "SOME-GPU-PC", local["20250101_CNT0102_P1"]),  # posed there
    ])
    assert orch._export_missed_to_central() == 1
    assert sent == ["20250101_CNT0102_P1"]


def test_this_machines_row_older_than_the_local_one_is_sent_again(tmp_path, monkeypatch):
    orch, db, sent = _orch(tmp_path, monkeypatch, ["20250101_CNT0101_P1"])
    _central(tmp_path, [("20250101_CNT0101_P1", db._hostname, "2000-01-01T00:00:00")])
    assert orch._export_missed_to_central() == 1
    assert sent == ["20250101_CNT0101_P1"]


def test_a_backlog_is_sent_a_batch_per_pass(tmp_path, monkeypatch):
    vids = ["20250101_CNT01%02d_P1" % i for i in range(1, 8)]
    orch, db, sent = _orch(tmp_path, monkeypatch, vids)
    _central(tmp_path, [])
    monkeypatch.setattr(o.ProcessingOrchestrator, "_SHARED_RECORD_BATCH", 3)
    assert orch._export_missed_to_central() == 3
    assert len(sent) == 3


def test_no_shared_record_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    assert orch._export_missed_to_central() == 0
