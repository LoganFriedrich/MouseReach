"""At startup the processing server sends the shared record what it lacks.

WHY (2026-10-22): an export that failed right after archiving (a locked or
unreachable share) was a warning and never retried, so five archived videos
from the August/September outages were missing from watcher_central.db for
good.
"""
import sqlite3

import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB


def test_archived_videos_missing_from_the_shared_record_are_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    central = tmp_path / "watcher_central.db"
    con = sqlite3.connect(central)
    con.execute("CREATE TABLE videos (video_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO videos VALUES ('20250101_CNT0101_P1')")
    con.commit()
    con.close()
    db = WatcherDB(tmp_path / "local.db")
    for v in ("20250101_CNT0101_P1", "20250101_CNT0102_P1"):
        db.register_video(video_id=v, source_path="x")
        db.force_state(v, "archived", reason="test fixture")
    db.register_video(video_id="20250101_CNT0103_P1", source_path="x")   # not archived
    sent = []
    monkeypatch.setattr(db, "export_to_central_db", lambda v, path=None: sent.append(v))
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = db

    assert orch._export_missed_to_central() == 1
    assert sent == ["20250101_CNT0102_P1"]


def test_no_shared_record_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    assert orch._export_missed_to_central() == 0
