"""The watcher's filing step refuses a features file this run did not make,
and a failed video cannot be reset straight to 'processed'.

WHY: a re-run copies the filed generation down, the old features file
included. The filing step checked only that a features file existed, so a
re-run whose kinematics failed -- or a failed video a person reset to
'processed' -- was filed with new outputs beside an earlier run's kinematics.
"""
import os
import time

import pytest

from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import refuse_filing_without_features

V = "20990101_ZZZ0101_P4"


def _row(db, upto):
    db.register_video(video_id=V, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving", "failed"):
        db.update_state(V, st)
        if st == upto:
            return


def _folder(tmp_path, features_age):
    d = tmp_path / "Processing"
    d.mkdir()
    (d / f"{V}.mp4").write_bytes(b"v")
    for s in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        (d / f"{V}{s}").write_text("{}")
    (d / f"{V}_processing_manifest.json").write_text("{}")
    f = d / f"{V}_features.json"
    f.write_text("{}")
    t = time.time() - features_age
    os.utime(f, (t, t))
    return d


def test_an_earlier_runs_features_file_is_not_filed(tmp_path):
    db = WatcherDB(db_path=tmp_path / "w.db")
    _row(db, "processed")
    d = _folder(tmp_path, features_age=3600)
    assert refuse_filing_without_features(db, V, d) is True
    assert db.get_video(V)["state"] == "failed"
    assert (d / f"{V}_features.json").exists(), "nothing moved or deleted"


def test_this_runs_features_file_is_filed(tmp_path):
    db = WatcherDB(db_path=tmp_path / "w.db")
    _row(db, "processed")
    d = _folder(tmp_path, features_age=0)
    assert refuse_filing_without_features(db, V, d) is False
    assert db.get_video(V)["state"] == "processed"


def test_a_failed_video_cannot_be_put_straight_to_processed(tmp_path):
    db = WatcherDB(db_path=tmp_path / "w.db")
    _row(db, "failed")
    try:
        db.update_state(V, "processed")
    except Exception:
        pass
    assert db.get_video(V)["state"] == "failed"
    assert "processed" not in WatcherDB.RESET_STEP_CHOICES
