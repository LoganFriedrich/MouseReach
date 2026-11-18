"""The filing step files a run only with that run's own provenance manifest.

WHY: when Processing held no manifest, the filing step accepted the one in the
FILED folder -- meant for a retry after an older filing step had moved the
manifest and stopped before the features file. But the filed folder normally
holds the PREVIOUS generation's manifest, so a run whose own manifest was never
written (a row put to 'processed' without making its kinematics) was filed
beside the old provenance, and the version scan then read the old versions.
A filed manifest now counts only when it is not older than the features file
being filed. And a row coming back from 'unresolvable' can no longer go
straight to 'processed': it re-runs, so every check runs again.

Built under tmp_path with invented ids; no real folder or database.
"""
import os

import pytest

import mousereach.archive.core as core
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import refuse_filing_without_features

V = "20990101_ZZZ0101_P9"
DAY = 24 * 3600


@pytest.fixture
def setup(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    filed = tmp_path / "Analyzed" / "ZZZ01"
    filed.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(core, "get_archive_destination", lambda v: filed)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=V, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed"):
        db.update_state(V, st)
    (proc / f"{V}.mp4").write_bytes(b"v")
    (proc / f"{V}_features.json").write_text("{}")
    return db, proc, filed


def _age(path, seconds):
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


def test_the_previous_generations_filed_manifest_does_not_stand_in(setup):
    db, proc, filed = setup
    old = filed / f"{V}_processing_manifest.json"
    old.write_text('{"pipeline_versions": {"segmenter": "an older one"}}')
    _age(old, 30 * DAY)                      # written by an earlier run
    assert refuse_filing_without_features(db, V, proc) is True
    row = db.get_video(V)
    assert row["state"] == "failed"
    assert "manifest" in (row.get("error_message") or "")


def test_a_manifest_an_earlier_attempt_of_this_run_filed_still_counts(setup):
    db, proc, filed = setup
    feats = proc / f"{V}_features.json"
    m = filed / f"{V}_processing_manifest.json"
    m.write_text("{}")
    t = feats.stat().st_mtime + 5            # stamped after the features file
    os.utime(m, (t, t))
    assert refuse_filing_without_features(db, V, proc) is False
    assert db.get_video(V)["state"] == "processed"


def test_an_unresolvable_row_cannot_be_put_straight_to_processed(tmp_path):
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=V, source_path="x")
    db.mark_unresolvable(V, "no file for it on this node")
    assert db.get_video(V)["state"] == "unresolvable"
    with pytest.raises(ValueError):
        db.update_state(V, "processed")
    assert db.get_video(V)["state"] == "unresolvable"
    # It re-runs instead: the way back is through the work steps.
    db.update_state(V, "processing")
