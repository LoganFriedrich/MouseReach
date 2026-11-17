"""A route whose last write fails still leaves the row and the folder agreeing.

The route published the bundle, then wrote the review manifest (a plain
write), then recorded the hold in the database. A manifest write that failed
on a passing hold raised with the files already in the queue: the row was
never set (the dispatcher then marked the video failed, and its retry needed an
mp4 that had moved), and with no manifest no review tool listed the bundle.

Now the hold is recorded right after the move, a failed manifest write is
logged, and the return scan writes a missing manifest again from the routing
record. Built under tmp_path with invented ids.
"""
import json

import pytest

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    review = tmp_path / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    proc = tmp_path / "Processing"
    proc.mkdir()
    for n in (f"{VID}.mp4", f"{VID}DLC_model_x.h5", f"{VID}_segments.json",
              f"{VID}_pellet_outcomes.json"):
        (proc / n).write_text("{}", encoding="utf-8")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        db.update_state(VID, st)
    return proc, triage, db


def test_failed_manifest_write_still_records_the_hold(env, monkeypatch):
    proc, triage, db = env
    real = rg._write_review_manifest

    def refuse(*a, **k):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(rg, "_write_review_manifest", refuse)
    try:
        rg.route_to_queue(VID, proc, triage, "2 triaged segment(s) unresolved: 1, 2",
                          db=db, db_state="triage")
    except PermissionError:
        pass
    # The folder and the row agree: the bundle is in triage and so is the row.
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "triage"
    assert not (triage / VID / f"{VID}_manifest.json").exists()

    # The return scan writes the missing manifest again from the routing record.
    monkeypatch.setattr(rg, "_write_review_manifest", real)
    out = rr.repair_unfinished_routes(force=True)
    assert out["manifests_written"] == 1
    man = json.loads((triage / VID / f"{VID}_manifest.json").read_text(encoding="utf-8"))
    assert man["provenance"]["routed_reason"].startswith("2 triaged segment(s)")
    assert man["video_stem"] == VID


def test_a_folder_down_to_its_queue_records_is_not_given_a_manifest(env):
    _, triage, _ = env
    b = triage / VID
    b.mkdir()
    (b / f"{VID}_routing.json").write_text(json.dumps({"routed_reason": "x"}))
    assert rr.repair_unfinished_routes(force=True)["manifests_written"] == 0
    assert not (b / f"{VID}_manifest.json").exists()


def test_a_rebuilt_manifest_keeps_the_step_that_sent_the_video(env, monkeypatch):
    """The manifest carries the step that sent the video (the return resumes
    just after it). A manifest rebuilt by the return scan must carry the same
    step, or the return resumes at the queue's default step instead."""
    proc, triage, db = env
    real = rg._write_review_manifest

    def refuse(*a, **k):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(rg, "_write_review_manifest", refuse)
    rg.route_to_queue(VID, proc, triage, "assignment could not run", db=db,
                      db_state="triage", held_at_step="assignment")
    monkeypatch.setattr(rg, "_write_review_manifest", real)
    assert rr.repair_unfinished_routes(force=True)["manifests_written"] == 1
    man = json.loads((triage / VID / f"{VID}_manifest.json").read_text(encoding="utf-8"))
    assert man["provenance"]["held_at_step"] == "assignment"
