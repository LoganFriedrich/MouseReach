"""A run the gate cannot send to triage (the video is in deep review) leaves
nothing behind in Processing.

A video is in one place at a time. When the gate asks for triage but the video
is in deep review, the route is refused and the video stays in deep review --
and this run's files used to stay in the local Processing folder as a second
place the video was in. Now this run's outputs are deleted, an input the
deep-review bundle holds identically is deleted, and an input that could be
the only copy is set aside (never deleted). Files outside the processing
folder are never touched. Built under tmp_path.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.review_gate import run_gate

VID = "20990101_ABC9901_P1"
POSE = f"{VID}DLC_resnet50_testshuffle1_100.h5"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "work"
    proc = root / "Processing"
    proc.mkdir(parents=True)
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    (deep / VID).mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", root, raising=False)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        db.update_state(VID, st)
    # A run whose gate decision is TRIAGE: one pellet the cascade flagged.
    (proc / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [0, 100]}))
    (proc / f"{VID}_reaches.json").write_text("{}")
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "triaged"}]}))
    (proc / f"{VID}.mp4").write_bytes(b"the video")
    (proc / POSE).write_bytes(b"a pose only this run has")
    (deep / VID / f"{VID}.mp4").write_bytes(b"the video")      # identical copy
    return root, proc, triage, deep, db


def test_the_run_leaves_processing_and_the_video_stays_in_deep_review(env):
    root, proc, triage, deep, db = env
    decision = run_gate(VID, proc, db, mp4_path=proc / f"{VID}.mp4")
    assert decision == "triage"
    assert not (triage / VID).exists()
    assert db.get_video(VID)["state"] == "deep_review"
    assert not any(p.name.startswith(VID) for p in proc.iterdir())   # one place only
    # The pose the deep-review bundle does not have is kept, set aside.
    kept = list((root / "_finished_elsewhere").rglob(POSE))
    assert len(kept) == 1 and kept[0].read_bytes() == b"a pose only this run has"
    # The identical video and the outputs are not kept twice.
    assert not list((root / "_finished_elsewhere").rglob(f"{VID}.mp4"))
    assert not list((root / "_finished_elsewhere").rglob(f"{VID}_*.json"))
    # The deep-review bundle is untouched.
    assert sorted(p.name for p in (deep / VID).iterdir()) == [f"{VID}.mp4"]


def test_a_route_from_anywhere_else_never_discards(env, tmp_path):
    """Only the gate's processing folder is cleared; route_to_queue itself,
    called on a filed folder, refuses and moves or deletes nothing."""
    from mousereach.watcher.review_gate import route_to_queue
    root, proc, triage, deep, db = env
    filed = tmp_path / "Analyzed" / "P" / "C"
    filed.mkdir(parents=True)
    (filed / f"{VID}.mp4").write_bytes(b"filed")
    route_to_queue(VID, filed, triage, "test", db=db, db_state="triage")
    assert (filed / f"{VID}.mp4").read_bytes() == b"filed"
