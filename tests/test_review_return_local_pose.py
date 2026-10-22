"""The review-return scan: a pose left in this node's processing folder is found,
a return that cannot go ahead is not retried every scan, and an unchanged triage
bundle is not re-read every scan.

WHY: an older divert between queues moved only the mp4 into deep review and left
the pose and every result in the processing folder of the node that ran the
video. Two human-cleared videos could then never return; each scan searched the
bundle, its manifest, Analyzed and staging, found nothing, logged an ERROR and
tried again two minutes later -- thousands of times from 2026-09-28 -- at about
5 s a try. Separately, each scan re-read every triage bundle's files (7-8 s of
every two-minute scan) although almost none had changed.
"""
import json

import mousereach.pipeline.versions as vmod
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.watcher.db import WatcherDB

STEM = "20240101_ABC0301_P3"
NEW = "DLC_newmodel"


def _wire(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path)
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", None)
    monkeypatch.setattr(vmod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": NEW}})
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(rr, "_status_cache", {})
    monkeypatch.setattr(rr, "_ensure_durable_review", lambda *a, **k: None)
    proc = tmp_path / "proc"
    proc.mkdir()
    return triage, deep, proc


def _split_bundle(deep, proc):
    """The September shape: the bundle holds the mp4 and the human's clear
    marker; pose and results stayed in the processing folder."""
    b = deep / STEM
    b.mkdir()
    (b / f"{STEM}.mp4").write_bytes(b"video")
    (b / f"{STEM}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "tester"}), encoding="utf-8")
    pose = proc / f"{STEM}{NEW}.h5"
    pose.write_bytes(b"pose")
    (proc / f"{STEM}_segments.json").write_text("{}", encoding="utf-8")
    return b, pose


# ---------------------------------------------------------------- the resolver

def test_a_current_model_pose_in_the_processing_folder_is_found(tmp_path, monkeypatch):
    _, deep, proc = _wire(monkeypatch, tmp_path)
    b, pose = _split_bundle(deep, proc)
    assert rr._resolve_inputs(b, STEM) == (b / f"{STEM}.mp4", None)
    assert rr._resolve_inputs(b, STEM, processing_dir=proc) == (b / f"{STEM}.mp4", pose)


def test_an_old_model_pose_in_the_processing_folder_is_refused(tmp_path, monkeypatch):
    _, deep, proc = _wire(monkeypatch, tmp_path)
    b = deep / STEM
    b.mkdir()
    (proc / f"{STEM}DLC_oldmodel.h5").write_bytes(b"old pose")
    assert rr._resolve_inputs(b, STEM, processing_dir=proc)[1] is None


def test_the_pose_the_videos_row_names_is_found(tmp_path, monkeypatch):
    _, deep, proc = _wire(monkeypatch, tmp_path)
    b = deep / STEM
    b.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pose = elsewhere / f"{STEM}{NEW}.h5"
    pose.write_bytes(b"pose")
    assert rr._resolve_inputs(b, STEM, recorded_pose=str(pose))[1] == pose


# ---------------------------------------------------------- the whole return

def test_a_split_bundle_now_returns(tmp_path, monkeypatch):
    _, deep, proc = _wire(monkeypatch, tmp_path)
    b, pose = _split_bundle(deep, proc)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=STEM, source_path=str(proc / f"{STEM}.mp4"))
    db.force_state(STEM, "deep_review", reason="test setup", dlc_output_path=str(pose))

    summary = rr.scan_review_queues(db, proc)

    assert summary["deep_returned"] == 1
    assert not b.exists()
    row = db.get_video(STEM)
    assert row["state"] == "processing"
    assert row["dlc_output_path"] == str(pose)
    assert (proc / f"{STEM}.mp4").read_bytes() == b"video"


def test_a_return_that_cannot_go_ahead_waits_an_hour(tmp_path, monkeypatch):
    _, deep, proc = _wire(monkeypatch, tmp_path)
    b = deep / STEM
    b.mkdir()
    (b / f"{STEM}.mp4").write_bytes(b"video")
    (b / f"{STEM}_deep_review_cleared.json").write_text("{}", encoding="utf-8")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=STEM, source_path=str(b / f"{STEM}.mp4"))
    db.force_state(STEM, "deep_review", reason="test setup")
    tries = []
    real = rr._resolve_inputs
    monkeypatch.setattr(rr, "_resolve_inputs",
                        lambda *a, **k: (tries.append(1), real(*a, **k))[1])

    rr.scan_review_queues(db, proc)
    rr.scan_review_queues(db, proc)
    assert len(tries) == 1, "the second scan does not try again within the hour"
    assert b.exists()

    rr._return_retry_after[STEM] = 0                 # the hour has passed
    rr.scan_review_queues(db, proc)
    assert len(tries) == 2


# --------------------------------------------------------- the triage cache

def _triage_bundle(triage):
    b = triage / STEM
    b.mkdir()
    (b / f"{STEM}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.9, "boundaries": [0, 100, 200]}),
        encoding="utf-8")
    (b / f"{STEM}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "outcome": "triaged",
                                  "flagged_for_review": True}]}),
        encoding="utf-8")
    return b


def test_an_unchanged_triage_bundle_is_read_once(tmp_path, monkeypatch):
    triage, _, proc = _wire(monkeypatch, tmp_path)
    b = _triage_bundle(triage)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)
    reads = []
    real = rr.triage_status
    monkeypatch.setattr(rr, "triage_status",
                        lambda bundle, stem: (reads.append(stem), real(bundle, stem))[1])

    class DB:
        def get_video(self, vid):
            return {"video_id": vid, "state": "triage"}

    rr.scan_review_queues(DB(), proc)
    rr.scan_review_queues(DB(), proc)
    assert reads == [STEM], "read on the first scan only"

    # A person saves an answer: the bundle changed, so it is read again.
    (b / f"{STEM}_causal_review.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "answers": {"reviewed": True}}]}),
        encoding="utf-8")
    rr.scan_review_queues(DB(), proc)
    assert reads == [STEM, STEM]
