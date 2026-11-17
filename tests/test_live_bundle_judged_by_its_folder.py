"""Whether a queue folder is a video's LIVE bundle is read from the folder,
not from one machine's database row alone.

A bundle escalated from the triage window before the window recorded the move
kept its row at 'triage' while the video -- and the reviewer's saved review --
sat in the deep-review queue. Judged by the row alone, the next route of that
video into deep review (from the filed results, or from a processing folder)
took the reviewer's live bundle for an old copy and moved it out of the queue.
Now a published bundle, or a folder holding a person's file, is live unless the
route is a pipeline run that started after the folder was last routed or
written. Built under tmp_path with invented ids; no real folder or database.
"""
import json
import os
import time

import pytest

import mousereach.watcher.review_gate as rg
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB

VID = "20990101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    review = nas / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    analyzed = nas / "Analyzed"
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path / "proc_root", raising=False)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        db.update_state(VID, st)
    return tmp_path, triage, deep, analyzed, db


def _files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n, text in names.items():
        (folder / n).write_text(text, encoding="utf-8")
    return folder


def _escalated_bundle(deep, routed_at):
    """A deep-review bundle escalated from the triage window: a routing
    record, the run's files and the reviewer's saved review."""
    b = _files(deep / VID, {
        f"{VID}.mp4": "video",
        f"{VID}_segments.json": '{"boundary_source": "algo"}',
        f"{VID}_causal_review.json": '{"reviewer": "person-a", "answers": [1]}',
        f"{VID}_manifest.json": "{}",
    })
    (b / f"{VID}_routing.json").write_text(json.dumps(
        {"routed_at": routed_at, "routed_reason": "reviewer escalated"}),
        encoding="utf-8")
    return b


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def _route(*a, **k):
    """Route, and say whether it was refused (any error) -- the test judges
    by what is on disk afterwards."""
    try:
        rg.route_to_queue(*a, **k)
        return False
    except Exception:
        return True


def test_a_reviewers_bundle_is_live_although_the_row_says_triage(env):
    tmp_path, triage, deep, analyzed, db = env
    db.force_state(VID, "triage", reason="window moved it without the database")
    live = _escalated_bundle(deep, "2099-01-01T00:00:00")
    before = _snapshot(live)
    filed = _files(analyzed / "PROJ" / "PROJ01", {
        f"{VID}.mp4": "video",
        f"{VID}_segments.json": '{"boundary_source": "algo", "generation": "filed"}',
        f"{VID}_pellet_outcomes.json": "{}"})
    filed_before = _snapshot(filed)
    refused = _route(VID, filed, deep, "mislabel", db=db, db_state="deep_review")
    assert refused
    assert _snapshot(live) == before               # the reviewer's bundle is untouched
    assert _snapshot(filed) == filed_before        # the filed results too
    assert not (tmp_path / "nas" / "Review" / "_Problematic").exists()


def test_a_run_that_started_before_the_bundle_was_routed_does_not_replace_it(env):
    tmp_path, triage, deep, analyzed, db = env
    # The row's run started in the past; the bundle was routed after it.
    live = _escalated_bundle(deep, "2099-01-01T00:00:00")
    before = _snapshot(live)
    proc = _files(tmp_path / "Processing", {
        f"{VID}.mp4": "video", f"{VID}_segments.json": '{"boundary_source": "algo"}'})
    refused = _route(VID, proc, deep, "segmentation_failed", db=db,
                     db_state="deep_review")
    assert refused
    assert _snapshot(live) == before
    assert (proc / f"{VID}.mp4").is_file()


def test_a_run_that_started_after_the_folder_was_last_touched_supersedes_it(env):
    tmp_path, triage, deep, analyzed, db = env
    old = _escalated_bundle(deep, "2000-01-01T00:00:00")
    then = time.time() - 86400
    for f in old.iterdir():
        os.utime(f, (then, then))
    proc = _files(tmp_path / "Processing", {
        f"{VID}.mp4": "video", f"{VID}_segments.json": '{"boundary_source": "algo"}'})
    # The reviewer's review is also in the durable review store (the review
    # window saves it there as well), so the old copy holds nothing without a
    # current copy elsewhere.
    durable = tmp_path / "review_records" / "reviews" / f"{VID}_causal_review.json"
    durable.parent.mkdir(parents=True)
    durable.write_bytes((old / f"{VID}_causal_review.json").read_bytes())
    bundle = rg.route_to_queue(VID, proc, deep, "segmentation_failed", db=db,
                               db_state="deep_review")
    assert (bundle / f"{VID}.mp4").is_file()
    # The old copy was a leftover: the review is in the store, the video is
    # the same file the run brings, the rest machine output and records.
    assert json.loads(durable.read_text(encoding="utf-8"))["reviewer"] == "person-a"
    assert not (bundle / f"{VID}_causal_review.json").exists()
    assert not (tmp_path / "nas" / "Review" / "_Problematic").exists()


def test_machine_outputs_only_are_an_old_copy_whatever_their_age(env):
    tmp_path, triage, deep, analyzed, db = env
    _files(deep / VID, {f"{VID}_reaches.json": '{"run": "old"}',
                        f"{VID}_segments.json": '{"boundary_source": "algo"}'})
    proc = _files(tmp_path / "Processing", {f"{VID}.mp4": "video"})
    bundle = rg.route_to_queue(VID, proc, deep, "segmentation_failed", db=db,
                               db_state="deep_review")
    assert (bundle / f"{VID}.mp4").is_file()
    assert not (bundle / f"{VID}_reaches.json").exists()
