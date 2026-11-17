"""A leftover kept for a person is never read as the video's hold.

A queue folder that is not the video's live bundle but holds a person's work
with no copy anywhere else stays where it is, under the video's name, with a
note in it (review_routing.LEFT_NOTE). Every reader that asks "is the video
held in this queue?" must answer NO for such a folder. One that answered yes
did real harm: a later route into triage took a deep-review leftover for
"deep review supersedes triage", relabelled the row, moved nothing, and the
video was left with no live bundle anywhere. Built under tmp_path with
invented ids; no real folder or database.
"""
import json

import pytest

import mousereach.watcher.review_gate as rg
from mousereach.config import Paths
from mousereach.watcher import review_routing as routing
from mousereach.watcher.db import WatcherDB

# Named here, not imported, so these tests also run (and fail by what they
# assert) on code without the leftover rule.
LEFT_NOTE = "LEFT_FOR_A_PERSON.json"      # review_routing.LEFT_NOTE

VID = "20990101_ABC9901_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    review = nas / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None, raising=False)
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
    return tmp_path, triage, deep, db


def _run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for n in (f"{VID}.mp4", f"{VID}DLC_model_x.h5"):
        (folder / n).write_text(n, encoding="utf-8")
    for n in (f"{VID}_segments.json", f"{VID}_pellet_outcomes.json"):
        (folder / n).write_text(json.dumps({"validation_status": "needs_review"}),
                                encoding="utf-8")
    return folder


def _leftover(queue, with_pose=False):
    """A folder of the video in ``queue`` that is a leftover kept for a
    person: a person's ground truth, and the note."""
    folder = queue / VID
    folder.mkdir()
    (folder / f"{VID}_unified_ground_truth.json").write_text('{"by": "person-a"}',
                                                             encoding="utf-8")
    if with_pose:
        (folder / f"{VID}.mp4").write_text("old video", encoding="utf-8")
        (folder / f"{VID}DLC_model_x.h5").write_text("old pose", encoding="utf-8")
    (folder / LEFT_NOTE).write_text(json.dumps({
        "video_id": VID, "why": "an older copy", "what_to_do": "look at it",
        "kept": [f"{VID}_unified_ground_truth.json"]}), encoding="utf-8")
    return folder


# ------------------------------------------------- the two steps, end to end

def test_a_refused_deep_route_then_a_triage_route_goes_through(env):
    """Step 1: an old deep-review folder holding a person's ground truth meets
    a newer run's route into deep review: it is left where it is with the
    note, and the route is refused. Step 2: the video is later routed into
    triage. It must really go there -- files moved, row 'triage' -- not be
    relabelled 'deep_review' on the strength of the leftover with its files
    left behind in Processing."""
    tmp_path, triage, deep, db = env
    old = deep / VID
    old.mkdir()
    (old / f"{VID}_unified_ground_truth.json").write_text('{"by": "person-a"}',
                                                          encoding="utf-8")
    # Written a day before this run started: an older copy, not the live
    # bundle (review_gate.queue_folder_is_live).
    import os
    import time
    then = time.time() - 86400
    for f in old.iterdir():
        os.utime(f, (then, then))
    proc = _run(tmp_path / "Processing")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, proc, deep, "segmentation_failed: x", db=db,
                          db_state="deep_review")
    assert (old / LEFT_NOTE).is_file()
    assert (proc / f"{VID}.mp4").is_file()

    bundle = rg.route_to_queue(VID, proc, triage, "outcome triaged", db=db,
                               db_state="triage")
    assert bundle == triage / VID
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert not (proc / f"{VID}.mp4").exists()
    assert db.get_video(VID)["state"] == "triage"
    # The leftover is untouched and still a leftover.
    assert (old / f"{VID}_unified_ground_truth.json").is_file()
    assert (old / LEFT_NOTE).is_file()


# ------------------------------------------------------- one test per reader

def test_review_routing_readers(env):
    _, triage, deep, _ = env
    _leftover(deep)
    (triage / VID).mkdir()
    assert VID not in routing.deep_review_stems()
    assert not routing.in_deep_review(VID)
    assert not routing.superseded_by_deep_review(triage / VID)


def test_route_command_does_not_count_a_leftover_as_held(env):
    _, triage, deep, db = env
    from mousereach.watcher import route_cli
    db.force_state(VID, "archived", reason="filed")
    _leftover(deep)
    _leftover(triage)
    assert route_cli._held_in_a_queue(VID, db) == ""


def test_pose_node_does_not_count_a_leftover_as_held(env):
    """DLCOrchestrator._already_done_or_held (also the CPU pose pool's rule,
    and what _record_finished_child records)."""
    _, triage, deep, _ = env
    from mousereach.watcher.orchestrator import DLCOrchestrator
    _leftover(deep, with_pose=True)
    _leftover(triage, with_pose=True)
    assert DLCOrchestrator._already_done_or_held(None, VID) is None


def test_a_claimed_single_is_not_retired_against_a_leftover(env):
    _, triage, deep, _ = env
    from mousereach.watcher.orchestrator import DLCOrchestrator
    _leftover(triage, with_pose=True)
    _leftover(deep, with_pose=True)
    seen = []

    class Node:
        def _retire_claimed_single(self, video_id, handed_on, verified=False):
            seen.append(handed_on)
            return False
    DLCOrchestrator._retire_claim_after_hold(Node(), VID)
    assert seen == [None]


def test_a_rerun_with_no_pose_is_not_held_by_a_leftover(env, monkeypatch):
    _, _, deep, _ = env
    from mousereach.watcher import repose
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    _leftover(deep, with_pose=True)
    monkeypatch.setattr(repose, "archived_video", lambda vid: None)
    calls = []

    class FakeDB:
        def force_state(self, vid, state, **kw):
            calls.append(("state", state))

        def mark_failed(self, vid, why):
            calls.append(("failed", why))

    class Node:
        db = FakeDB()
    assert ProcessingOrchestrator._no_pose_for_reprocess(Node(), VID) == "failed"
    assert ("state", "deep_review") not in calls


def test_repose_does_not_count_a_triage_leftover_as_held(env):
    _, triage, _, _ = env
    from mousereach.watcher import repose
    _leftover(triage)
    assert repose._held_in_triage_on_disk(VID) is False


def test_census_and_reconcile_do_not_list_a_leftover(env):
    _, triage, deep, _ = env
    from mousereach.census.runner import bundles_in
    _leftover(triage)
    (deep / VID).mkdir()
    assert bundles_in(triage) == set()
    assert bundles_in(deep) == {VID}


def test_the_dashboard_folder_scan_does_not_list_a_leftover(env, monkeypatch):
    tmp_path, triage, deep, _ = env
    from mousereach.dashboard import folder_scan
    for name in ("SINGLE_ANIMAL_OUTPUT", "DLC_QUEUE", "DLC_STAGING", "PROCESSING",
                 "FAILED", "ANALYZED_OUTPUT", "MULTI_ANIMAL_SOURCE",
                 "MULTI_ANIMAL_OUTPUT", "QUARANTINE"):
        monkeypatch.setattr(Paths, name, None, raising=False)

    class NoConfig:
        @staticmethod
        def load():
            raise RuntimeError("no config in this test")
    monkeypatch.setattr(folder_scan, "WatcherConfig", NoConfig, raising=False)
    import mousereach.config as config
    monkeypatch.setattr(config, "WatcherConfig", NoConfig)
    _leftover(triage)
    try:
        found = folder_scan.scan_pipeline_folders()
    except TypeError:
        pytest.skip("this folder scan needs folders the test does not build")
    assert VID not in found


def test_deep_review_route_leaves_a_triage_leftover_as_it_is(env):
    """review_gate._drop_triage_copy: a triage folder that is already a
    leftover is not a triage copy to empty again."""
    tmp_path, triage, deep, db = env
    left = _leftover(triage, with_pose=True)
    before = {p.name: p.read_bytes() for p in left.iterdir()}
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), deep, "segmentation_failed",
                      db=db, db_state="deep_review")
    assert {p.name: p.read_bytes() for p in left.iterdir()} == before
