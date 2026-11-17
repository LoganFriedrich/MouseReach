"""A folder in a queue that is not the video's live bundle is a leftover.

A route meets <queue>/<video> before it moves anything. The LIVE bundle of a
video that is held there stops it. Anything else -- an empty folder a return
left, a folder down to its queue records, an old copy of a video that has
since been analysed again, what is left of a triage copy after the video
moved into deep review -- is a leftover: what has a current copy elsewhere is
deleted and the route goes on (refusing all of them made every later run of
such a video fail). What a person wrote, or a video with no copy elsewhere,
stays where it is, listed in the attention panel; the route waits for it
instead of merging into it. Nothing is moved to a set-aside folder. Built
under tmp_path with invented ids; no real folder or database.
"""
import json

import pytest

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.config import Paths
from mousereach.watcher import review_routing as routing

# Named here, not imported, so these tests also run (and fail by what they
# assert) on code without the leftover rule.
LEFT_NOTE = "LEFT_FOR_A_PERSON.json"      # review_routing.LEFT_NOTE
from mousereach.watcher.db import WatcherDB

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
    for n in (f"{VID}.mp4", f"{VID}DLC_model_x.h5", f"{VID}_segments.json",
              f"{VID}_pellet_outcomes.json"):
        # Machine outputs are written as the pipeline writes them (JSON):
        # one that cannot be read counts as a person's file.
        (folder / n).write_text(json.dumps({"file": n}) if n.endswith(".json") else n,
                                encoding="utf-8")
    return folder


def _written_before_this_run(folder):
    """The old copy was last written a day before the run being routed
    started (review_gate.queue_folder_is_live: only a run that started after
    the folder was last touched supersedes a folder a person wrote in)."""
    import os
    import time
    then = time.time() - 86400
    for f in folder.iterdir():
        os.utime(f, (then, then))


def _attention(tmp_path, kind):
    files = list((tmp_path / "nas" / "attention").glob("*_queue_folders_for_a_person.json"))
    return [i for f in files
            for i in json.loads(f.read_text(encoding="utf-8"))["items"]
            if i.get("kind") == kind]


def _nothing_set_aside(tmp_path):
    assert not (tmp_path / "nas" / "Review" / "_Problematic").exists()


def test_an_empty_folder_left_by_a_return_does_not_block(env):
    tmp_path, triage, _, db = env
    (triage / VID).mkdir()
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "triage"


def test_a_records_only_folder_is_deleted_and_does_not_block(env):
    tmp_path, triage, _, db = env
    (triage / VID).mkdir()
    (triage / VID / f"{VID}_routing.json").write_text('{"old": true}', encoding="utf-8")
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    record = json.loads((triage / VID / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert "old" not in record                    # the new route's own record
    assert _attention(tmp_path, "left_for_a_person") == []   # nothing for a person
    _nothing_set_aside(tmp_path)


def test_an_old_copy_with_a_persons_file_stays_and_the_route_waits(env):
    """The video was analysed again (its row is no longer held), so the
    folder in the queue is an old copy. Its machine output is deleted; the
    ground truth a person wrote has no copy elsewhere, so it stays where it
    is, listed, and the new run is NOT merged into that folder: the route is
    refused and the video stays in Processing. Once a person has dealt with
    the folder, the route goes through."""
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}_unified_ground_truth.json").write_text('{"by": "person-a"}',
                                                          encoding="utf-8")
    (old / f"{VID}_reaches.json").write_text('{"run": "old"}', encoding="utf-8")
    _written_before_this_run(old)
    proc = _run(tmp_path / "Processing")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    assert {p.name for p in old.iterdir()} == {
        LEFT_NOTE, f"{VID}_unified_ground_truth.json"}
    assert (proc / f"{VID}.mp4").is_file()                  # nothing moved
    assert db.get_video(VID)["state"] == "processing"
    note = json.loads((old / LEFT_NOTE).read_text(encoding="utf-8"))
    assert note["kept"] == [f"{VID}_unified_ground_truth.json"]
    items = _attention(tmp_path, "left_for_a_person")
    assert [(i["video_id"], i["needs_person"]) for i in items] == [(VID, True)]
    _nothing_set_aside(tmp_path)
    # A person deals with the folder; the route then goes through and the
    # list is current again.
    import shutil
    shutil.rmtree(old)
    rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    rr.repair_unfinished_routes(force=True)
    assert _attention(tmp_path, "left_for_a_person") == []


def test_the_live_bundle_of_a_held_video_refuses_and_is_listed(env):
    tmp_path, triage, _, db = env
    live = triage / VID
    _run(live)
    db.force_state(VID, "triage", reason="test")
    before = {p.name: p.read_bytes() for p in live.iterdir()}
    src = _run(tmp_path / "Processing")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, src, triage, "reason", db=db, db_state="triage")
    assert {p.name: p.read_bytes() for p in live.iterdir()} == before
    assert (src / f"{VID}.mp4").is_file()
    items = _attention(tmp_path, "queue_folder_in_the_way")
    assert [i["video_id"] for i in items] == [VID]
    # A person deals with the folder: the item drops off by itself.
    import shutil
    shutil.rmtree(live)
    rr.repair_unfinished_routes(force=True)
    assert _attention(tmp_path, "queue_folder_in_the_way") == []


def test_a_triage_leftover_with_ground_truth_stays_listed_and_is_never_merged(env):
    """Into deep review while the triage copy holds a person's ground truth:
    the ground truth stays in the triage folder, with a note, listed (the
    review's record is in the durable store). When the video later leaves
    deep review and is gated to triage again, the route is refused -- never
    merged into that folder -- and goes through once a person has dealt with
    it."""
    tmp_path, triage, deep, db = env
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}_causal_review.json").write_text('{"a": 1}', encoding="utf-8")
    (copy / f"{VID}_unified_ground_truth.json").write_text('{"gt": 1}', encoding="utf-8")
    proc = _run(tmp_path / "Processing")
    (proc / f"{VID}_causal_review.json").write_text('{"b": 2}', encoding="utf-8")
    rg.route_to_queue(VID, proc, deep, "seg failed", db=db, db_state="deep_review")

    assert {p.name for p in copy.iterdir()} == {
        LEFT_NOTE, f"{VID}_unified_ground_truth.json", f"{VID}_causal_review.json"}
    assert (copy / f"{VID}_unified_ground_truth.json").read_text(
        encoding="utf-8") == '{"gt": 1}'
    assert [i["video_id"] for i in _attention(tmp_path, "left_for_a_person")] == [VID]
    assert (deep / VID / f"{VID}_causal_review.json").read_text(
        encoding="utf-8") == '{"b": 2}'                     # the run's own, untouched
    _nothing_set_aside(tmp_path)

    # The video leaves deep review and is analysed again; the gate holds it.
    import shutil
    shutil.rmtree(deep / VID)
    db.force_state(VID, "processing", reason="test")
    proc2 = _run(tmp_path / "Processing2")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, proc2, triage, "1 triaged segment(s) unresolved: 2",
                          db=db, db_state="triage")
    assert (copy / f"{VID}_unified_ground_truth.json").is_file()
    assert (proc2 / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "processing"
    shutil.rmtree(copy)                                     # a person dealt with it
    bundle = rg.route_to_queue(VID, proc2, triage, "1 triaged segment(s) unresolved: 2",
                               db=db, db_state="triage")
    assert (bundle / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "triage"


def test_a_stale_bundle_holding_a_persons_file_stays_and_is_listed(env):
    tmp_path, triage, _, db = env
    b = triage / VID
    b.mkdir()
    (b / f"{VID}_deep_review_cleared.json").write_text('{"by": "person-a"}',
                                                       encoding="utf-8")
    ok = rr._retire_stale_bundle(b, VID, "archived", why="stale bundle")
    assert ok
    assert (b / f"{VID}_deep_review_cleared.json").is_file() and (b / LEFT_NOTE).is_file()
    assert [i["video_id"] for i in _attention(tmp_path, "left_for_a_person")] == [VID]
    _nothing_set_aside(tmp_path)
    # Deleted by a person: the item goes away on the next refresh.
    import shutil
    shutil.rmtree(b)
    rr.repair_unfinished_routes(force=True)
    assert _attention(tmp_path, "left_for_a_person") == []


def test_an_old_copys_review_is_never_written_into_the_durable_store(env):
    """The old copy's answers were made on its own cuts; the readers match the
    store's copy to the new run's cuts by frame span. So a review the store
    does not hold is never put there from a leftover: it stays where it is,
    listed, and the route waits. One the store already holds is deleted and
    the route goes on."""
    from mousereach.review.causal_review_io import durable_review_path
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}_causal_review.json").write_text('{"answered": 1}', encoding="utf-8")
    _written_before_this_run(old)
    proc = _run(tmp_path / "Processing")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    assert not durable_review_path(VID).exists()
    assert (old / f"{VID}_causal_review.json").is_file()
    durable_review_path(VID).parent.mkdir(parents=True, exist_ok=True)
    durable_review_path(VID).write_text('{"answered": 1}', encoding="utf-8")
    rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    assert not (triage / VID / f"{VID}_causal_review.json").exists()
    assert (triage / VID / f"{VID}.mp4").is_file()


def test_a_stale_deep_review_folder_does_not_block_an_escalation_from_triage(env):
    """Held in triage, with an old deep-review folder of machine outputs only
    (no queue record, nothing a person wrote): it is not the live bundle, so
    its machine outputs are deleted and the Escalate goes through."""
    tmp_path, triage, deep, db = env
    live = _run(triage / VID)
    db.force_state(VID, "triage", reason="test")
    stale = deep / VID
    stale.mkdir()
    (stale / f"{VID}_segments.json").write_text('{"boundary_source": "algo"}',
                                                encoding="utf-8")
    bundle = rg.escalate_to_deep_review(VID, live, "escalated", db=db)
    assert (bundle / f"{VID}.mp4").is_file()
    assert (bundle / f"{VID}_segments.json").read_text(encoding="utf-8") == \
        json.dumps({"file": f"{VID}_segments.json"})
    assert db.get_video(VID)["state"] == "deep_review"
    assert not live.exists()


def test_an_old_copy_holding_the_only_video_stays_and_is_listed(env):
    """The new route brings no video; the old copy in the queue holds one. It
    is never deleted or filed away where nobody looks: it stays where it is,
    listed, and the route waits."""
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}.mp4").write_text("the only video", encoding="utf-8")
    (old / f"{VID}_reaches.json").write_text('{"run": "old"}', encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text('{"run": "new"}', encoding="utf-8")
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, src, triage, "reason", db=db, db_state="triage")
    assert (old / f"{VID}.mp4").read_text(encoding="utf-8") == "the only video"
    assert not (old / f"{VID}_reaches.json").exists()      # a machine output
    assert [i["video_id"] for i in _attention(tmp_path, "left_for_a_person")] == [VID]
    _nothing_set_aside(tmp_path)
