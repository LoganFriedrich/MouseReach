"""A folder in a queue that is not the video's live bundle never blocks its routes.

A route meets <queue>/<video> before it moves anything. Only the LIVE bundle of
a video that is held there stops it. Everything else -- an empty folder a
return left, a folder down to its queue records, an old copy of a video that
has since been analysed again, what is left of a triage copy after the video
moved into deep review -- is taken out of the queue (kept for a person when a
person wrote anything in it, and listed in the attention panel), and the route
goes on. Refusing all of them made every later run of such a video fail.
Built under tmp_path with invented ids; no real folder or database.
"""
import json

import pytest

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.config import Paths
from mousereach.watcher import review_routing as routing
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
        (folder / n).write_text(n, encoding="utf-8")
    return folder


def _attention(tmp_path, pattern):
    files = list((tmp_path / "nas" / "attention").glob(pattern))
    return [i for f in files
            for i in json.loads(f.read_text(encoding="utf-8"))["items"]]


def test_an_empty_folder_left_by_a_return_does_not_block(env):
    tmp_path, triage, _, db = env
    (triage / VID).mkdir()
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "triage"


def test_a_records_only_folder_is_set_aside_and_does_not_block(env):
    tmp_path, triage, _, db = env
    (triage / VID).mkdir()
    (triage / VID / f"{VID}_routing.json").write_text('{"old": true}', encoding="utf-8")
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    aside = list((tmp_path / "nas" / "Review" / "_Problematic").glob(
        f"_superseded_queue_folders_*/{VID}/{VID}_routing.json"))
    assert aside and aside[0].read_text(encoding="utf-8") == '{"old": true}'
    assert _attention(tmp_path, "kept_for_a_person.json") == []   # nothing for a person


def test_an_old_copy_with_a_persons_file_is_kept_for_a_person(env):
    """The video was analysed again (its row is no longer held), so the
    folder in the queue is an old copy: it leaves the queue whole, kept and
    listed, and the new run is routed."""
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}_unified_ground_truth.json").write_text('{"by": "person-a"}',
                                                          encoding="utf-8")
    (old / f"{VID}_reaches.json").write_text("old run", encoding="utf-8")
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert not (triage / VID / f"{VID}_unified_ground_truth.json").exists()
    kept = tmp_path / "nas" / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / f"{VID}_unified_ground_truth.json").read_text(
        encoding="utf-8") == '{"by": "person-a"}'
    note = json.loads((kept / "SET_ASIDE.json").read_text(encoding="utf-8"))
    assert note["person_files"] == [f"{VID}_unified_ground_truth.json"]
    items = _attention(tmp_path, "kept_for_a_person.json")
    assert [(i["video_id"], i["needs_person"]) for i in items] == [(VID, True)]


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
    items = _attention(tmp_path, "*_queue_folders_in_the_way.json")
    assert [i["video_id"] for i in items] == [VID]
    # A person deals with the folder: the item drops off by itself.
    import shutil
    shutil.rmtree(live)
    rr.repair_unfinished_routes(force=True)
    assert _attention(tmp_path, "*_queue_folders_in_the_way.json") == []


def test_a_triage_leftover_leaves_the_queue_and_never_blocks_the_video_again(env):
    """Into deep review while the triage copy holds a different review: the
    rest of the triage copy is kept for a person OUTSIDE the queue. When the
    video later leaves deep review and is gated to triage again, nothing is
    in the way -- the run is not refused, not marked failed, and the scan has
    no bundle without its video to divert."""
    tmp_path, triage, deep, db = env
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}_causal_review.json").write_text('{"a": 1}', encoding="utf-8")
    proc = _run(tmp_path / "Processing")
    (proc / f"{VID}_causal_review.json").write_text('{"b": 2}', encoding="utf-8")
    rg.route_to_queue(VID, proc, deep, "seg failed", db=db, db_state="deep_review")

    assert not copy.exists()
    kept = tmp_path / "nas" / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / f"{VID}_causal_review.json").read_text(encoding="utf-8") == '{"a": 1}'
    assert [i["video_id"] for i in _attention(tmp_path, "kept_for_a_person.json")] == [VID]

    # The video leaves deep review and is analysed again; the gate holds it.
    import shutil
    shutil.rmtree(deep / VID)
    db.force_state(VID, "processing", reason="test")
    bundle = rg.route_to_queue(VID, _run(tmp_path / "Processing2"), triage,
                               "1 triaged segment(s) unresolved: 2", db=db,
                               db_state="triage")
    assert (bundle / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "triage"
    assert not (deep / VID).exists()


def test_a_stale_bundle_holding_a_persons_file_leaves_the_queue_and_is_listed(env):
    tmp_path, triage, _, db = env
    b = triage / VID
    b.mkdir()
    (b / f"{VID}_deep_review_cleared.json").write_text('{"by": "person-a"}',
                                                       encoding="utf-8")
    ok = rr._retire_stale_bundle(b, VID, "archived", why="stale bundle")
    assert ok and not b.exists()
    kept = tmp_path / "nas" / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / f"{VID}_deep_review_cleared.json").is_file()
    assert [i["video_id"] for i in _attention(tmp_path, "kept_for_a_person.json")] == [VID]
    # Deleted by a person: the item goes away on the next refresh.
    import shutil
    shutil.rmtree(kept)
    rr.repair_unfinished_routes(force=True)
    assert _attention(tmp_path, "kept_for_a_person.json") == []


def test_an_old_copy_leaving_the_queue_keeps_its_review_in_the_durable_store(env):
    """The new bundle has no review: without the durable copy the person
    would be asked again what they already answered."""
    from mousereach.review.causal_review_io import durable_review_path
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}_causal_review.json").write_text('{"answered": 1}', encoding="utf-8")
    assert not durable_review_path(VID).exists()
    rg.route_to_queue(VID, _run(tmp_path / "Processing"), triage, "reason",
                      db=db, db_state="triage")
    assert durable_review_path(VID).read_text(encoding="utf-8") == '{"answered": 1}'


def test_a_stale_deep_review_folder_does_not_block_an_escalation_from_triage(env):
    """Held in triage, with an old deep-review folder: the deep-review folder
    is not the live bundle (the state belongs to the other queue), so it is
    set aside and the Escalate goes through."""
    tmp_path, triage, deep, db = env
    live = _run(triage / VID)
    db.force_state(VID, "triage", reason="test")
    stale = deep / VID
    stale.mkdir()
    (stale / f"{VID}_segments.json").write_text("old cuts", encoding="utf-8")
    bundle = rg.escalate_to_deep_review(VID, live, "escalated", db=db)
    assert (bundle / f"{VID}.mp4").is_file()
    assert (bundle / f"{VID}_segments.json").read_text(encoding="utf-8") == \
        f"{VID}_segments.json"
    assert db.get_video(VID)["state"] == "deep_review"
    assert not live.exists()


def test_an_old_copy_holding_the_only_video_is_kept_for_a_person(env):
    """The new route brings no video; the old copy in the queue holds one. It
    is not filed away where nobody looks: it is kept for a person and listed."""
    tmp_path, triage, _, db = env
    old = triage / VID
    old.mkdir()
    (old / f"{VID}.mp4").write_text("the only video", encoding="utf-8")
    (old / f"{VID}_reaches.json").write_text("old run", encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text("new", encoding="utf-8")
    rg.route_to_queue(VID, src, triage, "reason", db=db, db_state="triage")
    kept = tmp_path / "nas" / "Review" / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / f"{VID}.mp4").read_text(encoding="utf-8") == "the only video"
    assert [i["video_id"] for i in _attention(tmp_path, "kept_for_a_person.json")] == [VID]
