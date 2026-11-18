"""What the dashboard's attention panel offers for review-queue holds.

A bundle that a route left without all its files gets buttons (move the
missing files in now; accept the bundle without a file that is gone). The
other holds -- a route refused because of a folder, a leftover folder kept
for a person -- are listed with what to do and have no buttons. No action
moves anything into a set-aside folder, and none deletes a file that may be
the only good copy. Built under tmp_path with invented ids; no real folder
or database.
"""
import json
import threading

import pytest

import mousereach.watcher.review_gate as rg
from mousereach.config import Paths
from mousereach.watcher import attention_actions as aa
from mousereach.watcher import review_routing as rt
from mousereach.watcher.attention import read_items
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
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path / "proc_root", raising=False)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    # Never this computer's own watcher database: an action given no db opens
    # it (db_location.open_watcher_db_if_present). A test passes its own db.
    from mousereach.watcher import db_location
    monkeypatch.setattr(db_location, "open_watcher_db_if_present",
                        lambda: (None, "no watcher database in this test"))
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        db.update_state(VID, st)
    return tmp_path, nas, triage, deep, db


def _files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n, text in names.items():
        (folder / n).write_text(text, encoding="utf-8")
    return folder


def _items(nas, kind):
    return [i for i in read_items(nas) if i.get("kind") == kind]


def _no_set_aside_folder(nas):
    """Nothing was set aside anywhere: the review root holds only the two
    queues (and the claims folder)."""
    review = nas / "Review"
    assert {p.name for p in review.iterdir()} <= {"deep_review", "triage", ".return_claims"}


def test_a_refused_route_is_listed_with_what_to_do_and_no_button(env):
    """The triage queue holds the live bundle (a person's review in it); a
    newer run's route from Processing is refused, listed, and nothing is
    offered that would move or delete either copy."""
    tmp_path, nas, triage, deep, db = env
    db.force_state(VID, "triage", reason="held")
    live = _files(triage / VID, {
        f"{VID}.mp4": "video", f"{VID}_causal_review.json": '{"by": "person-a"}',
        f"{VID}_manifest.json": "{}"})
    proc = _files(tmp_path / "Processing", {
        f"{VID}.mp4": "newer video", f"{VID}_reaches.json": "newer reaches"})
    with pytest.raises(rg.AlreadyInQueueError):
        rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    [item] = _items(nas, "queue_folder_in_the_way")
    assert "finish its review as usual" in item["says"]
    # A line saying what to do, and no button: nothing is offered that would
    # move or delete the live bundle.
    actions = aa.actions_for(item)
    assert actions and all(key.startswith("note:") for key, _, _ in actions)
    with pytest.raises(aa.ActionRefused):
        aa.perform(item, "use_new_run", db=db)
    with pytest.raises(aa.ActionRefused):
        aa.perform(item, "delete_leftover", db=db)
    assert (live / f"{VID}_causal_review.json").read_text(encoding="utf-8") == '{"by": "person-a"}'
    assert (proc / f"{VID}.mp4").read_text(encoding="utf-8") == "newer video"
    _no_set_aside_folder(nas)


def _unfinished_bundle(tmp_path, queue, entry_source, in_bundle):
    b = _files(queue / VID, {f"{VID}_segments.json": '{"boundary_source": "algo"}'})
    if in_bundle:
        (b / f"{VID}.mp4").write_text("bundle copy", encoding="utf-8")
    (b / f"{VID}_routing.json").write_text(json.dumps({
        "routed_at": "2099-01-01T00:00:00", "routed_by": "SOME-HOST",
        "failed_files": [{"name": f"{VID}.mp4", "source": str(entry_source),
                          "error": "held open"}]}), encoding="utf-8")
    return b


def test_an_unfinished_route_can_be_accepted_without_a_file_that_is_gone(env):
    tmp_path, nas, triage, deep, db = env
    b = _unfinished_bundle(tmp_path, deep, tmp_path / "gone" / f"{VID}.mp4",
                           in_bundle=False)
    item = {"kind": "route_unfinished", "video_id": VID, "folder": str(b)}
    assert [k for k, _, _ in aa.actions_for(item)] == ["finish_now",
                                                       f"accept_missing:{VID}.mp4"]
    aa.perform(item, f"accept_missing:{VID}.mp4")
    record = json.loads((b / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["failed_files"] == []
    [resolved] = record["resolved_by_a_person"]
    assert resolved["action"] == "accept_missing" and resolved["error"] == "held open"


def test_two_differing_copies_get_no_button_and_nothing_changes(env):
    """The file is in the bundle AND still at its old place, and the two
    differ: which is right is a person's call, so no button deletes or
    replaces either; the line says where both are."""
    tmp_path, nas, triage, deep, db = env
    other = _files(tmp_path / "nas" / "staging", {f"{VID}.mp4": "full copy"}) / f"{VID}.mp4"
    b = _unfinished_bundle(tmp_path, deep, other, in_bundle=True)
    item = {"kind": "route_unfinished", "video_id": VID, "folder": str(b)}
    actions = aa.actions_for(item)
    assert [k for k, _, _ in actions] == ["finish_now", f"note:{VID}.mp4"]
    assert str(other) in actions[1][2]
    for key in (f"keep_bundle:{VID}.mp4", f"use_source:{VID}.mp4",
                f"accept_missing:{VID}.mp4"):
        with pytest.raises(aa.ActionRefused):
            aa.perform(item, key)
    assert (b / f"{VID}.mp4").read_text(encoding="utf-8") == "bundle copy"
    assert other.read_text(encoding="utf-8") == "full copy"
    _no_set_aside_folder(nas)


def test_accepting_is_refused_while_another_program_holds_the_claim(env):
    from mousereach.watcher import review_return as rr
    tmp_path, nas, triage, deep, db = env
    b = _unfinished_bundle(tmp_path, deep, tmp_path / "gone" / f"{VID}.mp4",
                           in_bundle=False)
    item = {"kind": "route_unfinished", "video_id": VID, "folder": str(b)}
    held, done = threading.Event(), threading.Event()

    def holder():
        assert rr._claim_return(VID)
        held.set()
        done.wait(30)
        rr._release_return_claim(VID)

    t = threading.Thread(target=holder)
    t.start()
    try:
        held.wait(30)
        with pytest.raises(aa.ActionRefused):
            aa.perform(item, f"accept_missing:{VID}.mp4")
    finally:
        done.set()
        t.join()
    record = json.loads((b / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in record["failed_files"]] == [f"{VID}.mp4"]


def test_a_file_at_the_source_of_a_video_running_again_gets_no_button(env, monkeypatch):
    """The route's source is this machine's processing folder and the video
    runs there again (its row is 'processing', not held in the queue): no
    button may take the running video's file, and the dialog says why."""
    import socket
    from mousereach.watcher import db_location
    tmp_path, nas, triage, deep, db = env
    monkeypatch.setattr(db_location, "open_watcher_db_if_present", lambda: (db, None))
    # A login that can name its watcher database (db_location.watcher_db_unnamed_reason).
    monkeypatch.setattr(db_location, "watcher_db_unnamed_reason", lambda: None)
    running = _files(tmp_path / "Processing", {f"{VID}.mp4": "running video"}) / f"{VID}.mp4"
    b = _unfinished_bundle(tmp_path, triage, running, in_bundle=False)
    item = {"kind": "route_unfinished", "video_id": VID, "folder": str(b),
            "host": socket.gethostname()}
    actions = aa.actions_for(item)
    assert [k for k, _, _ in actions] == ["finish_now", f"note:{VID}.mp4"]
    assert "processing" in actions[1][2]
    aa.perform(item, "finish_now")
    with pytest.raises(aa.ActionRefused):
        aa.perform(item, f"accept_missing:{VID}.mp4")
    assert running.read_text(encoding="utf-8") == "running video"
    assert not (b / f"{VID}.mp4").exists()


def test_writers_of_the_list_never_lose_each_others_items(env):
    """The watcher, a review window and the dashboard on one machine all
    update the same list; each update is read, changed and written under the
    list's lock, so concurrent writers never drop another's item."""
    tmp_path, nas, triage, deep, db = env
    errors = []

    def writer(prefix):
        try:
            for n in range(15):
                rt.note_for_a_person("queue_folder_in_the_way", f"{prefix}{n}",
                                     tmp_path / f"{prefix}{n}", says="x")
        except Exception as e:          # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(_items(nas, "queue_folder_in_the_way")) == 45
