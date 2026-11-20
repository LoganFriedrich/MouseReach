"""A route that could not move every file is finished later, or not done at all.

A route goes on when a file cannot be moved (typically the mp4, still held
open) and lists it under failed_files. Nothing ever moved such a file in
afterwards: the reviewer's bundle lacked it and Processing kept a file of a
video whose row said it was in a queue. Now the return scan moves it in and
lists the bundle for a person until it is whole. A route out of the FILED
folder must never leave the filed generation half moved, so it is all or
nothing. Built under tmp_path with invented ids.
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
    review = tmp_path / "nas" / "Review"      # the queues live on the NAS
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(rr, "_unfinished_listed", [None], raising=False)
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        db.update_state(VID, st)
    return tmp_path, triage, db


def _source(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for n in (f"{VID}.mp4", f"{VID}DLC_model_x.h5", f"{VID}_segments.json",
              f"{VID}_pellet_outcomes.json"):
        # Machine outputs are written as the pipeline writes them (JSON):
        # one that cannot be read counts as a person's file.
        (folder / n).write_text(json.dumps({"file": n}) if n.endswith(".json") else n,
                                encoding="utf-8")
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def _lock(monkeypatch, name):
    """Make every move of ``name`` OUT of its current folder fail, as a file
    held open does. Returns a switch that releases the lock."""
    real = routing._safe_move
    held = {"on": True}

    def move(src, dst):
        if held["on"] and src.name == name:
            raise PermissionError(32, "being used by another process")
        return real(src, dst)

    monkeypatch.setattr(routing, "_safe_move", move)
    return held


def test_a_file_left_behind_is_moved_in_by_the_scan(env, monkeypatch):
    tmp_path, triage, db = env
    proc = tmp_path / "Processing"
    _source(proc)
    lock = _lock(monkeypatch, f"{VID}.mp4")
    rg.route_to_queue(VID, proc, triage, "1 triaged segment(s) unresolved: 2",
                      db=db, db_state="triage")
    bundle = triage / VID
    assert not (bundle / f"{VID}.mp4").exists()
    assert (proc / f"{VID}.mp4").is_file()

    # The scan sees an ordinary unanswered triage bundle.
    monkeypatch.setattr(rr, "_cached_triage_status", lambda b, s, sig: type("S", (), {
        "seg_failed": False, "seg_pending_reseg": False, "has_triage": True,
        "unresolved": [2], "too_short": [], "seg_corrected": []})())

    def scan():
        monkeypatch.setattr(rr, "_route_repair_due", [0.0], raising=False)
        return rr.scan_review_queues(db, tmp_path / "proc2")

    # Still held at the next scan: listed for a person, nothing lost.
    scan()
    listed = json.loads(next((tmp_path / "nas" / "attention").glob(
        "*_unfinished_routes.json")).read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in listed] == [VID]

    # Released: the scan moves it in and the bundle is whole.
    lock["on"] = False
    out = scan()
    assert (bundle / f"{VID}.mp4").read_text(encoding="utf-8") == f"{VID}.mp4"
    assert not any(p.name.startswith(VID) for p in proc.iterdir())
    assert out["route_files_finished"] == 1
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["failed_files"] == [] and [e["name"] for e in record["finished_later"]] == [f"{VID}.mp4"]
    # The whole entry is kept -- what went wrong and where the file was --
    # with how and when it was finished: it is the evidence of the route.
    [entry] = record["finished_later"]
    assert entry["source"] == str(proc / f"{VID}.mp4") and entry["error"]
    assert entry["finished"] == "moved in" and entry["finished_at"]
    listed = json.loads(next((tmp_path / "nas" / "attention").glob(
        "*_unfinished_routes.json")).read_text(encoding="utf-8"))["items"]
    assert listed == []


def test_a_route_out_of_the_filed_folder_is_all_or_nothing(env, monkeypatch):
    tmp_path, triage, db = env
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    before = _source(filed)
    _lock(monkeypatch, f"{VID}_segments.json")
    with pytest.raises(OSError):
        rg.route_to_queue(VID, filed, triage, "bench check", db=db,
                          db_state="triage", all_or_nothing=True)
    assert {p.name: p.read_bytes() for p in filed.iterdir()} == before
    assert not (triage / VID).exists()
    assert not (triage / routing.INCOMING_DIR_NAME / VID).exists()
    assert db.get_video(VID)["state"] == "processing"


def _filed_with_a_question(tmp_path, monkeypatch, db):
    from mousereach.watcher import route_cli
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    _source(filed)
    (filed / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"}]}), encoding="utf-8")
    (filed / f"{VID}_segments.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "start_frame": 100, "end_frame": 190}]}), encoding="utf-8")
    db.force_state(VID, "archived", reason="test")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    monkeypatch.setattr(route_cli, "_find_outcomes",
                        lambda analyzed, vid: filed / f"{VID}_pellet_outcomes.json")
    return filed, {p.name: p.read_bytes() for p in filed.iterdir()}


def test_the_route_command_routes_out_of_the_filed_folder_all_or_nothing(env, monkeypatch):
    """A route that fails puts every file back -- and writes no review flag
    into the filed results: nothing was routed, so nothing changes."""
    from mousereach.watcher import route_cli
    tmp_path, triage, db = env
    filed, before = _filed_with_a_question(tmp_path, monkeypatch, db)
    _lock(monkeypatch, f"{VID}_segments.json")
    res = route_cli.route_video(VID, "triage", "bench check", [1], db=db)
    assert not res["routed"] and res["error"]
    assert {p.name: p.read_bytes() for p in filed.iterdir()} == before
    assert not (triage / VID).exists()


def test_a_route_refused_because_the_video_is_held_leaves_the_filed_folder_alone(
        env, monkeypatch):
    """The held bundle appears between the check and the move: the question
    goes into the held bundle, and the filed results stay as they were."""
    from mousereach.watcher import route_cli
    tmp_path, triage, db = env
    filed, before = _filed_with_a_question(tmp_path, monkeypatch, db)
    held = triage / VID
    held.mkdir()
    (held / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"}]}), encoding="utf-8")
    (held / f"{VID}.mp4").write_text("held run", encoding="utf-8")
    monkeypatch.setattr(route_cli, "_held_in_a_queue", lambda vid, db: "")
    db.force_state(VID, "triage", reason="test")
    res = route_cli.route_video(VID, "triage", "bench check", [1], db=db)
    assert res["held"] and not res["routed"] and res["flagged"] == [1]
    assert {p.name: p.read_bytes() for p in filed.iterdir()} == before
    flagged = json.loads((held / f"{VID}_pellet_outcomes.json").read_text(
        encoding="utf-8"))["segments"][0]
    assert flagged["flagged_for_review"] is True


def test_a_routed_question_is_written_into_the_bundle(env, monkeypatch):
    from mousereach.watcher import route_cli
    tmp_path, triage, db = env
    filed, _ = _filed_with_a_question(tmp_path, monkeypatch, db)
    res = route_cli.route_video(VID, "triage", "bench check", [1], db=db)
    assert res["routed"] and res["flagged"] == [1] and not res["error"]
    seg = json.loads((triage / VID / f"{VID}_pellet_outcomes.json").read_text(
        encoding="utf-8"))["segments"][0]
    assert seg["flagged_for_review"] is True and seg["flag_reason"] == "bench check"
    # The filed copy stays, unflagged, until the new run replaces it (filed is
    # final, 2026-11-19).
    assert not json.loads(filed.joinpath(f"{VID}_pellet_outcomes.json").read_text(
        encoding="utf-8"))["segments"][0].get("flagged_for_review")


def test_a_file_a_person_moved_in_by_hand_finishes_the_route(env, tmp_path):
    """The attention panel asks a person to move a file that will not move by
    itself into the bundle. Once it is there (and gone from its source), the
    entry leaves the record -- keeping the record's modified time -- and the
    bundle is no longer listed."""
    _, triage, _ = env
    bundle = triage / VID
    bundle.mkdir()
    (bundle / f"{VID}.mp4").write_text("video", encoding="utf-8")
    (bundle / f"{VID}_manifest.json").write_text("{}", encoding="utf-8")
    record = bundle / f"{VID}_routing.json"
    record.write_text(json.dumps({"routed_at": "2099-01-01T00:00:00",
                                  "source_dir": str(tmp_path / "gone"),
                                  "failed_files": [{"name": f"{VID}.mp4",
                                                    "error": "in use"}]}),
                      encoding="utf-8")
    mtime = record.stat().st_mtime_ns
    out = rr.repair_unfinished_routes(force=True)
    assert out["route_files_finished"] == 1
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["failed_files"] == [] and [e["name"] for e in data["finished_later"]] == [f"{VID}.mp4"]
    assert record.stat().st_mtime_ns == mtime
    assert _listed(tmp_path) == []


def test_a_manual_fix_of_a_deep_review_route_empties_the_triage_copy(env, monkeypatch):
    """The mp4 of a route into deep review is held open, so the triage copy is
    kept and listed. A person moves the mp4 in by hand, as the panel says: the
    next pass crosses it off, empties the triage copy (a person's review it
    holds is in the durable store already, so its copy here is deleted --
    never moved into the deep-review bundle) and the listing goes away."""
    tmp_path, triage, db = env
    deep = Paths.DEEP_REVIEW
    copy = triage / VID
    copy.mkdir()
    (copy / f"{VID}_causal_review.json").write_text('{"a": 1}', encoding="utf-8")
    (copy / f"{VID}_pellet_outcomes.json").write_text('{"run": "old"}', encoding="utf-8")
    from mousereach.review.causal_review_io import durable_review_path
    durable_review_path(VID).parent.mkdir(parents=True, exist_ok=True)
    durable_review_path(VID).write_text('{"a": 1}', encoding="utf-8")
    proc = tmp_path / "Processing"
    proc.mkdir()
    for n in (f"{VID}.mp4", f"{VID}_segments.json"):
        (proc / n).write_text(n, encoding="utf-8")
    lock = _lock(monkeypatch, f"{VID}.mp4")
    rg.route_to_queue(VID, proc, deep, "segmentation_failed: x", db=db,
                      db_state="deep_review")
    lock["on"] = False
    assert copy.is_dir()
    in_the_way = list((tmp_path / "nas" / "attention").glob("*_queue_folders_for_a_person.json"))
    assert in_the_way and json.loads(in_the_way[0].read_text(encoding="utf-8"))["items"]

    (proc / f"{VID}.mp4").rename(deep / VID / f"{VID}.mp4")   # the person's fix
    rr.repair_unfinished_routes(force=True)

    record = json.loads((deep / VID / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["failed_files"] == []
    assert not copy.exists()
    assert not (deep / VID / f"{VID}_causal_review.json").exists()
    assert json.loads(in_the_way[0].read_text(encoding="utf-8"))["items"] == []
    assert _listed(tmp_path) == []


# ---------------------------------------------------------------------------
# A file held open for real (not a simulated refusal). On Windows a file open
# without delete sharing cannot be renamed, so a move copies it and then fails
# to delete the source -- the copy is left at the destination.
# ---------------------------------------------------------------------------

def _held_open(path):
    return open(path, "rb")


def test_a_route_out_of_the_filed_folder_leaves_no_copy_of_a_held_file(env):
    tmp_path, triage, db = env
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    before = _source(filed)
    with _held_open(filed / f"{VID}.mp4"):
        with pytest.raises(OSError):
            rg.route_to_queue(VID, filed, triage, "bench check", db=db,
                              db_state="triage", all_or_nothing=True)
    assert {p.name: p.read_bytes() for p in filed.iterdir()} == before
    assert not (triage / VID).exists()
    assert not (triage / routing.INCOMING_DIR_NAME / VID).exists()
    assert not list(triage.rglob(f"{VID}*"))      # no copy anywhere in the queue


def test_a_pipeline_route_leaves_a_held_file_in_one_place_only(env, monkeypatch):
    tmp_path, triage, db = env
    proc = tmp_path / "Processing"
    _source(proc)
    with _held_open(proc / f"{VID}.mp4"):
        rg.route_to_queue(VID, proc, triage, "1 triaged segment(s) unresolved: 2",
                          db=db, db_state="triage")
    bundle = triage / VID
    assert (proc / f"{VID}.mp4").is_file()
    assert not (bundle / f"{VID}.mp4").exists()   # not in two places
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in record["failed_files"]] == [f"{VID}.mp4"]
    out = rr.repair_unfinished_routes(force=True)
    assert out["route_files_finished"] == 1
    assert (bundle / f"{VID}.mp4").is_file()
    assert not (proc / f"{VID}.mp4").exists()


def _record_with_copy_in_both_places(triage, proc, routed_by, same=True):
    import socket
    bundle = triage / VID
    bundle.mkdir()
    proc.mkdir(parents=True)
    (bundle / f"{VID}.mp4").write_text("video", encoding="utf-8")
    (proc / f"{VID}.mp4").write_text("video" if same else "vid", encoding="utf-8")
    (bundle / f"{VID}_manifest.json").write_text("{}", encoding="utf-8")
    record = {"routed_at": "2099-01-01T00:00:00", "source_dir": str(proc),
              "failed_files": [{"name": f"{VID}.mp4", "error": "in use",
                                "source": str(proc / f"{VID}.mp4")}]}
    if routed_by:
        record["routed_by"] = socket.gethostname() if routed_by == "here" else routed_by
    (bundle / f"{VID}_routing.json").write_text(json.dumps(record), encoding="utf-8")
    return bundle


def _listed(tmp_path):
    files = list((tmp_path / "nas" / "attention").glob("*_unfinished_routes.json"))
    return json.loads(files[0].read_text(encoding="utf-8"))["items"] if files else []


def test_an_identical_copy_left_at_the_source_is_removed(env):
    tmp_path, triage, _ = env
    proc = tmp_path / "Processing"
    bundle = _record_with_copy_in_both_places(triage, proc, "here")
    out = rr.repair_unfinished_routes(force=True)
    assert out["route_files_finished"] == 1
    assert not (proc / f"{VID}.mp4").exists()
    assert (bundle / f"{VID}.mp4").read_text(encoding="utf-8") == "video"
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["failed_files"] == [] and [e["name"] for e in record["finished_later"]] == [f"{VID}.mp4"]
    assert _listed(tmp_path) == []


def test_a_different_copy_left_at_the_source_is_listed_not_deleted(env):
    tmp_path, triage, _ = env
    proc = tmp_path / "Processing"
    bundle = _record_with_copy_in_both_places(triage, proc, "here", same=False)
    rr.repair_unfinished_routes(force=True)
    assert (proc / f"{VID}.mp4").read_text(encoding="utf-8") == "vid"
    assert (bundle / f"{VID}.mp4").read_text(encoding="utf-8") == "video"
    assert [i["video_id"] for i in _listed(tmp_path)] == [VID]


def test_another_machines_local_source_is_never_touched(env):
    """The source is a path on the routing machine's own disk: the same path
    here is a different file (this machine's own run, say)."""
    tmp_path, triage, _ = env
    proc = tmp_path / "Processing"
    bundle = _record_with_copy_in_both_places(triage, proc, "other-node")
    before = (bundle / f"{VID}_routing.json").read_bytes()
    out = rr.repair_unfinished_routes(force=True)
    assert out["route_files_finished"] == 0
    assert (proc / f"{VID}.mp4").is_file()
    assert (bundle / f"{VID}_routing.json").read_bytes() == before
    assert _listed(tmp_path) == []             # the routing machine lists it


def test_a_record_without_its_machine_is_listed_not_acted_on(env):
    tmp_path, triage, _ = env
    proc = tmp_path / "Processing"
    _record_with_copy_in_both_places(triage, proc, None)
    rr.repair_unfinished_routes(force=True)
    assert (proc / f"{VID}.mp4").is_file()
    assert [i["video_id"] for i in _listed(tmp_path)] == [VID]


def test_a_route_records_the_machine_that_made_it(env):
    import socket
    tmp_path, triage, db = env
    proc = tmp_path / "Processing"
    _source(proc)
    rg.route_to_queue(VID, proc, triage, "reason", db=db, db_state="triage")
    record = json.loads((triage / VID / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["routed_by"] == socket.gethostname()


def test_the_scan_never_takes_a_file_of_a_video_running_again(env, monkeypatch):
    """A bundle left in the triage queue with a file missing while the video
    runs again in Processing (its row says 'processing'). The run's mp4 kept
    its modified time when it was copied in, so it looks older than the
    route. The scan must neither move it into the old bundle nor delete it as
    an 'identical copy': it lists the entry for a person."""
    import os
    tmp_path, triage, db = env
    proc = tmp_path / "Processing"
    _source(proc)
    lock = _lock(monkeypatch, f"{VID}.mp4")
    rg.route_to_queue(VID, proc, triage, "1 triaged segment(s) unresolved: 2",
                      db=db, db_state="triage")
    lock["on"] = False
    bundle = triage / VID
    assert not (bundle / f"{VID}.mp4").exists()
    # The video goes back to Processing and runs again; its mp4 (copied with
    # its old modified time) sits at the route's source.
    db.force_state(VID, "processing", reason="running again")
    mp4 = proc / f"{VID}.mp4"
    old = mp4.stat().st_mtime - 3600
    os.utime(mp4, (old, old))

    out = rr.repair_unfinished_routes(force=True, db=db)
    assert out["route_files_finished"] == 0
    assert mp4.read_text(encoding="utf-8") == f"{VID}.mp4"          # not taken
    assert not (bundle / f"{VID}.mp4").exists()
    [item] = _listed(tmp_path)
    assert item["video_id"] == VID and "processing" in item["says"]

    # The same with an identical copy already in the bundle: not deleted.
    (bundle / f"{VID}.mp4").write_text(f"{VID}.mp4", encoding="utf-8")
    rr.repair_unfinished_routes(force=True, db=db)
    assert mp4.is_file()

    # Held in the queue again: the scan finishes the route as usual.
    db.force_state(VID, "triage", reason="held again")
    rr.repair_unfinished_routes(force=True, db=db)
    assert not mp4.exists()
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert record["failed_files"] == []


def test_a_routing_record_is_read_only_when_it_changed(env, monkeypatch):
    """The repair pass runs every ten minutes over every bundle of both
    queues. A record is opened again only when its size or modified time
    changed; the listing alone says whether a manifest is missing. WHY:
    opening every record on the share each pass kept the watcher on the
    share for minutes."""
    tmp_path, triage, db = env
    for n in range(5):
        b = triage / f"20990101_ABC99{n:02d}_P1"
        b.mkdir()
        (b / f"{b.name}.mp4").write_text("v", encoding="utf-8")
        (b / f"{b.name}_manifest.json").write_text("{}", encoding="utf-8")
        (b / f"{b.name}_routing.json").write_text('{"failed_files": []}', encoding="utf-8")
    reads = []
    real = routing.read_routing_manifest
    monkeypatch.setattr(routing, "read_routing_manifest",
                        lambda b, s: reads.append(s) or real(b, s))
    rr._record_flags_cache.clear()
    rr.repair_unfinished_routes(force=True, db=db)
    assert len(reads) == 5
    rr.repair_unfinished_routes(force=True, db=db)
    assert len(reads) == 5                       # nothing changed: nothing read
    changed = triage / "20990101_ABC9902_P1" / "20990101_ABC9902_P1_routing.json"
    changed.write_text('{"failed_files": [], "x": 1}', encoding="utf-8")
    rr.repair_unfinished_routes(force=True, db=db)
    assert reads[5:] == ["20990101_ABC9902_P1"]


def test_a_route_from_triage_the_row_never_recorded_is_finished(env, monkeypatch):
    """A review window in a login that names no watcher database moves a video
    from triage into deep review: the folder moves, the row stays 'triage',
    and the mp4 (open in the window) stays in the triage folder. The scan
    must bring the row in line, move the mp4 in and empty the triage folder;
    the person's review, which moved with the bundle, is kept. Before, the
    repair refused ('not held in this queue') and the video sat in both
    queues for good."""
    tmp_path, triage, db = env
    deep = Paths.DEEP_REVIEW
    db.update_state(VID, "triage")
    copy = triage / VID
    _source(copy)
    (copy / f"{VID}_causal_review.json").write_text('{"answers": [1]}', encoding="utf-8")
    lock = _lock(monkeypatch, f"{VID}.mp4")
    rg.route_to_queue(VID, copy, deep, "reviewer escalated from triage", db=None,
                      db_state="deep_review")
    bundle = deep / VID
    assert (copy / f"{VID}.mp4").is_file() and not (bundle / f"{VID}.mp4").exists()
    assert db.get_video(VID)["state"] == "triage"

    # Still held open: the row is brought in line, the file stays listed.
    rr.repair_unfinished_routes(force=True, db=db)
    assert db.get_video(VID)["state"] == "deep_review"
    assert (copy / f"{VID}.mp4").is_file()
    assert [i["video_id"] for i in _listed(tmp_path)] == [VID]

    # Released: moved in, triage folder gone, the review kept in the bundle.
    lock["on"] = False
    out = rr.repair_unfinished_routes(force=True, db=db)
    assert out["route_files_finished"] == 1
    assert (bundle / f"{VID}.mp4").read_text(encoding="utf-8") == f"{VID}.mp4"
    assert not copy.exists()
    assert json.loads((bundle / f"{VID}_causal_review.json").read_text(
        encoding="utf-8")) == {"answers": [1]}
    assert _listed(tmp_path) == []


def test_a_row_is_not_relabelled_for_a_route_from_processing(env, monkeypatch):
    """Only a route FROM the video's own triage folder is caught up: a deep-
    review bundle routed from Processing whose row says 'triage' is a case
    for a person, and the row is left alone."""
    tmp_path, triage, db = env
    proc = tmp_path / "Processing"
    _source(proc)
    rg.route_to_queue(VID, proc, Paths.DEEP_REVIEW, "segmentation_failed: x",
                      db=None, db_state="deep_review")
    db.update_state(VID, "triage")
    assert rr._catch_up_route_from_triage(Paths.DEEP_REVIEW / VID, VID, db) is False
    assert db.get_video(VID)["state"] == "triage"
