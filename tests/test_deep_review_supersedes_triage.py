"""Deep review supersedes triage: a video in deep review is never in triage too.

A video that is in the deep-review queue is not allowed to be in the triage
queue as well. Routing into triage while a deep-review bundle exists creates
nothing (the video stays in deep review); routing into deep review deletes the
video's triage bundle once the deep-review bundle is complete; and every reader
that finds both treats the video as deep review only. A router that did not
check used to route videos already in deep review into triage from their filed
results, so a reviewer could answer a triage copy holding older results.
Built under tmp_path; no real folder or database.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.review_gate import route_to_queue
from mousereach.watcher.review_routing import (
    deep_review_stems, in_deep_review, superseded_by_deep_review,
)

VID = "20990101_ABC9901_P1"


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "deep_review"):
        db.update_state(VID, st)
    return tmp_path, triage, deep, db


def _bundle(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_text("x")
    return folder


def test_routing_into_triage_is_refused_while_in_deep_review(queues):
    tmp_path, triage, deep, db = queues
    _bundle(deep / VID, f"{VID}.mp4", f"{VID}_segments.json")
    source = _bundle(tmp_path / "filed", f"{VID}.mp4", f"{VID}_pellet_outcomes.json")
    out = route_to_queue(VID, source, triage, "bench check", db=db, db_state="triage")
    assert out == deep / VID
    assert not (triage / VID).exists()
    assert not (triage / ".incoming" / VID).exists()
    assert sorted(p.name for p in source.iterdir()) == [
        f"{VID}.mp4", f"{VID}_pellet_outcomes.json"]           # nothing moved
    assert db.get_video(VID)["state"] == "deep_review"


def test_the_route_command_does_not_flag_or_route_it(queues, monkeypatch):
    from mousereach.watcher import route_cli
    tmp_path, triage, deep, db = queues
    _bundle(deep / VID, f"{VID}.mp4")
    filed = _bundle(tmp_path / "filed", f"{VID}.mp4")
    outcomes = filed / f"{VID}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"segments": [{"segment_num": 1}]}))
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path)
    monkeypatch.setattr(route_cli, "_find_outcomes", lambda analyzed, vid: outcomes)
    res = route_cli.route_video(VID, "triage", "bench check", [1], db=db)
    assert res["in_deep_review"] and not res["routed"]
    assert "flagged_for_review" not in outcomes.read_text()
    assert not (triage / VID).exists()


def test_routing_into_deep_review_deletes_the_triage_copy(queues):
    tmp_path, triage, deep, db = queues
    stale = _bundle(triage / VID, f"{VID}.mp4", f"{VID}_causal_review.json")
    source = _bundle(tmp_path / "Processing", f"{VID}.mp4", f"{VID}_segments.json")
    route_to_queue(VID, source, deep, "segmentation_failed", db=db,
                   db_state="deep_review")
    assert (deep / VID / f"{VID}.mp4").is_file()
    assert not stale.exists()


def test_escalating_from_triage_leaves_no_triage_copy(queues):
    from mousereach.watcher.review_gate import escalate_to_deep_review
    tmp_path, triage, deep, db = queues
    db.force_state(VID, "triage", reason="test")
    b = _bundle(triage / VID, f"{VID}.mp4", f"{VID}_segments.json")
    escalate_to_deep_review(VID, b, "reviewer escalated", db=db)
    assert (deep / VID / f"{VID}.mp4").is_file()
    assert not (triage / VID).exists()
    assert db.get_video(VID)["state"] == "deep_review"


def test_a_triage_copy_is_kept_when_files_failed_to_move(queues, monkeypatch):
    from mousereach.watcher import review_gate
    tmp_path, triage, deep, db = queues
    stale = _bundle(triage / VID, f"{VID}.mp4")
    bundle = _bundle(deep / VID, f"{VID}_segments.json")
    (bundle / f"{VID}_routing.json").write_text(json.dumps(
        {"failed_files": [{"name": f"{VID}.mp4", "error": "in use"}]}))
    review_gate._drop_triage_copy(VID, bundle)
    assert stale.is_dir()


def test_readers_treat_the_video_as_deep_review_only(queues, monkeypatch):
    tmp_path, triage, deep, db = queues
    _bundle(deep / VID, f"{VID}.mp4")
    copy = _bundle(triage / VID, f"{VID}.mp4", f"{VID}_manifest.json")
    assert in_deep_review(VID) and VID in deep_review_stems()
    assert superseded_by_deep_review(copy)
    assert not superseded_by_deep_review(deep / VID)

    from mousereach.dashboard import folder_scan
    assert folder_scan._PRIORITY["deep_review"] > folder_scan._PRIORITY["triage"]


def test_the_return_scan_never_returns_a_superseded_triage_copy(queues, monkeypatch):
    from mousereach.watcher import review_return
    tmp_path, triage, deep, db = queues
    _bundle(deep / VID, f"{VID}.mp4", f"{VID}_segments.json")
    # A triage copy that looks fully answered -- it would normally go back.
    _bundle(triage / VID, f"{VID}.mp4", f"{VID}_extra.json")
    returned = []
    monkeypatch.setattr(review_return, "_return_to_processing",
                        lambda *a, **k: returned.append(a[1]) or True)
    monkeypatch.setattr(review_return, "_cached_triage_status",
                        lambda b, s, sig: type("S", (), {
                            "seg_failed": False, "seg_pending_reseg": False,
                            "has_triage": True, "unresolved": [], "too_short": [],
                            "seg_corrected": []})())
    review_return.scan_review_queues(db, tmp_path / "Processing")
    assert VID not in returned
    assert (triage / VID).is_dir()           # readers ignore it; they do not delete


def test_a_census_counts_it_in_deep_review_only(queues):
    from mousereach.census import locate_sessions  # noqa: F401  (import check)
    import inspect
    from mousereach.census import runner
    src = inspect.getsource(runner)
    assert 'locations["triage"] = set(locations["triage"]) - set(locations["deep_review"])' in src


def test_reconcile_judges_it_as_deep_review_only():
    import inspect
    from mousereach.watcher import reconcile
    assert 'places.discard("triage")' in inspect.getsource(reconcile)
