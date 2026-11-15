"""A crystallized video is never moved.

mousereach-crystallize locks a filed video for a publication: its results must
stay exactly as they are until a person unlocks it. Routing into a review
queue, the review return and the version scan's divert used to force their way
through the lock (force_state), so a locked video's results could leave the
filed folder. Now every one of them refuses before anything moves, and
force_state itself will not take a row out of 'crystallized' except for the
unlock command. Built under tmp_path; no real folder or database.
"""
import pytest

from mousereach.config import Paths
from mousereach.watcher.db import CrystallizedError, WatcherDB

VID = "20990101_ABC9901_P1"


def _locked_db(path):
    db = WatcherDB(db_path=path)
    db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving", "archived", "crystallized"):
        db.update_state(VID, st)
    return db


@pytest.fixture
def filed(tmp_path, monkeypatch):
    folder = tmp_path / "Analyzed" / "PROJ" / "COHORT"
    folder.mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_pellet_outcomes.json", f"{VID}_features.json"):
        (folder / name).write_text("{}" if name.endswith(".json") else "v")
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    return folder, _locked_db(tmp_path / "w.db")


def _untouched(folder):
    return sorted(p.name for p in folder.iterdir()) == sorted(
        [f"{VID}.mp4", f"{VID}_features.json", f"{VID}_pellet_outcomes.json"])


def test_route_to_queue_refuses_before_moving(filed):
    from mousereach.watcher.review_gate import route_to_queue
    folder, db = filed
    with pytest.raises(CrystallizedError):
        route_to_queue(VID, folder, Paths.TRIAGE_REVIEW, "test", db=db, db_state="triage")
    assert _untouched(folder)
    assert db.get_video(VID)["state"] == "crystallized"


def test_the_route_command_reports_a_lock(filed, monkeypatch):
    from mousereach.watcher import route_cli
    folder, db = filed
    monkeypatch.setattr(route_cli, "_find_outcomes",
                        lambda analyzed, vid: folder / f"{vid}_pellet_outcomes.json")
    res = route_cli.route_video(VID, "triage", "test", [1], db=db)
    assert res["locked"] and not res["routed"] and "crystallized" in res["error"]
    assert _untouched(folder)


def test_the_review_return_refuses(filed, tmp_path, monkeypatch):
    from mousereach.watcher import review_return
    _, db = filed
    bundle = tmp_path / "triage" / VID
    bundle.mkdir(parents=True)
    (bundle / f"{VID}.mp4").write_text("v")
    pose = bundle / f"{VID}DLC_resnet50_testshuffle1_100.h5"
    pose.write_text("p")
    monkeypatch.setattr(review_return, "_claim_return", lambda stem: True)
    monkeypatch.setattr(review_return, "_release_return_claim", lambda stem: None)
    # Everything a return needs is found, so only the lock can stop it (the
    # code before the lock check moved the files and forced 'processing').
    monkeypatch.setattr(review_return, "_resolve_inputs",
                        lambda *a, **k: (bundle / f"{VID}.mp4", pose))
    monkeypatch.setattr(review_return, "_ensure_durable_review", lambda *a, **k: None)
    proc = tmp_path / "Processing"
    assert review_return._return_to_processing(bundle, VID, proc, db, "test") is False
    assert (bundle / f"{VID}.mp4").is_file()
    assert db.get_video(VID)["state"] == "crystallized"


def test_the_version_scan_divert_refuses(filed):
    from mousereach.watcher.reprocessor import ReprocessingScanner
    folder, db = filed
    scanner = ReprocessingScanner.__new__(ReprocessingScanner)
    scanner.db = db
    scanner.archive_dir = folder.parent.parent
    scanner._divert_mislabel_to_deep_review(VID)
    assert _untouched(folder)


def test_force_state_cannot_leave_crystallized(filed):
    _, db = filed
    with pytest.raises(CrystallizedError):
        db.force_state(VID, "processing", reason="test")
    assert db.get_video(VID)["state"] == "crystallized"


def test_the_unlock_command_still_unlocks(filed):
    from mousereach.pipeline.versions import uncrystallize_videos
    _, db = filed
    assert uncrystallize_videos(db, [VID]) == 1
    assert db.get_video(VID)["state"] == "archived"
