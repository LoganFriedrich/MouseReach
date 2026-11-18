"""A publication lock is a record beside the filed results, honoured by every machine.

WHY: mousereach-crystallize used to lock a video only in the watcher database
of the machine that ran it. Every machine keeps its own database, so the lock
was invisible to any other machine's version scan, and a lock written into the
wrong database looked set and protected nothing. Now locking writes
``<video>_crystallized.json`` beside the filed results -- the one place every
machine reads -- and the version scan, the route into a review queue, filing,
the start of a re-run, the dashboard's Re-run button and the re-pose command
all honour it, whatever machine set it. Unlocking removes it.

Built under tmp_path: a filed tree, databases made by the test, and hand-written
records standing for a lock set from another machine.
"""
import json
import sys

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import CrystallizedError, WatcherDB

VID = "20990101_ZZZ9901_P1"
LOCK = "_crystallized.json"


def _archived_db(path, vid=VID):
    db = WatcherDB(db_path=path)
    db.register_video(video_id=vid, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete",
               "processing", "processed", "archiving", "archived"):
        db.update_state(vid, st)
    return db


@pytest.fixture
def filed(tmp_path, monkeypatch):
    """A filed video in the folder the archive step files it into, and the
    database of a machine that did NOT set any lock (its row is 'archived')."""
    from mousereach.archive.core import filed_folder
    analyzed = tmp_path / "Analyzed"
    folder = filed_folder(VID, analyzed)
    folder.mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_pellet_outcomes.json",
                 f"{VID}_features.json", f"{VID}_processing_manifest.json"):
        (folder / name).write_text("{}" if name.endswith(".json") else "v")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    return folder, _archived_db(tmp_path / "other_machine.db")


def _lock_from_elsewhere(folder, vid=VID, label="LABEL_A"):
    """The record a lock set on another machine leaves beside the results."""
    (folder / f"{vid}{LOCK}").write_text(
        json.dumps({"video_id": vid, "label": label, "crystallized_by": "someone",
                    "crystallized_at": "2099-01-01T00:00:00"}), encoding="utf-8")


def test_locking_writes_the_record_beside_the_filed_results(filed, tmp_path):
    from mousereach.pipeline.versions import crystallize_videos, uncrystallize_videos
    folder, _ = filed
    db = _archived_db(tmp_path / "server.db")
    assert crystallize_videos(db, video_ids=[VID], label="LABEL_A") == 1
    record = folder / f"{VID}{LOCK}"
    assert record.is_file(), "the lock lives only in one machine's database"
    assert json.loads(record.read_text(encoding="utf-8"))["label"] == "LABEL_A"
    assert db.get_video(VID)["state"] == "crystallized"
    assert not [p for p in folder.iterdir() if p.name.startswith(".")], "temp file left"

    assert uncrystallize_videos(db, label="LABEL_A") == 1
    assert not record.exists()
    assert db.get_video(VID)["state"] == "archived"


def test_a_video_whose_filed_results_are_missing_is_not_locked(filed, tmp_path):
    from mousereach.pipeline.versions import crystallize_videos
    folder, _ = filed
    for p in folder.iterdir():
        p.unlink()
    db = _archived_db(tmp_path / "server.db")
    failures = []
    assert crystallize_videos(db, video_ids=[VID], label="L", failures=failures) == 0
    assert failures and failures[0][0] == VID
    assert db.get_video(VID)["state"] == "archived"


def test_another_machines_record_refuses_the_route_into_a_queue(filed):
    from mousereach.watcher.db import crystallized_refusal
    from mousereach.watcher.review_gate import route_to_queue
    folder, db = filed
    _lock_from_elsewhere(folder)
    why = crystallized_refusal(db, VID, "routed")
    assert why and "LABEL_A" in why, "a lock set on another machine was not seen"
    with pytest.raises(CrystallizedError):
        route_to_queue(VID, folder, Paths.TRIAGE_REVIEW, "test", db=db, db_state="triage")
    assert (folder / f"{VID}_features.json").exists()


def test_the_record_is_honoured_with_no_database_at_all(filed):
    from mousereach.watcher.db import crystallized_refusal
    folder, _ = filed
    _lock_from_elsewhere(folder)
    assert crystallized_refusal(None, VID, "backfilled")


def test_filing_never_replaces_locked_results(filed, tmp_path, monkeypatch):
    from mousereach.archive import core
    folder, _ = filed
    _lock_from_elsewhere(folder)
    proc = tmp_path / "Processing"
    proc.mkdir()
    (proc / f"{VID}_features.json").write_text('{"new": true}')
    (proc / f"{VID}.mp4").write_text("new video")
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    result = core.archive_video(VID, verbose=False, skip_ready_check=True,
                                source_dir=proc)
    assert not result["success"]
    assert (folder / f"{VID}_features.json").read_text() == "{}", "locked results replaced"
    assert (proc / f"{VID}_features.json").exists()


def test_the_version_scan_never_marks_a_video_locked_elsewhere(filed, tmp_path, monkeypatch):
    import mousereach.pipeline.versions as versions_mod
    import mousereach.review.causal_review_io as crio
    from mousereach.watcher.reprocessor import ReprocessingScanner
    folder, _ = filed
    (folder / f"{VID}_processing_manifest.json").write_text(json.dumps(
        {"dlc_model": {"dlc_scorer": "MODEL_A"},
         "pipeline_versions": {"segmenter": "0.1.0", "reach_detector": "1.0.0",
                               "outcome_detector": "1.0.0", "assignment": "1.0.0",
                               "kinematic_extractor": "1.0.0"}}))
    _lock_from_elsewhere(folder)
    monkeypatch.setattr(versions_mod, "get_current_versions", lambda root=None: {
        "versions": {"dlc_scorer": "MODEL_A", "segmenter": "1.0.0",
                     "reach_detector": "1.0.0", "outcome_detector": "1.0.0",
                     "assignment": "1.0.0", "kinematic_extractor": "1.0.0"}})
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})

    forced = []

    class StubDB:
        def _get_connection(self):
            class C:
                def execute(self, *a):
                    return [(VID,)]

                def close(self):
                    pass
            return C()

        def get_videos_in_state(self, state):
            return [{"video_id": VID, "state": "archived"}] if state == "archived" else []

        def force_state(self, video_id, state, **k):
            forced.append((video_id, state))

        def update_state(self, video_id, state, **k):
            forced.append((video_id, state))

        def set_fields(self, *a, **k):
            pass

        def register_video(self, *a, **k):
            pass

    summary = ReprocessingScanner(StubDB(), tmp_path).scan(mark_outdated=True,
                                                           adopt_orphans=False)
    assert (VID, "outdated") not in forced, "a locked video was marked for a re-run"
    assert summary.get("locked_by_record") == 1


def test_a_rerun_of_a_locked_video_never_starts(filed, tmp_path):
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    folder, db = filed
    db.update_state(VID, "outdated", reprocess_scope="segmentation")
    _lock_from_elsewhere(folder)
    orch = object.__new__(ProcessingOrchestrator)
    orch.db = db
    started = []
    orch._archive_reachable = lambda: started.append("reachable") or True
    orch._no_pose_for_reprocess = lambda vid: started.append("pose")
    orch.staging_dir = None
    orch._reprocess_video({"id": VID, "data": db.get_video(VID)})
    assert started == [], "the re-run of a locked video was started"
    assert db.get_video(VID)["state"] == "archived"


def test_uncrystallize_by_label_removes_a_record_with_no_row_here(filed, tmp_path):
    from mousereach.pipeline.versions import uncrystallize_videos
    folder, _ = filed
    _lock_from_elsewhere(folder, label="LABEL_B")
    db = WatcherDB(db_path=tmp_path / "empty.db")
    assert uncrystallize_videos(db, label="LABEL_B") == 1
    assert not (folder / f"{VID}{LOCK}").exists()


def test_the_crystallize_command_exits_1_when_a_video_is_not_locked(filed, tmp_path, monkeypatch, capsys):
    from mousereach.pipeline import versions
    from mousereach.watcher import cli
    folder, _ = filed
    for p in folder.iterdir():
        p.unlink()
    path = tmp_path / "server.db"
    _archived_db(path)
    monkeypatch.setattr(versions, "publication_lock_gate", lambda: (True, None))
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: path)
    monkeypatch.setattr(sys, "argv", ["mousereach-crystallize", "--videos", VID,
                                      "--label", "LABEL_A"])
    with pytest.raises(SystemExit) as e:
        cli.main_crystallize()
    assert e.value.code == 1
    assert VID in capsys.readouterr().err


def test_an_unreachable_filed_tree_counts_as_locked(tmp_path, monkeypatch):
    """A filed tree that is configured but not there (an unmapped drive) shows
    no record -- which proves nothing, so it refuses instead of reading as
    "unlocked"."""
    from mousereach.watcher.db import crystallized_refusal
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "unmapped" / "Analyzed")
    why = crystallized_refusal(None, VID, "routed")
    assert why and "cannot tell" in why


def test_process_animal_never_queues_a_video_locked_elsewhere(filed, tmp_path, monkeypatch, capsys):
    """No row on this machine, but a lock record beside the filed results:
    the video is not claimed or copied for a new pose."""
    from mousereach.archive.core import filed_folder
    from mousereach.watcher import cli
    vid = "20240101_ABC9902_P1"          # a name the singles listing accepts
    folder = filed_folder(vid, Paths.ANALYZED_OUTPUT)
    folder.mkdir(parents=True)
    (folder / f"{vid}_processing_manifest.json").write_text("{}")
    _lock_from_elsewhere(folder, vid=vid)
    nas, work = tmp_path / "nas", tmp_path / "work"
    for name, d in {"NAS_ROOT": nas, "NAS_DRIVE": nas,
                    "MULTI_ANIMAL_SOURCE": nas / "Multi",
                    "SINGLE_ANIMAL_OUTPUT": nas / "Single_Animal",
                    "DLC_STAGING": nas / "Posed", "PROCESSING_ROOT": work,
                    "PROCESSING": work / "Processing",
                    "DLC_QUEUE": work / "DLC_Queue"}.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    shared = Paths.SINGLE_ANIMAL_OUTPUT / f"{vid}.mp4"
    shared.write_bytes(b"video")
    WatcherDB(db_path=tmp_path / "here.db")
    monkeypatch.setattr(cli, "_resolve_db_path", lambda: tmp_path / "here.db")
    monkeypatch.setattr("socket.gethostname", lambda: "NODE-T")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(sys, "argv", ["mousereach-watch-process-animal", "ABC9902"])
    try:
        cli.main_process_animal()
    except SystemExit:
        pass
    assert shared.is_file(), "a locked video was claimed for a new pose"
    assert not (Paths.DLC_QUEUE / f"{vid}.mp4").exists()
    assert "crystallized" in capsys.readouterr().out
