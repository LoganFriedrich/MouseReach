"""Copying or moving a file onto itself keeps it -- and so does a re-run whose
pose is already in the local processing folder.

WHY: shutil raises SameFileError (an OSError) when source and destination are
one file, and safe_copy's partial-copy cleanup then unlinked the destination,
which IS the source. A video that failed its first pass and had Re-run pressed
keeps its row's pose path, which points into the local processing folder; the
re-run's copy-down then "copied" every file of the video onto itself -- the
video, the pose, a person's cuts, review files -- and deleted each one.

Built under tmp_path with invented ids; no real folder or database.
"""
from types import SimpleNamespace

import pytest

from mousereach.config import Paths
from mousereach.watcher import transfer
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20990101_ABC0301_P1"
POSE = f"{VID}DLC_resnet50_TestModelshuffle1_100.h5"


def test_safe_copy_onto_itself_keeps_the_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("keep me", encoding="utf-8")
    assert transfer.safe_copy(f, tmp_path / "a.json") is True
    assert f.read_text(encoding="utf-8") == "keep me"


def test_safe_move_onto_itself_keeps_the_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("keep me", encoding="utf-8")
    assert transfer.safe_move(f, tmp_path / "." / "a.json") is True
    assert f.read_text(encoding="utf-8") == "keep me"


@pytest.fixture
def orch(tmp_path, monkeypatch):
    root = tmp_path / "proc_root"
    proc = root / "Processing"
    proc.mkdir(parents=True)
    nas = tmp_path / "nas"
    nas.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", root, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    (nas / "Analyzed").mkdir()
    o = object.__new__(ProcessingOrchestrator)
    o.config = SimpleNamespace(max_retries=3, max_local_pending=4)
    o.db = WatcherDB(tmp_path / "w.db")
    o.processing_dir = proc
    o.staging_dir = None
    o.hostname = "NODE-T"
    o._score_waits = {}
    o._archive_backoff = {}
    return o


def test_rerun_of_a_failed_never_filed_video_with_a_local_pose_keeps_its_files(
        orch, monkeypatch):
    from mousereach.watcher import repose
    import mousereach.watcher.orchestrator as o
    proc = orch.processing_dir
    names = [f"{VID}.mp4", POSE, f"{VID}_segments.json", f"{VID}_causal_review.json"]
    for n in names:
        (proc / n).write_text(f"content of {n}", encoding="utf-8")
    # A first pass that failed, then Re-run pressed: 'outdated' with the row's
    # pose path still pointing into the processing folder.
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test",
                        dlc_output_path=str(proc / POSE))
    for _ in range(3):
        orch.db.mark_failed(VID, "boom")
    orch.db.force_state(VID, "outdated", reason="Re-run pressed",
                        reprocess_scope="segmentation")
    monkeypatch.setattr(repose, "declared_scorer", lambda: "")
    monkeypatch.setattr(o, "archived_pose_candidates", lambda *a, **k: [])
    monkeypatch.setattr("mousereach.archive.core.get_archive_destination",
                        lambda vid: Paths.ANALYZED_OUTPUT / "PROJ" / "COHORT")
    monkeypatch.setattr(orch, "_archive_reachable", lambda: True, raising=False)
    ran = []
    monkeypatch.setattr(orch, "_run_pipeline", lambda work: ran.append(work["id"]),
                        raising=False)

    orch._reprocess_video({"id": VID, "data": orch.db.get_video(VID)})

    for n in names:
        assert (proc / n).read_text(encoding="utf-8") == f"content of {n}", n
    assert ran == [VID]
    assert orch.db.get_video(VID)["state"] == "processing"
