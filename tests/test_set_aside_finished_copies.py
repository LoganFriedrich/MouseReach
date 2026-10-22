"""A GPU node sets aside local copies of videos already finished, at startup.

WHY (plan Step 7, 2026-10-22): copies of videos finished long ago sat in a
node's own pose queue and processing folder for good -- never picked (their
rows are finished), never removed; the Lab DLC PC held 34, the oldest from
2025-07. A copy moves only when the node's row says the video is finished AND
its results are in Analyzed; it is moved, never deleted.
"""
from datetime import date

import mousereach.archive.core as archive_core
import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB


def _node(tmp_path, monkeypatch):
    proc_root = tmp_path / "local"
    queue, processing = proc_root / "DLC_Queue", proc_root / "Processing"
    queue.mkdir(parents=True)
    processing.mkdir(parents=True)
    analyzed = tmp_path / "Analyzed" / "CNT01"
    analyzed.mkdir(parents=True)
    for name, value in (("PROCESSING_ROOT", proc_root), ("DLC_QUEUE", queue),
                        ("PROCESSING", processing)):
        monkeypatch.setattr(Paths, name, value, raising=False)
    monkeypatch.setattr(archive_core, "get_archive_destination", lambda stem: analyzed)
    orch = object.__new__(o.DLCOrchestrator)
    orch.db = WatcherDB(tmp_path / "w.db")
    return orch, queue, processing, analyzed, proc_root


def _video(db, folder, stem, state, analyzed=None, with_pose=True):
    (folder / f"{stem}.mp4").write_bytes(b"v")
    if with_pose:
        (folder / f"{stem}DLC_resnet101_X.h5").write_bytes(b"p")
    db.register_video(video_id=stem, source_path=str(folder / f"{stem}.mp4"))
    if state != "discovered":
        db.force_state(stem, state, reason="test fixture")
    if analyzed is not None:
        (analyzed / f"{stem}_processing_manifest.json").write_text("{}", encoding="utf-8")


def test_finished_copies_with_results_are_set_aside_and_nothing_else(tmp_path, monkeypatch):
    orch, queue, processing, analyzed, root = _node(tmp_path, monkeypatch)
    _video(orch.db, queue, "20250625_CNT0103_P4", "outdated", analyzed)      # finished: moves
    _video(orch.db, processing, "20250811_CNT0303_P4", "archived", analyzed)  # finished: moves
    _video(orch.db, queue, "20250701_CNT0111_P1", "dlc_queued", analyzed)    # active: stays
    _video(orch.db, queue, "20250702_CNT0112_P1", "archived")                # no results: stays
    (queue / "20250703_CNT0113_P1.mp4").write_bytes(b"v")                    # no row: stays

    assert orch._set_aside_finished_copies() == 2

    kept = root / "_finished_elsewhere" / date.today().isoformat()
    assert sorted(p.name for p in (kept / "DLC_Queue").iterdir()) == [
        "20250625_CNT0103_P4.mp4", "20250625_CNT0103_P4DLC_resnet101_X.h5"]
    assert (kept / "Processing" / "20250811_CNT0303_P4.mp4").is_file()
    assert sorted(p.name for p in queue.iterdir() if p.suffix == ".mp4") == [
        "20250701_CNT0111_P1.mp4", "20250702_CNT0112_P1.mp4", "20250703_CNT0113_P1.mp4"]
