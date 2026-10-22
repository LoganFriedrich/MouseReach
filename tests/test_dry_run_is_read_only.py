"""`mousereach-watch --dry-run` reads only.

WHY: the install guide has every new machine run the dry run before its
watcher starts, and it used to build the whole watcher (start-up recovery) and
call the real scans -- registering files and moving videos between states, in
the local database and the shared one (104 recordings advanced on one machine,
2026-08-30). The preview opens the database read-only and lists folders; these
tests prove the database bytes and the folder tree are unchanged after it.
"""
import hashlib

import pytest

import mousereach.config as mc
import mousereach.watcher.work_priority as wp
from mousereach.config import Paths, WatcherConfig
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.preview import ReadOnlyDB, preview


def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob("*"))


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def site(tmp_path, monkeypatch):
    nas, proc = tmp_path / "nas", tmp_path / "proc"
    folders = {
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "DLC_QUEUE": proc / "DLC_Queue",
    }
    for name, d in folders.items():
        d.mkdir(parents=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(mc, "PROCESSING_ROOT", proc, raising=False)
    wp.invalidate_lab_policy()
    db_path = proc / "watcher.db"
    db = WatcherDB(db_path)
    db.register_video(video_id="20250101_CNT0101_P1", source_path="x.mp4")
    db.force_state("20250101_CNT0101_P1", "dlc_complete", reason="test fixture")
    db.register_collage("20250102_CNT0101,CNT0102,CNT0000_P1.mkv", "x.mkv")
    db.force_collage_state("20250102_CNT0101,CNT0102,CNT0000_P1.mkv", "stable")
    yield tmp_path, folders, db_path
    wp.invalidate_lab_policy()


def test_the_processing_preview_changes_nothing(site):
    root, folders, db_path = site
    stem = "20250103_CNT0101_P2"
    (folders["DLC_STAGING"] / f"{stem}.mp4").write_bytes(b"v")
    (folders["DLC_STAGING"] / f"{stem}DLC_resnet101_X.h5").write_bytes(b"p")
    before, sha = _tree(root), _sha(db_path)

    lines = preview(WatcherConfig({"mode": "processing_server", "max_local_pending": 2}), db_path)

    assert _tree(root) == before and _sha(db_path) == sha
    text = "\n".join(lines)
    assert "posed videos it would newly take in: 1" in text
    assert "It would take next: intake 20250101_CNT0101_P1" in text


def test_the_gpu_node_preview_changes_nothing(site):
    root, folders, db_path = site
    (folders["MULTI_ANIMAL_SOURCE"] / "20250104_CNT0101,CNT0102,CNT0000_P1.mkv").write_bytes(b"r")
    (folders["SINGLE_ANIMAL_OUTPUT"] / "20250105_CNT0101_P1.mp4").write_bytes(b"s")
    before, sha = _tree(root), _sha(db_path)

    lines = preview(WatcherConfig({"mode": "dlc_pc"}), db_path)

    assert _tree(root) == before and _sha(db_path) == sha
    text = "\n".join(lines)
    assert "recordings it would newly register: 1" in text
    assert "Shared cut videos waiting to be posed: 1" in text
    assert "It would take next:" in text


def test_a_machine_with_no_database_yet_gets_a_preview_not_a_database(tmp_path):
    missing = tmp_path / "none" / "watcher.db"
    lines = preview(WatcherConfig({"mode": "processing_server"}), missing)
    assert not missing.exists() and not missing.parent.exists()
    assert "Videos in this machine's database: 0" in lines


def test_the_stand_in_database_cannot_write(tmp_path):
    db = ReadOnlyDB(tmp_path / "w.db")
    for name in ("force_state", "update_state", "register_video", "mark_failed",
                 "mark_unresolvable", "register_collage", "force_collage_state"):
        assert not hasattr(db, name), name
