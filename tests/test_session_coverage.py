"""Where each listed session is, from shared storage alone.

WHY: once a list's recordings are imported, its sessions spread across the
pipeline, and nothing said how far the list had got without asking every
machine. One stage per session, first match wins (watcher/coverage.py).
"""
import json
import sqlite3

import pytest

import mousereach.watcher.coverage as cov
from mousereach.config import Paths


def _share(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    dirs = {
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "REVIEW_ROOT": nas / "Processing" / "Review",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    return nas, dirs


S = [f"20220101_ASPA04{n:02d}_P1" for n in range(1, 13)]


def test_every_session_is_placed_in_exactly_one_stage(tmp_path, monkeypatch):
    nas, d = _share(tmp_path, monkeypatch)
    from mousereach.archive.core import get_archive_destination
    done = get_archive_destination(S[0])
    done.mkdir(parents=True, exist_ok=True)
    (done / f"{S[0]}_processing_manifest.json").write_text("{}", encoding="utf-8")
    (d["REVIEW_ROOT"] / "triage" / S[1]).mkdir(parents=True)
    (d["REVIEW_ROOT"] / "_deprioritized_not_in_paper_set" / "triage" / S[2]).mkdir(parents=True)
    (d["SINGLE_ANIMAL_OUTPUT"] / f"{S[4]}.mp4").write_bytes(b"v")
    (d["SINGLE_ANIMAL_OUTPUT"] / ".inflight" / "NODE-A").mkdir(parents=True)
    (d["SINGLE_ANIMAL_OUTPUT"] / ".inflight" / "NODE-A" / f"{S[5]}.mp4").write_bytes(b"v")
    (d["DLC_STAGING"] / f"{S[6]}.mp4").write_bytes(b"v")
    # recordings: one never claimed (S[7]), one being cut (S[8]), one cut (S[9])
    for i, n in ((7, 1), (8, 2), (9, 3)):
        animals = ",".join([S[i].split("_")[1]] + [f"ASPA09{k:02d}" for k in range(n, n + 7)])
        (d["MULTI_ANIMAL_SOURCE"] / f"20220101_{animals}_P1.mkv").write_bytes(b"r")
    db = sqlite3.connect(nas / "watcher_central.db")
    db.execute("CREATE TABLE videos (video_id TEXT, state TEXT)")
    db.execute("CREATE TABLE pipeline_videos (video_id TEXT, state TEXT)")
    db.execute("CREATE TABLE pipeline_collages (filename TEXT, state TEXT)")
    db.execute("INSERT INTO videos VALUES (?, 'dlc_running')", (S[3],))
    db.execute("INSERT INTO videos VALUES (?, 'archived')", (S[0],))     # results win anyway
    rec = {i: next(n.name for n in d["MULTI_ANIMAL_SOURCE"].iterdir()
                   if S[i].split("_")[1] in n.name) for i in (8, 9)}
    db.execute("INSERT INTO pipeline_collages VALUES (?, 'cropping')", (rec[8],))
    db.execute("INSERT INTO pipeline_collages VALUES (?, 'cropped')", (rec[9],))
    db.commit()
    db.close()

    stage = cov.session_coverage(S[:11], nas)
    assert stage == {
        S[0]: "analysed",
        S[1]: "in review",
        S[2]: "review, parked",
        S[3]: "being posed",
        S[4]: "waiting to be posed",
        S[5]: "being posed",
        S[6]: "posed, waiting for analysis",
        S[7]: "recording waiting to be cut",
        S[8]: "recording being cut",
        S[9]: "cut; on a node, not yet in the shared record",
        S[10]: "not imported",
    }


def test_the_command_writes_counts_beside_the_list(tmp_path, monkeypatch):
    nas, d = _share(tmp_path, monkeypatch)
    lst = tmp_path / "sessions.txt"
    lst.write_text("# test\n20220101_D01_P1\nnot a session\n", encoding="utf-8")
    assert cov.main(["--sessions-file", str(lst)]) == 0
    out = json.loads((tmp_path / "sessions.coverage.json").read_text(encoding="utf-8"))
    assert out["sessions"] == 1
    assert out["by_stage"] == {"not imported": 1}
    assert out["not_session_names"] == ["not a session"]


def _record(path, counter_writes=1):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x)")
    for i in range(counter_writes):
        con.execute("INSERT INTO t VALUES (?)", (i,))
        con.commit()
    con.close()


def test_the_record_is_read_from_a_copy_taken_between_writes(tmp_path):
    """2026-10-29: reading the shared record in place waited 10 minutes on another
    machine's lock and failed. The report reads a local copy instead."""
    from mousereach.watcher import coverage
    db = tmp_path / "watcher_central.db"
    _record(db)
    copy = coverage._settled_copy(db, tmp_path / "copy.db", tries=1, pause=0)
    assert copy.read_bytes() == db.read_bytes()


def test_no_copy_is_taken_while_a_write_is_under_way(tmp_path, monkeypatch):
    from mousereach.watcher import coverage
    db = tmp_path / "watcher_central.db"
    _record(db)
    journal = db.with_name(db.name + "-journal")
    journal.write_bytes(b"in progress")           # a writer's rollback journal
    with pytest.raises(RuntimeError, match="being written"):
        coverage._settled_copy(db, tmp_path / "copy.db", tries=2, pause=0)
    journal.unlink()                              # the write finished
    assert coverage._settled_copy(db, tmp_path / "copy.db", tries=1, pause=0).is_file()


def test_a_copy_that_a_write_overtook_is_taken_again(tmp_path, monkeypatch):
    from mousereach.watcher import coverage
    db = tmp_path / "watcher_central.db"
    _record(db)
    real_copy, calls = coverage.shutil.copyfile, []

    def copy_then_someone_writes(src, dst):
        real_copy(src, dst)
        if not calls:                             # first copy: a write lands mid-copy
            con = sqlite3.connect(src); con.execute("INSERT INTO t VALUES (99)"); con.commit(); con.close()
        calls.append(1)
    monkeypatch.setattr(coverage.shutil, "copyfile", copy_then_someone_writes)
    copy = coverage._settled_copy(db, tmp_path / "copy.db", tries=3, pause=0)
    assert len(calls) == 2
    assert sqlite3.connect(copy).execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
