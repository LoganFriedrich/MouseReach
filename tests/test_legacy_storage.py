"""Videos left on old storage: what the pipeline already has, and a verified bring-in.

WHY: recordings that exist only in a retired pipeline folder are invisible to
every machine that could process them, and the old storage cannot be retired
while they are there (found 2026-10-21: a cohort's Flat-tray sessions, cut and
uncut, existed nowhere else). The inventory must never bring in what the
pipeline already has under another name or in another form, and a copy must be
verified before the pipeline can see it.
"""
import json
import os
import sqlite3
import time

import mousereach.pipeline.legacy_storage as ls
from mousereach.config import Paths

REC = "20260716_CNT0501,CNT0502,CNT0000,CNT0000,CNT0000,CNT0000,CNT0000,CNT0000"


def _setup(tmp_path, monkeypatch):
    old, nas = tmp_path / "old", tmp_path / "nas"
    dirs = {
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "UNSUPPORTED_TRAY_RETURN": nas / "Unanalyzed" / "Unsupported_Tray_Type",
        "ANALYZED_OUTPUT": nas / "Analyzed",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    old.mkdir()
    return old, nas, dirs


def _put(path, data=b"v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _rows(old, nas):
    return {r["name"]: r for r in ls.inventory(old, ls.PipelineIndex(nas))}


def test_names_are_read_as_recordings_cut_videos_or_neither():
    rec = ls.classify(REC + "_F1")
    assert rec["kind"] == "recording" and rec["tray"] == "F"
    assert rec["sessions"] == ["20260716_CNT0501_F1", "20260716_CNT0502_F1"]  # blanks skipped
    assert ls.classify("20260716_CNT0501_F1")["kind"] == "cut"
    assert ls.classify(REC + "_P4(LONG)")["kind"] == "other"
    assert ls.classify("20251222_CNT0401,CNT407,CNT0000_P1")["bad_ids"] == ["CNT407"]


def test_what_the_pipeline_already_has_is_never_missing(tmp_path, monkeypatch):
    old, nas, d = _setup(tmp_path, monkeypatch)
    # same file
    _put(old / "Unanalyzed" / "20250101_CNT0101_P1.mp4", b"abc")
    _put(nas / "Analyzed" / "CNT01" / "20250101_CNT0101_P1.mp4", b"abc")
    # same name, other size
    _put(old / "Unanalyzed" / "20250101_CNT0102_P1.mp4", b"")
    _put(nas / "Analyzed" / "CNT01" / "20250101_CNT0102_P1.mp4", b"abc")
    # results only (the video itself archived elsewhere)
    _put(old / "Unanalyzed" / "20250101_CNT0103_P1.mp4")
    _put(nas / "Analyzed" / "CNT01" / "20250101_CNT0103_P1_processing_manifest.json", b"{}")
    # known to the shared record only (being posed on a GPU machine)
    _put(old / "Unanalyzed" / "20250101_CNT0104_P1.mp4")
    db = sqlite3.connect(nas / "watcher_central.db")
    db.execute("CREATE TABLE videos (video_id TEXT)")
    db.execute("INSERT INTO videos VALUES ('20250101_CNT0104_P1')")
    db.commit()
    db.close()
    # a recording renamed on the pipeline (animals in another order)
    _put(old / "Multi" / "20250101_CNT0105,CNT0106,CNT0000_P2.mkv")
    _put(d["MULTI_ANIMAL_SOURCE"] / "20250101_CNT0106,CNT0105,CNT0000_P2.mkv")
    # a cut video waiting in a GPU machine's claim folder (a dot-folder)
    _put(old / "Unanalyzed" / "20250101_CNT0107_P1.mkv")
    _put(d["SINGLE_ANIMAL_OUTPUT"] / ".inflight" / "NODE-A" / "20250101_CNT0107_P1.mp4")

    rows = _rows(old, nas)

    assert rows["20250101_CNT0101_P1.mp4"]["status"] == ls.SAME_FILE
    assert rows["20250101_CNT0102_P1.mp4"]["status"] == ls.DIFFERENT_SIZE
    assert rows["20250101_CNT0103_P1.mp4"]["status"] == ls.HAS_SESSION
    assert rows["20250101_CNT0104_P1.mp4"]["status"] == ls.HAS_SESSION
    assert rows["20250101_CNT0105,CNT0106,CNT0000_P2.mkv"]["status"] == ls.HAS_ALL_SESSIONS
    assert rows["20250101_CNT0107_P1.mkv"]["status"] == ls.HAS_SESSION
    assert not any(r["destination"] for r in rows.values())


def test_names_a_person_must_look_at_are_left_alone(tmp_path, monkeypatch):
    old, nas, _ = _setup(tmp_path, monkeypatch)
    _put(old / "x" / (REC + "_P4(LONG).mkv"))
    _put(old / "x" / "20251222_CNT0401,CNT407,CNT0000_P1.mkv")
    _put(old / "x" / "20210924_ABS2_13-15_f.mp4")
    rows = _rows(old, nas)
    assert rows[REC + "_P4(LONG).mkv"]["status"] == ls.NOT_PIPELINE_NAME
    assert rows["20251222_CNT0401,CNT407,CNT0000_P1.mkv"]["status"] == ls.BAD_ID
    assert rows["20210924_ABS2_13-15_f.mp4"]["status"] == ls.NOT_PIPELINE_NAME
    assert not any(r["destination"] for r in rows.values())


def test_recordings_come_in_and_cut_videos_never_do(tmp_path, monkeypatch):
    # WHY: the input to MouseReach is the recording; the pipeline's own cut
    # records which recording each single came from (2026-10-23).
    old, nas, d = _setup(tmp_path, monkeypatch)
    # Pillar recording, nothing cut: the intake
    _put(old / "a" / "20250102_CNT0101,CNT0102,CNT0000_P1.mkv")
    # Flat recording and both its cut videos: the recording to the intake (Flat
    # trays go where Pillar goes); the cut videos stay -- it is cut again
    _put(old / "b" / (REC + "_F1.mkv"))
    _put(old / "c" / "20260716_CNT0501_F1.mp4")
    _put(old / "c" / "20260716_CNT0502_F1.mp4")
    # A cut video whose recording is nowhere: named for the rebuild route
    _put(old / "c" / "20250104_CNT0101_P1.mp4")

    rows = _rows(old, nas)
    dest = {n: r["destination"] for n, r in rows.items()}

    assert dest["20250102_CNT0101,CNT0102,CNT0000_P1.mkv"] == str(
        d["MULTI_ANIMAL_SOURCE"] / "20250102_CNT0101,CNT0102,CNT0000_P1.mkv")
    assert dest[REC + "_F1.mkv"] == str(d["MULTI_ANIMAL_SOURCE"] / (REC + "_F1.mkv"))
    assert rows["20260716_CNT0501_F1.mp4"]["status"] == ls.RECORDING_COMES_IN
    assert dest["20260716_CNT0501_F1.mp4"] == ""
    assert rows["20250104_CNT0101_P1.mp4"]["status"] == ls.NO_RECORDING
    assert dest["20250104_CNT0101_P1.mp4"] == ""


def test_a_copy_is_verified_and_never_replaces_a_file(tmp_path, monkeypatch):
    old, nas, d = _setup(tmp_path, monkeypatch)
    src = _put(old / "20250102_CNT0101_P1.mp4", b"video bytes")
    ledger = tmp_path / "ledger.jsonl"
    row = {"old_path": str(src), "destination": str(d["SINGLE_ANIMAL_OUTPUT"] / src.name)}

    assert ls.bring_in(row, ledger) == "copied"
    assert (d["SINGLE_ANIMAL_OUTPUT"] / src.name).read_bytes() == b"video bytes"
    entry = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert entry["bytes"] == 11 and len(entry["sha256"]) == 64
    assert src.read_bytes() == b"video bytes"                      # old storage untouched

    (d["SINGLE_ANIMAL_OUTPUT"] / src.name).write_bytes(b"other")
    assert ls.bring_in(row, ledger).startswith("not copied: a file of that name")
    assert (d["SINGLE_ANIMAL_OUTPUT"] / src.name).read_bytes() == b"other"


def test_a_copy_that_does_not_match_is_removed(tmp_path, monkeypatch):
    old, nas, d = _setup(tmp_path, monkeypatch)
    src = _put(old / "20250102_CNT0101_P1.mp4", b"video bytes")
    import mousereach.pipeline.fsutil as fsutil
    calls = iter(["aaa", "bbb"])
    monkeypatch.setattr(fsutil, "sha256_file", lambda p: next(calls))
    row = {"old_path": str(src), "destination": str(d["SINGLE_ANIMAL_OUTPUT"] / src.name)}

    assert "SHA-256" in ls.bring_in(row, tmp_path / "ledger.jsonl")
    assert list(d["SINGLE_ANIMAL_OUTPUT"].iterdir()) == []         # neither copy nor .part
    assert not (tmp_path / "ledger.jsonl").exists()


def test_the_command_reports_brings_in_and_then_reads_zero(tmp_path, monkeypatch):
    old, nas, d = _setup(tmp_path, monkeypatch)
    src = _put(old / "Unanalyzed" / "20250102_CNT0101,CNT0102,CNT0000_P1.mkv", b"v1")
    _put(old / "Unanalyzed" / "notes.mp4", b"x")
    t = time.time() - 3600
    os.utime(src, (t, t))

    assert ls.main(["--from", str(old), "--label", "old"]) == 0
    summary = json.loads((nas / "legacy_inventory" / "old.summary.json").read_text(encoding="utf-8"))
    assert summary["only_here"] == 1
    assert not (d["MULTI_ANIMAL_SOURCE"] / src.name).exists()      # report only

    assert ls.main(["--from", str(old), "--label", "old", "--bring-in"]) == 0
    assert (d["MULTI_ANIMAL_SOURCE"] / src.name).read_bytes() == b"v1"
    assert abs(src.stat().st_mtime - t) < 2                        # old file not touched
    summary = json.loads((nas / "legacy_inventory" / "old.summary.json").read_text(encoding="utf-8"))
    assert summary["only_here"] == 0

    assert ls.main(["--from", str(old), "--label", "old"]) == 0   # a re-run agrees
    summary = json.loads((nas / "legacy_inventory" / "old.summary.json").read_text(encoding="utf-8"))
    assert summary["only_here"] == 0
    assert summary["by_status"] == {ls.SAME_FILE: 1, ls.NOT_PIPELINE_NAME: 1}


def test_overlapping_roots_are_refused(tmp_path, monkeypatch):
    old, nas, _ = _setup(tmp_path, monkeypatch)
    assert ls.main(["--from", str(nas / "Unanalyzed")]) == 1
    assert ls.main(["--from", str(nas)]) == 1


def test_copies_in_several_old_folders_are_brought_in_once(tmp_path, monkeypatch):
    """12 recordings sat in two old folders at once (2026-10-22). Same size:
    one video, brought in once, and "only here" reaches zero. Different
    sizes: a person decides which is right."""
    old, nas, d = _setup(tmp_path, monkeypatch)
    name = "20250105_CNT0101,CNT0102,CNT0000_P1.mkv"
    _put(old / "a" / name, b"same")
    _put(old / "b" / name, b"same")
    other = "20250106_CNT0101,CNT0102,CNT0000_P1.mkv"
    _put(old / "a" / other, b"one")
    _put(old / "b" / other, b"other")

    rows = ls.inventory(old, ls.PipelineIndex(nas))
    by = {(r["folder"], r["name"]): r for r in rows}
    assert by[("a", name)]["status"] == ls.ONLY_HERE
    assert by[("b", name)]["status"] == ls.SAME_AS_ANOTHER
    assert by[("b", name)]["same_as"] == by[("a", name)]["old_path"]
    assert {by[(f, other)]["status"] for f in "ab"} == {ls.COPIES_DIFFER}

    assert ls.main(["--from", str(old), "--label", "old", "--bring-in"]) == 0
    assert (d["MULTI_ANIMAL_SOURCE"] / name).read_bytes() == b"same"
    assert not (d["MULTI_ANIMAL_SOURCE"] / other).exists()
    summary = json.loads((nas / "legacy_inventory" / "old.summary.json").read_text(encoding="utf-8"))
    assert summary["only_here"] == 0
    assert summary["by_status"][ls.SAME_FILE] == 2
    assert summary["by_status"][ls.COPIES_DIFFER] == 2
