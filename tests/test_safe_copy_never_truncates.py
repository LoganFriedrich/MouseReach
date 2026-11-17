"""A copy interrupted mid-file never leaves a truncated file under the final name.

WHY: safe_copy used to copy straight onto the destination name. A copy cut off
part-way (the process killed, the machine restarted, the share dropped) left a
truncated video under its real name; the next pass found a file of the
expected name, took it for the video and could queue it for pose. The copy now
goes to '<name>.part' and is renamed into place only when complete, and no
picker takes a '.part' file for one of a video's files.
"""
import shutil

import pytest

import mousereach.archive.core as core
from mousereach.watcher import transfer

V = "20990101_ZZZ0101_P1"


def test_an_interrupted_copy_leaves_nothing_under_the_final_name(tmp_path, monkeypatch):
    src = tmp_path / "src" / f"{V}.mp4"
    src.parent.mkdir()
    src.write_bytes(b"x" * 1000)
    dst = tmp_path / "dst" / f"{V}.mp4"

    def half_then_killed(a, b, *k, **kw):
        with open(b, "wb") as f:
            f.write(b"x" * 400)
        raise KeyboardInterrupt          # not an OSError: nothing cleans up

    monkeypatch.setattr(shutil, "copy2", half_then_killed)
    with pytest.raises(KeyboardInterrupt):
        transfer.safe_copy(src, dst)
    assert not dst.exists(), "a truncated copy sits under the video's real name"


def test_a_completed_copy_is_whole_and_leaves_no_temporary_file(tmp_path):
    src = tmp_path / f"{V}.mp4"
    src.write_bytes(b"y" * 1000)
    dst = tmp_path / "out" / f"{V}.mp4"
    assert transfer.safe_copy(src, dst)
    assert dst.read_bytes() == b"y" * 1000
    assert [p.name for p in dst.parent.iterdir()] == [dst.name]


def test_a_failed_copy_leaves_no_temporary_file(tmp_path, monkeypatch):
    src = tmp_path / f"{V}.mp4"
    src.write_bytes(b"y" * 1000)
    dst = tmp_path / "out" / f"{V}.mp4"

    def fails(a, b, *k, **kw):
        with open(b, "wb") as f:
            f.write(b"y" * 10)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copy2", fails)
    assert transfer.safe_copy(src, dst) is False
    assert not any(dst.parent.iterdir())


def test_filing_never_takes_a_leftover_part_file(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    (proc / f"{V}.mp4").write_bytes(b"v")
    (proc / f"{V}_features.json").write_text("{}")
    (proc / f"{V}.mp4.part").write_bytes(b"half")
    monkeypatch.setattr(core.Paths, "PROCESSING", proc)
    names = {p.name for p in core.get_video_files(V)}
    assert f"{V}.mp4.part" not in names
    assert {f"{V}.mp4", f"{V}_features.json"} <= names
