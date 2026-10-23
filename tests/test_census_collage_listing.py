"""The census finds collages by NAME, without asking the share about each file.

WHY (2026-10-23): collect_collages called is_file() on every entry under its
roots -- one round trip to the share per file, about 164,000 under one archive
root -- and the nightly census ran past its hour. It now uses the walk's own
listing. These tests pin that the result is unchanged: recursive, files only,
collage-shaped names only. Built under tmp_path.
"""
from pathlib import Path

from mousereach.census.expected_sessions import collect_collages


def test_finds_collages_in_subfolders_and_skips_everything_else(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    keep = [tmp_path / "20250101_CNT0101,CNT0102_P1.mkv",
            tmp_path / "sub" / "deeper" / "20250102_CNT0103,CNT0104_P1.MP4"]
    for p in keep:
        p.write_bytes(b"")
    (tmp_path / "20250101_CNT0101_P1.mp4").write_bytes(b"")          # a single, not a collage
    (tmp_path / "sub" / "a,b_notes.txt").write_bytes(b"")             # wrong kind of file
    (tmp_path / "sub" / "20250103_X,Y_P1.mkv").mkdir()                # a FOLDER named like one
    got = sorted(collect_collages([tmp_path, tmp_path / "missing"]))
    assert got == sorted(keep)


def test_no_file_is_asked_about_one_by_one(tmp_path, monkeypatch):
    (tmp_path / "20250101_A,B_P1.mkv").write_bytes(b"")
    def refuse(self, *a, **k):
        raise AssertionError("is_file() called per entry")
    monkeypatch.setattr(Path, "is_file", refuse)
    assert len(collect_collages([tmp_path])) == 1
