"""Routing takes a video's files by name from the folder listing.

The listing answers "is this a file?" itself; asking the share again for every
entry of a cohort's Analyzed folder cost minutes per video (2026-10-24).
"""
from mousereach.watcher.review_routing import _bundle_files


def test_only_this_videos_files_and_only_files(tmp_path):
    for name in ("20250101_CNT0312_P2.mp4", "20250101_CNT0312_P2_segments.json",
                 "20250101_CNT0312_P2DLC_resnet101_X_100000.h5",
                 "20250101_CNT0312_P21.mp4", "20250101_CNT0312_P21_segments.json",
                 "20250101_CNT0313_P2.mp4"):
        (tmp_path / name).write_text("x")
    (tmp_path / "20250101_CNT0312_P2_dir").mkdir()          # a folder is not a file
    got = sorted(p.name for p in _bundle_files(tmp_path, "20250101_CNT0312_P2"))
    assert got == ["20250101_CNT0312_P2.mp4",
                   "20250101_CNT0312_P2DLC_resnet101_X_100000.h5",
                   "20250101_CNT0312_P2_segments.json"]


def test_a_missing_folder_has_no_files(tmp_path):
    assert _bundle_files(tmp_path / "gone", "x") == []
