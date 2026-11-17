"""The clear marker is all-or-nothing, and a save leaves no empty archive.

Marker: a crash between creating the marker and writing it must not leave an
empty marker (the record of who cleared the video and when would be lost while
every reader still treats the video as cleared).
Archive: saving cuts for a video with nothing to archive must not leave an
empty archive folder behind.
"""
import json
import os
from pathlib import Path

import pytest

from mousereach.review import release_cli


def test_marker_crash_leaves_no_empty_marker(tmp_path, monkeypatch):
    marker = tmp_path / "x.deep_review_cleared"

    def boom(fd):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        release_cli._write_marker_once(marker, {"by": "someone"})
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == []  # no temp file left either


def test_marker_written_whole_and_create_only(tmp_path):
    marker = tmp_path / "x.deep_review_cleared"
    assert release_cli._write_marker_once(marker, {"by": "a"}) is True
    assert json.loads(marker.read_text())["by"] == "a"
    assert release_cli._write_marker_once(marker, {"by": "b"}) is False
    assert json.loads(marker.read_text())["by"] == "a"
    assert [p.name for p in tmp_path.iterdir()] == [marker.name]


def test_marker_never_creates_the_folder(tmp_path):
    gone = tmp_path / "no_such_bundle" / "m.json"
    assert release_cli._write_marker_once(gone, {"by": "a"}) is False
    assert not gone.parent.exists()


def test_invalidate_makes_archive_only_when_it_moves_something(tmp_path):
    pytest.importorskip("qtpy")
    from mousereach.review.fix_segmentation_widget import invalidate_stale_downstream
    seg = tmp_path / "v_segments.json"
    seg.write_text("{}")
    arch = tmp_path / "_archived" / "before"
    assert invalidate_stale_downstream(seg, "v", arch) == []
    assert not arch.exists()
    (tmp_path / "v_reaches.json").write_text("{}")
    assert invalidate_stale_downstream(seg, "v", arch) == ["v_reaches.json"]
    assert (arch / "v_reaches.json").is_file()
