"""The features file is written whole or not at all.

WHY: it was written in place (open for writing, then json.dump). Opening for
writing empties the file first, so a crash, a full disk or a value that
cannot be written as JSON part-way through left a truncated
<video>_features.json in Processing, or an emptied previous one. It now goes
through fsutil.write_json_atomic like every other output: a temporary file
swapped in, so the old file stays exactly as it was when the write fails.

Built under tmp_path with invented ids; no real folder or database.
"""
from pathlib import Path
from unittest import mock

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator

V = "20990101_ZZZ0101_P8"
OLD = '{"an earlier": "features file"}'


@pytest.fixture
def server(tmp_path, monkeypatch):
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o.staging_dir = None
    o.db.register_video(video_id=V, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(V, st)
    for s in ("_reaches.json", "_pellet_outcomes.json", "_processing_manifest.json"):
        (proc / f"{V}{s}").write_text("{}")
    (proc / f"{V}_features.json").write_text(OLD)
    return o


class _Unwritable:
    """An extractor whose result cannot be written as JSON part-way through:
    the first key writes, the second does not."""
    VERSION = "test"

    def extract(self, *a, **k):
        return mock.Mock(n_segments=1,
                         to_dict=lambda: {"written": 1, "not writable": object()})

    def extract_session_only(self, *a, **k):
        return self.extract()


@pytest.mark.parametrize("session_only", [False, True])
def test_a_write_that_fails_part_way_leaves_the_old_file_whole(server, session_only):
    with mock.patch("mousereach.kinematics.core.feature_extractor.FeatureExtractor",
                    _Unwritable):
        why = server._make_features(V, server.processing_dir, Path("pose.h5"),
                                    session_only)
    assert why, "a failed write must hold the video"
    feats = server.processing_dir / f"{V}_features.json"
    assert feats.read_text() == OLD, "the features file was truncated or emptied"
    # No temporary file is left for a mover to carry along.
    assert sorted(p.name for p in server.processing_dir.iterdir()
                  if p.name.startswith(".")) == []
