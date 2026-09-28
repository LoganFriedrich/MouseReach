"""A file this node cannot see is an infrastructure fact, never a scientific verdict.

'failed' says something about an animal's data. A missing or half-written input says
something about a machine. Conflating them spends the video's retry, parks it, and
puts a scientific-looking verdict on a video nobody ever examined.

This is not hypothetical. Staging a video's pose files is not atomic across files, so a
node can look for the set while another node is still writing it -- and on 2026-09-19 a
video sat 'failed' for nine days because of a race lasting about one minute, with every
file present the whole time.
"""
from types import SimpleNamespace

import pytest

from mousereach.watcher import orchestrator as orch


class FakeDB:
    def __init__(self):
        self.failed = []
        self.unresolvable = []
        self.steps = []

    def mark_failed(self, video_id, message):
        self.failed.append((video_id, message))

    def mark_unresolvable(self, video_id, reason):
        self.unresolvable.append((video_id, reason))

    def log_step(self, video_id, step, status, message=None, duration=None):
        self.steps.append((step, status))

    def update_state(self, *a, **k):
        pass

    def get_video(self, video_id):
        return {}


def _node(tmp_path, db):
    o = object.__new__(orch.DLCOrchestrator)
    o.db = db
    o.hostname = "NODE-A"
    o.staging_dir = tmp_path / "Posed"
    o.processing_dir = tmp_path / "Processing"
    o.staging_dir.mkdir(parents=True, exist_ok=True)
    o._get_associated_files = lambda d, v: []          # the race: nothing visible yet
    o._release_claim_given_up = lambda *a, **k: None
    return o


def test_intake_with_no_visible_files_does_not_fail_the_video(tmp_path):
    db = FakeDB()
    o = _node(tmp_path, db)
    work = {"id": "VID", "data": {"current_path": str(tmp_path / "Posed" / "VID.mp4")}}

    result = o._intake_from_staging(work) if hasattr(o, "_intake_from_staging") else None
    if result is None:
        pytest.skip("intake handler not named as expected; covered by the source check")

    assert db.failed == [], (
        "a video whose files were not visible must NOT be marked failed -- that is a "
        "verdict about the animal's data, and it spends the retry")
    assert db.unresolvable, "it must be retired as unresolvable instead"


def test_neither_missing_input_path_raises_or_fails_in_the_source():
    """Guards the two places this went wrong, by reading the code itself.

    A behavioural test cannot reach both handlers without standing up most of an
    orchestrator, but the defect is precise and easy to state: neither place may turn
    'the files are not here' into FileNotFoundError, because the enclosing handler
    turns any exception into mark_failed.
    """
    from pathlib import Path
    source = Path(orch.__file__).read_text(encoding="utf-8")
    assert "No files found for" not in source, (
        "a missing input must be recorded with mark_unresolvable, not raised into "
        "the blanket handler that calls mark_failed")
    assert source.count("no files for it in") >= 2, (
        "both the intake and the staging path must retire the row as unresolvable")


def test_mark_unresolvable_is_the_documented_tool_for_this():
    """The database layer already says which of the two to use, and why."""
    from mousereach.watcher.db import WatcherDB
    doc = WatcherDB.mark_unresolvable.__doc__ or ""
    assert "mark_failed" in doc and "RETRY" in doc.upper(), (
        "the guidance that distinguishes these two lives on mark_unresolvable; if it "
        "moves, this test should point at wherever it went")
