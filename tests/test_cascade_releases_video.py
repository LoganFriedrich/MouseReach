"""The outcome cascade never leaves its video handle open on exit.

WHY: the CV artifact gate keeps a one-entry VideoCapture cache for
performance, and it used to be released only on the watcher's ARCHIVE path.
A video the gate ROUTED TO REVIEW kept its mp4 open until the next video
evicted it, so the routing move failed with "being used by another process"
and the mp4 silently stayed behind in Processing (43 review bundles missing
their mp4 in one day, 2026-09-10). The cascade entry now releases the cache
on every exit -- including the exception path, which this test pins.
"""
import pytest

import mousereach.outcomes.v6_cascade.cv_artifact_gate as gate
from mousereach.outcomes.v6_cascade.detector import detect_outcomes_v6_cascade


class DummyCap:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_cache_released_even_when_the_cascade_raises(monkeypatch):
    cap = DummyCap()
    monkeypatch.setitem(gate._cap_cache, "path", "X:/somewhere.mp4")
    monkeypatch.setitem(gate._cap_cache, "cap", cap)

    with pytest.raises(Exception):
        # Garbage inputs: the impl must raise, and the wrapper's finally must
        # still close the cached capture.
        detect_outcomes_v6_cascade(None, [(0, 10)], [(1, 2)], video_id="t")

    assert cap.released
    assert gate._cap_cache["cap"] is None
    assert gate._cap_cache["path"] is None
