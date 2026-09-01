"""The outdated two-way door and the bulk release check.

'outdated' must not be a one-way state (declaration accidents marked 1,255
rows in August 2026 with no way back), and release completeness must be
judged on human.outcome -- answers.reviewed has never been written by
anything and is False on every review file in existence.
"""

import json

from mousereach.watcher.db import VIDEO_TRANSITIONS
from mousereach.review.release_cli import _bundle_dirs, _completeness


def test_outdated_can_return_to_archived():
    assert "archived" in VIDEO_TRANSITIONS["outdated"]


def _bundle(tmp_path, stem, segments):
    b = tmp_path / stem
    b.mkdir()
    (b / f"{stem}_causal_review.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8")
    return b


def test_completeness_counts_human_outcome_not_answers_reviewed():
    # answers.reviewed is False everywhere and must be irrelevant.
    segs = [
        {"segment_num": 1, "human": {"outcome": "retrieved"},
         "answers": {"reviewed": False}},
        {"segment_num": 2, "human": {"outcome": "missed"},
         "answers": {"reviewed": False}},
    ]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        b = _bundle(Path(d), "20990101_ABC9901_P1", segs)
        assert _completeness(b, "20990101_ABC9901_P1") == (2, 2)


def test_partial_review_is_not_complete(tmp_path):
    segs = [
        {"segment_num": 1, "human": {"outcome": "retrieved"}},
        {"segment_num": 2, "human": {"outcome": None}},
        {"segment_num": 3, "human": {}},
    ]
    b = _bundle(tmp_path, "20990101_ABC9901_P2", segs)
    assert _completeness(b, "20990101_ABC9901_P2") == (1, 3)


def test_missing_review_file_is_zero(tmp_path):
    b = tmp_path / "20990101_ABC9901_P3"
    b.mkdir()
    assert _completeness(b, "20990101_ABC9901_P3") == (0, 0)


def test_bundle_dirs_skip_dot_directories(tmp_path):
    (tmp_path / "20990101_ABC9901_P1").mkdir()
    (tmp_path / ".claims").mkdir()
    (tmp_path / "a_file.txt").write_text("x", encoding="utf-8")
    names = [b.name for b in _bundle_dirs(tmp_path)]
    assert names == ["20990101_ABC9901_P1"]
