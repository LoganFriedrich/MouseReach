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


def test_routing_reason_gates_segmentation_release(tmp_path):
    from mousereach.review.release_cli import (
        _names_segmentation, _routing_reason, _boundary_source_human)
    stem = "20990101_ABC9901_P1"
    b = tmp_path / stem
    b.mkdir()
    (b / f"{stem}_routing.json").write_text(json.dumps(
        {"routed_reason": "reviewer escalated from triage: bad segmentation"}),
        encoding="utf-8")
    (b / f"{stem}_segments.json").write_text(json.dumps(
        {"boundary_source": "human", "boundaries": [1, 2]}), encoding="utf-8")
    assert _routing_reason(b, stem).startswith("reviewer escalated")
    assert _names_segmentation(_routing_reason(b, stem))
    assert _boundary_source_human(b, stem)
    # A QC-routed bundle must NOT read as segmentation-releasable: the clear
    # marker is a blanket human-clear token the gate honors, so a cut-fix
    # releasing it would clear a concern nobody addressed.
    assert not _names_segmentation("qc_needs_review")
    assert not _names_segmentation("")
    assert _names_segmentation("bench_disagreement_segment_mislabel")


def test_ghost_segment_flags_are_ignored(tmp_path):
    """A triage flag naming a segment that does not exist is unanswerable and
    must not hold the bundle forever (seven live bundles were flagged on
    segment 21 of a 20-segment video)."""
    from mousereach.review.triage_status import triage_status
    stem = "20990101_ABC9905_P1"
    (tmp_path / f"{stem}_segments.json").write_text(json.dumps({
        "overall_confidence": 0.9,
        "boundaries": list(range(0, 2100, 100)),  # 21 boundaries -> 20 segments
    }), encoding="utf-8")
    (tmp_path / f"{stem}_pellet_outcomes.json").write_text(json.dumps({
        "segments": [
            {"segment_num": 20, "outcome": "triaged", "flagged_for_review": True},
            {"segment_num": 21, "outcome": "triaged", "flagged_for_review": True},
        ],
    }), encoding="utf-8")
    st = triage_status(tmp_path, stem)
    assert 20 in st.triaged
    assert 21 not in st.triaged  # the ghost is dropped, not waited on forever


def test_classify_and_release_shared_engine(tmp_path):
    """The GUI button and the CLI share classify_queue/release_finished; the
    release must touch exactly the finished bundles and nothing else."""
    from mousereach.review.release_cli import classify_queue, release_finished

    def mk(stem, review_segments=None, seg_doc=None, routing=None):
        b = tmp_path / stem
        b.mkdir()
        if review_segments is not None:
            (b / f"{stem}_causal_review.json").write_text(
                json.dumps({"segments": review_segments}), encoding="utf-8")
        if seg_doc is not None:
            (b / f"{stem}_segments.json").write_text(
                json.dumps(seg_doc), encoding="utf-8")
        if routing is not None:
            (b / f"{stem}_routing.json").write_text(
                json.dumps({"routed_reason": routing}), encoding="utf-8")
        return b

    mk("20990101_ABC9901_P1",
       review_segments=[{"segment_num": 1, "human": {"outcome": "retrieved"}}])
    mk("20990101_ABC9902_P1", seg_doc={"boundary_source": "human"},
       routing="escalated: bad segmentation")
    mk("20990101_ABC9903_P1", seg_doc={"boundary_source": "human"},
       routing="qc_needs_review")
    mk("20990101_ABC9904_P1",
       review_segments=[{"segment_num": 1, "human": {"outcome": None}},
                        {"segment_num": 2, "human": {"outcome": "missed"}}])

    c = classify_queue(tmp_path)
    assert [s for s, *_ in c["complete"]] == ["20990101_ABC9901_P1"]
    assert [s for s, *_ in c["fixed_release"]] == ["20990101_ABC9902_P1"]
    assert [s for s, *_ in c["fixed_held"]] == ["20990101_ABC9903_P1"]

    n, failures = release_finished(tmp_path, c)
    assert n == 2 and not failures
    assert (tmp_path / "20990101_ABC9901_P1"
            / "20990101_ABC9901_P1_deep_review_cleared.json").is_file()
    assert (tmp_path / "20990101_ABC9902_P1"
            / "20990101_ABC9902_P1_deep_review_cleared.json").is_file()
    # The held-back and partial bundles must be untouched.
    assert not (tmp_path / "20990101_ABC9903_P1"
                / "20990101_ABC9903_P1_deep_review_cleared.json").exists()
    assert not (tmp_path / "20990101_ABC9904_P1"
                / "20990101_ABC9904_P1_deep_review_cleared.json").exists()
