"""Every check of "did a person answer this pellet?" uses is_answered.

Three places counted a record as answered when its ``human.outcome`` was set:
the re-segmentation window's release-on-save, the bulk release check and the
census. A pellet left with no outcome picked is saved with the algorithm's
"triaged" in its outcome (and ``reviewed: False``); a pellet sent back with
"are you sure?" keeps its old outcome under ``reviewed: False``. Both are open
questions, and those three places released or counted them as answered.
"""
import json

from mousereach.census.review_completeness import completeness_of_document
from mousereach.review.causal_review_widget import segment_answered
from mousereach.review.release_cli import _completeness, classify_queue

STEM = "20990101_ABC9901_P1"


def _answer(n, start=None):
    rec = {"segment_num": n, "human": {"outcome": "untouched"}, "answers": {}}
    if start is not None:
        rec["segment_span"] = {"start": start, "end": start + 99}
    return rec


def _left_open(n):
    """A pellet saved with no outcome picked, as the review window writes it."""
    return {"segment_num": n, "human": {"outcome": "triaged"},
            "answers": {"reviewed": False}}


def _rechecked(n):
    """A pellet sent back with "are you sure?": old outcome kept, marked open."""
    return {"segment_num": n, "human": {"outcome": "retrieved"},
            "answers": {"reviewed": False, "recheck": {
                "asked_at": "2099-01-02T10:00:00", "question": "sure?"}}}


def _bundle(root, review, seg_doc=None, stem=STEM):
    b = root / stem
    b.mkdir(parents=True)
    (b / f"{stem}_causal_review.json").write_text(json.dumps({"segments": review}),
                                                  encoding="utf-8")
    if seg_doc is not None:
        (b / f"{stem}_segments.json").write_text(json.dumps(seg_doc), encoding="utf-8")
    return b


def test_release_check_does_not_count_a_pellet_left_open(tmp_path):
    b = _bundle(tmp_path, [_answer(1), _left_open(2)])
    assert _completeness(b, STEM) == (1, 2)


def test_release_check_does_not_count_a_rechecked_pellet(tmp_path):
    b = _bundle(tmp_path, [_answer(1), _rechecked(2)])
    assert _completeness(b, STEM) == (1, 2)


def test_release_check_matches_answers_to_the_current_cuts_by_frames(tmp_path):
    # Two answers about frames 0-99 and 100-199; the cuts now make segments
    # at 1000-1099 and 1100-1199, so neither answer is about today's footage.
    seg = {"boundaries": [1000, 1100, 1200], "boundary_source": "human"}
    b = _bundle(tmp_path, [_answer(1, start=0), _answer(2, start=100)], seg)
    assert _completeness(b, STEM) == (0, 2)


def test_bulk_release_leaves_a_bundle_with_an_open_pellet(tmp_path):
    _bundle(tmp_path, [_answer(1), _left_open(2)], stem="20990101_ABC9901_E1")
    cls = classify_queue(tmp_path)
    assert cls["complete"] == []
    assert cls["partial"] == [("20990101_ABC9901_E1", 1, 2)]


def test_release_on_save_needs_every_segment_answered():
    from mousereach.review.causal_review_widget import review_complete
    assert review_complete([_answer(1), _answer(2)])
    assert not review_complete([_answer(1), _left_open(2)])
    assert not review_complete([_answer(1), _rechecked(2)])
    assert not review_complete([])


def test_a_pellet_left_open_in_this_session_is_not_done():
    seg = {"segment_num": 2}
    assert not segment_answered(seg, {2: _left_open(2)})
    assert segment_answered(seg, {2: _answer(2)})


def test_census_counts_answers_not_outcomes():
    doc = {"segments": [_answer(1), _left_open(2), _rechecked(3)]}
    assert completeness_of_document(doc)[:2] == (1, 3)


def test_recheck_never_reopens_a_pellet_nobody_decided(tmp_path, monkeypatch):
    """An older record kept the algorithm's "triaged" with no reviewed flag:
    nobody decided it, so there is no answer to ask about again."""
    import mousereach.watcher.route_cli as rc
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{STEM}_segments.json").write_text(
        json.dumps({"boundaries": [100, 200, 300]}), encoding="utf-8")
    undecided = {"segment_num": 1, "segment_span": {"start": 100, "end": 199},
                 "human": {"outcome": "triaged"}, "answers": {}}
    (folder / f"{STEM}_causal_review.json").write_text(
        json.dumps({"segments": [undecided]}), encoding="utf-8")
    assert rc.reopen_answered(STEM, folder, [1]) == []
