"""A segment counts as answered only when a person's answer is about its frames.

WHY: the release check, the review gate and the queue views counted an answer by
the segment NUMBER it was written against, while the results step matches it by
FRAMES. After a person re-cut a video, an old answer for "segment 7" -- about
other footage -- counted as answering today's segment 7: the video left review,
the results step correctly ignored the answer, and today's segment 7 reached the
data with nobody having watched it. A nightly check then sent the same videos
back every night (2026-10-21). And a segmentation a person made is logged as
made, however short its segments (Logan, 2026-10-21).
"""
import json

from mousereach.review import causal_review_io as io
from mousereach.review import truth_resolver as tr
from mousereach.review.triage_status import resolved_segments, triage_status
from mousereach.watcher import review_gate

STEM = "20240101_ABC0101_P1"

# Today's segmentation: a person added a boundary at 150, so everything after it
# is one number later than in the cut the first answers were written against.
TODAY = {"boundaries": [0, 100, 150, 300, 450], "boundary_source": "human"}
# An answer written against the OLD cut, where segment 3 was frames 300-449 --
# today's segment 4 -- plus one written against today's cut for segment 3.
REVIEW = {"segments": [
    {"segment_num": 3, "segment_span": {"start": 300, "end": 449},
     "human": {"outcome": "retrieved"}, "answers": {"reviewed": True}},
]}


def test_an_old_answer_counts_for_the_segment_whose_frames_it_is_about():
    assert resolved_segments(REVIEW, TODAY) == {4}
    assert resolved_segments(REVIEW) == {3}          # the old, number-only reading


def test_a_placeholder_is_never_an_answer():
    doc = {"segments": [{"segment_num": 1, "segment_span": {"start": 0, "end": 99},
                         "answers": {"reviewed": False}}]}
    assert resolved_segments(doc, TODAY) == set()


def _bundle(tmp_path, flagged, review):
    b = tmp_path / STEM
    b.mkdir()
    (b / f"{STEM}_segments.json").write_text(json.dumps(TODAY), encoding="utf-8")
    (b / f"{STEM}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "triaged" if n in flagged else "retrieved"}
        for n in (1, 2, 3, 4)]}), encoding="utf-8")
    (b / f"{STEM}_causal_review.json").write_text(json.dumps(review), encoding="utf-8")
    return b


def test_the_release_check_does_not_count_an_answer_about_other_frames(tmp_path):
    """The nightly loop: today's segment 3 is triaged; the only answer numbered 3
    is about today's segment 4. The bundle must stay until 3 is watched."""
    b = _bundle(tmp_path, flagged={3}, review=REVIEW)
    st = triage_status(b, STEM)
    assert 3 in st.unresolved
    assert not st.fully_resolved


def test_the_release_check_counts_an_answer_about_the_right_frames(tmp_path):
    b = _bundle(tmp_path, flagged={4}, review=REVIEW)
    assert triage_status(b, STEM).fully_resolved


def test_the_gate_counts_answers_by_frames(tmp_path, monkeypatch):
    b = _bundle(tmp_path, flagged={3}, review=REVIEW)
    monkeypatch.setattr(io, "resolve_review_path",
                        lambda vid, d=None: b / f"{STEM}_causal_review.json")
    assert review_gate._review_resolved_segments(STEM, b) == {4}


# ------------------------------------------------- a person's short segments

SHORT = {"boundaries": [0, 100, 103, 300], "boundary_source": "human"}   # segment 2: 3 frames
SHORT_REVIEW = {"segments": [
    {"segment_num": 2, "segment_span": {"start": 100, "end": 102},
     "human": {"outcome": "untouched"}, "answers": {"reviewed": True}},
]}


def test_a_short_segment_a_person_made_is_real():
    current = io.segments_from_segmentation(SHORT)
    assert io.answered_segments(SHORT_REVIEW, current, human_segmentation=True) == {2}
    assert resolved_segments(SHORT_REVIEW, SHORT) == {2}


def test_a_short_segment_the_algorithm_made_is_still_an_artifact():
    algo = dict(SHORT, boundary_source="algorithm")
    current = io.segments_from_segmentation(algo)
    assert io.answered_segments(SHORT_REVIEW, current, human_segmentation=False) == set()
    assert resolved_segments(SHORT_REVIEW, algo) == set()


def test_the_results_step_applies_an_answer_on_a_short_human_segment():
    current = io.segments_from_segmentation(SHORT)
    out = tr._seg_overrides_from_review(SHORT_REVIEW, "human_review", current,
                                        human_segmentation=True)
    assert out[2]["outcome"] == "untouched"
    assert tr._seg_overrides_from_review(SHORT_REVIEW, "human_review", current) == {}
