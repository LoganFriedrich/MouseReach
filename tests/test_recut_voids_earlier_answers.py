"""A re-cut voids the answers given before it.

Rule (owner ruling 2026-11-18, narrowing R2): when a person re-cuts a video,
an earlier answer stands only on a pellet whose frames the re-cut did not
change at all. Every other earlier answer -- its pellet moved, even by one
frame, or the files do not show which cuts it was given on -- is void: fitting
it onto the new cuts by frame overlap or by segment number is a guess about
footage nobody judged on the new cuts. It is kept in the file as a record and
dropped: the pellet takes the re-run algorithms' call and is asked only if
they flag it (owner ruling 2026-11-21).

Before: every reader matched earlier answers to the new cuts by frame overlap
(and span-less ones by number), so a person's answer about one stretch of video
was applied to a differently cut pellet.

Now: when a person set the current cuts (the segments file records when, and
the cuts they replaced), an answer given before that time counts only when its
frames are a pellet of the replaced cuts AND the same frames are a current
pellet (causal_review_io.answer_survives_recut). Answers given after, and
every answer on a video whose cuts are the segmenter's own, are matched as
before.

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json

import pytest

import mousereach.review.causal_review_io as io
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

from test_causal_ownership_by_source import VID, _outcomes, run_dir  # noqa: F401

CUT_AT = "2099-03-02T10:00:00"
BEFORE, AFTER = "2099-03-01T09:00:00", "2099-03-02T11:00:00"


# The cuts the person replaced: pellet 1 was 950-1990, pellet 2 1991-2999,
# pellet 3 3000-3999 (unmoved by the re-cut).
REPLACED = [950, 1991, 3000, 4000]


def _seg_doc(human=True, replaced=REPLACED):
    d = {"boundaries": [1000, 2000, 3000, 4000]}
    if human:
        d.update(boundary_source="human", corrected_at=CUT_AT)
        if replaced is not None:
            d["algo_boundaries"] = replaced
    return d


def _rec(span, when, outcome="displaced_sa", num=1):
    r = {"segment_num": num, "pellet_num": num, "reviewer": "reviewer_a", "reviewed_at": when,
         "human": {"outcome": outcome, "causal_reach": None}, "answers": {}}
    if span:
        r["segment_span"] = {"start": span[0], "end": span[1]}
    return r


def _answered(doc, human=True, replaced=REPLACED):
    seg_doc = _seg_doc(human, replaced)
    return io.answered_segments(doc, io.segments_from_segmentation(seg_doc),
                                human_segmentation=human)


def test_an_answer_given_before_a_recut_on_moved_cuts_is_void():
    # given on frames 950-1990; pellet 1 is now 1000-1999
    doc = {"segments": [_rec((950, 1990), BEFORE)]}
    assert _answered(doc) == set()
    restored, unmatched, notes = io.restore_review_for_segments(
        doc, io.segments_from_segmentation(_seg_doc()), human_segmentation=True)
    assert restored == {}
    assert unmatched and unmatched[0]["reviewed_at"] == BEFORE     # kept as a record
    assert any("re-cut voids it" in n for n in notes)


def test_an_answer_on_a_pellet_whose_cuts_did_not_move_stands():
    # pellet 3 is 3000-3999 before and after the re-cut
    doc = {"segments": [_rec((3000, 3999), BEFORE, num=3)]}
    assert _answered(doc) == {3}


def test_an_unmoved_span_that_was_no_pellet_of_the_replaced_cuts_is_void():
    # 2000-2999 is a current pellet, but no pellet of the replaced cuts: the
    # record does not show the cuts it was given on (a save could stamp the
    # current frames on an old answer), so it is treated as moved
    doc = {"segments": [_rec((2000, 2999), BEFORE, num=2)]}
    assert _answered(doc) == set()


def test_without_the_replaced_cuts_on_file_an_earlier_answer_is_void():
    doc = {"segments": [_rec((3000, 3999), BEFORE, num=3)]}
    assert _answered(doc, replaced=None) == set()


def test_an_answer_given_after_the_recut_is_matched_as_usual():
    doc = {"segments": [_rec((1000, 1999), AFTER), _rec((2010, 2990), AFTER, num=2)]}
    assert _answered(doc) == {1, 2}


def test_an_answer_with_no_frames_given_before_a_recut_is_not_applied_by_number():
    doc = {"segments": [_rec(None, BEFORE, num=2)]}
    assert _answered(doc) == set()


def test_on_the_segmenters_own_cuts_matching_is_unchanged():
    doc = {"segments": [_rec((950, 1990), BEFORE)]}
    assert _answered(doc, human=False) == {1}


def test_the_results_drop_an_answer_voided_by_a_recut(run_dir, monkeypatch):
    """Kinematics: the person re-cut the video after answering; the old answer
    (frames 0-120, pellet 1 is now 0-99) is not applied, and the algorithm's
    call is written in its place (owner ruling 2026-11-21: a voided answer is
    dropped, never re-asked). Until then the extractor refused
    (RecutVoidedAnswers) until the pellet was asked again."""
    folder, pose = run_dir
    monkeypatch.setattr(io, "review_dirs", lambda *a, **k: [folder])
    _outcomes(folder, outcome="retrieved")
    (folder / f"{VID}_segments.json").write_text(json.dumps(
        {"boundaries": [0, 100], "boundary_source": "human", "corrected_at": CUT_AT}),
        encoding="utf-8")
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": BEFORE, "segments": [{
            "segment_num": 1, "segment_span": {"start": 0, "end": 120},
            "reviewer": "reviewer_a", "reviewed_at": BEFORE,
            "human": {"outcome": "untouched"}, "answers": {}}]}), encoding="utf-8")
    out = FeatureExtractor().extract(
        pose, folder / f"{VID}_reaches.json", folder / f"{VID}_pellet_outcomes.json",
        review_path=folder / f"{VID}_causal_review.json").to_dict()
    assert out["segments"][0]["outcome"] == "retrieved"


def test_the_voiding_note_has_a_plain_wording():
    # plain_review_note only: the review window shows no re-cut note on screen
    # (a reviewer is never told a video was re-cut); the note is logged.
    doc = {"segments": [_rec((950, 1990), BEFORE)]}
    _, notes = io.index_review_by_segment(
        doc, io.segments_from_segmentation(_seg_doc()), human_segmentation=True)
    words = [io.plain_review_note(n, span=(1000, 1999)) for n in notes]
    assert "earlier answer for frames 950-1990 was given before the re-cut -- answer again" in words


def test_a_save_after_the_recut_does_not_bring_a_void_answer_back(tmp_path, monkeypatch):
    """The window saves a placeholder for the voided pellet; the save keeps
    the earlier copy's answer as a record, but it stays void on re-reading."""
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    stem = "20990101_TEST0101_P1"
    (tmp_path / f"{stem}_causal_review.json").write_text(json.dumps(
        {"video_stem": stem, "reviewed_at": BEFORE, "segments": [_rec((950, 1990), BEFORE)]}))
    placeholder = io.build_segment_record(
        segment_num=1, pellet_num=1, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=None, human_causal_reach=None,
        is_phantom=False, agreed=True, answers={"reviewed": False}, notes="",
        segment_span={"start": 1000, "end": 1999})
    out = io.save_causal_review(stem, tmp_path, [placeholder], provenance={},
                                reviewer="reviewer_b")
    saved = json.loads(out.read_text())
    assert _answered(saved) == set(), "the void answer must not count after a save"
    kept = [r for r in saved["segments"] + saved.get("unmatched_answers", [])
            if (r.get("human") or {}).get("outcome") == "displaced_sa"]
    assert kept and kept[0]["reviewed_at"] == BEFORE      # kept as a record
