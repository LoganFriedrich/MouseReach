"""The review tool restores saved answers by the frames they cover.

WHY (2026-10-22): the tool restored a saved answer by segment NUMBER. After a
re-cut renumbered the segments, an answer about other footage sat on today's
segment N: the tool counted N as done and never showed it, while the release
check, which matches by frames, counted N unanswered and kept the video held.
Nobody could finish such a video. The person's fresh answer for the old frames
was saved beside the old one, so 59 review files carried two records for the
same frames, and the results step used whichever came first in the file.
"""
from mousereach.review import causal_review_io as io


def _rec(num, span, outcome="displaced_sa", reach=(8818, 8830, 65), reviewed=True):
    answers = {"triage_kind": "both_uncertain"}
    if not reviewed:
        answers = {"reviewed": False}
    return {
        "segment_num": num, "pellet_num": num,
        "segment_span": {"start": span[0], "end": span[1]},
        "algo": {"outcome": "triaged", "causal_reach": None, "interaction_frame": None},
        "human": {"outcome": outcome if reviewed else None,
                  "causal_reach": ({"start": reach[0], "end": reach[1], "reach_id": reach[2]}
                                   if reviewed and reach else None),
                  "is_phantom": False, "agreed": False},
        "answers": answers, "notes": "",
    }


# today's cut: a segment was inserted before the reviewed stretch
CURRENT = [{"segment_num": 5, "start_frame": 7000, "end_frame": 8786},
           {"segment_num": 6, "start_frame": 8787, "end_frame": 10628}]


def test_an_answer_goes_back_on_the_segment_whose_frames_it_covers():
    doc = {"segments": [_rec(5, (8787, 10628))]}
    restored, unmatched, _ = io.restore_review_for_segments(doc, CURRENT)
    assert sorted(restored) == [6], "segment 5 is other footage now; nobody answered it"
    assert restored[6]["segment_num"] == 6
    assert restored[6]["segment_span"] == {"start": 8787, "end": 10628}
    assert unmatched == []


def test_two_records_with_the_same_answer_count_once():
    doc = {"segments": [_rec(5, (8787, 10628)), _rec(6, (8787, 10628))]}
    restored, unmatched, _ = io.restore_review_for_segments(doc, CURRENT)
    assert sorted(restored) == [6] and unmatched == []
    assert io.answered_segments(doc, CURRENT) == {6}


def test_two_answers_that_disagree_are_neither_applied():
    doc = {"segments": [_rec(5, (8787, 10628), outcome="retrieved"),
                        _rec(6, (8787, 10628), outcome="untouched", reach=None)]}
    mapping, notes = io.index_review_by_segment(doc, CURRENT)
    assert 6 not in mapping
    assert any("disagree" in n for n in notes)
    assert io.answered_segments(doc, CURRENT) == set(), "the segment goes back to a person"
    outcomes = {"segments": [{"segment_num": 6, "outcome": "triaged", "causal_reach_id": None}]}
    applied = io.apply_review_overrides(outcomes, mapping)
    assert applied["segments"][0]["outcome"] == "triaged"
    assert applied["segments"][0]["outcome_source"] == "algo"


def test_an_answer_beats_a_placeholder_for_the_same_frames_whatever_the_order():
    doc = {"segments": [_rec(5, (8787, 10628), reviewed=False), _rec(6, (8787, 10628))]}
    mapping, _ = io.index_review_by_segment(doc, CURRENT)
    assert mapping[6]["human"]["outcome"] == "displaced_sa"


def test_an_answer_that_matches_no_segment_is_kept_and_survives_a_save(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    gone = _rec(9, (20000, 21000), outcome="retrieved")
    doc = {"segments": [gone, _rec(6, (8787, 10628))]}
    restored, unmatched, _ = io.restore_review_for_segments(doc, CURRENT)
    assert sorted(restored) == [6] and unmatched == [gone]

    io.save_causal_review("V", tmp_path, list(restored.values()), {}, reviewer="r",
                          unmatched_answers=unmatched)
    saved, _ = io.load_causal_review("V", tmp_path)
    assert saved[io.UNMATCHED_ANSWERS_KEY] == [gone]
    assert [s["segment_num"] for s in saved["segments"]] == [6]
    # reopened again: carried once more, not duplicated, never applied
    restored2, unmatched2, _ = io.restore_review_for_segments(saved, CURRENT)
    assert sorted(restored2) == [6] and unmatched2 == [gone]


def test_a_left_blank_record_never_outranks_a_decision_for_the_same_frames():
    """Seen in the corpus: the blank sat first in the file and was applied, so
    the video went back to triage although the person had answered."""
    doc = {"segments": [_rec(5, (8787, 10628), outcome="triaged", reach=None),
                        _rec(6, (8787, 10628))]}
    mapping, notes = io.index_review_by_segment(doc, CURRENT)
    assert mapping[6]["human"]["outcome"] == "displaced_sa"
    assert not any("disagree" in n for n in notes)


def test_an_answer_whose_reach_lies_outside_its_frames_is_not_applied():
    """The corpus case: segment 5's record says frames 7513-9353 but names
    reach 5693-5700 -- the answer belongs to the footage before."""
    current = [{"segment_num": 5, "start_frame": 7513, "end_frame": 9353}]
    doc = {"segments": [_rec(5, (7513, 9353), reach=(5693, 5700, 12))]}
    mapping, notes = io.index_review_by_segment(doc, current)
    assert mapping == {}
    assert any("outside the frames" in n for n in notes)
    assert io.answered_segments(doc, current) == set()
    restored, unmatched, _ = io.restore_review_for_segments(doc, current)
    assert restored == {} and len(unmatched) == 1, "kept in the file, never applied"
