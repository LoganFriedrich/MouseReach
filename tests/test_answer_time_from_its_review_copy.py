"""An answer saved without a time of its own takes its review copy's time.

Records saved before records carried their own "reviewed_at" have only the
file's. newest_statements matched each copy's answers as a bare
{"segments": [...]}, so those answers had no time at all, counted as given
before the latest re-cut, and were voided -- although the file says they were
given AFTER it. The rule: a re-cut voids only answers given before it.
"""
import mousereach.review.causal_review_io as cio

# A person re-cut the video on day 1 (old cuts -> new cuts, every pellet moved).
OLD = [100, 1100, 2100, 3100]
NEW = [150, 1150, 2150, 3150]
SEG_DOC = {"boundaries": NEW, "boundary_source": "human",
           "corrected_at": "2099-01-01T12:00:00", "algo_boundaries": OLD}


def _answer(sn, start, end, reach):
    return {"segment_num": sn, "segment_span": {"start": start, "end": end},
            "human": {"outcome": "retrieved", "causal_reach": {"start": reach, "end": reach + 10}},
            "answers": {"triage_kind": "outcome_uncertain", "outcome_pick": "retrieved"}}


def _copy(reviewed_at):
    # No per-record time: only the file's.
    return {"type": "causal_review", "reviewer": "someone", "reviewed_at": reviewed_at,
            "segments": [_answer(1, 150, 1149, 400), _answer(2, 1150, 2149, 1500)]}


def test_answers_given_after_the_recut_stand_when_only_the_file_has_a_time():
    current = cio.segments_from_segmentation(SEG_DOC)
    chosen, unmatched, notes = cio.newest_statements([_copy("2099-02-01T09:00:00")], current, True)
    assert sorted(chosen) == [1, 2], notes
    assert unmatched == []
    # The chosen record carries the time it was given, for every later reader.
    assert chosen[1]["reviewed_at"] == "2099-02-01T09:00:00"


def test_merged_review_keeps_them_too():
    current = cio.segments_from_segmentation(SEG_DOC)
    doc, _ = cio.merge_review_copies([_copy("2099-02-01T09:00:00")], current, True)
    assert [r["segment_num"] for r in doc["segments"]] == [1, 2]


def test_answers_given_before_the_recut_are_still_voided():
    current = cio.segments_from_segmentation(SEG_DOC)
    chosen, _, notes = cio.newest_statements([_copy("2098-12-01T09:00:00")], current, True)
    assert chosen == {}
    assert any("before a re-cut" in n for n in notes)
