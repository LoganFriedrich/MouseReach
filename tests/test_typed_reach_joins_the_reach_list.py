"""A reach a reviewer typed that the detector missed joins the reach list.

WHY (2026-10-26): the review window's "Undetected reach (enter frames manually)"
row swapped a typed reach for any detected reach touching it by one frame, and
the results step resolved a typed reach only as a pointer into the detected
reaches -- so a reach the detector missed was never added (no kinematics), and
its answer was lost or landed on a neighbouring reach (of 199 typed reaches on
2026-10-25: 14 lost the causal reach, 6 pointed to another reach). Now one rule
decides when a typed reach IS a detected reach (a detected reach covers at least
half its frames), and otherwise the typed reach is added exactly as a
supplementary ground-truth reach is, and is the causal reach. The results step
also counts only real answers (is_answered).
"""
import mousereach.review.truth_resolver as tr
from mousereach.review.causal_review_io import typed_reach_match

REACHES = [{"reach_id": 7, "start_frame": 100, "end_frame": 120},
           {"reach_id": 8, "start_frame": 200, "end_frame": 230}]


def test_a_detected_reach_covering_half_the_typed_frames_is_that_reach():
    assert typed_reach_match(105, 124, REACHES)["reach_id"] == 7      # 16 of 20 frames
    assert typed_reach_match(118, 140, REACHES) is None               # 3 of 23 frames: missed
    assert typed_reach_match(150, 160, REACHES) is None               # touches nothing
    # the review window's list uses start/end
    assert typed_reach_match(200, 210, [{"start": 195, "end": 215, "reach_id": 3}])["reach_id"] == 3


def _data(review_human, answers=None):
    reaches = {"video_name": "V", "segments": [
        {"segment_num": 2, "start_frame": 0, "end_frame": 400, "reaches": [dict(r) for r in REACHES]}]}
    outcomes = {"segments": [{"segment_num": 2, "outcome": "triaged", "causal_reach_id": None}]}
    review = {"reviewer": "someone", "segments": [{
        "segment_num": 2, "segment_span": {"start": 0, "end": 400},
        "human": review_human, "answers": answers or {"outcome_pick": review_human.get("outcome")}}]}
    return reaches, outcomes, review


def _resolve(monkeypatch, reaches, outcomes, review):
    monkeypatch.setattr(tr, "_durable_review_doc", lambda stem: None)
    monkeypatch.setattr(tr, "_review_doc", lambda stem, base: review if base == "PRIMARY" else None)
    import mousereach.review.causal_review_io as cio
    monkeypatch.setattr(cio, "find_gt", lambda *a, **k: None)
    return tr.resolve_truth_layers(reaches, outcomes, video_stem="V", primary_dir="PRIMARY")


def test_a_missed_reach_is_added_with_the_persons_frames_and_is_the_causal_reach(monkeypatch):
    r, o, rev = _data({"outcome": "retrieved", "causal_reach": {"start": 300, "end": 318}})
    reaches, outcomes = _resolve(monkeypatch, r, o, rev)
    seg = reaches["segments"][0]
    added = [x for x in seg["reaches"] if x.get("reach_source") == "human_review"]
    assert len(added) == 1 and (added[0]["start_frame"], added[0]["end_frame"]) == (300, 318)
    assert added[0]["reach_id"] == 9                     # a new id, no clash with 7 or 8
    assert seg["n_reaches"] == 3
    o2 = outcomes["segments"][0]
    assert o2["causal_reach_id"] == 9 and o2["outcome"] == "retrieved"
    assert o2["outcome_source"] == "human_review"


def test_a_typed_reach_a_detected_reach_covers_points_at_that_reach(monkeypatch):
    r, o, rev = _data({"outcome": "displaced_sa", "causal_reach": {"start": 202, "end": 226}})
    reaches, outcomes = _resolve(monkeypatch, r, o, rev)
    assert outcomes["segments"][0]["causal_reach_id"] == 8
    assert len(reaches["segments"][0]["reaches"]) == 2   # nothing added


def test_a_partly_overlapping_typed_reach_replaces_the_detected_one_like_ground_truth(monkeypatch):
    r, o, rev = _data({"outcome": "retrieved", "causal_reach": {"start": 118, "end": 140}})
    reaches, outcomes = _resolve(monkeypatch, r, o, rev)
    frames = [(x["start_frame"], x["end_frame"], x.get("reach_source")) for x in reaches["segments"][0]["reaches"]]
    assert frames == [(118, 140, "human_review"), (200, 230, "algo")]
    assert outcomes["segments"][0]["causal_reach_id"] == 9


def test_a_pellet_left_open_is_not_applied_as_a_human_answer(monkeypatch):
    r, o, rev = _data({"outcome": "triaged", "causal_reach": None},
                      answers={"triage_kind": "both_uncertain"})
    reaches, outcomes = _resolve(monkeypatch, r, o, rev)
    assert outcomes["segments"][0]["outcome_source"] == "algo"
    assert len(reaches["segments"][0]["reaches"]) == 2


def test_a_typed_reach_longer_than_any_reach_is_never_added(monkeypatch):
    """A mistyped frame number (35183-35914, 732 frames, 2026-10-26) would become a
    'reach' measured over 24 seconds of video."""
    r, o, rev = _data({"outcome": "retrieved", "causal_reach": {"start": 300, "end": 1031}})
    reaches, outcomes = _resolve(monkeypatch, r, o, rev)
    assert len(reaches["segments"][0]["reaches"]) == 2
    assert outcomes["segments"][0]["causal_reach_id"] is None


def test_the_review_window_refuses_a_typed_reach_longer_than_any_reach():
    from mousereach.review.causal_review_io import answer_problem
    rec = {"segment_num": 2, "segment_span": {"start": 0, "end": 2000},
           "human": {"outcome": "retrieved",
                     "causal_reach": {"start": 300, "end": 1031, "typed": [300, 1031]}},
           "answers": {"outcome_pick": "retrieved"}}
    assert "runs 732 frames" in answer_problem(rec)
    rec["human"]["causal_reach"] = {"start": 300, "end": 318, "typed": [300, 318]}
    assert answer_problem(rec) is None
