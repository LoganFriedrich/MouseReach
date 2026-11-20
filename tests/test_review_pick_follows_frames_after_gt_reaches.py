"""A review's causal-reach pick is found by its FRAMES in the final reach list.

The review is matched against the detected reaches; ground-truth reaches can
then replace that list and renumber it. The pick's id then named another reach:
on 2026-11-20 a pick of reach 66 (frames 9519-9534) was filed as reach 66 of the
ground-truth list (frames 9494-9504), so the wrong reach was credited.
"""
from mousereach.review.truth_resolver import _apply_outcome_layers


def _outcomes():
    return {"segments": [{"segment_num": 6, "outcome": "triaged"}]}


def _gt_reaches():
    # the list after ground truth replaced the detected reaches, renumbered
    return {"segments": [{"segment_num": 6, "reaches": [
        {"reach_id": 66, "start_frame": 9494, "end_frame": 9504, "reach_source": "ground_truth"},
        {"reach_id": 67, "start_frame": 9519, "end_frame": 9534, "reach_source": "ground_truth"}]}]}


def _review(**kw):
    ov = {"outcome": "retrieved", "causal_reach_id": 66, "pick_frames": (9519, 9534),
          "source": "human_review", "reviewer": "someone"}
    ov.update(kw)
    return {6: ov}


def test_review_pick_is_found_by_frames():
    out = _apply_outcome_layers(_outcomes(), [_review()], _gt_reaches())
    seg = out["segments"][0]
    assert seg["outcome"] == "retrieved" and seg["causal_reach_id"] == 67
    assert seg["causal_reach_source"] == "human_review"


def test_pick_with_no_reach_in_final_list_goes_to_the_assignment():
    out = _apply_outcome_layers(_outcomes(), [_review(pick_frames=(20000, 20010))], _gt_reaches())
    seg = out["segments"][0]
    assert seg["outcome"] == "retrieved" and "causal_reach_id" not in seg


def test_without_frames_the_id_is_kept():
    out = _apply_outcome_layers(_outcomes(), [_review(pick_frames=None)], _gt_reaches())
    assert out["segments"][0]["causal_reach_id"] == 66


def test_ground_truth_wins_and_review_pick_found_by_frames():
    gt = {6: {"outcome": "retrieved", "interaction_frame": None, "source": "ground_truth"}}
    out = _apply_outcome_layers(_outcomes(), [_review(), gt], _gt_reaches())
    seg = out["segments"][0]
    assert seg["causal_reach_id"] == 67 and seg["causal_reach_source"] == "human_review"
