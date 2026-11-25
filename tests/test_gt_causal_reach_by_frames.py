"""A ground-truth touched pellet gets its causal reach from FRAMES, and a
ground-truth outcome no longer throws away the rest of a person's review.

Ground truth stores no causal reach a person chose: the ground-truth tool has
no control for it, and any id in the file was copied from the algorithm's
reach list of the run the file was made on. The ground-truth outcome used to
replace the review's whole answer for the pellet, with that stored id (usually
none), and the key's presence then blocked the reach assignment's answer too:
touched pellets were filed with no causal reach, and the review's pick and
ignore windows were discarded.

Invented ids; files under tmp_path only.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.review.truth_resolver import resolve_truth_layers

STEM = "20990101_ABC9901_P1"


@pytest.fixture(autouse=True)
def _only_these_files(monkeypatch):
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    import mousereach.review.bundle_media as bm
    monkeypatch.setattr(bm, "filed_media_dirs", lambda stem: [])
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", None, raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", None, raising=False)


def _reaches():
    # Two detected reaches in pellet 1, ids as THIS run numbered them.
    return {"video_name": STEM, "segments": [
        {"segment_num": 1, "start_frame": 0, "end_frame": 999, "reaches": [
            {"reach_id": 31, "reach_num": 1, "start_frame": 100, "end_frame": 130},
            {"reach_id": 32, "reach_num": 2, "start_frame": 400, "end_frame": 440}]}]}


def _outcomes():
    return {"video_name": STEM, "segments": [{"segment_num": 1, "outcome": "triaged"}]}


def _gt(folder, outcome="retrieved", ifr=420, stored_id=5):
    (folder / f"{STEM}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": STEM,
        "reaches": {"exhaustive": False, "reaches": []},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": outcome, "interaction_frame": ifr,
             "causal_reach_id": stored_id, "determined": True}]},
    }), encoding="utf-8")


def _review(folder, outcome, pick=None, ranges=None):
    human = {"outcome": outcome}
    if pick:
        human["causal_reach"] = {"reach_id": pick[0], "start": pick[1], "end": pick[2]}
    (folder / f"{STEM}_causal_review.json").write_text(json.dumps({
        "video_name": STEM, "segments": [{
            "segment_num": 1, "segment_span": {"start": 0, "end": 999},
            "human": human, "answers": {"abnormal_ranges": ranges or []}}]}),
        encoding="utf-8")


def _resolve(folder):
    r, o = resolve_truth_layers(_reaches(), _outcomes(), video_stem=STEM,
                                primary_dir=folder, extra_gt_dirs=(folder,))
    return r, o["segments"][0]


def test_the_reach_holding_the_interaction_frame_is_causal_not_the_stored_id(tmp_path):
    _gt(tmp_path, ifr=420, stored_id=5)        # id 5 is from an older run
    _, seg = _resolve(tmp_path)
    assert seg["outcome_source"] == "ground_truth"
    assert seg["causal_reach_id"] == 32
    assert seg["causal_reach_source"] == "ground_truth"


def test_without_an_anchor_the_reach_assignment_is_left_to_answer(tmp_path):
    _gt(tmp_path, ifr=None, stored_id=None)
    _, seg = _resolve(tmp_path)
    # No key: the extractor merges the reach assignment's pick (a present
    # key, even None, would mean "a person said no reach did this").
    assert "causal_reach_id" not in seg


def test_the_reviews_pick_is_kept_when_it_gave_the_same_outcome(tmp_path):
    _gt(tmp_path, ifr=None)
    _review(tmp_path, "retrieved", pick=(31, 100, 130))
    _, seg = _resolve(tmp_path)
    assert (seg["outcome"], seg["outcome_source"]) == ("retrieved", "ground_truth")
    assert seg["causal_reach_id"] == 31
    assert seg["causal_reach_source"] == "human_review"


def test_the_reviews_ignore_windows_carry_through_a_ground_truth_outcome(tmp_path):
    _gt(tmp_path, ifr=420)
    window = [{"start_frame": 600, "end_frame": 650, "reason": "tail"}]
    _review(tmp_path, "retrieved", pick=(32, 400, 440), ranges=window)
    _, seg = _resolve(tmp_path)
    assert seg["abnormal_ranges"] == window


def test_an_untouched_ground_truth_pellet_has_no_causal_reach(tmp_path):
    _gt(tmp_path, outcome="untouched", ifr=None, stored_id=7)
    _, seg = _resolve(tmp_path)
    assert "causal_reach_id" in seg and seg["causal_reach_id"] is None


def test_a_ground_truth_reach_never_shares_an_id_with_a_detected_one(tmp_path):
    # A determined ground-truth reach labelled id 32 (its id in an older run)
    # at frames no detected reach covers; this run's reach 32 is elsewhere.
    (tmp_path / f"{STEM}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": STEM,
        "reaches": {"exhaustive": False, "reaches": [
            {"reach_id": 32, "segment_num": 1, "start_frame": 700, "end_frame": 720,
             "start_determined": True, "end_determined": True}]},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 710,
             "determined": True}]},
    }), encoding="utf-8")
    r, seg = _resolve(tmp_path)
    ids = [x["reach_id"] for x in r["segments"][0]["reaches"]]
    assert len(ids) == len(set(ids)) == 3
    causal = [x for x in r["segments"][0]["reaches"] if x["reach_id"] == seg["causal_reach_id"]]
    assert [(x["start_frame"], x["reach_source"]) for x in causal] == [(700, "ground_truth")]


def test_a_review_pick_a_ground_truth_reach_replaced_follows_its_frames(tmp_path):
    # The review picked detected reach 31 (frames 100-130); a determined
    # ground-truth reach (98-135) replaced it and was renumbered 90. The pick
    # is found again by its frames (truth_resolver._apply_outcome_layers):
    # the ground-truth reach IS the reach the person picked,
    # so it is the causal reach (a dangling id 31 would leave the pellet
    # to the reach assignment).
    (tmp_path / f"{STEM}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": STEM,
        "reaches": {"exhaustive": False, "reaches": [
            {"reach_id": 90, "segment_num": 1, "start_frame": 98, "end_frame": 135,
             "start_determined": True, "end_determined": True}]},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "retrieved", "determined": True}]},
    }), encoding="utf-8")
    _review(tmp_path, "retrieved", pick=(31, 100, 130))
    _, seg = _resolve(tmp_path)
    assert seg["causal_reach_id"] == 90
    assert seg["causal_reach_source"] == "human_review"


def test_a_review_pick_with_no_reach_left_goes_to_the_assignment(tmp_path):
    # The ground-truth reach list (exhaustive) has no reach at the picked
    # frames: the pick names nothing in the final list, so the id must not
    # dangle -- the reach assignment answers.
    (tmp_path / f"{STEM}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": STEM,
        "reaches": {"exhaustive": True, "reaches": [
            {"reach_id": 90, "segment_num": 1, "start_frame": 400, "end_frame": 440,
             "start_determined": True, "end_determined": True}]},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "retrieved", "determined": True}]},
    }), encoding="utf-8")
    _review(tmp_path, "retrieved", pick=(31, 100, 130))
    _, seg = _resolve(tmp_path)
    assert "causal_reach_id" not in seg
