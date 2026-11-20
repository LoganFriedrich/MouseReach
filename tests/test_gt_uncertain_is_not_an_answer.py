"""A ground-truth outcome marked determined but valued "uncertain" is no answer.

Rule: human answers are taken by precedence -- ground truth > deep review >
triage -- and the highest layer that GAVE AN ANSWER wins; a video is never held
over the disagreement. "uncertain" says the person could not tell, so it does
not outrank a review's answer.

WHY: the review gate counted a determined "uncertain" as settled (and a ground
truth with every outcome determined as fully certified), while the kinematics
applied "uncertain" as the pellet's outcome over a triage answer. The gate
passed the video, its features said "uncertain", and the archive step refused
to file it. All three readers now share causal_review_io.gt_outcome_is_answer.
"""
import json
import os

import pytest

VID = "20990101_XYZ0101_P1"


def _video(folder, n=3, flagged=(2,)):
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": k, "outcome": "triaged" if k in flagged else "untouched",
         "flagged_for_review": k in flagged} for k in range(1, n + 1)]}), encoding="utf-8")
    (folder / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [1000 * i for i in range(n + 1)]}),
        encoding="utf-8")
    old = (folder / f"{VID}_pellet_outcomes.json").stat().st_mtime - 60
    os.utime(folder / f"{VID}_segments.json", (old, old))


def _gt_doc(uncertain=(2,), n=3):
    return {"reaches": {"exhaustive": True},
            "outcomes": {"exhaustive": True, "segments": [
                {"segment_num": k, "determined": True,
                 "outcome": "uncertain" if k in uncertain else "untouched"}
                for k in range(1, n + 1)]}}


@pytest.fixture(autouse=True)
def empty_gt_index(monkeypatch):
    import mousereach.review.causal_review_io as cio
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})


def test_helper():
    from mousereach.review.causal_review_io import gt_outcome_is_answer
    assert gt_outcome_is_answer({"determined": True, "outcome": "retrieved"})
    assert gt_outcome_is_answer({"outcome": "untouched"})
    assert not gt_outcome_is_answer({"determined": True, "outcome": "uncertain"})
    assert not gt_outcome_is_answer({"determined": True, "outcome": "triaged"})
    assert not gt_outcome_is_answer({"determined": True, "outcome": None})
    assert not gt_outcome_is_answer({"determined": False, "outcome": "retrieved"})


def test_gate_does_not_count_uncertain_as_answered(tmp_path):
    from mousereach.review.triage_status import gt_resolved_segments
    import mousereach.watcher.review_gate as rg
    _video(tmp_path)
    (tmp_path / f"{VID}_unified_ground_truth.json").write_text(
        json.dumps(_gt_doc()), encoding="utf-8")
    full, segs = gt_resolved_segments(VID, extra_dirs=(tmp_path,))
    assert not full and segs == {1, 3}
    decision, _, st = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_TRIAGE and 2 in st.unresolved


def test_kinematics_take_the_review_answer_under_an_uncertain_gt():
    from mousereach.review.truth_resolver import (_apply_outcome_layers,
                                                  _seg_overrides_from_gt)
    gt_layer, _ = _seg_overrides_from_gt(_gt_doc())
    assert 2 not in gt_layer and gt_layer[1]["outcome"] == "untouched"
    review = {2: {"outcome": "retrieved", "causal_reach_id": 7, "source": "human_review",
                  "reviewer": "someone"}}
    outcomes = {"segments": [{"segment_num": k, "outcome": "triaged"} for k in (1, 2, 3)]}
    reaches = {"segments": [{"segment_num": 2, "reaches": [
        {"reach_id": 7, "start_frame": 1100, "end_frame": 1110}]}]}
    out = _apply_outcome_layers(outcomes, [review, gt_layer], reaches)
    seg2 = out["segments"][1]
    assert seg2["outcome"] == "retrieved"
    assert seg2["outcome_source"] == "human_review"
    assert seg2["causal_reach_id"] == 7
    assert out["segments"][0]["outcome_source"] == "ground_truth"


def test_triage_lift_skips_uncertain_gt():
    from mousereach.triage.gt_resolve import resolve_triaged_outcomes
    doc = {"video_id": VID, "segments": [
        {"segment_num": k, "outcome": "triaged", "flagged_for_review": True}
        for k in (1, 2, 3)]}
    out, res = resolve_triaged_outcomes(doc, _gt_doc(), None)
    seg2 = out["segments"][1]
    assert seg2["flagged_for_review"] is True and seg2["outcome"] == "triaged"
    assert out["segments"][0]["triage_cleared"] is True
