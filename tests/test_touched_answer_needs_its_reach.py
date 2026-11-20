"""A touched pellet's answer names the reach that touched it.

WHY: the review window saved "retrieved" (or another touched outcome) with no
causal reach as a complete answer. The results then shipped that pellet with
NO causal reach: the person's empty pick was read as "a person decided there
is no causal reach", which blocked the reach assignment's answer. And an ignore
window drawn over exactly the reach picked as causal left the pellet's own
reach out of the measurements.

Now: the window refuses both answers (answer_problem says why), and a stored
record of the first shape leaves the causal reach to the reach assignment.

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json

import pytest

import mousereach.review.causal_review_io as crio
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

from test_causal_ownership_by_source import VID, _outcomes, run_dir  # noqa: F401


def _rec(outcome, reach=None, windows=None):
    answers = {"triage_kind": "both_uncertain"}
    if windows:
        answers["abnormal_ranges"] = [{"start_frame": a, "end_frame": b, "reason": ""}
                                      for a, b in windows]
    return crio.build_segment_record(
        segment_num=3, pellet_num=3, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome,
        human_causal_reach=({"start": reach[0], "end": reach[1], "reach_id": 9}
                            if reach else None),
        is_phantom=False, agreed=False, answers=answers, notes="",
        segment_span={"start": 1000, "end": 2000},
        # The person checked the reach's first and last frame (owner rule
        # 2026-11-19; causal_review_io.CAUSAL_BOUNDS_KEY).
        causal_reach_bounds=({"start": reach[0], "end": reach[1]} if reach else None))


@pytest.mark.parametrize("outcome", ["retrieved", "displaced_sa", "displaced_outside"])
def test_a_touched_outcome_with_no_reach_is_not_recorded(outcome):
    why = crio.answer_problem(_rec(outcome))
    assert why and "no reach is picked" in why and "untouched" in why


def test_a_touched_outcome_with_its_reach_is_recorded():
    assert crio.answer_problem(_rec("retrieved", reach=(1200, 1215))) is None


def test_untouched_needs_no_reach():
    assert crio.answer_problem(_rec("untouched")) is None


def test_an_ignore_window_over_the_whole_causal_reach_is_not_recorded():
    why = crio.answer_problem(_rec("retrieved", reach=(1200, 1215), windows=[(1195, 1220)]))
    assert why and "covers the whole reach" in why


def test_an_ignore_window_elsewhere_is_recorded():
    assert crio.answer_problem(
        _rec("retrieved", reach=(1200, 1215), windows=[(1500, 1530)])) is None


def test_a_stored_touched_answer_with_no_reach_gets_the_assignments_reach(run_dir, monkeypatch):
    """The review says retrieved and names no reach; the reach assignment names
    reach 2. The person's outcome stands and the causal reach is reach 2."""
    folder, pose = run_dir
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [folder])
    _outcomes(folder, outcome="displaced_sa")
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "segments": [{
            "segment_num": 1, "segment_span": {"start": 0, "end": 99},
            "human": {"outcome": "retrieved", "causal_reach": None},
            "answers": {"outcome_override": "retrieved"}}]}), encoding="utf-8")
    feats = FeatureExtractor().extract(
        pose, folder / f"{VID}_reaches.json", folder / f"{VID}_pellet_outcomes.json",
        review_path=folder / f"{VID}_causal_review.json").to_dict()
    seg = feats["segments"][0]
    assert seg["outcome"] == "retrieved"
    assert seg["causal_reach_id"] == 2
    assert [r["reach_id"] for r in seg["reaches"] if r.get("causal_reach")] == [2]
