"""A human's causal pick is resolved by its FRAMES, not by the reach id it was
written against: reach ids are renumbered on every run."""
from mousereach.review.truth_resolver import _seg_overrides_from_review


def _review(pick):
    return {"reviewer": "test", "segments": [{
        "segment_num": 3,
        "segment_span": {"start": 3960, "end": 5797},
        "human": {"outcome": "displaced_sa", "causal_reach": pick},
        "answers": {"reviewed": True},
    }]}


# This run's reaches for segment 3: the reviewer's reach exists again as id 36.
CURRENT = [{
    "segment_num": 3, "start_frame": 3960, "end_frame": 5797,
    "reaches": [
        {"reach_id": 35, "start_frame": 3980, "end_frame": 3984},
        {"reach_id": 36, "start_frame": 3990, "end_frame": 4004},
        {"reach_id": 37, "start_frame": 4017, "end_frame": 4034},
    ],
}]


def test_stale_id_with_frames_resolves_to_this_runs_reach():
    out = _seg_overrides_from_review(_review({"start": 3989, "end": 4004, "reach_id": 42}),
                                     "human_review", CURRENT)
    assert out[3]["causal_reach_id"] == 36
    assert out[3]["review_causal_reach_id"] == 42


def test_frames_only_review_still_resolves():
    out = _seg_overrides_from_review(_review({"start": 3989, "end": 4004}), "human_review", CURRENT)
    assert out[3]["causal_reach_id"] == 36
    assert out[3]["review_causal_reach_id"] is None


def test_stored_id_is_the_fallback_when_nothing_overlaps():
    out = _seg_overrides_from_review(_review({"start": 5000, "end": 5010, "reach_id": 36}),
                                     "human_review", CURRENT)
    assert out[3]["causal_reach_id"] == 36


def test_no_segments_means_no_frame_matching():
    out = _seg_overrides_from_review(_review({"start": 3989, "end": 4004, "reach_id": 42}),
                                     "human_review", None)
    assert out[3]["causal_reach_id"] == 42
