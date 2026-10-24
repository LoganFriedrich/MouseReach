"""Every 'untouched' call opens at the end of its segment, reaches or not.

An untouched call is judged by where the pellet is when the segment ends. The
review tool opened one with no reaches 30 frames before the end, but one with
reaches (every hand-score mismatch the algorithm called untouched) on its
reach -- in triage, a median of 172 frames and up to 1,825 frames before the
end -- so the reviewer had to skip forward to see the pellet (Logan,
2026-10-24: "Why aren't they the same?").
"""
from mousereach.review.causal_review_widget import CausalReviewWidget as W


class _Dims:
    def set_current_step(self, axis, value):
        self.step = value


class _Viewer:
    def __init__(self):
        self.dims = _Dims()


class _Tool:
    """Just the parts of the review window that pick where a pellet opens."""
    WINDOW_PAD = W.WINDOW_PAD
    OPEN_LEAD = W.OPEN_LEAD
    _opens_at_end = W._opens_at_end
    _relevant_reach = W._relevant_reach
    _most_likely_reach = W._most_likely_reach
    _candidate_reach_ids = W._candidate_reach_ids
    _compute_segment_window = W._compute_segment_window
    _load_frame_window = W._load_frame_window
    _navigate_to_segment = W._navigate_to_segment

    def __init__(self, records=None):
        self.n_frames = 20000
        self.dlc_df = None
        self._review_records = records or {}
        self.viewer = _Viewer()

    def opens_at(self, seg):
        self._load_frame_window(self._compute_segment_window(seg))
        self._navigate_to_segment(seg)
        return self.viewer.dims.step


def _seg(outcome, reaches=(), causal=None):
    return {"segment_num": 4, "start_frame": 1000, "end_frame": 3000,
            "outcome": outcome, "interaction_frame": None, "causal_reach": causal,
            "all_reaches": [{"reach_id": i, "start": s, "end": e}
                            for i, (s, e) in enumerate(reaches)]}


def test_untouched_with_reaches_opens_where_untouched_without_reaches_does():
    early_reach = _seg("untouched", reaches=[(1200, 1240)])
    no_reach = _seg("untouched")
    assert _Tool().opens_at(early_reach) == _Tool().opens_at(no_reach) == 3000 - 30


def test_a_touched_call_still_opens_on_its_reach():
    seg = _seg("retrieved", reaches=[(1200, 1240)], causal={"start": 1200, "end": 1240})
    assert _Tool().opens_at(seg) == 1200 - W.OPEN_LEAD


def test_a_reach_the_reviewer_chose_still_comes_first():
    # The reviewer changed the call and picked the reach that did it.
    records = {4: {"human": {"outcome": "retrieved",
                             "causal_reach": {"start": 1200, "end": 1240}}}}
    seg = _seg("untouched", reaches=[(1200, 1240)])
    assert _Tool(records).opens_at(seg) == 1200 - W.OPEN_LEAD
