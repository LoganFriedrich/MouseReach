"""A triage answer records the call on EVERY reach of its segment, and the
causal reach's frames as the person checked them.

"Naming the causal reach of a segment ALSO answers every other reach in that
segment -- they all missed." "'untouched' / 'no reach interacted' => every
reach of the segment is a miss." "A causal answer always includes the person
checking that reach's start/end (always shown, even if not asked)."
"Untouched segments: no reach bounds judged."

Record fields (causal_review_io.REACH_CALLS_KEY / CAUSAL_BOUNDS_KEY):
``human.reach_calls`` and ``human.causal_reach_bounds``.

Real Qt widgets (offscreen) for the window part; invented ids.
"""
from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QApplication

from mousereach.review import causal_review_io as io
from mousereach.review.causal_review_widget import CausalReviewWidget as W

REACHES = [{"start": 1100, "end": 1110, "reach_id": 1},
           {"start": 1200, "end": 1215, "reach_id": 2},
           {"start": 1300, "end": 1320, "reach_id": 3}]
SPAN = {"start": 1000, "end": 2000}


def _calls(rec):
    return {(c["reach_id"], c["call"]) for c in rec["human"]["reach_calls"]}


def test_naming_the_causal_reach_makes_every_other_reach_a_miss():
    rec = io.build_segment_record(
        segment_num=3, pellet_num=3, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome="retrieved",
        human_causal_reach={"start": 1200, "end": 1215, "reach_id": 2},
        is_phantom=False, agreed=False, answers={}, segment_span=SPAN,
        segment_reaches=REACHES, causal_reach_bounds={"start": 1198, "end": 1217})
    assert _calls(rec) == {(1, "miss"), (2, "causal"), (3, "miss")}
    causal = next(c for c in rec["human"]["reach_calls"] if c["call"] == "causal")
    assert (causal["start"], causal["end"], causal["bounds"]) == (1198, 1217, "human")
    assert (causal["detected_start"], causal["detected_end"]) == (1200, 1215)
    assert all(c["bounds"] == "detected" for c in rec["human"]["reach_calls"]
               if c["call"] == "miss")
    assert rec["human"]["causal_reach_bounds"] == {"start": 1198, "end": 1217}
    # The picked reach stays as picked, so it still matches the reach list.
    assert rec["human"]["causal_reach"] == {"start": 1200, "end": 1215, "reach_id": 2}


@pytest.mark.parametrize("outcome", ["untouched", "abnormal_exception"])
def test_no_reach_interacted_makes_every_reach_a_miss_with_no_bounds(outcome):
    rec = io.build_segment_record(
        segment_num=3, pellet_num=3, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome, human_causal_reach=None,
        is_phantom=False, agreed=False, answers={}, segment_span=SPAN,
        segment_reaches=REACHES)
    assert _calls(rec) == {(1, "miss"), (2, "miss"), (3, "miss")}
    assert "causal_reach_bounds" not in rec["human"]


def test_an_undecided_answer_says_nothing_about_the_reaches():
    assert io.segment_reach_calls("triaged", None, REACHES) is None
    assert io.segment_reach_calls(None, None, REACHES) is None


def test_a_reach_the_detector_missed_is_recorded_as_the_causal_one():
    calls = io.segment_reach_calls(
        "displaced_sa", {"start": 1500, "end": 1512, "typed": [1500, 1512]}, REACHES,
        {"start": 1501, "end": 1512})
    assert [c["call"] for c in calls] == ["miss", "miss", "miss", "causal"]
    assert calls[-1] == {"reach_id": None, "start": 1501, "end": 1512, "call": "causal",
                         "detected": False, "bounds": "human"}


def test_a_causal_answer_without_its_frames_checked_is_not_recorded():
    rec = {"segment_num": 3, "pellet_num": 3, "segment_span": SPAN,
           "human": {"outcome": "retrieved",
                     "causal_reach": {"start": 1200, "end": 1215, "reach_id": 2}},
           "answers": {}}
    assert "were not recorded" in io.answer_problem(rec)
    assert "Checked" not in io.answer_problem(rec)
    rec["human"]["causal_reach_bounds"] = {"start": 1215, "end": 1200}
    assert "before its first frame" in io.answer_problem(rec)
    rec["human"]["causal_reach_bounds"] = {"start": 2100, "end": 2110}
    assert "outside segment" in io.answer_problem(rec)
    rec["human"]["causal_reach_bounds"] = {"start": 1199, "end": 1216}
    assert io.answer_problem(rec) is None


# ---- the window -------------------------------------------------------------

_KEEP = []
SEG = {"segment_num": 3, "pellet_num": 3, "start_frame": 1000, "end_frame": 2000,
       "is_boundary": False, "outcome": "triaged", "causal_reach": None,
       "interaction_frame": 1205, "all_reaches": REACHES, "flagged": False}


def _window():
    _KEEP.append(QApplication.instance() or QApplication([]))
    w = SimpleNamespace(
        _q_widgets={}, _segment_list=[SEG], _current_seg_idx=0,
        _panel_kind="both_uncertain", _pinned_reach=None, _segments_data={},
        _review_records={}, n_frames=5000, viewer=None, _restored_panel=None,
        _candidate_reach_ids=lambda seg: None, _goto_frame=lambda *a: None,
        _jump_to_selected_reach=lambda *a: None)
    for name in ("_snap_to_detected_reach", "_read_reach_pick", "_collect_answers",
                 "_collect_answers_unchecked", "_segmentation_is_human",
                 "_fill_causal_bounds", "_checked_causal_bounds",
                 "_sync_reach_picker_to_outcome"):
        setattr(w, name, getattr(W, name).__get__(w))
    _KEEP.append(W._make_outcome_picker(w, SEG, require_pick=True))
    _KEEP.append(W._make_reach_picker(w, SEG))
    return w


def test_the_window_always_shows_the_causal_reachs_frames_to_check():
    w = _window()
    rp = w._q_widgets["_reach_picker"]
    w._q_widgets["_outcome_combo"].setCurrentText("retrieved")
    # The reach nearest the interaction frame is pre-picked; its frames are
    # shown, and naming it confirms them (no separate tick).
    assert (rp["bounds_start"].value(), rp["bounds_end"].value()) == (1200, 1215)
    assert rp["bounds_checked"].isHidden()
    rp["bounds_end"].setValue(1218)
    rec = w._collect_answers()
    assert rec["human"]["causal_reach_bounds"] == {"start": 1200, "end": 1218}
    assert _calls(rec) == {(1, "miss"), (2, "causal"), (3, "miss")}


def test_the_window_records_every_reach_a_miss_for_untouched():
    w = _window()
    w._q_widgets["_outcome_combo"].setCurrentText("untouched")
    w._sync_reach_picker_to_outcome("untouched")   # the panel's own hook-up
    rp = w._q_widgets["_reach_picker"]
    assert not rp["bounds_checked"].isEnabled()
    rec = w._collect_answers()
    assert rec["human"]["outcome"] == "untouched"
    assert _calls(rec) == {(1, "miss"), (2, "miss"), (3, "miss")}
    assert "causal_reach_bounds" not in rec["human"]
