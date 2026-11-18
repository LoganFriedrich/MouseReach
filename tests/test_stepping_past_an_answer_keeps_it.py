"""Stepping past an answered pellet in the review window keeps the answer.

WHY: the window put a saved answer back on screen by reach ID, which a re-run
renumbers, so another reach was pre-selected; an outcome the dropdown does not
list (abnormal_exception, saved when an ignore window is set with no reach)
could not be put back at all. Next / Prev / Save then read the panel and stored
it as the person's answer, so merely stepping past the pellet rewrote it.

Now: the reach is put back by its frames, an unlisted outcome is added to the
list, and a restored answer the person did not change is kept exactly as
stored (reviewer and time included).

Real Qt widgets (offscreen); invented ids.
"""
import copy
from types import SimpleNamespace

from qtpy.QtWidgets import QApplication

from mousereach.review import causal_review_io as io
from mousereach.review.causal_review_widget import CausalReviewWidget as W

_KEEP = []

# This run's reaches: the re-run renumbered them. The reach the person picked
# (frames 1200-1215, saved as id 9) is id 12 now; id 9 is another reach.
REACHES = [{"start": 1100, "end": 1110, "reach_id": 9},
           {"start": 1200, "end": 1215, "reach_id": 12}]
SEG = {"segment_num": 3, "pellet_num": 3, "start_frame": 1000, "end_frame": 2000,
       "is_boundary": False, "outcome": "triaged", "causal_reach": None,
       "interaction_frame": None, "all_reaches": REACHES, "flagged": False}


def _window(saved):
    _KEEP.append(QApplication.instance() or QApplication([]))
    w = SimpleNamespace(
        _q_widgets={}, _segment_list=[SEG], _current_seg_idx=0,
        _panel_kind="both_uncertain", _pinned_reach=None, _segments_data={},
        _review_records={3: saved}, n_frames=5000, viewer=None,
        _candidate_reach_ids=lambda seg: None, _goto_frame=lambda *a: None,
        _jump_to_selected_reach=lambda *a: None)
    for name in ("_snap_to_detected_reach", "_read_reach_pick",
                 "_collect_answers_unchecked", "_segmentation_is_human"):
        setattr(w, name, getattr(W, name).__get__(w))
    _KEEP.append(W._make_outcome_picker(w, SEG, require_pick=True))
    _KEEP.append(W._make_reach_picker(w, SEG))
    w._restored_panel = None
    W._restore_answers(w, 3)
    return w


def _saved(outcome, reach=None, answers=None):
    rec = io.build_segment_record(
        segment_num=3, pellet_num=3, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome, human_causal_reach=reach,
        is_phantom=False, agreed=False, answers=answers or {"triage_kind": "both_uncertain"},
        notes="", segment_span={"start": 1000, "end": 2000})
    rec["reviewer"], rec["reviewed_at"] = "reviewer_a", "2099-01-01T00:00:00"
    return rec


def test_the_picked_reach_is_put_back_by_its_frames_after_a_renumbering():
    saved = _saved("retrieved", {"start": 1200, "end": 1215, "reach_id": 9})
    w = _window(saved)
    pick = W._read_reach_pick(w, w._q_widgets["_reach_picker"])
    assert (pick["start"], pick["end"]) == (1200, 1215)


def test_stepping_past_an_unchanged_answer_keeps_the_stored_record():
    saved = _saved("retrieved", {"start": 1200, "end": 1215, "reach_id": 9})
    w = _window(copy.deepcopy(saved))
    rec = W._collect_answers(w)
    assert rec["human"]["causal_reach"]["start"] == 1200
    assert rec["reviewer"] == "reviewer_a" and rec["reviewed_at"] == "2099-01-01T00:00:00"
    assert io._answer_signature(rec) == io._answer_signature(saved)


def test_an_abnormal_exception_answer_is_shown_and_kept():
    saved = _saved("abnormal_exception", None,
                   {"triage_kind": "both_uncertain", "outcome_pick": "abnormal_exception",
                    "abnormal_ranges": [{"start_frame": 1300, "end_frame": 1340, "reason": ""}]})
    w = _window(copy.deepcopy(saved))
    assert w._q_widgets["_outcome_combo"].currentText() == "abnormal_exception"
    rec = W._collect_answers(w)
    assert rec["human"]["outcome"] == "abnormal_exception"
    assert rec["answers"]["abnormal_ranges"] == saved["answers"]["abnormal_ranges"]


def test_a_changed_answer_is_collected_as_the_persons_new_answer():
    saved = _saved("retrieved", {"start": 1200, "end": 1215, "reach_id": 9})
    w = _window(copy.deepcopy(saved))
    w._q_widgets["_outcome_combo"].setCurrentText("displaced_sa")
    rec = W._collect_answers(w)
    assert rec["human"]["outcome"] == "displaced_sa"
    assert "reviewer" not in rec          # stamped by the save, as a new answer


def test_changing_an_answer_and_changing_it_back_saves_what_is_on_screen():
    """Restore A, change to B (stored), change back to A: A is what the person
    now says, not the stored B."""
    saved = _saved("retrieved", {"start": 1200, "end": 1215, "reach_id": 9})
    w = _window(copy.deepcopy(saved))
    combo = w._q_widgets["_outcome_combo"]
    combo.setCurrentText("displaced_sa")
    w._review_records[3] = W._collect_answers(w)          # Save Review stores B
    combo.setCurrentText("retrieved")
    rec = W._collect_answers(w)
    assert rec["human"]["outcome"] == "retrieved"
