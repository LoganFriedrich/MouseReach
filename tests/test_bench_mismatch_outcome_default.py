"""A hand-score mismatch opens with the algorithm's outcome selected.

The bench check sends a finished pellet back when the hand-scored sheet
disagrees with the algorithm; the algorithm is usually right (Logan,
2026-10-24), so its outcome is pre-selected and "Save Segment + Next" means
"the algorithm was right". A pellet the algorithm could not decide still opens
on "-- select outcome --".
"""
from types import SimpleNamespace

from qtpy.QtWidgets import QApplication

from mousereach.review.causal_review_widget import (
    OUTCOME_CHOICES, OUTCOME_PICK_SENTINEL, CausalReviewWidget, is_bench_mismatch,
)

BENCH = ("bench disagreement (never reviewed): the hand score says retrieved; "
         "the algorithm says untouched")


_KEEP = []   # the panels: Qt deletes a dropdown whose panel is garbage-collected


def _combo(seg, require_pick=True):
    _KEEP.append(QApplication.instance() or QApplication([]))
    stub = SimpleNamespace(_q_widgets={})
    _KEEP.append(CausalReviewWidget._make_outcome_picker(stub, seg, require_pick=require_pick))
    return stub._q_widgets["_outcome_combo"]


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_a_hand_score_mismatch_opens_on_the_algorithms_outcome():
    combo = _combo({"outcome": "untouched", "flag_reason": BENCH})
    assert combo.currentText() == "untouched"
    assert OUTCOME_PICK_SENTINEL not in _items(combo)
    assert _items(combo) == OUTCOME_CHOICES          # every outcome can still be chosen


def test_a_pellet_the_algorithm_could_not_decide_still_asks():
    combo = _combo({"outcome": "triaged", "flag_reason": "fell_through_all_committing_stages"})
    assert combo.currentText() == OUTCOME_PICK_SENTINEL
    combo = _combo({"outcome": "untouched", "flag_reason": "cv_artifact_landing_no_pellet_arrival"})
    assert combo.currentText() == OUTCOME_PICK_SENTINEL


def test_the_mismatch_is_recognised_however_the_reason_continues():
    assert is_bench_mismatch(BENCH)
    assert is_bench_mismatch("Bench disagreement: the hand score says untouched")
    assert not is_bench_mismatch("reviewer escalated from triage: bench disagreement noted")
    assert not is_bench_mismatch(None)
