"""A speed bump before saving an answer that BOTH the algorithm and the tray
sheet's hand score contradict.

WHY: answers that look like reviewer fatigue were found among filed pellets
("retrieved" where the algorithm and the hand score both said displaced). The
person watching the video outranks both readings, so the review window never
blocks or changes the answer -- it asks once more, and "Take another look"
saves nothing and stays on the same question.
"""
import json
from types import SimpleNamespace

import pytest

import mousereach.watcher.hand_scores as hs
from mousereach.review import double_disagreement as dd
from mousereach.review.causal_review_widget import CausalReviewWidget

VID = "20250709_ABC0207_P1"
UNTOUCHED, DISPLACED, RETRIEVED = 0, 1, 2


# ---------------------------------------------------------------------------
# The decision (pure functions)
# ---------------------------------------------------------------------------

def test_disagreeing_with_both_asks():
    assert dd.double_disagreement("retrieved", "displaced_sa", DISPLACED)
    assert dd.double_disagreement("untouched", "retrieved", RETRIEVED)
    assert dd.double_disagreement("displaced_sa", "untouched", UNTOUCHED)


def test_agreeing_with_either_does_not_ask():
    # agrees with the sheet only
    assert not dd.double_disagreement("retrieved", "displaced_sa", RETRIEVED)
    # agrees with the algorithm only
    assert not dd.double_disagreement("displaced_sa", "displaced_sa", RETRIEVED)


def test_nothing_to_compare_does_not_ask():
    assert not dd.double_disagreement("retrieved", "displaced_sa", None)   # no hand score
    assert not dd.double_disagreement("retrieved", None, DISPLACED)        # no algo outcome
    assert not dd.double_disagreement("retrieved", "triaged", DISPLACED)   # algo undecided
    assert not dd.double_disagreement(None, "displaced_sa", DISPLACED)
    assert not dd.double_disagreement("abnormal", "displaced_sa", RETRIEVED)
    assert not dd.double_disagreement("retrieved", "displaced_sa", 7)      # not a sheet code


def test_the_sheet_cannot_tell_inside_from_outside_the_sa():
    # The sheet has one "displaced": outside-the-SA against it is agreement.
    assert not dd.disagrees_with_sheet("displaced_outside", DISPLACED)
    assert not dd.disagrees_with_sheet("displaced_sa", DISPLACED)
    assert not dd.double_disagreement("displaced_outside", "untouched", DISPLACED)


def test_the_algorithm_cannot_tell_left_the_sa_from_eaten():
    # The algorithm calls a pellet that left the SA unseen "retrieved".
    assert not dd.disagrees_with_algorithm("displaced_outside", "retrieved")
    assert not dd.double_disagreement("displaced_outside", "retrieved", UNTOUCHED)
    # ...but it did see a pellet land in the SA: retrieved against that differs.
    assert dd.disagrees_with_algorithm("retrieved", "displaced_sa")
    assert dd.disagrees_with_algorithm("displaced_outside", "displaced_sa")


def test_the_box_says_plainly_what_disagrees():
    t = dd.dialog_text("retrieved", "displaced_sa", DISPLACED)
    assert t.startswith("The algorithm said displaced (in SA). The tray sheet says "
                        "displaced. You answered retrieved.")
    t.encode("ascii")


# ---------------------------------------------------------------------------
# The hand score lookup (reuses the before-filing check's loader)
# ---------------------------------------------------------------------------

def _scores(tmp_path, monkeypatch, rows):
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n" + "".join(
        "%s,%d,%d\n" % (VID, p, s) for p, s in rows.items()), encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f)})
    hs._CACHE.update(key=None)


def test_hand_score_lookup(tmp_path, monkeypatch):
    _scores(tmp_path, monkeypatch, {3: 1})
    assert hs.pellet_hand_score(VID, 3) == (1, "")
    assert hs.pellet_hand_score(VID, 4)[0] is None
    assert hs.pellet_hand_score("20250709_ABC0299_P1", 3)[0] is None


def test_hand_score_lookup_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "settings", lambda: {})
    assert hs.pellet_hand_score(VID, 3)[0] is None
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(tmp_path / "missing.csv")})
    hs._CACHE.update(key=None)
    score, why = hs.pellet_hand_score(VID, 3)
    assert score is None and "could not be read" in why


# ---------------------------------------------------------------------------
# The review window's glue, on a stub (no napari viewer needed)
# ---------------------------------------------------------------------------

def _record(seg, human, algo):
    return {"segment_num": seg, "pellet_num": seg, "algo": {"outcome": algo},
            "human": {"outcome": human}, "answers": {"triage_kind": "outcome_uncertain"}}


class _Label:
    text = ""

    def setText(self, t):
        self.text = t


def _stub(record, answer_box, records=None):
    """A stand-in for the window with one segment open; ``answer_box`` is what
    the reviewer clicks (True = keep, False = take another look)."""
    calls = {"asked": [], "shown": 0, "saved_video": 0}
    stub = SimpleNamespace(
        _video_stem=VID, _review_records=dict(records or {}), _current_seg_idx=0,
        _triage_only=True, _status_label=_Label(),
        _collect_answers=lambda: json.loads(json.dumps(record)),
        _next_visible_idx=lambda i: i + 1,
    )
    stub._ask_keep_answer = lambda text: (calls["asked"].append(text), answer_box)[1]
    stub._show_current_segment = lambda: calls.__setitem__("shown", calls["shown"] + 1)
    stub._save_and_next_video = lambda: calls.__setitem__("saved_video", calls["saved_video"] + 1)
    stub._passes_double_check = lambda rec: CausalReviewWidget._passes_double_check(stub, rec)
    return stub, calls


@pytest.fixture(autouse=True)
def _quiet_qt(monkeypatch):
    import mousereach.review.causal_review_widget as w
    monkeypatch.setattr(w, "show_info", lambda *a, **k: None)


def test_take_another_look_saves_nothing_and_stays(tmp_path, monkeypatch):
    _scores(tmp_path, monkeypatch, {3: DISPLACED})
    stub, calls = _stub(_record(3, "retrieved", "displaced_sa"), answer_box=False)
    CausalReviewWidget._save_segment_and_advance(stub)
    assert len(calls["asked"]) == 1
    assert stub._review_records == {}          # nothing recorded
    assert stub._current_seg_idx == 0          # same question
    assert calls["shown"] == 0 and calls["saved_video"] == 0
    assert "LOOK AGAIN" in stub._status_label.text


def test_keep_my_answer_saves_and_advances_with_a_note(tmp_path, monkeypatch):
    _scores(tmp_path, monkeypatch, {3: DISPLACED})
    stub, calls = _stub(_record(3, "retrieved", "displaced_sa"), answer_box=True)
    CausalReviewWidget._save_segment_and_advance(stub)
    rec = stub._review_records[3]
    assert rec["human"]["outcome"] == "retrieved"   # never changed
    note = rec["answers"][dd.FIELD]
    assert note["shown"] and note["kept"]
    assert (note["algo_outcome"], note["hand_score"], note["hand_word"]) == (
        "displaced_sa", DISPLACED, "displaced")
    assert stub._current_seg_idx == 1 and calls["shown"] == 1
    # asked once: the later whole-video save does not ask about it again
    assert CausalReviewWidget._passes_double_check(stub, _record(3, "retrieved", "displaced_sa"))
    assert len(calls["asked"]) == 1


def test_no_box_when_either_agrees_or_no_hand_score(tmp_path, monkeypatch, capsys):
    _scores(tmp_path, monkeypatch, {3: RETRIEVED})
    stub, calls = _stub(_record(3, "retrieved", "displaced_sa"), answer_box=False)
    CausalReviewWidget._save_segment_and_advance(stub)
    assert calls["asked"] == [] and 3 in stub._review_records
    assert dd.FIELD not in stub._review_records[3]["answers"]

    # no hand-score row for this pellet: saved as usual, one line says so
    stub, calls = _stub(_record(5, "retrieved", "displaced_sa"), answer_box=False)
    CausalReviewWidget._save_segment_and_advance(stub)
    assert calls["asked"] == [] and 5 in stub._review_records
    out = capsys.readouterr().out
    assert "check not run" in out and "segment 5" in out
    out.encode("ascii")
