"""The review window records how long a person spends on each pellet.

WHY: "time to settle one pellet" used to be estimated from the gaps between
save times, which measure the session, not the pellet. Each segment's record
now carries ``timing``: when the pellet was shown, when its answer was taken,
and the seconds the person was actively at the window in between (window
active, input within the last minute). A revisit adds a visit and never
overwrites one. Records saved before this have no ``timing`` and must keep
working everywhere.

No GUI is needed: the clock takes an injected time source, and the widget's
record-storing method is called on a stand-in object.

The video id is synthetic (see test_review_durability.py for why).
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.review import causal_review_io as crio
from mousereach.review import pellet_timing as pt
from mousereach.review.causal_review_io import (
    build_segment_record, is_answered, restore_review_for_segments,
    save_causal_review, update_corpus_index,
)

STEM = "20990101_ZZZ9901_P1"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def nas(tmp_path, monkeypatch):
    from mousereach.config import Paths
    root = tmp_path / "nas"
    root.mkdir()
    monkeypatch.setattr(Paths, "NAS_ROOT", root, raising=False)
    return root


def _rec(n, outcome=None):
    return build_segment_record(
        segment_num=n, pellet_num=n, algo_outcome="triaged", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome, human_causal_reach=None,
        is_phantom=False, agreed=False,
        answers={"reviewed": False} if outcome is None else {"triage_kind": "outcome_uncertain"},
        segment_span={"start": 1000 * n, "end": 1000 * n + 999})


# --- the active-time clock ------------------------------------------------

def _tick_for(acc, clock, seconds, active=True):
    for _ in range(int(seconds)):
        clock.t += 1.0
        acc.tick(active)


def test_clock_counts_only_while_active_and_recently_used():
    clock = FakeClock()
    acc = pt.ActiveTimeAccumulator(idle_limit=60, clock=clock)
    acc.reset()                       # showing the pellet counts as input
    _tick_for(acc, clock, 30)
    assert acc.seconds == pytest.approx(30)
    _tick_for(acc, clock, 20, active=False)      # another program in front
    assert acc.seconds == pytest.approx(30)
    _tick_for(acc, clock, 30)                    # back, but no input for 50..80 s
    assert acc.seconds == pytest.approx(40)      # counts up to 60 s after the last input
    acc.note_input()
    _tick_for(acc, clock, 5)
    assert acc.seconds == pytest.approx(45)


def test_a_stalled_timer_adds_at_most_the_idle_limit():
    clock = FakeClock()
    acc = pt.ActiveTimeAccumulator(idle_limit=60, clock=clock)
    acc.reset()
    clock.t += 10
    acc.note_input()
    clock.t += 3600                   # the computer slept; input is an hour old
    acc.tick(True)
    assert acc.seconds == 0
    acc.note_input()
    clock.t += 40                     # one long load right after a click
    acc.tick(True)
    assert acc.seconds == pytest.approx(40)


def test_reset_starts_a_new_visit_from_zero():
    clock = FakeClock()
    acc = pt.ActiveTimeAccumulator(clock=clock)
    acc.reset()
    _tick_for(acc, clock, 10)
    acc.reset()
    assert acc.seconds == 0


# --- merging visits into a record -----------------------------------------

def test_a_revisit_adds_a_visit_and_never_overwrites():
    v1 = pt.make_visit("2026-11-14T10:00:00-06:00", None, 12.34, "triage_review")
    r = pt.add_visit(_rec(1, "triaged"), None, v1)
    assert r["timing"]["answered_at"] is None
    assert r["timing"]["active_seconds"] == 12.3
    v2 = pt.make_visit("2026-11-14T11:00:00-06:00", "2026-11-14T11:00:40-06:00", 30,
                       "triage_review")
    r2 = pt.add_visit(_rec(1, "retrieved"), r["timing"], v2)
    t = r2["timing"]
    assert [v["shown_at"] for v in t["visits"]] == [v1["shown_at"], v2["shown_at"]]
    assert t["shown_at"] == v1["shown_at"]                 # first time on screen
    assert t["answered_at"] == "2026-11-14T11:00:40-06:00"  # the answer that stands
    assert t["active_seconds"] == pytest.approx(42.3)       # total over visits
    assert t["tool"] == "triage_review"
    assert "timing" not in _rec(1, "retrieved")             # inputs untouched


def test_carry_keeps_timing_and_leaves_untimed_records_alone():
    r = pt.add_visit(_rec(1, "retrieved"), None,
                     pt.make_visit("2026-11-14T10:00:00-06:00", "2026-11-14T10:00:09-06:00",
                                   9, "resegmentation_review"))
    again = pt.carry_timing(_rec(1, "retrieved"), r["timing"])
    assert again["timing"] == r["timing"]
    plain = _rec(1, "retrieved")
    assert pt.carry_timing(plain, None) is plain
    assert pt.carry_timing(plain, "garbage") is plain


def test_now_iso_carries_a_utc_offset():
    s = pt.now_iso()
    assert s[-6] in "+-" and s[-3] == ":"


# --- the widget's store path, without a GUI -------------------------------

def _stand_in(stem=STEM):
    from mousereach.review.causal_review_widget import CausalReviewWidget as W
    clock = FakeClock()
    w = SimpleNamespace(
        _video_stem=stem, _review_records={}, _carried_timing={}, _open_visit=None,
        _pellet_clock=pt.ActiveTimeAccumulator(clock=clock),
        _deep_review=False, _triage_only=True)
    for name in ("_begin_visit", "_store_record", "_timing_tool"):
        setattr(w, name, getattr(W, name).__get__(w))
    return w, clock


def test_widget_store_closes_one_visit_and_stamps_answer_only_when_answered():
    w, clock = _stand_in()
    w._begin_visit(1)
    _tick_for(w._pellet_clock, clock, 7)
    w._store_record(_rec(1, "triaged"))                     # left open
    t = w._review_records[1]["timing"]
    assert t["answered_at"] is None and t["active_seconds"] == 7
    assert t["tool"] == "triage_review"
    # Stored again without being shown again (a save right after Next):
    # same timing, no new visit.
    w._store_record(_rec(1, "triaged"))
    assert len(w._review_records[1]["timing"]["visits"]) == 1
    # Shown again and answered: a second visit.
    w._begin_visit(1)
    w._begin_visit(1)                                       # re-show keeps the visit
    _tick_for(w._pellet_clock, clock, 5)
    w._store_record(_rec(1, "retrieved"))
    rec = w._review_records[1]
    assert is_answered(rec)
    assert len(rec["timing"]["visits"]) == 2
    assert rec["timing"]["answered_at"] is not None
    assert rec["timing"]["active_seconds"] == 12


def test_widget_uses_carried_timing_from_an_earlier_session():
    w, clock = _stand_in()
    earlier = pt.add_visit(_rec(2, "triaged"), None,
                           pt.make_visit("2026-11-13T09:00:00-06:00", None, 20,
                                         "triage_review"))["timing"]
    w._carried_timing = {2: earlier}
    w._begin_visit(2)
    _tick_for(w._pellet_clock, clock, 4)
    w._store_record(_rec(2, "displaced_sa"))
    t = w._review_records[2]["timing"]
    assert len(t["visits"]) == 2 and t["active_seconds"] == 24


def test_tool_names_follow_the_mode():
    w, _ = _stand_in()
    assert w._timing_tool() == "triage_review"
    w._deep_review = True
    assert w._timing_tool() == "resegmentation_review"


# --- saved files: timing survives, old files still work -------------------

def test_timing_is_saved_restored_and_indexed(tmp_path, nas, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    timed = pt.add_visit(_rec(1, "retrieved"), None,
                         pt.make_visit("2026-11-14T10:00:00-06:00",
                                       "2026-11-14T10:00:30-06:00", 25, "triage_review"))
    monkeypatch.setattr(crio, "_get_timestamp", lambda: "2026-11-14T10:00:31")
    path = save_causal_review(STEM, folder, [timed, _rec(2)], provenance={},
                              reviewer="reviewer_a")
    doc = json.loads(path.read_text(encoding="utf-8"))
    by = {s["segment_num"]: s for s in doc["segments"]}
    assert by[1]["timing"]["active_seconds"] == 25
    assert "timing" not in by[2]

    current = [{"segment_num": n, "start_frame": 1000 * n, "end_frame": 1000 * n + 999}
               for n in (1, 2)]
    restored, unmatched, _ = restore_review_for_segments(doc, current)
    assert restored[1]["timing"] == by[1]["timing"]
    assert unmatched == []

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    ip = update_corpus_index(STEM, path, [by[1]], "reviewer_a", "2026-11-14T10:00:31",
                             index_dir=index_dir)
    entries = json.loads(ip.read_text(encoding="utf-8"))["entries"]
    assert entries[f"{STEM}__seg1"]["timing"]["active_seconds"] == 25


def test_timing_alone_never_changes_who_answered(tmp_path, nas, monkeypatch):
    """A revisit that only adds time is not a new answer: the per-segment
    reviewer stays with whoever gave the answer."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    monkeypatch.setattr(crio, "_get_timestamp", lambda: "2026-11-01T09:00:00")
    save_causal_review(STEM, folder, [_rec(1, "retrieved")], provenance={},
                       reviewer="reviewer_a")
    looked_again = pt.add_visit(_rec(1, "retrieved"), None,
                                pt.make_visit("2026-11-14T10:00:00-06:00",
                                              "2026-11-14T10:00:05-06:00", 5,
                                              "triage_review"))
    monkeypatch.setattr(crio, "_get_timestamp", lambda: "2026-11-14T10:00:06")
    path = save_causal_review(STEM, folder, [looked_again], provenance={},
                              reviewer="reviewer_b")
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert seg["reviewer"] == "reviewer_a"
    assert seg["timing"]["active_seconds"] == 5


def test_old_records_without_timing_still_read_everywhere(tmp_path):
    from mousereach.review.causal_review_io import apply_review_overrides
    from mousereach.review.triage_status import resolved_segments
    old = {"type": "causal_review", "schema_version": "1.0", "segments": [_rec(1, "retrieved")]}
    assert resolved_segments(old) == {1}
    current = [{"segment_num": 1, "start_frame": 1000, "end_frame": 1999}]
    restored, _, _ = restore_review_for_segments(old, current)
    assert "timing" not in restored[1]
    outcomes = {"segments": [{"segment_num": 1, "outcome": "triaged"}]}
    applied = apply_review_overrides(outcomes, {1: old["segments"][0]})
    assert applied["segments"][0]["outcome"] == "retrieved"
