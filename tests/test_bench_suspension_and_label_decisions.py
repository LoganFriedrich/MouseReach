"""Sessions whose hand-score comparison the lab suspended, and a person's
decision about a label mismatch (watcher/hand_scores.suspension,
record_decision; watcher/label_decision_cli.py).

WHY: for sessions whose tray labels disagree with the
sheet, while nobody knows whether the file names or the sheet rows are the
swapped ones, "we can't draw any conclusions about manual score mismatch.
Leave that element out for now when calling triage on exclusively these
videos." And: "the sheets just are what they are, and if they're wrong ...
that speaks to human error rate, which is kind of something we want to
capture by leaving it in" -- so a label mismatch is released by a person's
decision (the name is wrong, or the sheet is a human error kept as it is and
recorded), never by asking anyone to edit the sheet.
"""
import json

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.review_return as rr
import mousereach.watcher.review_gate as rg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_label_mismatch_hold import (  # noqa: E402
    CALLS, OTHER, P3_ROW, P4_ROW, VID, FakeDB,
                                            _bench_bundle, _doc, _orch, _outcomes,
                                            _setup, _wire, _with)

ENTRY = {"video_key": VID, "reason": "tray labels disagree with the sheet; which is "
         "swapped is not known yet", "set_by": "the lab", "set_at": "2026-11-18T10:00:00"}


def _suspend(monkeypatch, entries):
    cfg = dict(hs.settings())
    cfg["suspended_sessions"] = entries
    monkeypatch.setattr(hs, "settings", lambda: cfg)


def _bench_flagged(doc):
    return {s["segment_num"] for s in doc["segments"]
            if str(s.get("flag_reason") or "").startswith("bench")}


# ----------------------------------------------------------- suspension --

def test_a_suspended_session_gets_no_bench_flags_and_no_label_hold(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    _outcomes(folder, CALLS, flags={4: "bench disagreement: the hand score says untouched; "
                                       "the algorithm says displaced_sa"})
    _suspend(monkeypatch, [ENTRY])
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    doc = _doc(folder)
    assert _bench_flagged(doc) == set()
    # Every other reason to triage still stands: pellet 13 the detector could not call.
    assert doc["segments"][12]["flagged_for_review"] is True
    assert doc["hand_scores"]["verdict"] == "suspended"
    assert "tray labels disagree" in doc["hand_scores"]["reason"]
    # The review window's answer check has no score to weigh, and says why.
    score, why = hs.score_from_outcomes(doc, 2)
    assert score is None and "suspended" in why and "tray labels" in why
    assert not orch._held_for_hand_scores(VID)


def test_other_sessions_are_compared_as_before(tmp_path, monkeypatch):
    own = _with(P4_ROW, {2: 0})
    folder = _setup(tmp_path, monkeypatch, {VID: own})
    _suspend(monkeypatch, [dict(ENTRY, video_key=OTHER)])
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True
    assert _bench_flagged(_doc(folder)) == {2}


def test_removing_the_entry_restores_the_comparison(tmp_path, monkeypatch):
    own = _with(P4_ROW, {2: 0})
    folder = _setup(tmp_path, monkeypatch, {VID: own})
    _suspend(monkeypatch, [ENTRY])
    assert hs.check(VID, folder).status == hs.SUSPENDED
    _suspend(monkeypatch, [])
    assert hs.check(VID, folder).status == hs.DISAGREES


def test_bundles_already_in_triage_are_reevaluated_without_bench_flags(tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    # Held for bench reasons only, nothing else open, nobody answered: released.
    calls = dict(P4_ROW)
    only_bench = _bench_bundle(triage, _with(P4_ROW, {2: 0, 5: 1}), calls=calls)
    # Held for bench reasons AND a pellet the detector could not call: kept,
    # bench flags taken back so the window does not show them.
    other = "20250101_ABC0103_P3"
    with_other = _bench_bundle(triage, P3_ROW, calls=CALLS, stem=other)
    _setup(tmp_path, monkeypatch, {VID: _with(P4_ROW, {2: 0, 5: 1}), other: P3_ROW},
           calls=calls, folder=tmp_path / "unused")
    _suspend(monkeypatch, [ENTRY, dict(ENTRY, video_key=other)])
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert [a[1] for a in returned] == [VID]
    assert summary.get("bench_refreshed") == 1
    doc = _doc(with_other, other)
    assert _bench_flagged(doc) == set()
    assert doc["segments"][12]["flagged_for_review"] is True     # still asked
    assert doc["hand_scores"]["verdict"] == "suspended"
    assert only_bench.exists()        # the return itself is stubbed here


# ------------------------------------------------- decisions on a mismatch --

def test_a_sheet_error_decision_files_it_with_the_disagreement_recorded(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    seg = folder / f"{VID}_segments.json"
    seg.write_text(json.dumps({"overall_confidence": 0.9,
                               "boundaries": [1000 * i for i in range(21)]}), encoding="utf-8")
    import os
    import time
    os.utime(seg, (time.time() - 3600,) * 2)
    _outcomes(folder, P4_ROW)               # every pellet called: nothing else to ask
    orch = _orch(tmp_path, monkeypatch)
    # A sheet-error decision recorded while the label-mismatch hold existed
    # (since retired) still stands: filed with the disagreement recorded.
    mm = hs.label_mismatch(VID, hs.algo_classes(_doc(folder)), {VID: P3_ROW, OTHER: P4_ROW})
    hs.record_decision(VID, hs.DECISION_SHEET_ERROR, "a reviewer", mismatch=mm)
    assert orch._hand_score_precheck(VID, folder) is True       # goes on
    doc = _doc(folder)
    assert _bench_flagged(doc) == set()
    kept = doc["hand_scores"]
    assert kept["verdict"] == "sheet_error"
    assert kept["decision"]["by"] == "a reviewer"
    assert kept["label_mismatch"]["matched_rows"][0]["key"] == OTHER
    decision, _, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN                        # filed, not triaged
    # The record can be counted: one line, who, when, which rows.
    lines = hs.decisions_path().read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[0])
    assert (e["decision"], e["own_row"]["key"], e["matched_rows"][0]["key"]) == (
        "sheet_error", VID, OTHER)
    assert e["at"]


def test_a_recorded_name_wrong_decision_changes_nothing_now(tmp_path, monkeypatch):
    """Retired hold: the video is compared with its own row."""
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    hs.record_decision(VID, hs.DECISION_NAME_WRONG, "a reviewer", new_name=OTHER)
    assert hs.check(VID, folder).status == hs.DISAGREES
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    assert _bench_flagged(_doc(folder))                          # asked in triage


def test_a_decision_needs_who_and_for_a_name_the_right_name(tmp_path, monkeypatch):
    import pytest
    _orch(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        hs.record_decision(VID, hs.DECISION_SHEET_ERROR, "")
    with pytest.raises(ValueError):
        hs.record_decision(VID, hs.DECISION_NAME_WRONG, "a reviewer")
    assert hs.decision_for(VID) is None


def test_labels_right_compares_as_usual_and_a_real_disagreement_goes_to_triage(
        tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    hs.record_decision(VID, hs.DECISION_LABELS_RIGHT, "a reviewer")
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    assert _bench_flagged(_doc(folder))                          # asked in triage
    assert hs.check(VID, folder).status == hs.DISAGREES
