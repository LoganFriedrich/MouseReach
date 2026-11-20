"""The label-mismatch hold is RETIRED (owner rule 2026-11-20: "Just trust what
is written down"). A video whose results match ANOTHER tray's hand scores
better than its own is no longer held: it is compared with its OWN row, like
every video, and goes on as named -- a disagreement goes to triage as an
ordinary bench disagreement. hand_scores.label_mismatch() itself is kept (it
describes the rows in a sheet-error decision recorded while the hold existed),
so its rule is still tested.

Still true from 2026-11-18: a triage bundle held over its hand scores is
compared again when the scores change: a corrected sheet must not leave a
person looking at old scores and disagreements that no longer exist, and an
unanswered bundle whose only questions went away must go back to the pipeline
and be filed.
"""
import json
import os
import time

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

VID = "20250101_ABC0102_P3"
OTHER = "20250101_ABC0102_P4"


def _with(row, changes):
    out = dict(row)
    out.update(changes)
    return out


# The algorithm's calls for VID: they match P4's row on every pellet and P3's
# on only some. 0 untouched, 1 displaced, 2 retrieved.
CALLS = {1: 2, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
         11: 1, 12: 1, 14: 2, 15: 1, 16: 2, 17: 1, 18: 1, 19: 2, 20: 1}
P4_ROW = _with(CALLS, {13: 2})
P3_ROW = {1: 2, 2: 1, 3: 1, 4: 0, 5: 1, 6: 2, 7: 2, 8: 1, 9: 2, 10: 2,
          11: 2, 12: 1, 13: 1, 14: 1, 15: 2, 16: 2, 17: 1, 18: 2, 19: 1, 20: 2}
P1_ROW = {n: (1 if n % 2 else 2) for n in range(1, 21)}
WORD = {0: "untouched", 1: "displaced_sa", 2: "retrieved"}


def _write_scores(tmp_path, rows):
    f = tmp_path / "hand.csv"
    lines = ["video_key,pellet,score"]
    for key, row in rows.items():
        for p, s in sorted(row.items()):
            lines.append("%s,%d,%d" % (key, p, s))
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    f.with_suffix(".json").write_text(json.dumps(
        {"last_successful_import": {"ABC_01": "2026-11-17T20:00:00"}}), encoding="utf-8")
    # A new modification time each write, as a real re-import gives.
    t = time.time() + len(list(tmp_path.glob("*"))) + _write_scores.n
    _write_scores.n += 10
    os.utime(f, (t, t))
    os.utime(f.with_suffix(".json"), (t, t))
    hs._CACHE.update(key=None)
    return f


_write_scores.n = 0


def _setup(tmp_path, monkeypatch, rows, calls=CALLS, folder=None, answered=()):
    f = _write_scores(tmp_path, rows)
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": ["ABC"]})
    folder = folder or (tmp_path / "work")
    folder.mkdir(parents=True, exist_ok=True)
    _outcomes(folder, calls)
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set(answered))
    return folder


def _outcomes(folder, calls, stem=VID, flags=None):
    segs = []
    for n in range(1, 21):
        s = {"segment_num": n, "outcome": WORD[calls[n]] if n in calls else "triaged"}
        if n not in calls:
            s.update(flagged_for_review=True, flag_reason="fell_through_all_committing_stages")
        if flags and n in flags:
            s.update(flagged_for_review=True, flag_reason=flags[n])
        segs.append(s)
    (folder / f"{stem}_pellet_outcomes.json").write_text(json.dumps({"segments": segs}),
                                                         encoding="utf-8")


def _doc(folder, stem=VID):
    return json.loads((folder / f"{stem}_pellet_outcomes.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the rule --

def test_a_swap_is_a_label_mismatch_naming_the_other_tray():
    scores = {VID: P3_ROW, OTHER: P4_ROW, "20250101_ABC0102_P1": P1_ROW}
    mm = hs.label_mismatch(VID, CALLS, scores)
    assert mm is not None
    assert [(m.key, m.agree, m.compared) for m in mm.matches] == [(OTHER, 19, 19)]
    assert (mm.own.agree, mm.own.compared) == (7, 19)
    text = hs.describe_mismatch(mm)
    assert "P4 (19/19)" in text and "not P3's (7/19)" in text
    assert "--sheet-error" in text and "--name-wrong" in text and "manual_corrections" in text
    assert "correct the sheet" not in text


def test_a_genuine_disagreement_with_no_better_row_is_not_one():
    # Several pellets disagree with the own row, and no other tray matches.
    own = _with(CALLS, {13: 1})
    for n in (2, 3, 4, 6, 7):
        own[n] = 0
    scores = {VID: own, "20250101_ABC0102_P1": P1_ROW}
    assert hs.label_mismatch(VID, CALLS, scores) is None


def test_trays_that_merely_look_alike_are_not_a_swap():
    # Another tray identical to the own row except on 3 pellets the algorithm
    # sides with it on: below the margin, a scoring question, not a label.
    other = dict(P4_ROW)
    own = dict(P4_ROW)
    for n in (2, 3, 4):
        own[n] = 0
    assert hs.label_mismatch(VID, CALLS, {VID: own, OTHER: other}) is None
    for n in (2, 3, 4, 6):
        own[n] = 0
    assert hs.label_mismatch(VID, CALLS, {VID: own, OTHER: other}) is not None


def test_too_few_pellets_cannot_tell_a_swap_from_chance():
    few = {n: CALLS[n] for n in list(CALLS)[:9]}
    assert hs.label_mismatch(VID, few, {VID: {n: 0 for n in few}, OTHER: P4_ROW}) is None


# ------------------------------------------------------- the pipeline hold --

def _orch(tmp_path, monkeypatch):
    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    return orch


def _attention(tmp_path):
    return json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json")
                      .read_text())["items"]


def test_a_swap_is_not_held_it_goes_on_as_named(tmp_path, monkeypatch):
    """Retired hold: compared with its OWN row, its disagreements flagged for
    triage like any other, nothing listed as a label mismatch."""
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    assert hs.check(VID, folder).status == hs.DISAGREES
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    assert not orch._held_for_hand_scores(VID)
    assert orch._waiting_for_hand_scores(VID) is False
    att = tmp_path / "nas" / "attention" / "HOST_hand_scores.json"
    assert not att.exists() or all(i.get("kind") != "label_mismatch"
                                   for i in _attention(tmp_path))
    doc = _doc(folder)
    assert doc["hand_scores"]["verdict"] == "disagrees"
    assert "label_mismatch" not in doc["hand_scores"]
    bench = {s["segment_num"] for s in doc["segments"]
             if str(s.get("flag_reason") or "").startswith("bench")}
    assert bench == {n for n, c in CALLS.items() if P3_ROW[n] != c}
    decision, _, _ = rg.evaluate_gate(VID, folder)
    assert decision != rg.DECISION_CLEAN                         # triage, not filed


def test_a_genuine_disagreement_still_goes_to_triage(tmp_path, monkeypatch):
    own = dict(P4_ROW)
    own[2] = 0
    folder = _setup(tmp_path, monkeypatch, {VID: own, "20250101_ABC0102_P1": P1_ROW})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    seg2 = _doc(folder)["segments"][1]
    assert seg2["flagged_for_review"] and seg2["flag_reason"].startswith("bench disagreement")
    assert not orch._held_for_hand_scores(VID)


def test_a_hold_written_before_the_retirement_goes_on_after_a_restart(
        tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    ready = orch.db.get_video(VID).get("processing_started_at") or "2026-11-19T10:00:00"
    if not orch.db.get_video(VID).get("processing_started_at"):
        orch.db.conn.execute("UPDATE videos SET processing_started_at=? WHERE video_id=?",
                             (ready, VID)) if hasattr(orch.db, "conn") else None
    att = tmp_path / "nas" / "attention"
    att.mkdir(parents=True, exist_ok=True)
    (att / "HOST_hand_scores.json").write_text(json.dumps({"items": [{
        "kind": "label_mismatch", "needs_person": True, "video_id": VID,
        "cohort_id": "ABC_01", "ready_since": ready, "says": "old hold"}]}),
        encoding="utf-8")
    restored = orch._restore_score_waits()
    for w in restored.values():
        assert "label_mismatch" not in w                  # a plain hold now
    orch._score_waits = restored
    assert orch._waiting_for_hand_scores(VID) is False   # checked again at once
    assert orch._hand_score_precheck(VID, folder) is True
    assert all(i.get("kind") != "label_mismatch" for i in _attention(tmp_path))


def test_census_counts_it(tmp_path, monkeypatch):
    from mousereach.census import runner
    assert "label_mismatch" in runner.HAND_SCORE_HOLD_KINDS


# ------------------------------------------------- bundles already in triage --

def _wire(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path, raising=False)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    import mousereach.archive.core as ac
    monkeypatch.setattr(ac, "get_archive_destination",
                        lambda stem: tmp_path / "Analyzed" / "ABC" / "ABC01")
    monkeypatch.setattr(rr, "BENCH_RECHECK_QUIET_S", 0)
    rr._bench_cache.clear()
    return triage


def _bench_bundle(triage, row_compared, calls=CALLS, stem=VID):
    """A triage bundle as the gate left it: outcomes with a bench flag on
    every pellet that disagreed with ``row_compared``, and that row kept."""
    b = triage / stem
    b.mkdir(parents=True)
    # Sound cuts, older than the outcomes (the gate and the return scan read
    # newer cuts as outcomes out of date).
    seg = b / f"{stem}_segments.json"
    seg.write_text(json.dumps({"overall_confidence": 0.9,
                               "boundaries": [1000 * i for i in range(21)]}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(seg, (old, old))
    flags = {n: "bench disagreement: the hand score says %s; the algorithm says %s"
             % (hs.HAND_WORDS[row_compared[n]], WORD[c])
             for n, c in calls.items() if row_compared.get(n) != c}
    _outcomes(b, calls, stem=stem, flags=flags)
    doc = _doc(b, stem)
    doc["hand_scores"] = {"pellets": {str(k): v for k, v in sorted(row_compared.items())},
                          "verdict": "disagrees"}
    (b / f"{stem}_pellet_outcomes.json").write_text(json.dumps(doc), encoding="utf-8")
    return b


class FakeDB:
    def get_video(self, vid):
        return {"video_id": vid, "state": "triage"}

    def states_of(self, vids):
        return {}


def test_a_bundle_in_triage_over_a_swap_stays_in_triage(tmp_path, monkeypatch):
    """Retired hold: a bundle whose results match another tray is not moved out
    of triage; the person answers its disagreements like any other."""
    triage = _wire(monkeypatch, tmp_path)
    _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    b = _bench_bundle(triage, P3_ROW)
    review = b / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"segments": [{"segment_num": 4, "answers": {}}]}),
                      encoding="utf-8")
    before = review.read_bytes()
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("moved")))
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert not summary.get("bench_label_mismatch")
    assert b.exists()
    assert review.read_bytes() == before            # the answers are not touched


def test_an_unanswered_bench_bundle_whose_sheet_was_corrected_goes_back_and_is_filed(
        tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    calls = dict(P4_ROW)                      # every pellet called, nothing else open
    old_row = _with(P4_ROW, {2: 0, 5: 1})
    b = _bench_bundle(triage, old_row, calls=calls)
    _setup(tmp_path, monkeypatch, {VID: P4_ROW}, calls=calls, folder=tmp_path / "unused")
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert [a[4] for a in returned] == ["hand_scores_agree"]
    assert summary.get("bench_release") == 1
    # On the return the pipeline compares again: the stale flags go, and the
    # gate has nothing to ask -- the video goes on to kinematics and filing.
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, b) is True
    assert not any(s.get("flagged_for_review") for s in _doc(b)["segments"])
    decision, _, _ = rg.evaluate_gate(VID, b)
    assert decision == rg.DECISION_CLEAN


def test_a_still_disagreeing_bundle_shows_the_current_scores(tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    calls = dict(P4_ROW)
    old_row = _with(P4_ROW, {2: 0, 5: 1})
    new_row = _with(P4_ROW, {7: 2})          # corrected sheet: 2 and 5 fixed, 7 differs
    b = _bench_bundle(triage, old_row, calls=calls)
    _setup(tmp_path, monkeypatch, {VID: new_row}, calls=calls, folder=tmp_path / "unused")
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("moved")))
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert summary.get("bench_refreshed") == 1
    doc = _doc(b)
    flagged = {s["segment_num"] for s in doc["segments"]
               if str(s.get("flag_reason") or "").startswith("bench")}
    assert flagged == {7}
    assert doc["hand_scores"]["pellets"]["2"] == 1 and doc["hand_scores"]["pellets"]["7"] == 2
    ok, _ = hs.score_from_outcomes(doc, 2)
    assert ok == 1                             # the window shows the corrected score


def test_a_corrected_sheet_never_loses_a_persons_answers(tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    calls = dict(P4_ROW)
    old_row = _with(P4_ROW, {2: 0, 5: 1})
    b = _bench_bundle(triage, old_row, calls=calls)
    review = b / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"segments": [
        {"segment_num": 2, "human": {"outcome": "displaced_sa"},
         "answers": {"reviewed": True}}]}), encoding="utf-8")
    before = review.read_bytes()
    _setup(tmp_path, monkeypatch, {VID: P4_ROW}, calls=calls, folder=tmp_path / "unused",
           answered={2})
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a[4]) or True)
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    # A person answered something: kept in triage, refreshed in place, the
    # answer untouched; the disagreement that went away (pellet 5) is no
    # longer flagged, the answered pellet keeps its flag as the record of why
    # it was asked.
    assert summary.get("bench_refreshed") == 1 and returned == []
    assert review.read_bytes() == before
    flagged = {s["segment_num"] for s in _doc(b)["segments"]
               if str(s.get("flag_reason") or "").startswith("bench")}
    assert flagged == {2}
    assert _doc(b)["hand_scores"]["verdict"] == "agrees"
    # The next scan: every remaining question is answered, so the ordinary
    # return releases it -- nothing is stranded.
    import time as _t
    old = _t.time() - 7200
    for f in b.iterdir():
        os.utime(f, (old, old))
    rr._status_cache.clear()
    rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert returned == ["triage_cleared"]
    assert review.read_bytes() == before


def test_a_bundle_being_answered_now_is_left_alone(tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    monkeypatch.setattr(rr, "BENCH_RECHECK_QUIET_S", 3600)
    _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    _bench_bundle(triage, P3_ROW)
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("moved")))
    rr.scan_review_queues(FakeDB(), tmp_path / "proc")


def test_the_decision_command_refuses_new_decisions_and_keeps_the_record(
        tmp_path, monkeypatch, capsys):
    from mousereach.watcher import label_decision_cli as cli
    _orch(tmp_path, monkeypatch)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    hs.record_decision(VID, hs.DECISION_SHEET_ERROR, "a reviewer")    # the history
    for args in (["--sheet-error", OTHER, "--by", "x"],
                 ["--name-wrong", OTHER, "--as", VID, "--by", "x"],
                 ["--labels-right", OTHER, "--by", "x"]):
        assert cli.main(args) == 1
        assert "retired" in capsys.readouterr().err
    lines = hs.decisions_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1                                         # nothing recorded
    assert cli.main(["--decisions"]) == 0
    assert VID in capsys.readouterr().out
    assert cli.main(["--list"]) == 0
    assert "retired" in capsys.readouterr().out


def test_a_suspended_bundle_with_another_question_is_not_released_again_and_again(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    triage = _wire(monkeypatch, tmp_path)
    calls = dict(P4_ROW)
    b = _bench_bundle(triage, _with(P4_ROW, {2: 0}), calls=calls)
    f = _write_scores(tmp_path, {VID: _with(P4_ROW, {2: 0})})
    monkeypatch.setattr(hs, "settings", lambda: {
        "path": str(f), "required_projects": ["ABC"],
        "suspended_sessions": [{"video_key": VID, "reason": "labels in doubt"}]})
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    # The scan's own reading says pellet 9's causal reach is open (no outcome flag).
    st = SimpleNamespace(unresolved={2, 9})
    sig = rr._bundle_signature(b)
    assert rr.bench_recheck(b, VID, sig, st) == rr.BENCH_REFRESHED
    doc = _doc(b)
    assert not any(str(s.get("flag_reason") or "").startswith("bench") for s in doc["segments"])
    st = SimpleNamespace(unresolved={9})
    assert rr.bench_recheck(b, VID, rr._bundle_signature(b), st) is rr.BENCH_NOTHING
