"""A video whose results match ANOTHER tray's hand scores is held as a label
mismatch, outside the review queues, and released when the sheet or the name
is corrected (watcher/hand_scores.label_mismatch, orchestrator.
_hold_label_mismatch, review_return.bench_recheck).

WHY (owner ruling 2026-11-18): "that kind of thing isn't for triage to do.
That's not something any triage process could repair, so it's not something
to put into triage over." A triage reason must be shown and fixable in the
review window; a swapped tray in the file name or the sheet is neither.

And a triage bundle held over its hand scores is compared again when the
scores change: a corrected sheet must not leave a person looking at old
scores and disagreements that no longer exist, and an unanswered bundle whose
only questions went away must go back to the pipeline and be filed.
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


def test_a_swap_is_held_outside_the_queues_not_triaged(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    # An old bench flag from a comparison with the wrong row is taken back.
    _outcomes(folder, CALLS, flags={4: "bench disagreement: the hand score says untouched; "
                                       "the algorithm says displaced_sa"})
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    monkeypatch.setattr(rg, "route_deep_review", lambda *a, **k: routed.append(a))
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is False      # held, before the gate
    assert routed == []
    assert orch.db.get_video(VID)["state"] == "processing"
    [item] = _attention(tmp_path)
    assert item["kind"] == "label_mismatch" and item["needs_person"] is True
    assert item["matched_rows"] == [{"key": OTHER, "agree": 19, "compared": 19}]
    assert item["own_row"]["agree"] == 7
    assert "not P3's (7/19)" in item["says"] and "held outside the review queues" in item["says"]
    doc = _doc(folder)
    assert not any(str(s.get("flag_reason") or "").startswith("bench")
                   for s in doc["segments"])
    assert doc["hand_scores"]["verdict"] == "label_mismatch"
    # held: not picked again while nothing changes
    assert orch._waiting_for_hand_scores(VID) is True


def test_a_genuine_disagreement_still_goes_to_triage(tmp_path, monkeypatch):
    own = dict(P4_ROW)
    own[2] = 0
    folder = _setup(tmp_path, monkeypatch, {VID: own, "20250101_ABC0102_P1": P1_ROW})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    seg2 = _doc(folder)["segments"][1]
    assert seg2["flagged_for_review"] and seg2["flag_reason"].startswith("bench disagreement")
    assert not orch._held_for_hand_scores(VID)


def test_a_corrected_sheet_releases_it(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is False
    # The sheet's P3 and P4 rows are swapped back; the next import rewrites the file.
    _write_scores(tmp_path, {VID: P4_ROW, OTHER: P3_ROW})
    assert orch._waiting_for_hand_scores(VID) is False          # checked again
    assert orch._hand_score_precheck(VID, folder) is True       # goes on
    assert not orch._held_for_hand_scores(VID)
    assert _attention(tmp_path) == []
    assert _doc(folder)["hand_scores"]["verdict"] == "agrees"


def test_a_corrected_name_releases_it(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is False
    # The name correction puts the real P3 recording's results under this
    # name (the originals kept in Archive/manual_corrections): its calls now
    # match P3's row. The scores file did not change.
    _outcomes(folder, {n: s for n, s in P3_ROW.items()})
    assert orch._waiting_for_hand_scores(VID) is False
    assert orch._hand_score_precheck(VID, folder) is True
    assert _attention(tmp_path) == []


def test_the_hold_survives_a_restart(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    orch = _orch(tmp_path, monkeypatch)
    orch._hand_score_precheck(VID, folder)
    restored = orch._restore_score_waits()
    assert restored[VID]["label_mismatch"] is True
    orch._score_waits = restored
    orch._write_score_attention()
    assert _attention(tmp_path)[0]["kind"] == "label_mismatch"


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


def test_a_bundle_in_triage_over_a_swap_moves_to_the_hold(tmp_path, monkeypatch):
    triage = _wire(monkeypatch, tmp_path)
    _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW})
    b = _bench_bundle(triage, P3_ROW)
    review = b / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"segments": [{"segment_num": 4, "answers": {}}]}),
                      encoding="utf-8")
    before = review.read_bytes()
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)
    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    assert [a[1] for a in returned] == [VID] and returned[0][4] == "label_mismatch"
    assert summary.get("bench_label_mismatch") == 1
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


def test_a_bundle_moved_to_the_hold_keeps_its_answers_through_the_real_return(
        tmp_path, monkeypatch):
    """No stubbed return: the bundle with a person's answer goes back to
    Processing through the ordinary return, the review file goes with it, and
    the pipeline's check holds it naming the answered pellet."""
    triage = _wire(monkeypatch, tmp_path)
    _setup(tmp_path, monkeypatch, {VID: P3_ROW, OTHER: P4_ROW}, folder=tmp_path / "unused")
    b = _bench_bundle(triage, P3_ROW)
    (b / f"{VID}.mp4").write_bytes(b"not a real video")
    (b / f"{VID}DLC_resnet50_testShuffle1_100.h5").write_bytes(b"not a real pose")
    review = b / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"segments": [
        {"segment_num": 4, "human": {"outcome": "displaced_sa"},
         "answers": {"reviewed": True}}]}), encoding="utf-8")
    before = review.read_bytes()
    import time as _t
    old = _t.time() - 7200
    for f in b.iterdir():
        os.utime(f, (old, old))
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: {4})
    proc = tmp_path / "proc"
    orch = _orch(tmp_path, monkeypatch)
    orch.db.force_state(VID, "triage", reason="test fixture")
    summary = rr.scan_review_queues(orch.db, proc)
    assert summary.get("bench_label_mismatch") == 1
    assert not (triage / VID).exists() or not any((triage / VID).iterdir())
    kept = proc / f"{VID}_causal_review.json"
    assert kept.read_bytes() == before                  # the answer travelled
    assert orch.db.get_video(VID)["state"] == "processing"
    assert orch._hand_score_precheck(VID, proc) is False            # held
    item = json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json")
                      .read_text())["items"][0]
    assert item["kind"] == "label_mismatch" and item["answers_kept"] == [4]
    assert "pellet(s) 4 are kept" in item["says"]


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
