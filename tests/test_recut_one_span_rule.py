"""One definition of a pellet's frames, and one re-cut rule, for the review
gate, the review window and the results step.

Rule (owner ruling 2026-11-18, narrowing R2: "If resegmenting doesn't alter,
then it's fine to keep"): an answer given before a person's re-cut stands
when its pellet's frames are exactly the same on the cuts it was given on and
on the current cuts. If they moved at all, or the files do not show which cuts
it was given on, it is void -- never carried by overlap or by number -- and
dropped (owner ruling 2026-11-21): its pellet takes the call of the algorithms
re-run on the new cuts and goes to triage only if they flag it. An answer
given after the re-cut always stands.

Before: the gate and the window framed a pellet from the segments file (it
ends the frame before the next cut) while the results step framed it from the
reaches file (it ends AT the next cut), and an answer given before a person's
re-cut survived only when its frames exactly matched a current pellet. On a
pellet whose cuts did not move the gate saw an exact match, counted the answer
and passed the video CLEAN; the results step never saw an exact match, voided
the answer and wrote the algorithm's call over the person's, with only a log
line. The earlier tests missed it because their reaches file ended the segment
the frame before the cut -- not what the reach detector writes.

Now: causal_review_io.pellet_span is the only place a pellet's frames are
defined and answer_survives_recut the only place the re-cut rule is decided;
the results step takes its pellets from the segments file (current_pellets).
The gate and the results step keep the same earlier answers (unmoved pellets)
and void the same ones, which are dropped. The one earlier answer that stands
but that no reader applies -- one a save kept aside (unmatched_answers) on a
pellet that did not move -- is asked again: by the gate, which sends the
video to triage, and by the results step, which refuses to write features
(RecutVoidedAnswers) and, in the watcher, routes the video to triage the same
way.

The fixtures use the files' real conventions: the reaches file is written by
the reach detector itself, run on a synthetic pose over the segments file, and
the review records carry the frames the review window stamps (pellet_span).

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as io
import mousereach.watcher.review_gate as rg
from mousereach.kinematics.core.feature_extractor import FeatureExtractor, RecutVoidedAnswers
from mousereach.reach.core.reach_detector import ReachDetector

VID = "20990101_ABC9911_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")
N_FRAMES = 400
OLD_CUTS = [0, 100, 190, 300]     # the cuts the person answered on
NEW_CUTS = [0, 100, 200, 300]     # the person's re-cut: pellet 1 unmoved, 2 and 3 moved
CUT_AT = "2020-03-02T10:00:00"
BEFORE, AFTER = "2020-03-01T09:00:00", "2020-03-02T11:00:00"


def _pose(folder):
    cols, data = [], []
    for p in POINTS:
        x = {"SABL": 90.0, "SABR": 110.0, "BOXL": 80.0, "BOXR": 120.0}.get(p, 100.0)
        for c, v in (("x", x), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(N_FRAMES, v))
    pose = folder / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as f:
        pickle.dump({"data": {"fps": 60.0}}, f)
    return pose


def _window_span(cuts, num):
    """The frames the review window stamps on an answer for pellet ``num``."""
    a, b = io.pellet_span(cuts, num - 1)
    return {"start": a, "end": b}


def _answer(num, when, outcome="untouched", cuts=OLD_CUTS):
    return {"segment_num": num, "pellet_num": num, "segment_span": _window_span(cuts, num),
            "reviewer": "reviewer_a", "reviewed_at": when,
            "human": {"outcome": outcome, "causal_reach": None}, "answers": {}}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A processing folder after a re-run on a person's re-cut: segments file
    (person's cuts), reaches file written by the reach detector over them,
    outcomes (the algorithm called every pellet 'retrieved', none flagged),
    and a reach assignment naming a causal reach in each pellet."""
    monkeypatch.setattr(io, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(io, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(io, "review_dirs", lambda *a, **k: [tmp_path], raising=False)
    pose = _pose(tmp_path)
    seg = {"boundaries": NEW_CUTS, "boundary_source": "human", "corrected_at": CUT_AT,
           "algo_boundaries": OLD_CUTS, "overall_confidence": 0.9, "total_frames": N_FRAMES}
    (tmp_path / f"{VID}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    det = ReachDetector()
    result = det.detect(pose, tmp_path / f"{VID}_segments.json")
    ReachDetector.save_results(result, tmp_path / f"{VID}_reaches.json")
    reaches = json.loads((tmp_path / f"{VID}_reaches.json").read_text(encoding="utf-8"))
    # One reach per pellet so 'retrieved' has a causal reach to credit.
    rid = 1
    for s in reaches["segments"]:
        a = s["start_frame"] + 20
        s["reaches"] = [{"reach_id": rid, "reach_num": 1, "start_frame": a, "end_frame": a + 10,
                         "apex_frame": a + 5, "duration_frames": 10}]
        s["n_reaches"] = 1
        rid += 1
    (tmp_path / f"{VID}_reaches.json").write_text(json.dumps(reaches), encoding="utf-8")
    pellets = len(NEW_CUTS) - 1
    (tmp_path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({
        "video_name": VID, "segments": [
            {"segment_num": n, "outcome": "retrieved", "flagged_for_review": False}
            for n in range(1, pellets + 1)]}), encoding="utf-8")
    (tmp_path / f"{VID}_reach_assignments.json").write_text(json.dumps({"reaches": [
        {"reach_id": n, "segment_num": n, "is_causal": True, "segment_ifr": None}
        for n in range(1, pellets + 1)]}), encoding="utf-8")
    return tmp_path, pose


def _review(folder, *records):
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": max(r["reviewed_at"] for r in records),
        "segments": list(records)}), encoding="utf-8")


def _extract(folder, pose):
    return FeatureExtractor().extract(
        pose, folder / f"{VID}_reaches.json", folder / f"{VID}_pellet_outcomes.json",
        review_path=folder / f"{VID}_causal_review.json").to_dict()


# ---------------------------------------------------------------------------
# The fixture is what the writers really write
# ---------------------------------------------------------------------------

def test_the_reaches_file_ends_a_segment_at_the_next_cut_and_adds_a_tail(run):
    folder, _ = run
    reaches = json.loads((folder / f"{VID}_reaches.json").read_text(encoding="utf-8"))
    frames = [(s["segment_num"], s["start_frame"], s["end_frame"]) for s in reaches["segments"]]
    assert frames == [(1, 0, 100), (2, 100, 200), (3, 200, 300), (4, 300, N_FRAMES)]


def test_every_reader_frames_a_pellet_the_same_way(run):
    folder, _ = run
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    reaches = json.loads((folder / f"{VID}_reaches.json").read_text(encoding="utf-8"))
    gate_window = [(s["segment_num"], s["start_frame"], s["end_frame"])
                   for s in io.segments_from_segmentation(seg_doc)]
    results_step = [(s["segment_num"], s["start_frame"], s["end_frame"])
                    for s in io.current_pellets(seg_doc, reaches)]
    assert gate_window == results_step == [(1, 0, 99), (2, 100, 199), (3, 200, 299)]
    # the detector's reaches are carried onto the pellets by number
    assert [len(p["reaches"]) for p in io.current_pellets(seg_doc, reaches)] == [1, 1, 1]


# ---------------------------------------------------------------------------
# The defect: gate CLEAN, results step silently voids (fails on 5546de0)
# ---------------------------------------------------------------------------

def test_an_unmoved_pellets_answer_is_kept_by_the_gate_and_the_results_step(run):
    """Pellet 1's cuts did not move (0-99 before and after); it was answered
    'untouched' before the re-cut. The gate counts the answer and passes the
    video CLEAN, and the results step applies the person's answer -- not the
    algorithm's 'retrieved'. On 0c36d0d both voided it and asked again."""
    folder, pose = run
    _review(folder, _answer(1, BEFORE))
    assert rg.recut_voided_open(VID, folder) == set()
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason
    seg1 = _extract(folder, pose)["segments"][0]
    assert seg1["outcome"] == "untouched"
    assert {r.get("outcome_source") for r in seg1["reaches"]} == {"human_review"}


def test_an_unmoved_pellet_keeps_its_answer_while_a_moved_ones_is_dropped(run):
    """Owner ruling 2026-11-21: pellet 2's frames moved, so its earlier
    answer is void and simply dropped -- not asked again (feb7c62 asked it
    again) -- and it takes the re-run algorithm's call."""
    folder, pose = run
    _review(folder, _answer(1, BEFORE), _answer(2, BEFORE, outcome="displaced_sa"))
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    current = io.segments_from_segmentation(seg_doc)
    doc = json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))
    assert io.answered_segments(doc, current, human_segmentation=True) == {1}
    assert io.recut_voided_segments(doc, current, human_segmentation=True) == set()
    segs = _extract(folder, pose)["segments"]
    assert segs[0]["outcome"] == "untouched"
    assert segs[1]["outcome"] == "retrieved"


def _replace_cuts(folder, old_cuts):
    """Say in the segments file that the person's re-cut replaced
    ``old_cuts`` (the fixer's algo_boundaries; None = not recorded)."""
    path = folder / f"{VID}_segments.json"
    seg = json.loads(path.read_text(encoding="utf-8"))
    if old_cuts is None:
        seg.pop("algo_boundaries", None)
    else:
        seg["algo_boundaries"] = old_cuts
    when = path.stat()
    path.write_text(json.dumps(seg), encoding="utf-8")
    # the same save as the fixture's: the later steps' files stay newer
    os.utime(path, ns=(when.st_atime_ns, when.st_mtime_ns))


def test_a_pellet_moved_by_one_frame_is_voided_and_dropped(run):
    """The person moved the second cut by ONE frame (199 -> 200): pellet 2
    was 100-198, is 100-199. Its earlier answer is void and dropped -- the
    gate passes the video (the algorithm flagged nothing) and the results
    step writes the algorithm's call -- while pellet 1 (0-99, unmoved) keeps
    its answer. Until 2026-11-21 pellet 2 was asked again."""
    folder, pose = run
    one_frame = [0, 100, 199, 300]
    _replace_cuts(folder, one_frame)
    _review(folder, _answer(1, BEFORE, cuts=one_frame),
            _answer(2, BEFORE, outcome="displaced_sa", cuts=one_frame))
    assert rg.recut_voided_open(VID, folder) == set()
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason
    segs = _extract(folder, pose)["segments"]
    assert segs[0]["outcome"] == "untouched"
    assert segs[1]["outcome"] == "retrieved"


def test_an_answer_with_no_record_of_its_cuts_is_dropped(run):
    """Pellet 1 keeps its earlier answer; pellet 2's earlier answer carries
    no frames, so nothing shows which cuts it was given on: it is treated as
    moved -- void and dropped, and counted as such."""
    folder, pose = run
    no_frames = dict(_answer(2, BEFORE, outcome="displaced_sa"))
    no_frames.pop("segment_span")
    _review(folder, _answer(1, BEFORE), no_frames)
    assert rg.recut_voided_open(VID, folder) == set()
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    recut = io.recut_of(io.segments_from_segmentation(seg_doc))
    assert io.recut_verdict(no_frames, None, recut) == io.RECUT_NO_RECORD
    segs = _extract(folder, pose)["segments"]
    assert segs[0]["outcome"] == "untouched"
    assert segs[1]["outcome"] == "retrieved"


def test_without_the_replaced_cuts_on_file_an_earlier_answer_is_dropped(run):
    # the segments file does not say which cuts the re-cut replaced: no
    # record, treated as moved, so the answer is dropped and not asked
    folder, pose = run
    _replace_cuts(folder, None)
    _review(folder, _answer(1, BEFORE))
    assert rg.recut_voided_open(VID, folder) == set()
    assert _extract(folder, pose)["segments"][0]["outcome"] == "retrieved"


def test_a_moved_pellets_answer_is_dropped_and_the_pellet_asked_only_if_flagged(run):
    """Owner ruling 2026-11-21. The voided answer flags nothing and holds
    nothing: the gate passes the video and the results step writes the
    algorithm's call. The pellet goes to triage only when the re-run
    algorithm flags it -- the same gate rule as any new video."""
    folder, pose = run
    _review(folder, _answer(2, BEFORE, outcome="displaced_sa"))
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason
    assert rg.flag_recut_voided(VID, folder) == []
    outcomes_path = folder / f"{VID}_pellet_outcomes.json"
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    assert not [s for s in outcomes["segments"] if s.get("flagged_for_review")]
    assert _extract(folder, pose)["segments"][1]["outcome"] == "retrieved"
    # the re-run algorithm flags pellet 2: triage, for the algorithm's reason
    outcomes["segments"][1].update(flagged_for_review=True, flag_reason="low_confidence")
    outcomes_path.write_text(json.dumps(outcomes), encoding="utf-8")
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_TRIAGE, reason
    assert rg.RECUT_FLAG_REASON not in reason


def _review_aside(folder, *records):
    """A review whose earlier answers a save kept aside (unmatched_answers)."""
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": max(r["reviewed_at"] for r in records),
        "segments": [], "unmatched_answers": list(records)}), encoding="utf-8")


def test_an_unmoved_pellets_answer_kept_aside_is_still_asked_again(run):
    """The one earlier answer still asked: pellet 1 did not move, so its
    answer stands, but a save kept it aside where no reader applies it.
    Unasked, the algorithm's call would replace an answer that stands."""
    folder, pose = run
    _review_aside(folder, _answer(1, BEFORE))
    assert rg.recut_voided_open(VID, folder) == {1}
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_TRIAGE and rg.RECUT_FLAG_REASON in reason, reason
    with pytest.raises(RecutVoidedAnswers) as refused:
        _extract(folder, pose)
    assert refused.value.segments == [1]


def test_a_moved_pellets_answer_kept_aside_is_dropped(run):
    folder, _ = run
    _review_aside(folder, _answer(2, BEFORE, outcome="displaced_sa"))
    assert rg.recut_voided_open(VID, folder) == set()


def test_answers_given_on_the_new_cuts_are_applied_by_both(run):
    """The convention matters for matching too: an answer given after the
    re-cut on pellet 1 is answered for the gate (CLEAN) and applied by the
    results step as the person's."""
    folder, pose = run
    _review(folder, _answer(1, BEFORE), _answer(1, AFTER, cuts=NEW_CUTS),
            _answer(2, AFTER, outcome="retrieved", cuts=NEW_CUTS),
            _answer(3, AFTER, outcome="retrieved", cuts=NEW_CUTS))
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason
    seg1 = _extract(folder, pose)["segments"][0]
    assert seg1["outcome"] == "untouched"
    assert {r.get("outcome_source") for r in seg1["reaches"]} == {"human_review"}


def test_on_the_segmenters_own_cuts_earlier_answers_stand_for_both(run):
    folder, pose = run
    seg_path = folder / f"{VID}_segments.json"
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    seg.update(boundary_source="algorithm", corrected_at=None)
    seg_path.write_text(json.dumps(seg), encoding="utf-8")
    _review(folder, _answer(1, BEFORE, cuts=NEW_CUTS))
    assert rg.recut_voided_open(VID, folder) == set()
    seg1 = _extract(folder, pose)["segments"][0]
    assert seg1["outcome"] == "untouched"
    assert {r.get("outcome_source") for r in seg1["reaches"]} == {"human_review"}


# ---------------------------------------------------------------------------
# The watcher answers the results step's refusal by routing to triage
# ---------------------------------------------------------------------------

class _DB:
    def __init__(self):
        self.steps = []

    def log_step(self, video_id, step, status, **kw):
        self.steps.append((step, status))


def test_the_watcher_routes_a_refused_video_to_triage_instead_of_failing_it(run, monkeypatch):
    from mousereach.watcher import orchestrator as orch
    folder, pose = run
    # pellet 1 unmoved, its answer kept aside: the one case still asked
    _review_aside(folder, _answer(1, BEFORE))
    (folder / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    routed = []
    monkeypatch.setattr(rg, "route_recut_voided_to_triage",
                        lambda vid, pdir, segs, **kw: routed.append((vid, list(segs))) or pdir)
    o = object.__new__(orch.ProcessingOrchestrator)
    o.db = _DB()
    why = o._make_features(VID, folder, pose, skip_outcomes=False)
    assert isinstance(why, orch.RoutedToReview), why
    assert routed == [(VID, [1])]
    assert not (folder / f"{VID}_features.json").exists()


def test_route_recut_voided_to_triage_flags_then_routes(run, monkeypatch):
    folder, _ = run
    calls = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda vid, src, dest, reason, **kw: calls.append((vid, reason, kw))
                        or (folder.parent / "triage" / vid))
    monkeypatch.setattr(rg.Paths, "TRIAGE_REVIEW", folder.parent / "triage", raising=False)
    rg.route_recut_voided_to_triage(VID, folder, [3])
    outcomes = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text(encoding="utf-8"))
    assert [s["segment_num"] for s in outcomes["segments"] if s.get("flagged_for_review")] == [3]
    (vid, reason, kw), = calls
    assert vid == VID and rg.RECUT_FLAG_REASON in reason and kw["db_state"] == "triage"


# ---------------------------------------------------------------------------
# Answering again after the re-cut ends the question (the real save path)
# ---------------------------------------------------------------------------

def _window_record(num, outcome, answered=True, cuts=NEW_CUTS):
    return io.build_segment_record(
        segment_num=num, pellet_num=num, algo_outcome="retrieved", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome if answered else None,
        human_causal_reach=None, is_phantom=False, agreed=False,
        answers={} if answered else {"reviewed": False}, notes="",
        segment_span=_window_span(cuts, num))


def test_the_same_answer_given_again_after_the_recut_counts(run, monkeypatch):
    """Pellet 2's cuts moved; its pre-cut answer is void (dropped). The
    pellet is then asked for another reason (the algorithm flags it) and the
    person gives the SAME answer on the new cuts. The save must stamp it
    with this save's time -- inheriting the voided answer's time (same
    number, same answer) would leave it void and ask it for ever."""
    folder, pose = run
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    # the earlier answers as the window saved them, on the old cuts
    _review(folder, dict(_window_record(1, "untouched", cuts=OLD_CUTS), reviewer="reviewer_a",
                         reviewed_at=BEFORE),
            dict(_window_record(2, "untouched", cuts=OLD_CUTS), reviewer="reviewer_a",
                 reviewed_at=BEFORE))
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    current = io.segments_from_segmentation(seg_doc)
    before = json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))
    assert io.answered_segments(before, current, human_segmentation=True) == {1}
    io.save_causal_review(VID, folder, [_window_record(1, "untouched"),
                                        _window_record(2, "untouched"),
                                        _window_record(3, "retrieved")],
                          provenance={}, reviewer="reviewer_b", cuts_set_at=CUT_AT,
                          cuts_replaced=OLD_CUTS)
    after = json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))
    assert io.answered_segments(after, current, human_segmentation=True) == {1, 2, 3}
    saved = {r["segment_num"]: r for r in json.loads(
        (folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))["segments"]}
    # pellet 1 did not move: its earlier answer is kept with who gave it and when;
    # pellet 2 moved: the answer given again is this save's
    assert (saved[1]["reviewer"], saved[1]["reviewed_at"]) == ("reviewer_a", BEFORE)
    assert saved[2]["reviewer"] == "reviewer_b" and saved[2]["reviewed_at"] > CUT_AT
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason
    seg2 = _extract(folder, pose)["segments"][1]
    assert seg2["outcome"] == "untouched"
    assert {r.get("outcome_source") for r in seg2["reaches"]} == {"human_review"}


def test_a_voided_answer_carried_by_a_save_stays_void(run, monkeypatch):
    """A save that leaves the voided pellet unanswered carries the old answer
    along as a record with its own (pre-cut) time: still void, never applied,
    and not asked (dropped, owner ruling 2026-11-21)."""
    folder, _ = run
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    _review(folder, dict(_window_record(2, "untouched", cuts=OLD_CUTS), reviewer="reviewer_a",
                         reviewed_at=BEFORE))
    io.save_causal_review(VID, folder, [_window_record(1, "retrieved"),
                                        _window_record(2, None, answered=False),
                                        _window_record(3, "retrieved")],
                          provenance={}, reviewer="reviewer_b", cuts_set_at=CUT_AT)
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    current = io.segments_from_segmentation(seg_doc)
    saved = json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))
    kept = [r for r in saved["segments"] + saved.get("unmatched_answers", [])
            if (r.get("human") or {}).get("outcome") == "untouched"
            and r.get("reviewed_at") == BEFORE]
    assert kept, "the voided answer is kept as a record"
    assert 2 not in io.answered_segments(saved, current, human_segmentation=True)
    assert rg.recut_voided_open(VID, folder) == set()


def test_a_reaches_file_made_on_other_cuts_is_refused_not_guessed(run):
    """A reaches file left from the old cuts beside the person's new cuts
    (a step command or backfill, where no gate checks for stale outputs):
    answers matched to the new pellets would land on the old reaches."""
    from mousereach.kinematics.core.feature_extractor import KinematicsProblem
    folder, pose = run
    reaches_path = folder / f"{VID}_reaches.json"
    reaches = json.loads(reaches_path.read_text(encoding="utf-8"))
    reaches["segments"][2]["start_frame"] = OLD_CUTS[2]          # pellet 3 began at 190
    reaches_path.write_text(json.dumps(reaches), encoding="utf-8")
    _review(folder, _answer(1, AFTER, cuts=NEW_CUTS))
    with pytest.raises(KinematicsProblem) as refused:
        _extract(folder, pose)
    assert "not made on the cuts" in str(refused.value)
