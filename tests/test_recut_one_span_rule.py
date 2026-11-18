"""One definition of a pellet's frames, and one re-cut rule, for the review
gate, the review window and the results step.

Rule (owner ruling R2): re-segmentation voids everything after it -- a
person's earlier outcome and causal-reach answers included. They are never
carried across a re-cut, by frames or by number, and their pellets are asked
again. Answers persist only through re-runs on the SAME cuts.

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
Every answer given before a person's re-cut is void and asked again -- by the
gate, which sends the video to triage, and by the results step, which refuses
to write features (RecutVoidedAnswers) and, in the watcher, routes the video
to triage the same way.

The fixtures use the files' real conventions: the reaches file is written by
the reach detector itself, run on a synthetic pose over the segments file, and
the review records carry the frames the review window stamps (pellet_span).

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json
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

def test_gate_and_results_step_agree_on_an_answer_given_before_the_recut(run):
    """Only pellet 1 (cuts unmoved) was answered, before the re-cut. On
    5546de0 the gate called it answered and passed the video CLEAN, and the
    results step voided it and wrote the algorithm's 'retrieved'. Now both
    void it: the gate sends the video to triage with pellet 1 flagged, and
    the results step refuses to write the algorithm's call in its place."""
    folder, pose = run
    _review(folder, _answer(1, BEFORE))
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_TRIAGE, reason
    assert rg.recut_voided_open(VID, folder) == {1}
    with pytest.raises(RecutVoidedAnswers) as refused:
        _extract(folder, pose)
    assert refused.value.segments == [1]


def test_an_unmoved_pellet_does_not_keep_an_answer_given_before_the_recut(run):
    # R2: answers persist only on the SAME cuts; a person's re-cut is new cuts
    # for the whole video, however many of its pellets kept their frames.
    folder, _ = run
    _review(folder, _answer(1, BEFORE))
    seg_doc = json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    current = io.segments_from_segmentation(seg_doc)
    doc = json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))
    assert io.answered_segments(doc, current, human_segmentation=True) == set()
    assert io.recut_voided_segments(doc, current, human_segmentation=True) == {1}


def test_a_moved_pellets_answer_is_voided_and_asked_again_never_dropped(run):
    folder, pose = run
    _review(folder, _answer(2, BEFORE, outcome="displaced_sa"))
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_TRIAGE and rg.RECUT_FLAG_REASON in reason, reason
    # the pellet is flagged in the outcomes file, so the triage window walks it
    assert rg.flag_recut_voided(VID, folder) == [2]
    outcomes = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text(encoding="utf-8"))
    flagged = {s["segment_num"]: s for s in outcomes["segments"] if s.get("flagged_for_review")}
    assert set(flagged) == {2}
    # and the results step never writes the algorithm's call over it
    with pytest.raises(RecutVoidedAnswers) as refused:
        _extract(folder, pose)
    assert refused.value.segments == [2]


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
    _review(folder, _answer(2, BEFORE, outcome="displaced_sa"))
    (folder / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    routed = []
    monkeypatch.setattr(rg, "route_recut_voided_to_triage",
                        lambda vid, pdir, segs, **kw: routed.append((vid, list(segs))) or pdir)
    o = object.__new__(orch.ProcessingOrchestrator)
    o.db = _DB()
    why = o._make_features(VID, folder, pose, skip_outcomes=False)
    assert isinstance(why, orch.RoutedToReview), why
    assert routed == [(VID, [2])]
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
