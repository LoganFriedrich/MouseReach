"""A pellet whose answer a re-cut voided is asked again; it never takes the
algorithm's call unasked.

Rule: a person's re-segmentation voids every earlier answer whose pellet's
frames it changed (or whose cuts the files do not show), and a voided answer
is ASKED AGAIN (test_recut_voids_earlier_answers.py covers the voiding).

Before: voiding only made the pellet unanswered. A video sent to deep review
for its cuts is released by saving them, re-runs from reaches, and an
unanswered pellet the algorithm did not flag takes the algorithm's call -- the
call the person had overruled -- with nobody asked. A filed video run again
did the same. The review gate now names those pellets (they hold the video in
triage) and flags them in the outcomes file, so the triage window walks them.

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json
import os

import pytest

import mousereach.review.causal_review_io as io
import mousereach.watcher.review_gate as rg
from mousereach.review.triage_status import triaged_segments

VID = "20990101_TEST9901_P1"
CUT_AT = "2099-03-02T10:00:00"
BEFORE, AFTER = "2099-03-01T09:00:00", "2099-03-02T11:00:00"


def _rec(span, when, outcome="untouched", num=1, answered=True):
    r = {"segment_num": num, "pellet_num": num, "reviewer": "reviewer_a",
         "reviewed_at": when, "human": {"outcome": outcome, "causal_reach": None},
         "answers": {} if answered else {"reviewed": False}}
    if span:
        r["segment_span"] = {"start": span[0], "end": span[1]}
    return r


# The cuts the person replaced: pellets 950-1990, 1991-2999, 3000-3999.
REPLACED = [950, 1991, 3000, 4000]


def _current(human=True, replaced=REPLACED):
    d = {"boundaries": [1000, 2000, 3000, 4000]}
    if human:
        d.update(boundary_source="human", corrected_at=CUT_AT)
        if replaced is not None:
            d["algo_boundaries"] = replaced
    return d


def _voided(doc, human=True, replaced=REPLACED):
    seg_doc = _current(human, replaced)
    return io.recut_voided_segments(doc, io.segments_from_segmentation(seg_doc),
                                    human_segmentation=human)


# ---------------------------------------------------------------------------
# Which pellets are asked again
# ---------------------------------------------------------------------------

def test_an_answer_on_moved_cuts_names_its_pellet():
    # given on frames 950-1990; pellet 1 is now 1000-1999
    assert _voided({"segments": [_rec((950, 1990), BEFORE)]}) == {1}


def test_a_pellet_answered_again_on_the_new_cuts_is_not_asked():
    doc = {"segments": [_rec((950, 1990), BEFORE), _rec((1000, 1999), AFTER)]}
    assert _voided(doc) == set()


def test_an_unchanged_pellet_is_not_asked_again():
    # pellet 3 is 3000-3999 on the replaced cuts and on the current ones
    assert _voided({"segments": [_rec((3000, 3999), BEFORE, num=3)]}) == set()


def test_an_unchanged_pellet_is_asked_when_the_replaced_cuts_are_not_on_file():
    assert _voided({"segments": [_rec((3000, 3999), BEFORE, num=3)]}, replaced=None) == {3}


def test_an_unchanged_answer_kept_only_among_the_unmatched_answers_is_asked():
    # no reader applies an aside answer, so one the re-cut would keep is
    # asked rather than left to the algorithm's call
    doc = {"segments": [], "unmatched_answers": [_rec((3000, 3999), BEFORE, num=3)]}
    assert _voided(doc) == {3}


def test_an_answer_with_no_frames_names_the_pellet_with_its_number():
    assert _voided({"segments": [_rec(None, BEFORE, num=3)]}) == {3}


def test_a_voided_placeholder_asks_nothing():
    assert _voided({"segments": [_rec((950, 1990), BEFORE, answered=False)]}) == set()


def test_on_the_segmenters_own_cuts_nothing_is_voided_or_asked():
    assert _voided({"segments": [_rec((950, 1990), BEFORE)]}, human=False) == set()


# ---------------------------------------------------------------------------
# The gate holds the video and flags the pellet for triage
# ---------------------------------------------------------------------------

@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """A processing folder after a re-run on cuts a person saved: the
    algorithm called every pellet without a flag, and the review holds an
    answer given on the old cuts for pellet 1 (frames moved)."""
    monkeypatch.setattr(io, "durable_review_dir", lambda: None)
    d = tmp_path / "work"
    d.mkdir()
    seg = dict(_current(), overall_confidence=0.9)
    (d / f"{VID}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    (d / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "retrieved"} for n in (1, 2, 3)]}),
        encoding="utf-8")
    (d / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": BEFORE,
        "segments": [_rec((950, 1990), BEFORE, outcome="untouched")]}), encoding="utf-8")
    return d


def test_the_gate_sends_a_voided_pellet_to_triage(run_folder):
    decision, reason, _ = rg.evaluate_gate(VID, run_folder)
    assert decision == rg.DECISION_TRIAGE, reason
    assert rg.RECUT_FLAG_REASON in reason


def test_the_gate_is_clean_once_the_pellet_is_answered_on_the_new_cuts(run_folder):
    p = run_folder / f"{VID}_causal_review.json"
    doc = json.loads(p.read_text())
    doc["segments"].append(_rec((1000, 1999), AFTER, outcome="untouched"))
    p.write_text(json.dumps(doc))
    decision, reason, _ = rg.evaluate_gate(VID, run_folder)
    assert decision == rg.DECISION_CLEAN, reason


def test_a_filed_video_run_again_flags_the_voided_pellet_before_it_is_routed(
        run_folder, tmp_path, monkeypatch):
    """The gate is all a filed video's re-run passes through when no algorithm
    is recomputed (orchestrator._run_checks_before_kinematics). The pellet
    must be flagged in the outcomes file the bundle carries, or the triage
    window would not walk it and the video would come straight back."""
    seen = {}

    def fake_route(video_id, folder, queue, reason, **kw):
        seen["reason"] = reason
        seen["outcomes"] = json.loads(
            (folder / f"{video_id}_pellet_outcomes.json").read_text())
        return tmp_path / "triage" / video_id

    monkeypatch.setattr(rg, "route_to_queue", fake_route)
    monkeypatch.setattr(rg.Paths, "TRIAGE_REVIEW", tmp_path / "triage", raising=False)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)
    decision = rg.run_gate(VID, run_folder)
    assert decision == rg.DECISION_TRIAGE
    assert triaged_segments(seen["outcomes"]) == {1}
    flagged = [s for s in seen["outcomes"]["segments"] if s.get("flagged_for_review")]
    assert flagged[0]["flag_reason"] == rg.RECUT_FLAG_REASON
    # the triage window shows only what is open on the pellet, no re-cut words
    from mousereach.review.causal_review_widget import plain_flag_reason
    assert plain_flag_reason(flagged[0]["flag_reason"]) == ""


def test_a_save_after_the_recut_does_not_hide_the_voided_answer_from_the_gate(
        run_folder, monkeypatch):
    """A save after the re-cut keeps the voided answer as a record outside the
    pellet list (unmatched_answers). The gate must still ask that pellet: a
    video cleared from Re-segmentation Review with it still open would
    otherwise be filed with the algorithm's call."""
    monkeypatch.setattr(io, "durable_review_path", lambda stem: None)
    placeholder = io.build_segment_record(
        segment_num=1, pellet_num=1, algo_outcome="retrieved", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=None, human_causal_reach=None,
        is_phantom=False, agreed=True, answers={"reviewed": False}, notes="",
        segment_span={"start": 1000, "end": 1999})
    out = io.save_causal_review(VID, run_folder, [placeholder], provenance={},
                                reviewer="reviewer_b")
    saved = json.loads(out.read_text())
    kept = [r for r in saved["segments"] + saved.get("unmatched_answers", [])
            if (r.get("human") or {}).get("outcome") == "untouched"]
    assert kept, "precondition: the voided answer is kept as a record"
    decision, reason, _ = rg.evaluate_gate(VID, run_folder)
    assert decision == rg.DECISION_TRIAGE, reason


def test_a_voided_answer_kept_among_the_unmatched_answers_is_still_asked():
    """Wherever a save keeps the voided answer -- in the pellet list or among
    the unmatched answers -- its pellet is asked again."""
    doc = {"segments": [], "unmatched_answers": [_rec((950, 1990), BEFORE)]}
    assert _voided(doc) == {1}


# ---------------------------------------------------------------------------
# A filed video run again: the whole pipeline run, with the real gate
# ---------------------------------------------------------------------------

REAL_RUN_GATE = rg.run_gate

from test_filed_rerun_skips_checks import VID as FILED_VID, rig  # noqa: E402,F401


def test_a_filed_kinematics_rerun_never_asks_the_voided_pellet_and_files_nothing(
        rig, tmp_path, monkeypatch):
    """A filed video re-run when only a downstream calculation changed runs no
    gate and asks nobody anything (lab decision 2026-11-19: a kinematics
    recalculation never holds or re-asks). Its review holds an answer given
    before a person re-cut it: nothing is routed to triage."""
    monkeypatch.setattr(rg, "run_gate", REAL_RUN_GATE)
    monkeypatch.setattr(io, "durable_review_dir", lambda: None)
    routed = {}

    def fake_route(video_id, folder, queue, reason, **kw):
        routed.update(queue=queue, outcomes=json.loads(
            (folder / f"{video_id}_pellet_outcomes.json").read_text()))
        return tmp_path / "triage" / video_id

    monkeypatch.setattr(rg, "route_to_queue", fake_route)
    monkeypatch.setattr(rg.Paths, "TRIAGE_REVIEW", tmp_path / "triage", raising=False)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)
    f = rig.folder
    (f / f"{FILED_VID}_segments.json").write_text(json.dumps(
        dict(_current(), overall_confidence=0.9)), encoding="utf-8")
    # The algorithm saw every pellet untouched (nothing it would triage); the
    # person had said pellet 1 was displaced, on the old cuts.
    (f / f"{FILED_VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "untouched"} for n in (1, 2, 3)]}), encoding="utf-8")
    (f / f"{FILED_VID}_causal_review.json").write_text(json.dumps({
        "video_name": FILED_VID, "reviewed_at": BEFORE,
        "segments": [_rec((950, 1990), BEFORE, outcome="displaced_sa")]}), encoding="utf-8")
    import mousereach.pipeline.pose_check as pc
    monkeypatch.setattr(pc, "check_pose", lambda *a, **k: (True, ""))
    rig.file_it()
    rig.run("kinematics")
    # Nothing was routed: no gate ran. (What the extractor does with the
    # voided answer -- refuse, so the re-run is given up and the filed results
    # stand -- is tested in test_filed_is_final.py; this rig's extractor is a
    # stand-in.)
    assert routed == {}
    assert rig.orch.db.get_video(FILED_VID)["state"] not in ("triage", "deep_review")


# ---------------------------------------------------------------------------
# Every copy of the review counts (the gate reads them merged)
# ---------------------------------------------------------------------------

@pytest.fixture
def durable(tmp_path, monkeypatch):
    store = tmp_path / "records"
    store.mkdir()
    monkeypatch.setattr(io, "durable_review_dir", lambda: store)
    monkeypatch.setattr(io, "gt_index", lambda refresh=False: {})
    return store


def test_an_answer_only_the_durable_copy_holds_is_still_asked_again(run_folder, durable):
    """The voided answer is in the durable copy only (the folder's copy is a
    placeholder). Read from one copy, the pellet took the algorithm's call."""
    (run_folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": BEFORE,
        "segments": [_rec((1000, 1999), None, answered=False)]}), encoding="utf-8")
    (durable / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": BEFORE,
        "segments": [_rec((950, 1990), BEFORE, outcome="untouched")]}), encoding="utf-8")
    # The folder's copy is the newer FILE, so a reader picking one copy by
    # file time sees only the placeholder.
    t = (run_folder / f"{VID}_segments.json").stat().st_mtime
    os.utime(durable / f"{VID}_causal_review.json", (t - 60, t - 60))
    os.utime(run_folder / f"{VID}_causal_review.json", (t + 60, t + 60))
    assert rg.recut_voided_open(VID, run_folder) == {1}
    decision, reason, _ = rg.evaluate_gate(VID, run_folder)
    assert decision == rg.DECISION_TRIAGE and rg.RECUT_FLAG_REASON in reason, reason


def test_a_new_answer_only_the_durable_copy_holds_ends_the_question(run_folder, durable):
    (durable / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "reviewed_at": AFTER,
        "segments": [_rec((1000, 1999), AFTER, outcome="untouched")]}), encoding="utf-8")
    t = (run_folder / f"{VID}_segments.json").stat().st_mtime
    os.utime(durable / f"{VID}_causal_review.json", (t - 60, t - 60))
    os.utime(run_folder / f"{VID}_causal_review.json", (t + 60, t + 60))
    assert rg.recut_voided_open(VID, run_folder) == set()
