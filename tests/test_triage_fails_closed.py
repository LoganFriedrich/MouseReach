"""A quality triage that cannot run holds the video for a person.

The unified quality triage (pipeline.triage.triage_video) is what the review
gate's quality verdict comes from. When it raised, the verdict stayed
'auto_approved' and the gate let the video through unchecked -- to kinematics
and filing. Now the verdict is QC_TRIAGE_FAILED and the gate holds the video in
deep review with a reason saying the check could not run; a person's
deep-review clear is honoured as for any quality hold, and Easy/Flat trays are
never held. A ``_triage.json`` from an earlier run is removed before triage
runs, so a crash cannot leave an old verdict beside new outputs.
Built under tmp_path; no real folder or database.
"""
import json
from unittest import mock

from mousereach.watcher.orchestrator import BaseOrchestrator
from mousereach.watcher.review_gate import (
    DECISION_CLEAN, DECISION_DEEP, QC_TRIAGE_FAILED, TRIAGE_FAILED_REASON,
    evaluate_gate,
)

PILLAR = "20990101_ABC9901_P1"
EASY = "20990101_ABC9902_E1"


def _orch():
    o = BaseOrchestrator.__new__(BaseOrchestrator)
    o.db = mock.Mock()
    return o


def _crash(**kw):
    raise RuntimeError("triage crashed")


def test_a_crashed_triage_gives_the_failed_verdict(tmp_path):
    verdict = _orch()._run_quality_triage(PILLAR, tmp_path, tmp_path / "p.h5", _crash)
    assert verdict == QC_TRIAGE_FAILED


def test_an_old_triage_record_is_removed_before_triage_runs(tmp_path):
    old = tmp_path / f"{PILLAR}_triage.json"
    old.write_text(json.dumps({"verdict": "auto_approved"}))
    _orch()._run_quality_triage(PILLAR, tmp_path, tmp_path / "p.h5", _crash)
    assert not old.exists()


def test_the_gate_holds_a_video_whose_triage_failed(tmp_path):
    (tmp_path / f"{PILLAR}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.9, "boundaries": [0]}))
    decision, reason, _ = evaluate_gate(PILLAR, tmp_path, qc_verdict=QC_TRIAGE_FAILED)
    assert decision == DECISION_DEEP
    assert reason == TRIAGE_FAILED_REASON


def test_a_persons_deep_review_clear_is_honoured(tmp_path):
    (tmp_path / f"{PILLAR}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.9, "boundaries": [0]}))
    (tmp_path / f"{PILLAR}_deep_review_cleared.json").write_text("{}")
    decision, _, _ = evaluate_gate(PILLAR, tmp_path, qc_verdict=QC_TRIAGE_FAILED)
    assert decision == DECISION_CLEAN


def test_easy_and_flat_are_still_never_held(tmp_path):
    decision, _, _ = evaluate_gate(EASY, tmp_path, qc_verdict=QC_TRIAGE_FAILED)
    assert decision == DECISION_CLEAN


def test_the_held_bundle_is_not_released_as_a_lifted_quality_hold(tmp_path):
    """The return scan releases a 'qc_needs_review' hold the current check
    would not make; a hold for a triage that could not run is not that."""
    from mousereach.watcher import review_return
    bundle = tmp_path / PILLAR
    bundle.mkdir()
    (bundle / f"{PILLAR}_routing.json").write_text(
        json.dumps({"routed_reason": TRIAGE_FAILED_REASON}))
    assert review_return._qc_hold_lifted(bundle, PILLAR, ("sig",)) is False
