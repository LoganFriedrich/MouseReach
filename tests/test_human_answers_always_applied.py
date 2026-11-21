"""A person's answer is ALWAYS applied, with no hold, queue or question
(owner rule 2026-11-21).

  (a) filing never completes while an answer for the video is missing from
      the features file about to be filed: the archive step makes the file
      again with the answer first (orchestrator._apply_answers_before_filing);
  (b) an answer for a FILED video reaches its filed results by a
      kinematics-only re-run that never holds (test_kinematics_only_rerun_
      keeps_filed_decisions.py) and that the version scan starts;
  (c) the two ways 66 filed videos were filed without answers, each
      reproduced here:
        1. ground truth beside the FILED results that the run in the
           processing folder never looked for;
        2. a features file that lacks an answer although the manifest's stamp
           says the review was applied (the stamp says a run SAW the file,
           not that the answer is in the result) -- caught only by checking
           the content (kinematics/core/answers_applied.py).

Built under tmp_path with invented ids; no real folder, database or
configuration.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.config import Paths
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

VID = "20990101_ABC0101_P1"
SCORER = "DLC_resnet50_TestModelshuffle1_100"
CURRENT = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
           "assignment": "1.0", "kinematic_extractor": FeatureExtractor.VERSION}
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


def _video_files(folder):
    """A pose file, one pellet with two detected reaches, the algorithm says
    displaced by reach 1; the pipeline's outputs for it."""
    n = 200
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = folder / f"{VID}{SCORER}.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as f:
        pickle.dump({"data": {"fps": 60.0}}, f)
    (folder / f"{VID}_reaches.json").write_text(json.dumps(
        {"video_name": VID, "total_frames": n, "n_segments": 1,
         "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 199,
                       "ruler_pixels": 34.0, "n_reaches": 2,
                       "reaches": [
                           {"reach_id": 1, "reach_num": 1, "start_frame": 20, "end_frame": 30,
                            "apex_frame": 25, "duration_frames": 11},
                           {"reach_id": 2, "reach_num": 2, "start_frame": 120, "end_frame": 135,
                            "apex_frame": 128, "duration_frames": 16}]}]}), encoding="utf-8")
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"video_name": VID, "segments": [{"segment_num": 1, "outcome": "displaced_sa",
                                          "start_frame": 0, "end_frame": 199}]}),
        encoding="utf-8")
    (folder / f"{VID}_reach_assignments.json").write_text(json.dumps(
        {"video_id": VID, "reaches": [{"reach_id": 1, "segment_num": 1, "is_causal": True}]}),
        encoding="utf-8")
    (folder / f"{VID}.mp4").write_bytes(b"x")
    return pose


def _features(folder, pose):
    feats = FeatureExtractor().extract(pose, folder / f"{VID}_reaches.json",
                                       folder / f"{VID}_pellet_outcomes.json")
    (folder / f"{VID}_features.json").write_text(json.dumps(feats.to_dict()), encoding="utf-8")
    return feats


def _review(folder, outcome="retrieved", reach=(120, 135, 2)):
    p = folder / f"{VID}_causal_review.json"
    p.write_text(json.dumps({
        "type": "causal_review", "schema_version": "1.1", "video_stem": VID,
        "reviewer": "someone", "reviewed_at": "2099-01-01T09:00:00",
        "segments": [{"segment_num": 1, "segment_span": {"start": 0, "end": 199},
                      "reviewed": True,
                      "human": {"outcome": outcome,
                                "causal_reach": {"start": reach[0], "end": reach[1],
                                                 "reach_id": reach[2]}},
                      "answers": {"outcome_pick": outcome,
                                  "causal_pick": {"start": reach[0], "end": reach[1],
                                                  "reach_id": reach[2]}}}]}),
        encoding="utf-8")
    return p


def _ground_truth(folder, outcome="untouched"):
    p = folder / f"{VID}_unified_ground_truth.json"
    p.write_text(json.dumps({
        "video_name": VID, "last_modified_at": "2099-01-01T09:00:00",
        "reaches": {"exhaustive": False, "reaches": []},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": outcome, "determined": True,
             "start_frame": 0, "end_frame": 199}]},
    }), encoding="utf-8")
    return p


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A processing folder and the video's filed folder under a share root."""
    import mousereach.config as config
    import mousereach.archive.core as arch
    import mousereach.pipeline.version_index as vi
    from types import SimpleNamespace
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    nas = tmp_path / "nas"
    filed = nas / "Analyzed" / "Project_A" / "ABC01"
    filed.mkdir(parents=True)
    proc = tmp_path / "Processing"
    proc.mkdir()
    (nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": dict(CURRENT, dlc_scorer=SCORER)}), encoding="utf-8")
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    monkeypatch.setattr(arch, "get_archive_destination", lambda vid: filed)
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {}, raising=False)
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    return SimpleNamespace(nas=nas, filed=filed, proc=proc)


# ---------------------------------------------------------------------------
# (c) 1. ground truth beside the filed results
# ---------------------------------------------------------------------------

def test_a_run_finds_ground_truth_filed_beside_the_video(lab):
    """REPRODUCES (c) 1: the run's folder holds no ground truth; the file sits
    beside the filed results. Before 2026-11-21 the extractor looked only in
    the run's folder and the processing-tree index, so the answer was not
    applied and the video was filed without it."""
    pose = _video_files(lab.proc)
    _ground_truth(lab.filed)
    feats = FeatureExtractor().extract(pose, lab.proc / f"{VID}_reaches.json",
                                       lab.proc / f"{VID}_pellet_outcomes.json")
    assert feats.to_dict()["segments"][0]["outcome"] == "untouched"


# ---------------------------------------------------------------------------
# (c) 2. the content check
# ---------------------------------------------------------------------------

def test_the_check_finds_an_answer_a_features_file_lacks(lab):
    from mousereach.kinematics.core.answers_applied import unapplied_answers
    pose = _video_files(lab.filed)
    _features(lab.filed, pose)                     # made before the review existed
    assert unapplied_answers(VID, lab.filed) == []
    _review(lab.filed)
    missing = unapplied_answers(VID, lab.filed)
    assert len(missing) == 1 and missing[0].startswith("pellet 1:")
    assert "outcome displaced_sa, the person's answer is retrieved" in missing[0]
    assert "causal reach frames 20-30, the person's is frames 120-135" in missing[0]
    # Made with the answer: nothing missing.
    _features(lab.filed, pose)
    assert unapplied_answers(VID, lab.filed) == []


def test_ground_truth_outranks_the_review_in_the_check(lab):
    """Ground truth > review: the features file must carry the ground
    truth's outcome, not the review's."""
    from mousereach.kinematics.core.answers_applied import unapplied_answers
    pose = _video_files(lab.filed)
    _review(lab.filed)
    _features(lab.filed, pose)
    assert unapplied_answers(VID, lab.filed) == []
    _ground_truth(lab.filed, outcome="untouched")
    assert "the person's answer is untouched (ground_truth)" in unapplied_answers(VID, lab.filed)[0]


def test_a_ground_truth_uncertain_is_no_answer_to_carry(lab):
    from mousereach.kinematics.core.answers_applied import unapplied_answers
    pose = _video_files(lab.filed)
    _review(lab.filed)
    _features(lab.filed, pose)
    _ground_truth(lab.filed, outcome="uncertain")
    assert unapplied_answers(VID, lab.filed) == []


class _DB:
    """The scanner's view of a watcher database, keeping mark_reason."""

    def __init__(self, archived=(), outdated=()):
        self.rows = {s: {"video_id": s, "state": "archived"} for s in archived}
        self.rows.update({s: {"video_id": s, "state": "outdated",
                              "reprocess_scope": "kinematics"} for s in outdated})

    def get_videos_in_state(self, state):
        return [dict(r) for r in self.rows.values() if r["state"] == state]

    def _get_connection(self):
        raise RuntimeError("no db in this test")

    def force_state(self, vid, state, reason=None, **kw):
        self.rows.setdefault(vid, {"video_id": vid}).update(dict(kw, state=state))

    def update_state(self, vid, state, **kw):
        self.force_state(vid, state, **kw)

    def set_fields(self, vid, **kw):
        self.rows[vid].update(kw)


def _manifest_saying_the_review_was_applied(lab, review):
    sha = hashlib.sha256(review.read_bytes()).hexdigest()
    (lab.filed / f"{VID}_processing_manifest.json").write_text(json.dumps({
        "video_stem": VID, "created_at": "2099-01-02T10:00:00",
        "dlc_model": {"dlc_scorer": SCORER}, "pipeline_versions": dict(CURRENT),
        "applied_review": {"file": review.name, "sha256": sha},
        "applied_ground_truth": None}), encoding="utf-8")


def test_the_version_scan_reruns_a_filed_video_whose_features_lack_an_answer(lab, monkeypatch):
    """REPRODUCES (c) 2: the manifest's stamp says the review was applied (a
    run saw it), the features file does not carry its answer (an older rule
    put it on another pellet). The stamps alone call the video current; the
    content check marks it for a kinematics-only re-run, and the scan's
    two-way door does not lift that mark on the next pass."""
    from mousereach.watcher.reprocessor import ReprocessingScanner
    pose = _video_files(lab.filed)
    _features(lab.filed, pose)                       # lacks the answer
    review = _review(lab.filed)
    _manifest_saying_the_review_was_applied(lab, review)
    st = review.stat()
    monkeypatch.setattr(crio, "review_path_index",
                        lambda stems: {VID: (review, st.st_mtime, st.st_size)})
    db = _DB(archived=[VID])
    summary = ReprocessingScanner(db, lab.nas).scan(mark_outdated=True, adopt_orphans=False)
    row = db.rows[VID]
    assert row["state"] == "outdated" and row["reprocess_scope"] == "kinematics"
    assert summary.get("answers_missing") == 1
    assert row["mark_reason"].startswith("a person's answer is not in the filed features file")
    # Next pass: still owed (the stamps would have said "current").
    summary = ReprocessingScanner(db, lab.nas).scan(mark_outdated=True, adopt_orphans=False)
    assert db.rows[VID]["state"] == "outdated"
    assert summary["unmarked_current"] == 0


def test_the_version_scan_leaves_a_filed_video_that_carries_every_answer(lab, monkeypatch):
    from mousereach.watcher.reprocessor import ReprocessingScanner
    pose = _video_files(lab.filed)
    review = _review(lab.filed)
    _features(lab.filed, pose)                       # made with the answer
    _manifest_saying_the_review_was_applied(lab, review)
    st = review.stat()
    monkeypatch.setattr(crio, "review_path_index",
                        lambda stems: {VID: (review, st.st_mtime, st.st_size)})
    db = _DB(archived=[VID])
    summary = ReprocessingScanner(db, lab.nas).scan(mark_outdated=True, adopt_orphans=False)
    assert summary["answers_missing"] == 0 and summary["answers_checked"] == 1
    assert db.rows[VID]["state"] == "archived"


# ---------------------------------------------------------------------------
# (a) the archive step applies a missing answer before it files
# ---------------------------------------------------------------------------

def _orchestrator(lab, tmp_path):
    from mousereach.watcher.db import WatcherDB
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.db.export_to_central_db = lambda vid: None
    o.processing_dir = lab.proc
    o.hostname = "NODE-T"
    o.db.register_video(video_id=VID, source_path="x")
    o.db.force_state(VID, "processed", reason="test")
    return o


def test_an_answer_saved_after_the_kinematics_is_applied_before_filing(lab, tmp_path):
    pose = _video_files(lab.proc)
    _features(lab.proc, pose)
    (lab.proc / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    _review(lab.proc)                                # saved after the kinematics step
    o = _orchestrator(lab, tmp_path)
    assert o._apply_answers_before_filing(VID, lab.proc) is True
    seg = json.loads((lab.proc / f"{VID}_features.json").read_text())["segments"][0]
    assert seg["outcome"] == "retrieved"
    from mousereach.kinematics.core.answers_applied import unapplied_answers
    assert unapplied_answers(VID, lab.proc) == []


def test_filing_does_not_complete_when_the_answer_cannot_be_applied(lab, tmp_path):
    pose = _video_files(lab.proc)
    _features(lab.proc, pose)
    (lab.proc / f"{VID}_processing_manifest.json").write_text("{}", encoding="utf-8")
    _review(lab.proc)
    pose.unlink()                                    # nothing to make kinematics from
    o = _orchestrator(lab, tmp_path)
    assert o._apply_answers_before_filing(VID, lab.proc) is False
    row = o.db.get_video(VID)
    assert row["state"] == "failed"
    assert "a person's answer could not be applied before filing" in row["error_message"]


def test_a_video_with_every_answer_in_it_files_without_a_second_run(lab, tmp_path):
    pose = _video_files(lab.proc)
    _review(lab.proc)
    _features(lab.proc, pose)
    before = (lab.proc / f"{VID}_features.json").stat().st_mtime_ns
    o = _orchestrator(lab, tmp_path)
    assert o._apply_answers_before_filing(VID, lab.proc) is True
    assert (lab.proc / f"{VID}_features.json").stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# (b) a kinematics-only re-run never asks about a re-cut
# ---------------------------------------------------------------------------

def test_a_kinematics_only_rerun_does_not_ask_about_answers_a_recut_voided(lab):
    """A filed video whose person's answer a re-cut voided: a new run would
    ask again in triage; a kinematics-only re-run of the filed video asks
    nothing and keeps the filed decision on that pellet."""
    from mousereach.kinematics.core.feature_extractor import RecutVoidedAnswers
    pose = _video_files(lab.proc)
    feats = _features(lab.proc, pose)
    (lab.filed / f"{VID}_features.json").write_text(json.dumps(feats.to_dict()),
                                                    encoding="utf-8")
    ex = FeatureExtractor()
    ex.recut_voided = []

    import mousereach.review.truth_resolver as tr
    real = tr.resolve_truth_layers

    def voided(*a, **k):
        out = real(*a, **k)
        if k.get("applied") is not None:
            k["applied"]["recut_voided"] = [1]
        return out
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(tr, "resolve_truth_layers", voided)
    try:
        with pytest.raises(RecutVoidedAnswers):
            FeatureExtractor().extract(pose, lab.proc / f"{VID}_reaches.json",
                                       lab.proc / f"{VID}_pellet_outcomes.json")
        out = ex.extract(pose, lab.proc / f"{VID}_reaches.json",
                         lab.proc / f"{VID}_pellet_outcomes.json",
                         keep_filed=lab.filed / f"{VID}_features.json")
    finally:
        mp.undo()
    assert out.to_dict()["segments"][0]["outcome"] == "displaced_sa"
    assert ex.recut_voided == [1]
