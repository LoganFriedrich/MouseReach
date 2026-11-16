"""Where the version scan says a filed video's re-run starts.

The scan marks a filed video 'outdated' with the stage its re-run starts from
(reprocess_scope). A run starting at 'kinematics' re-runs no algorithm and no
quality or hand-score check (orchestrator.filed_rerun_reuses_analysis); a run
starting at segmentation re-runs everything. So the scope must be the EARLIEST
stage that really needs it -- no earlier.

Built under tmp_path with invented ids; no real folder or database.
"""
import json

import pytest

import mousereach.review.causal_review_io as crio
from mousereach.pipeline.manifest import TRACKED_STAGES
from mousereach.watcher import reprocessor
from mousereach.watcher.reprocessor import ReprocessingScanner

SCORER = "DLC_resnet50_TestModelshuffle1_100"
STEM = "20990101_ABC0101_P1"
CURRENT = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
           "assignment": "1.0", "kinematic_extractor": "5.0"}


class DB:
    def __init__(self, archived=(), outdated=()):
        self.rows = {s: "archived" for s in archived}
        self.rows.update({s: "outdated" for s in outdated})
        self.marks = {}

    def get_videos_in_state(self, state):
        return [{"video_id": s} for s, st in self.rows.items() if st == state]

    def _get_connection(self):
        raise RuntimeError("no db in this test")

    def force_state(self, vid, state, reason=None, **kw):
        self.rows[vid] = state
        self.marks[vid] = kw.get("reprocess_scope")

    def update_state(self, vid, state, **kw):
        self.force_state(vid, state, **kw)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    import mousereach.config as config
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    nas = tmp_path / "nas"
    cohort = nas / "Analyzed" / "Project_A" / "ABC01"
    cohort.mkdir(parents=True)
    (nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": dict(CURRENT, dlc_scorer=SCORER)}), encoding="utf-8")
    # No saved review anywhere unless a test puts one there.
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    return nas, cohort


def _filed(cohort, versions, stem=STEM, **extra):
    m = {"video_stem": stem, "created_at": "2099-01-01T10:00:00",
         "dlc_model": {"dlc_scorer": SCORER}, "pipeline_versions": versions}
    m.update(extra)
    (cohort / f"{stem}_processing_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    (cohort / f"{stem}_features.json").write_text("{}", encoding="utf-8")
    (cohort / f"{stem}.mp4").write_bytes(b"x")


def _scan(nas, db):
    sc = ReprocessingScanner(db, nas)
    return sc.scan(mark_outdated=True, adopt_orphans=False)


# --- a stage the manifest never recorded ------------------------------------

def test_a_manifest_missing_only_its_kinematics_version_reruns_kinematics_only(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, kinematic_extractor="not_run"))
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert db.marks[STEM] == "kinematics"
    listed = summary["outdated_videos"][0]["stale_components"]
    assert "kinematic_extractor (not recorded)" in listed


def test_an_unrecorded_stage_starts_the_rerun_at_its_own_stage(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, reach_detector="unknown"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks[STEM] == "reach"


def test_an_unrecorded_stage_earlier_than_a_stale_one_wins(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, outcome_detector="", kinematic_extractor="4.9"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks[STEM] == "outcome"


def test_a_stale_assignment_starts_at_outcome_not_segmentation(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, assignment="0.9"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks[STEM] == "outcome"


def test_every_tracked_stage_has_a_start_stage():
    """A tracked stage with no line in the map is ignored, and a video whose
    only stale stage is ignored is re-run from segmentation."""
    assert set(TRACKED_STAGES) <= set(reprocessor._STAGE_FOR_COMPONENT)


# --- a filed video with no processing manifest ------------------------------
# WHY: the scan skipped a video with no manifest before it looked for a saved
# review, so a review saved for it never reached its filed kinematics.

def _review_for(tmp_path, monkeypatch, stem=STEM, mtime=4_000_000_000):
    import os
    p = tmp_path / "reviews" / f"{stem}_causal_review.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"segments": []}), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    monkeypatch.setattr(crio, "review_path_index",
                        lambda stems: {stem: (p, mtime, p.stat().st_size)})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: p)
    return p


def _filed_without_manifest(cohort, stem=STEM):
    (cohort / f"{stem}_features.json").write_text("{}", encoding="utf-8")
    (cohort / f"{stem}.mp4").write_bytes(b"x")


def test_a_review_saved_for_a_filed_video_with_no_manifest_reruns_its_kinematics(
        archive, tmp_path, monkeypatch):
    nas, cohort = archive
    _filed_without_manifest(cohort)
    _review_for(tmp_path, monkeypatch)
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert db.marks[STEM] == "kinematics"
    assert summary["no_manifest"] == 0


def test_a_filed_video_with_no_manifest_and_no_review_is_only_counted(archive):
    nas, cohort = archive
    _filed_without_manifest(cohort)
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert summary["no_manifest"] == 1
    assert STEM not in db.marks
