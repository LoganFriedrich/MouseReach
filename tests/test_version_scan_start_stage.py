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
    # The shared version index is never written by a test.
    import mousereach.pipeline.version_index as vi
    from types import SimpleNamespace
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
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
    assert db.marks.get(STEM) == "kinematics"
    listed = summary["outdated_videos"][0]["stale_components"]
    assert "kinematic_extractor (not recorded)" in listed


# Filed is final (2026-11-19): an algorithm stage never starts a filed
# video's re-run; only kinematics does (test_filed_is_final.py has more).

def test_an_unrecorded_algorithm_stage_marks_nothing(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, reach_detector="unknown"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert STEM not in db.marks


def test_an_unrecorded_algorithm_stage_beside_a_stale_extractor_reruns_kinematics_only(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, outcome_detector="", kinematic_extractor="4.9"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks.get(STEM) == "kinematics"


def test_a_stale_assignment_marks_nothing(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT, assignment="0.9"))
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert STEM not in db.marks


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
    assert db.marks.get(STEM) == "kinematics"
    assert summary["no_manifest"] == 0


def test_a_filed_video_with_no_manifest_and_no_review_is_only_counted(archive):
    nas, cohort = archive
    _filed_without_manifest(cohort)
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert summary["no_manifest"] == 1
    assert STEM not in db.marks


# --- ground truth saved or changed after filing ------------------------------
# WHY: the scan compared only the causal review with what the filed kinematics
# applied, so ground truth saved after filing reached the features only if
# some other reason re-ran the video (pipeline.manifest.ground_truth_applied).

def _gt(folder, last_modified_at, stem=STEM, body="a"):
    p = folder / f"{stem}_unified_ground_truth.json"
    p.write_text(json.dumps({"video_name": stem, "last_modified_at": last_modified_at,
                             "note": body}), encoding="utf-8")
    return p


def test_ground_truth_saved_after_filing_reruns_the_kinematics(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    _gt(cohort, "2099-01-02T09:00:00")          # after the manifest's created_at
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert db.marks.get(STEM) == "kinematics"
    assert summary["outdated_videos"][0]["stale_components"] == ["ground_truth"]


def test_ground_truth_older_than_the_run_is_not_a_reason(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    _gt(cohort, "2099-01-01T09:00:00")          # before created_at
    db = DB(archived=[STEM])
    assert _scan(nas, db)["current"] == 1
    assert STEM not in db.marks


def test_the_rerun_stamps_the_ground_truth_and_the_next_scan_leaves_it(archive, tmp_path):
    """After the re-run the stamp matches and the video is not marked again;
    a later change to the file marks it once more."""
    from mousereach.pipeline.manifest import record_kinematic_version
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    gt = _gt(cohort, "2099-01-02T09:00:00")
    # what the re-run does: the filed files are copied down, kinematics run,
    # the manifest is stamped with the ground truth beside the run, and filed.
    run = tmp_path / "run"
    run.mkdir()
    for f in cohort.iterdir():
        (run / f.name).write_bytes(f.read_bytes())
    assert record_kinematic_version(STEM, run, "5.0",
                                    ground_truth_path=run / gt.name)
    (cohort / f"{STEM}_processing_manifest.json").write_bytes(
        (run / f"{STEM}_processing_manifest.json").read_bytes())
    db = DB(archived=[STEM])
    assert _scan(nas, db)["current"] == 1
    _gt(cohort, "2099-01-03T09:00:00", body="changed")
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks.get(STEM) == "kinematics"


def test_kinematics_that_applied_no_ground_truth_owe_one_filed_later(archive):
    """A null stamp says the extractor applied NO ground truth (its lookup
    found none). A file filed beside the video is owed a re-run even though
    the file's own date is older than the manifest -- the case a stale
    ground-truth index produced: the file was there, the run did not see it."""
    nas, cohort = archive
    _filed(cohort, dict(CURRENT), applied_ground_truth=None)
    _gt(cohort, "2099-01-01T09:00:00")          # before created_at
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert db.marks.get(STEM) == "kinematics"


def test_a_marked_video_whose_ground_truth_is_still_owed_is_not_unmarked(archive):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    _gt(cohort, "2099-01-02T09:00:00")
    db = DB(outdated=[STEM])
    assert _scan(nas, db)["unmarked_current"] == 0
    assert db.rows[STEM] == "outdated"


def test_recording_retirement_agrees_about_ground_truth(archive):
    """Recording retirement must not call a video complete that the scan
    will re-run (collage_provenance.build_complete_stems)."""
    from mousereach.video_prep.core.collage_provenance import build_complete_stems
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    _gt(cohort, "2099-01-02T09:00:00")
    assert STEM not in build_complete_stems(nas / "Analyzed", nas)
    _gt(cohort, "2099-01-01T09:00:00")
    assert STEM in build_complete_stems(nas / "Analyzed", nas)


# --- a pass whose listing of the archive failed ------------------------------
# WHY: with the listing gone (a share hiccup) every filed video looks as if it
# had no manifest and no features file; marked on that, each one with a saved
# review or ground truth would get a re-run it does not owe.

def _listing_fails(monkeypatch, seen_gt=None):
    def _raise(self):
        if seen_gt is not None:
            self._filed_gt = dict(seen_gt)     # what a listing saw before it broke
        raise OSError("the share went away")
    monkeypatch.setattr(ReprocessingScanner, "_index_archive", _raise)


def test_a_pass_that_could_not_list_the_archive_marks_no_manifestless_video(
        archive, tmp_path, monkeypatch):
    nas, cohort = archive
    _filed_without_manifest(cohort)
    _review_for(tmp_path, monkeypatch)
    _listing_fails(monkeypatch)
    db = DB(archived=[STEM])
    summary = _scan(nas, db)
    assert STEM not in db.marks
    assert summary["no_manifest"] == 1


def test_a_listing_that_broke_part_way_is_no_ground_truth_reason(archive, monkeypatch):
    nas, cohort = archive
    _filed(cohort, dict(CURRENT))
    gt = _gt(cohort, "2099-01-02T09:00:00")
    _listing_fails(monkeypatch, seen_gt={STEM: (gt, gt.stat().st_mtime)})
    db = DB(archived=[STEM])
    _scan(nas, db)
    assert STEM not in db.marks
