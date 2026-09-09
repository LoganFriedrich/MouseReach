"""The scanner's "is the review applied?" question is answered by CONTENT
identity, never by cross-clock mtimes.

WHY: SMB-written review files carry the NAS server's clock; archived features
keep the local clock (copy2 preserves it). A ~20-minute NAS clock skew made
every fresh re-run's features compare OLDER than the review it had just
applied, so reviewed videos reprocessed in a loop all day (2026-09-09). The
manifest now records which review the kinematics applied (reviewed_at +
sha256, written by record_kinematic_version) and the scanner compares that.

Also: human-authored segmentation (boundary_source == 'human') is
version-exempt -- a human's cuts cannot go stale with a segmenter release
(20251009_CNT0308_P3 was swept into the Archive 53 times before this).
"""
import json
from pathlib import Path

import pytest

from mousereach.pipeline.fsutil import sha256_file
from mousereach.watcher.reprocessor import ReprocessingScanner
import mousereach.review.causal_review_io as crio


class FakeDB:
    def get_videos_in_state(self, state):
        return []


STEM = "19990101_TEST0001_P1"


@pytest.fixture
def scanner(tmp_path):
    (tmp_path / "Analyzed").mkdir()
    return ReprocessingScanner(FakeDB(), tmp_path)


@pytest.fixture
def review(tmp_path, monkeypatch):
    p = tmp_path / f"{STEM}_causal_review.json"
    p.write_text(json.dumps({
        "reviewer": "tester",
        "reviewed_at": "2026-09-09T14:11:01.598657",
        "segments": [{"segment_num": 3, "answers": {"reviewed": True}}],
    }), encoding="utf-8")
    monkeypatch.setattr(crio, "resolve_review_path",
                        lambda vid, primary_dir=None: p)
    return p


def test_matching_stamp_means_applied(scanner, review):
    manifest = {"applied_review": {"sha256": sha256_file(review)}}
    assert scanner._pending_review_path(STEM, manifest, {}) is None


def test_differing_stamp_means_pending(scanner, review):
    manifest = {"applied_review": {"sha256": "0" * 64}}
    assert scanner._pending_review_path(STEM, manifest, {}) == review


def test_no_stamp_falls_back_to_process_clock_fields_not_mtimes(
        scanner, review):
    """The skew case pinned: the review FILE's mtime is far newer than the
    features mtime (NAS clock ahead), but its in-file reviewed_at predates the
    manifest's created_at -- the run already applied it. Old code re-marked;
    new code does not."""
    manifest = {"created_at": "2026-09-09T14:14:30.000000"}   # after reviewed_at
    feats_mtime = {STEM: review.stat().st_mtime - 3600}       # mtime says stale
    assert scanner._pending_review_path(STEM, manifest, feats_mtime) is None


def test_no_stamp_review_saved_after_run_is_pending(scanner, review):
    manifest = {"created_at": "2026-09-09T14:05:00.000000"}   # before reviewed_at
    feats_mtime = {STEM: review.stat().st_mtime + 3600}
    assert scanner._pending_review_path(STEM, manifest, feats_mtime) == review


def test_reviewed_but_no_kinematics_yet_is_pending(scanner, review):
    assert scanner._pending_review_path(STEM, {}, {}) == review


# --- human-segmentation version exemption --------------------------------

def _seg_beside_manifest(tmp_path, source):
    mdir = tmp_path / "Analyzed" / "P" / "C"
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / f"{STEM}_processing_manifest.json"
    mpath.write_text("{}", encoding="utf-8")
    (mdir / f"{STEM}_segments.json").write_text(
        json.dumps({"boundary_source": source, "boundaries": []}),
        encoding="utf-8")
    return {STEM: mpath}


def test_human_segmentation_is_version_exempt(scanner, tmp_path):
    idx = _seg_beside_manifest(tmp_path, "human")
    comparison = {"is_current": False, "stale_components": ["segmenter"],
                  "needs_full_reprocess": False}
    out = scanner._drop_human_seg_staleness(STEM, idx, comparison)
    assert out["stale_components"] == []
    assert out["is_current"] is True


def test_algo_segmentation_still_goes_stale(scanner, tmp_path):
    idx = _seg_beside_manifest(tmp_path, "algo")
    comparison = {"is_current": False, "stale_components": ["segmenter"],
                  "needs_full_reprocess": False}
    out = scanner._drop_human_seg_staleness(STEM, idx, comparison)
    assert out["stale_components"] == ["segmenter"]
    assert out["is_current"] is False


def test_other_stale_stages_survive_the_exemption(scanner, tmp_path):
    idx = _seg_beside_manifest(tmp_path, "human")
    comparison = {"is_current": False,
                  "stale_components": ["segmenter", "outcome_detector"],
                  "needs_full_reprocess": False}
    out = scanner._drop_human_seg_staleness(STEM, idx, comparison)
    assert out["stale_components"] == ["outcome_detector"]
    assert out["is_current"] is False


# --- the stamp writer -----------------------------------------------------

def test_record_kinematic_version_stamps_applied_review(tmp_path, monkeypatch):
    import mousereach.pipeline.version_index as vi
    monkeypatch.setattr(vi, "VersionIndex",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    from mousereach.pipeline.manifest import record_kinematic_version

    mpath = tmp_path / f"{STEM}_processing_manifest.json"
    mpath.write_text(json.dumps({"pipeline_versions": {}}), encoding="utf-8")
    review = tmp_path / f"{STEM}_causal_review.json"
    review.write_text(json.dumps({"reviewed_at": "2026-09-09T14:11:01"}),
                      encoding="utf-8")

    assert record_kinematic_version(STEM, tmp_path, "2.1.0",
                                    review_path=review)
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    stamp = doc["applied_review"]
    assert stamp["reviewed_at"] == "2026-09-09T14:11:01"
    assert stamp["sha256"] == sha256_file(review)

    # A later run that applied NO review clears the stamp -- the manifest
    # describes the CURRENT features.
    assert record_kinematic_version(STEM, tmp_path, "2.1.0", review_path=None)
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    assert "applied_review" not in doc
