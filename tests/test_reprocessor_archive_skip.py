"""The version scanner never reads Analyzed/Archive/ as live data.

WHY: superseded outputs move to Analyzed/Archive/ and KEEP THEIR ORIGINAL
NAMES ({stem}_processing_manifest.json, {stem}_features.json,
{stem}_segments.json, {stem}_pellet_outcomes.json, {stem}DLC_...h5). Every
walk the scanner made of Analyzed (rglob, and the two-levels-deep index glob
-- Analyzed/Archive/<folder>/<file> is exactly that deep) picked those up
silently: an archived pose counted as "declared pose already on disk" (no
GPU re-pose ever queued), an archived manifest was adopted as a video, an
archived segments/outcomes file decided whether an ARCHIVE folder got routed
to deep review.

Each test builds a tmp Analyzed tree with a same-named decoy under
Analyzed/Archive/ and calls the real scanner code. Every test below fails
against the old rglob/glob code; the reason is stated per test. Tests that
place ONLY a decoy are order-independent. Tests with a live copy AND a decoy
also rely on the fact that the old walks visit "Archive" before "Connectome"
(alphabetical directory order on NTFS, where these tests run); they
additionally pin that live data is still found.
"""
import json
import os
import time

import pytest

import mousereach.pipeline.versions as versions_mod
import mousereach.review.causal_review_io as crio
import mousereach.watcher.review_gate as rg
from mousereach.watcher.reprocessor import (
    ReprocessingScanner,
    pose_scorers_in_archive,
)

OLD = "DLC_resnet50_MPSAOct27shuffle1_100000"
NEW = "DLC_resnet101_MPSAOct27shuffle3_100000"
STEM = "20240101_ABC0101_P1"
STEM_B = "20240101_ABC0102_P1"

STALE = {"segmenter": "1.0.0", "reach_detector": "1.0.0",
         "outcome_detector": "1.0.0", "assignment": "1.0.0",
         "kinematic_extractor": "1.0.0"}
CURRENT = {"versions": {"dlc_scorer": NEW, "segmenter": "1.0.0",
                        "reach_detector": "1.0.0", "outcome_detector": "1.0.0",
                        "assignment": "1.0.0", "kinematic_extractor": "1.0.0"}}


# --------------------------------------------------------------- tree helpers

def _analyzed(tmp_path):
    a = tmp_path / "Analyzed"
    a.mkdir(exist_ok=True)
    return a


def _live_cohort(tmp_path):
    d = _analyzed(tmp_path) / "Connectome" / "CNT01"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _decoy_dir(tmp_path, *parts):
    parts = parts or ("superseded_processing_root_3.1",)
    d = _analyzed(tmp_path).joinpath("Archive", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _live_pose_dir(tmp_path):
    d = _analyzed(tmp_path) / "Connectome" / "DLC Model 4" / "CNT01"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class StubDB:
    """Just enough WatcherDB for scan(): ``rows`` already have a db row (no
    adoption), ``archived`` are returned for state 'archived'."""

    def __init__(self, rows=(), archived=()):
        self.rows = list(rows)
        self.archived = list(archived)
        self.registered = []
        self.forced = []

    def _get_connection(self):
        rows = self.rows

        class C:
            def execute(self, *a):
                return [(r,) for r in rows]

            def close(self):
                pass
        return C()

    def get_videos_in_state(self, state):
        if state == "archived":
            return [{"video_id": v, "state": "archived"} for v in self.archived]
        return []

    def register_video(self, video_id, source_path=None, current_path=None, **k):
        self.registered.append((video_id, source_path))

    def force_state(self, video_id, state, **k):
        self.forced.append((video_id, state, k.get("reprocess_scope")))

    def update_state(self, *a, **k):
        pass


def _scanner(tmp_path, db=None):
    _analyzed(tmp_path)
    return ReprocessingScanner(db or StubDB(), tmp_path)


@pytest.fixture
def no_review(monkeypatch):
    # The real resolver looks at configured queue folders; keep the test on
    # tmp_path only.
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)


# ------------------------------------------------------------ pose index walk

def test_archived_pose_is_not_pose_present_live_model_folder_is(tmp_path):
    """Old code: rglob("*DLC*.h5") entered Analyzed/Archive/DLC Model 4.0/ and
    put STEM in the index, so the first assert failed."""
    _write_json(_decoy_dir(tmp_path, "DLC Model 4.0", "CNT01")
                / f"{STEM}{NEW}.h5", {})
    _write_json(_live_pose_dir(tmp_path) / f"{STEM_B}{NEW}.h5", {})

    index = pose_scorers_in_archive(tmp_path / "Analyzed")

    assert STEM not in index
    assert index.get(STEM_B) == {NEW}


def test_scan_queues_a_real_repose_when_the_declared_pose_is_only_archived(
        tmp_path, monkeypatch, no_review):
    """End to end through scan(): both manifests name the OLD model. STEM's
    declared (NEW) pose exists only under Analyzed/Archive -> it genuinely
    needs a GPU re-pose (scope 'full'). STEM_B's declared pose is live in
    DLC Model 4 -> re-run from segmentation. Old code counted the archived
    pose, gave STEM scope 'segmentation' and pose_already_current == 2."""
    coh = _live_cohort(tmp_path)
    for vid in (STEM, STEM_B):
        _write_json(coh / f"{vid}_processing_manifest.json",
                    {"dlc_model": {"dlc_scorer": OLD},
                     "pipeline_versions": STALE})
        (coh / f"{vid}.mp4").write_text("x", encoding="utf-8")
    (_decoy_dir(tmp_path, "DLC Model 4.0", "CNT01")
     / f"{STEM}{NEW}.h5").write_text("pose", encoding="utf-8")
    (_live_pose_dir(tmp_path) / f"{STEM_B}{NEW}.h5").write_text(
        "pose", encoding="utf-8")
    monkeypatch.setattr(versions_mod, "get_current_versions",
                        lambda root=None: CURRENT)
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])

    db = StubDB(rows=(STEM, STEM_B), archived=(STEM, STEM_B))
    summary = _scanner(tmp_path, db).scan(mark_outdated=True,
                                          adopt_orphans=False)

    assert (STEM, "outdated", "full") in db.forced
    assert (STEM_B, "outdated", "segmentation") in db.forced
    assert summary["pose_already_current"] == 1
    assert summary["outdated_full"] == 1


# ------------------------------------------------------- one-walk scan index

def test_scan_never_adopts_an_archived_manifest(tmp_path, monkeypatch,
                                                no_review):
    """Old code: glob("*/*/*_processing_manifest.json") matched
    Analyzed/Archive/superseded_processing_root_3.1/<manifest> and registered
    the ARCHIVED mp4 beside it as a live video."""
    decoy = _decoy_dir(tmp_path)
    _write_json(decoy / f"{STEM}_processing_manifest.json", {})
    (decoy / f"{STEM}.mp4").write_text("x", encoding="utf-8")
    coh = _live_cohort(tmp_path)
    _write_json(coh / f"{STEM_B}_processing_manifest.json", {})
    (coh / f"{STEM_B}.mp4").write_text("x", encoding="utf-8")
    monkeypatch.setattr(versions_mod, "get_current_versions",
                        lambda root=None: {"versions": {"dlc_scorer": "X"}})
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])

    db = StubDB()
    summary = _scanner(tmp_path, db).scan(mark_outdated=True)

    assert [r[0] for r in db.registered] == [STEM_B]
    assert summary["adopted"] == 1
    assert STEM not in summary["adopt_no_mp4"]


def test_scan_features_index_ignores_archived_features(tmp_path, monkeypatch):
    """A review is saved; the only features file is an ARCHIVED one, stamped
    newer than the review. The live video has no kinematics yet, so the review
    is pending and a kinematics re-run is owed. Old code: the depth-limited
    features glob indexed the archived file, its newer mtime said "review
    already applied", and the video was reported current."""
    coh = _live_cohort(tmp_path)
    # Non-empty: scan() counts an empty manifest dict as "no manifest".
    _write_json(coh / f"{STEM}_processing_manifest.json",
                {"pipeline_versions": STALE})
    feats = _write_json(_decoy_dir(tmp_path) / f"{STEM}_features.json", {})
    future = time.time() + 86400
    os.utime(feats, (future, future))
    review = _write_json(tmp_path / f"{STEM}_causal_review.json",
                         {"segments": []})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: review)
    monkeypatch.setattr(versions_mod, "get_current_versions",
                        lambda root=None: {"versions": {"dlc_scorer": "X"}})
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])
    monkeypatch.setattr(versions_mod, "compare_manifest_to_current",
                        lambda m, c: {"is_current": True,
                                      "stale_components": [],
                                      "needs_full_reprocess": False})

    db = StubDB(rows=(STEM,), archived=(STEM,))
    summary = _scanner(tmp_path, db).scan(mark_outdated=True)

    assert summary["review_triggered"] == 1
    assert summary["current"] == 0
    assert (STEM, "outdated", "kinematics") in db.forced


def test_pending_review_fallback_walk_ignores_archived_features(tmp_path,
                                                                monkeypatch):
    """Same case through the per-video fallback (no index passed). Old code:
    next(rglob(features)) returned the archived file -> None (not pending)."""
    feats = _write_json(_decoy_dir(tmp_path) / f"{STEM}_features.json", {})
    future = time.time() + 86400
    os.utime(feats, (future, future))
    review = _write_json(tmp_path / f"{STEM}_causal_review.json",
                         {"segments": []})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: review)

    sc = _scanner(tmp_path)
    assert sc._pending_review_path(STEM, {}, None) == review


# -------------------------------------------------------------- _load_manifest

@pytest.mark.parametrize("decoy_parts", [
    # two levels deep: the old glob("*/*/<manifest>") matched this
    ("superseded_processing_root_3.1",),
    # deeper: only the old rglob fallback matched this
    ("2026-09-01", "Connectome", "CNT01"),
])
def test_load_manifest_ignores_an_archived_manifest(tmp_path, decoy_parts):
    """Old code returned the archived manifest for a stem with no live one."""
    _write_json(_decoy_dir(tmp_path, *decoy_parts)
                / f"{STEM}_processing_manifest.json", {"which": "archive"})
    sc = _scanner(tmp_path)
    assert sc._load_manifest(STEM) is None
    assert sc._load_manifest_indexed(STEM, {}) is None


def test_load_manifest_ignores_a_pre_backfill_backup_under_archived(tmp_path):
    """Analyzed/_archived/<folder>/<manifest> is two levels deep, like a live
    Analyzed/<project>/<cohort>/<manifest>; backfill-manifest-versions run with
    --root Analyzed/<project> used to write original-named backups there. Old
    code (SUPERSEDED_DIR_NAMES == {"Archive"}) let _outside_superseded keep
    the hit and returned the backup for a stem with no live manifest."""
    backup = _analyzed(tmp_path) / "_archived" / "manifests_pre_version_backfill_x"
    backup.mkdir(parents=True)
    _write_json(backup / f"{STEM}_processing_manifest.json", {"which": "backup"})
    sc = _scanner(tmp_path)
    assert sc._load_manifest(STEM) is None
    assert sc._load_manifest_indexed(STEM, {}) is None


def test_load_manifest_returns_the_live_manifest_over_an_archived_one(
        tmp_path):
    """Old code's glob visited Analyzed/Archive before Analyzed/Connectome and
    returned the archived manifest first."""
    _write_json(_live_cohort(tmp_path) / f"{STEM}_processing_manifest.json",
                {"which": "live"})
    _write_json(_decoy_dir(tmp_path) / f"{STEM}_processing_manifest.json",
                {"which": "archive"})
    assert _scanner(tmp_path)._load_manifest(STEM) == {"which": "live"}


# --------------------------------------------------------- _segments_human_fixed

def test_segments_human_fixed_ignores_archived_segments(tmp_path):
    """Old code: next(rglob(segments)) read the archived human segments and
    said the boundaries were already fixed."""
    _write_json(_decoy_dir(tmp_path) / f"{STEM}_segments.json",
                {"boundary_source": "human"})
    assert _scanner(tmp_path)._segments_human_fixed(STEM) is False


@pytest.mark.parametrize("live_src, decoy_src, expected", [
    ("algo", "human", False),
    ("human", "algo", True),
])
def test_segments_human_fixed_reads_the_live_segments(tmp_path, live_src,
                                                      decoy_src, expected):
    """Old code read Analyzed/Archive first and returned the decoy's answer."""
    _write_json(_live_cohort(tmp_path) / f"{STEM}_segments.json",
                {"boundary_source": live_src})
    _write_json(_decoy_dir(tmp_path) / f"{STEM}_segments.json",
                {"boundary_source": decoy_src})
    assert _scanner(tmp_path)._segments_human_fixed(STEM) is expected


# ------------------------------------------------ _divert_mislabel_to_deep_review

@pytest.fixture
def routed(monkeypatch):
    calls = []
    monkeypatch.setattr(rg, "route_deep_review",
                        lambda vid, src, reason=None, db=None, **k:
                        calls.append((vid, src)))
    return calls


def test_an_archive_folder_is_never_routed_to_deep_review(tmp_path, routed):
    """Old code: next(rglob(pellet_outcomes)) found the archived file and
    routed its Analyzed/Archive/<folder> into the deep review queue."""
    _write_json(_decoy_dir(tmp_path) / f"{STEM}_pellet_outcomes.json", {})
    _scanner(tmp_path)._divert_mislabel_to_deep_review(STEM)
    assert routed == []


def test_divert_routes_the_live_bundle_not_the_archived_copy(tmp_path, routed):
    """Old code visited Analyzed/Archive first and routed the archive folder."""
    coh = _live_cohort(tmp_path)
    _write_json(coh / f"{STEM}_pellet_outcomes.json", {})
    _write_json(_decoy_dir(tmp_path) / f"{STEM}_pellet_outcomes.json", {})
    _scanner(tmp_path)._divert_mislabel_to_deep_review(STEM)
    assert routed == [(STEM, coh)]
