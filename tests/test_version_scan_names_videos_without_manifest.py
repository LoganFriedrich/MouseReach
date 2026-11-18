"""The version scan names the filed videos it cannot check, and lists them for
a person.

A filed video with no processing manifest has no recorded versions, so no
version scan ever marks it outdated: a newer algorithm never reaches it. The
scan used to add it to a bare count. It now names each one in its log line and
in its summary, and the watcher rewrites the attention list
attention/<host>_filed_without_manifest.json from every scan that listed the
archive (and leaves it alone when the listing failed, since then every filed
video looks manifest-less).

Built under tmp_path with invented ids; no real folder or database.
"""
import json
import logging
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_io as crio
from mousereach.config import Paths
from mousereach.watcher.orchestrator import ProcessingOrchestrator
from mousereach.watcher.reprocessor import ReprocessingScanner

SCORER = "DLC_resnet50_TestModelshuffle1_100"
WITH = "20990101_ABC0101_P1"
WITHOUT = "20990101_ABC0102_P1"
CURRENT = {"segmenter": "2.0", "reach_detector": "3.0", "outcome_detector": "4.0",
           "assignment": "1.0", "kinematic_extractor": "5.0"}


class DB:
    def __init__(self, archived=()):
        self.rows = {s: "archived" for s in archived}

    def get_videos_in_state(self, state):
        return [{"video_id": s} for s, st in self.rows.items() if st == state]

    def _get_connection(self):
        raise RuntimeError("no db in this test")

    def force_state(self, vid, state, reason=None, **kw):
        self.rows[vid] = state

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
    import mousereach.pipeline.version_index as vi
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    m = {"video_stem": WITH, "created_at": "2099-01-01T10:00:00",
         "dlc_model": {"dlc_scorer": SCORER}, "pipeline_versions": dict(CURRENT)}
    (cohort / f"{WITH}_processing_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    for stem in (WITH, WITHOUT):
        (cohort / f"{stem}_features.json").write_text("{}", encoding="utf-8")
        (cohort / f"{stem}.mp4").write_bytes(b"x")
    return nas


def test_a_filed_video_with_no_manifest_is_named_in_the_scan(archive, caplog):
    db = DB(archived=[WITH, WITHOUT])
    with caplog.at_level(logging.INFO, logger="mousereach.watcher.reprocessor"):
        summary = ReprocessingScanner(db, archive).scan(mark_outdated=True,
                                                        adopt_orphans=False)
    assert summary["no_manifest"] == 1
    assert summary.get("no_manifest_videos") == [WITHOUT]
    named = [r.getMessage() for r in caplog.records if WITHOUT in r.getMessage()]
    assert named, "the scan's log does not name the video"
    assert db.rows[WITHOUT] == "archived"     # named, not re-run by the scan


def _orchestrator(tmp_path, monkeypatch, summary):
    nas = tmp_path / "nas"
    nas.mkdir(exist_ok=True)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.hostname = "node-a"
    o.config = SimpleNamespace(reprocess_scan_interval_seconds=0)
    o._reprocess_scanner = SimpleNamespace(scan=lambda **k: dict(summary))
    o._publish_repose_requests = lambda: None
    o._write_failed_rerun_attention = lambda: None
    return o, nas / "attention" / "node-a_filed_without_manifest.json"


def test_the_watcher_lists_them_for_a_person(tmp_path, monkeypatch):
    o, listed = _orchestrator(tmp_path, monkeypatch, {
        "outdated": 0, "no_manifest": 1, "no_manifest_videos": [WITHOUT],
        "no_manifest_listed": True})
    o._maybe_review_reprocess_scan(force=True)
    assert listed.is_file(), "no attention list was written"
    items = json.loads(listed.read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in items] == [WITHOUT]
    assert items[0]["needs_person"] is True
    assert "Re-run video" in items[0]["says"]

    # A later scan that finds it with a manifest takes it off the list.
    o._reprocess_scanner = SimpleNamespace(scan=lambda **k: {
        "outdated": 0, "no_manifest": 0, "no_manifest_videos": [],
        "no_manifest_listed": True})
    o._maybe_review_reprocess_scan(force=True)
    assert json.loads(listed.read_text(encoding="utf-8"))["items"] == []


def test_a_scan_that_could_not_list_the_archive_leaves_the_list_alone(
        tmp_path, monkeypatch):
    o, listed = _orchestrator(tmp_path, monkeypatch, {
        "outdated": 0, "no_manifest": 1, "no_manifest_videos": [WITHOUT],
        "no_manifest_listed": True})
    o._maybe_review_reprocess_scan(force=True)
    before = listed.read_text(encoding="utf-8")
    o._reprocess_scanner = SimpleNamespace(scan=lambda **k: {
        "outdated": 0, "no_manifest": 2, "no_manifest_videos": [],
        "no_manifest_listed": False})
    o._maybe_review_reprocess_scan(force=True)
    assert listed.read_text(encoding="utf-8") == before


def test_a_video_the_listing_did_not_show_is_not_named(archive):
    """An 'archived' row whose files the listing does not show is not known to
    lack a manifest: it is counted, not named."""
    gone = "20990101_ABC0103_P1"
    db = DB(archived=[WITH, gone])
    summary = ReprocessingScanner(db, archive).scan(mark_outdated=True,
                                                    adopt_orphans=False)
    assert summary["no_manifest_videos"] == []
    assert summary["no_manifest_listed"] is True


def test_an_empty_listing_names_nobody_and_is_not_used(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    nas.mkdir()
    (nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": dict(CURRENT, dlc_scorer=SCORER)}), encoding="utf-8")
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    db = DB(archived=[WITHOUT])        # no Analyzed folder at all
    summary = ReprocessingScanner(db, nas).scan(mark_outdated=True,
                                                adopt_orphans=False)
    assert summary["no_manifest_videos"] == []
    assert summary["no_manifest_listed"] is False
def test_a_pose_node_that_also_processes_still_asks_for_re_poses(
        tmp_path, monkeypatch):
    """A graphics-card node with also_process on runs the version scan too.
    Writing the no-manifest list must work there and must not stop the scan
    before it publishes its re-pose requests."""
    from mousereach.watcher.orchestrator import DLCOrchestrator
    nas = tmp_path / "nas"
    nas.mkdir(exist_ok=True)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas)
    o = DLCOrchestrator.__new__(DLCOrchestrator)
    o.hostname = "node-b"
    o.handles_reprocessing = True
    o.adopts_orphans = False
    o.config = SimpleNamespace(reprocess_scan_interval_seconds=0, also_process=True)
    o._reprocess_scanner = SimpleNamespace(scan=lambda **k: {
        "outdated": 0, "no_manifest": 1, "no_manifest_videos": [WITHOUT],
        "no_manifest_listed": True})
    asked = []
    o._publish_repose_requests = lambda: asked.append(True)
    warnings = []
    monkeypatch.setattr("mousereach.watcher.orchestrator.logger.warning",
                        lambda msg, *a, **k: warnings.append(str(msg)))
    o._maybe_review_reprocess_scan(force=True)
    assert not [w for w in warnings if "Reprocess scan skipped" in w], warnings
    assert asked == [True], "the scan stopped before asking for re-poses"
    listed = nas / "attention" / "node-b_filed_without_manifest.json"
    items = json.loads(listed.read_text(encoding="utf-8"))["items"]
    assert [i["video_id"] for i in items] == [WITHOUT]
