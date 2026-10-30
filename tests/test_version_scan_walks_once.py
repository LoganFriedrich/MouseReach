"""The version scan walks the archive a bounded number of times, never once
per video.

WHY (2026-10-30): a video whose manifest the scan's index does not hold fell
back to ``iter_files(archive_dir, "<stem>_processing_manifest.json")`` -- a full
recursive walk of Analyzed to look for ONE file name. A video with no manifest
on disk never stops costing that, so it paid a whole-tree walk on every pass.
Measured on the live archive: 55 such videos of 5,965 with a supported tray, at
about 35 s a walk -- roughly 32 minutes of a 37-minute pass, during which the
watcher's work loop does nothing. The scans had been 6-50 s a week earlier.

What matters is the WALK COUNT, so that is what these tests assert: it must not
grow with the number of videos the index missed.
"""
import json

import pytest

import mousereach.pipeline.analyzed_tree as analyzed_tree
from mousereach.watcher.reprocessor import ReprocessingScanner

SCORER = "DLC_resnet101_MPSAOct27shuffle3_100000"


def _manifest(folder, stem, seg="2.0", scorer=SCORER):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (stem + "_processing_manifest.json")).write_text(json.dumps({
        "video_stem": stem, "created_at": "2026-01-01T00:00:00",
        "dlc_model": {"dlc_scorer": scorer},
        "components": {"segmentation": {"version": seg}}}), encoding="utf-8")
    (folder / (stem + "_features.json")).write_text("{}", encoding="utf-8")


@pytest.fixture
def counted_walks(monkeypatch):
    """Count every full-tree walk the scan performs."""
    calls = []
    real = analyzed_tree.iter_files

    def counting(root, pattern="*"):
        calls.append(pattern)
        return real(root, pattern)

    monkeypatch.setattr(analyzed_tree, "iter_files", counting)
    return calls


def _scanner(tmp_path, stems_with_manifests, all_stems, monkeypatch):
    import mousereach.config as config
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "CNT": {"folder": "Connectome", "cohort_folder": "{CODE}{NN}"}})
    nas = tmp_path / "nas"
    cohort = nas / "Analyzed" / "Connectome" / "CNT01"
    cohort.mkdir(parents=True)          # the tree exists even with no manifests in it
    for stem in stems_with_manifests:
        _manifest(cohort, stem)
    (nas / "pipeline_versions.json").write_text(json.dumps({
        "versions": {"dlc_scorer": SCORER, "segmenter": "2.0"}}), encoding="utf-8")

    class DB:
        def get_videos_in_state(self, state):
            return [{"video_id": s} for s in all_stems] if state == "archived" else []

        def _get_connection(self):
            raise RuntimeError("no db in this test")

    sc = ReprocessingScanner(DB(), nas)
    return sc


def _stems(n, start=1):
    return ["20260101_CNT01%02d_P1" % i for i in range(start, start + n)]


def test_videos_with_no_manifest_do_not_each_cost_a_walk(tmp_path, monkeypatch, counted_walks):
    """20 videos, none with a manifest: the old code walked the tree 20 times."""
    missing = _stems(20)
    sc = _scanner(tmp_path, [], missing, monkeypatch)
    summary = sc.scan(mark_outdated=False, adopt_orphans=False)
    assert summary["no_manifest"] == 20
    assert len(counted_walks) <= 1, counted_walks


def test_the_walk_count_does_not_grow_with_the_number_missed(tmp_path, monkeypatch, counted_walks):
    sc = _scanner(tmp_path, [], _stems(60), monkeypatch)
    sc.scan(mark_outdated=False, adopt_orphans=False)
    few = len(counted_walks)
    counted_walks.clear()
    sc2 = _scanner(tmp_path / "bigger", [], _stems(240), monkeypatch)
    sc2.scan(mark_outdated=False, adopt_orphans=False)
    assert len(counted_walks) == few, (few, len(counted_walks))


def test_a_manifest_outside_the_filed_folder_is_still_found(tmp_path, monkeypatch):
    """The fallback must still FIND things, not just be cheap: a manifest filed
    somewhere the index does not cover is found by the one deep walk."""
    stem = "20260101_CNT0199_P1"
    sc = _scanner(tmp_path, [], [stem], monkeypatch)
    _manifest(sc.archive_dir / "Connectome" / "SomewhereElse" / "deeper", stem)
    found = sc._load_manifest(stem)
    assert found["video_stem"] == stem
    # and the scan does not report it missing (whether it then judges it
    # current depends on the declaration, which is not what this test is about)
    summary = sc.scan(mark_outdated=False, adopt_orphans=False)
    assert summary["no_manifest"] == 0


def test_a_superseded_manifest_is_still_never_read(tmp_path, monkeypatch):
    """The exclusion the old walk had must survive: a copy under Archive/ keeps
    its name, and reading it would compare an older generation as live."""
    stem = "20260101_CNT0150_P1"
    sc = _scanner(tmp_path, [], [stem], monkeypatch)
    _manifest(sc.archive_dir / "Archive" / "old", stem, seg="1.0")
    assert sc._load_manifest(stem) is None
    assert sc.scan(mark_outdated=False, adopt_orphans=False)["no_manifest"] == 1


def test_the_deep_index_is_rebuilt_each_scan(tmp_path, monkeypatch):
    """The tree changes between passes, so the index must not be kept."""
    stem = "20260101_CNT0177_P1"
    sc = _scanner(tmp_path, [], [stem], monkeypatch)
    assert sc.scan(mark_outdated=False, adopt_orphans=False)["no_manifest"] == 1
    _manifest(sc.archive_dir / "Connectome" / "Elsewhere", stem)
    assert sc.scan(mark_outdated=False, adopt_orphans=False)["no_manifest"] == 0
