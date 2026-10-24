"""The re-run scan and recording retirement must answer two questions the same
way, or a video the scan will never re-run is never "complete" for retirement
and its recording sits in the intake for good (196 did, until 2026-10-24):

  * has this video's human review been applied?
      watcher/reprocessor.py  ReprocessingScanner._pending_review_path
      video_prep/core/collage_provenance.py  review_applied
  * is a person's segmentation exempt from the segmenter version?
      ReprocessingScanner._drop_human_seg_staleness
      collage_provenance._drop_human_seg_staleness

Each rule lives in both places (the video_prep package does not import the
watcher). This test runs both on the same cases: change one without the other
and it fails.
"""
import json
import os

import pytest

import mousereach.review.causal_review_io as crio
import mousereach.video_prep.core.collage_provenance as cp
from mousereach.pipeline.fsutil import sha256_file
from mousereach.watcher.reprocessor import ReprocessingScanner

STEM = "19990101_TEST0001_P1"


class FakeDB:
    def get_videos_in_state(self, state):
        return []


@pytest.fixture
def scanner(tmp_path):
    (tmp_path / "Analyzed").mkdir()
    return ReprocessingScanner(FakeDB(), tmp_path)


def _review(tmp_path, monkeypatch, reviewed_at, mtime):
    p = tmp_path / f"{STEM}_causal_review.json"
    body = {"segments": []}
    if reviewed_at:
        body["reviewed_at"] = reviewed_at
    p.write_text(json.dumps(body), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    monkeypatch.setattr(crio, "resolve_review_path", lambda vid, primary_dir=None: p)
    return p


CASES = [
    # (stamp, reviewed_at, created_at, review_mtime, features_mtime)
    ("match", "2026-10-01T14:51:31", "2026-10-01T15:01:30", 2_000, 1_000),
    ("other", "2026-10-01T14:51:31", "2026-10-01T15:01:30", 500, 1_000),
    (None, "2026-10-01T14:51:31", "2026-10-01T15:01:30", 2_000, 1_000),
    (None, "2026-10-01T15:30:00", "2026-10-01T15:01:30", 500, 1_000),
    (None, None, "2026-10-01T15:01:30", 500, 1_000),
    (None, None, "2026-10-01T15:01:30", 2_000, 1_000),
    (None, None, None, 500, 1_000),
]


@pytest.mark.parametrize("stamp,reviewed_at,created_at,r_mtime,f_mtime", CASES)
def test_both_say_the_same_about_a_review(scanner, tmp_path, monkeypatch, stamp,
                                          reviewed_at, created_at, r_mtime, f_mtime):
    p = _review(tmp_path, monkeypatch, reviewed_at, r_mtime)
    manifest = {}
    if created_at:
        manifest["created_at"] = created_at
    if stamp:
        manifest["applied_review"] = {"sha256": sha256_file(p) if stamp == "match" else "0" * 64}
    rerun_says_applied = scanner._pending_review_path(STEM, manifest, {STEM: f_mtime}) is None
    retirement_says_applied = cp.review_applied(manifest, p, review_mtime=r_mtime,
                                                features_mtime=f_mtime)
    assert rerun_says_applied == retirement_says_applied


@pytest.mark.parametrize("source", ["human", "algo"])
@pytest.mark.parametrize("stale", [["segmenter"], ["segmenter", "outcome_detector"]])
@pytest.mark.parametrize("full", [False, True])
def test_both_say_the_same_about_a_persons_segmentation(scanner, tmp_path, source, stale, full):
    folder = tmp_path / "Analyzed" / "P" / "C"
    folder.mkdir(parents=True)
    manifest_path = folder / f"{STEM}_processing_manifest.json"
    manifest_path.write_text("{}")
    (folder / f"{STEM}_segments.json").write_text(json.dumps({"boundary_source": source}))
    cmp = {"is_current": False, "stale_components": list(stale), "needs_full_reprocess": full}
    a = scanner._drop_human_seg_staleness(STEM, {STEM: manifest_path}, dict(cmp))
    b = cp._drop_human_seg_staleness(STEM, folder, dict(cmp))
    assert (a["is_current"], a["stale_components"]) == (b["is_current"], b["stale_components"])
