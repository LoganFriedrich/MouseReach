#!/usr/bin/env python3
"""A human review must survive anything the pipeline does to its surroundings.

A review is a fact about what a person saw in a stretch of video. It is not the
algorithms' output, nothing regenerates it, and re-review costs a human's time --
so it is the one artifact in this pipeline that must never depend on the
lifecycle of the directory it happens to be sitting in.

It used to. Reviews were written into the triage bundle the reviewer had open.
Bundles are transient: staging regenerates them, and returning a cleared bundle
MOVES its files onto one node's local disk and deletes the directory. A review
only reached storage that survives when its video was archived, so whether a
reviewer's work outlived a reprocess came down to timing.

Measured on 2026-08-24 across 1,686 reviewed videos:
    662  had a durable copy (their video had been archived)
    983  existed only on one machine's local disk
     41  existed only inside a Y: triage bundle -- one reprocess from gone
and the corpus index could not rebuild any of them, because it stored a file
path plus two summary fields and neither the frames the reviewer looked at nor
the reach they picked.

These tests pin the three properties that fix: the review is written somewhere
nothing regenerates, it is found from anywhere afterwards, and regenerating a
bundle carries it forward instead of dropping it.
"""

import json
import pytest
from pathlib import Path

from mousereach.review import causal_review_io as crio
from mousereach.review.causal_review_io import (
    build_segment_record,
    durable_review_dir,
    durable_review_path,
    resolve_review_path,
    save_causal_review,
    update_corpus_index,
)


@pytest.fixture
def nas(tmp_path, monkeypatch):
    """Point the durable store and corpus index at a temp tree."""
    from mousereach.config import Paths
    root = tmp_path / "nas"
    root.mkdir()
    monkeypatch.setattr(Paths, "NAS_ROOT", root, raising=False)
    return root


@pytest.fixture
def a_review():
    """One reviewed segment, carrying everything a re-review would have to redo."""
    return [build_segment_record(
        segment_num=7,
        pellet_num=7,
        algo_outcome="untouched",
        algo_causal_reach=None,
        algo_interaction_frame=None,
        human_outcome="displaced_sa",
        human_causal_reach={"start": 12345, "end": 12408},
        is_phantom=False,
        agreed=False,
        answers={"did_the_paw_touch_it": "yes"},
        notes="algo missed the touch",
        segment_span={"start": 12000, "end": 12600},
    )]


class TestDurableCopy:
    """The review is written where nothing regenerates it."""

    def test_save_writes_a_durable_copy_as_well_as_the_working_one(
            self, tmp_path, nas, a_review):
        bundle = tmp_path / "triage" / "20250624_CNT0104_P3"
        bundle.mkdir(parents=True)

        working = save_causal_review(
            "20250624_CNT0104_P3", bundle, a_review, provenance={}, reviewer="tester")

        assert working.is_file(), "the working copy is still written where the reviewer is"
        durable = durable_review_path("20250624_CNT0104_P3")
        assert durable.is_file(), "and a durable copy exists outside the bundle"
        assert durable.parent != bundle, "which is NOT inside the transient bundle"

    def test_the_durable_copy_holds_the_full_payload(self, tmp_path, nas, a_review):
        bundle = tmp_path / "triage" / "20250624_CNT0104_P3"
        bundle.mkdir(parents=True)
        save_causal_review("20250624_CNT0104_P3", bundle, a_review, provenance={})

        doc = json.loads(durable_review_path("20250624_CNT0104_P3").read_text())
        rec = doc["segments"][0]
        assert rec["segment_span"] == {"start": 12000, "end": 12600}, (
            "the frames the reviewer looked at")
        assert rec["human"]["causal_reach"] == {"start": 12345, "end": 12408}, (
            "and the reach they picked -- neither is re-derivable")
        assert rec["human"]["outcome"] == "displaced_sa"
        assert rec["answers"] == {"did_the_paw_touch_it": "yes"}

    def test_review_survives_the_bundle_being_wiped(self, tmp_path, nas, a_review):
        """The exact loss: the directory the review lived in is regenerated."""
        import shutil
        bundle = tmp_path / "triage" / "20250624_CNT0104_P3"
        bundle.mkdir(parents=True)
        save_causal_review("20250624_CNT0104_P3", bundle, a_review, provenance={})

        shutil.rmtree(bundle)                      # reprocess regenerates the bundle

        found = resolve_review_path("20250624_CNT0104_P3")
        assert found is not None, "the review is still findable with its bundle gone"
        assert json.loads(found.read_text())["segments"][0]["human"]["outcome"] == "displaced_sa"

    def test_a_bundle_only_review_is_unrecoverable_once_the_bundle_goes(
            self, tmp_path, nas, a_review):
        """What the old behaviour cost, pinned so it cannot come back quietly.

        With the review written ONLY into the bundle -- which is what
        save_causal_review used to do -- wiping the bundle leaves nothing to
        find, and the corpus index cannot rebuild it. This is the 41-video case.
        """
        import shutil
        stem = "20250624_CNT0104_P3"
        bundle = tmp_path / "triage" / stem
        bundle.mkdir(parents=True)
        (bundle / f"{stem}_causal_review.json").write_text(json.dumps(
            {"type": "causal_review", "video_stem": stem, "segments": a_review}))

        shutil.rmtree(bundle)

        assert resolve_review_path(stem) is None
        assert not durable_review_path(stem).exists()

    def test_save_still_works_with_nothing_configured(self, tmp_path, monkeypatch, a_review):
        """No NAS, no processing root: keep the working copy rather than refuse."""
        from mousereach.config import Paths
        monkeypatch.setattr(Paths, "NAS_ROOT", None, raising=False)
        monkeypatch.setattr(Paths, "PROCESSING_ROOT", None, raising=False)
        out = tmp_path / "somewhere"
        out.mkdir()
        written = save_causal_review("20250624_CNT0104_P3", out, a_review, provenance={})
        assert written.is_file()
        assert durable_review_dir() is None


class TestBundleRegeneration:
    """Regenerating a bundle carries the review forward."""

    def test_restore_puts_the_review_back_into_a_fresh_bundle(
            self, tmp_path, nas, a_review):
        from mousereach.review.staging import _restore_review_into_bundle
        import shutil

        stem = "20250624_CNT0104_P3"
        bundle = tmp_path / "triage" / stem
        bundle.mkdir(parents=True)
        save_causal_review(stem, bundle, a_review, provenance={})
        shutil.rmtree(bundle)

        bundle.mkdir(parents=True)                 # stage_video recreates it
        _restore_review_into_bundle(stem, bundle, lambda m: None)

        restored = bundle / f"{stem}_causal_review.json"
        assert restored.is_file(), (
            "stage_video(overwrite=True) on a bundle whose review was wiped "
            "must bring the review back")
        assert json.loads(restored.read_text())["segments"][0]["segment_span"] == {
            "start": 12000, "end": 12600}

    def test_restore_never_clobbers_a_newer_review_in_the_bundle(
            self, tmp_path, nas, a_review):
        """A reviewer may be editing the bundle copy right now."""
        from mousereach.review.staging import _restore_review_into_bundle
        import os, time

        stem = "20250624_CNT0104_P3"
        bundle = tmp_path / "triage" / stem
        bundle.mkdir(parents=True)
        save_causal_review(stem, bundle, a_review, provenance={})

        newer = bundle / f"{stem}_causal_review.json"
        doc = json.loads(newer.read_text())
        doc["segments"][0]["human"]["outcome"] = "retrieved"     # an in-progress edit
        newer.write_text(json.dumps(doc))
        os.utime(newer, (time.time() + 10, time.time() + 10))

        _restore_review_into_bundle(stem, bundle, lambda m: None)

        assert json.loads(newer.read_text())["segments"][0]["human"]["outcome"] == "retrieved"

    def test_stage_video_carries_the_review_forward(self):
        """The carry-forward runs on every staging path, not just by hand."""
        import inspect
        from mousereach.review import staging
        src = inspect.getsource(staging.stage_video)
        assert "_restore_review_into_bundle" in src


class TestManualResegmentation:
    """A reviewer's own boundaries are human-authored work too."""

    def test_a_restage_keeps_the_reviewers_cuts(self, tmp_path):
        """Without this, a restage silently re-cuts the video with the algorithm
        and the result still looks like a valid segmentation."""
        import inspect
        from mousereach.review import staging
        src = inspect.getsource(staging.stage_video)
        assert 'manual_resegmentation' in src
        assert 'boundaries_override = list(' in src, (
            'the manual boundaries must be fed back in as an override')


class TestReturnToProcessing:
    """A review must not be lost at the moment its bundle is torn down."""

    def test_durable_copy_is_made_before_the_bundle_is_emptied(
            self, tmp_path, nas, a_review):
        from mousereach.watcher.review_return import _ensure_durable_review

        stem = "20250624_CNT0104_P3"
        bundle = tmp_path / "triage" / stem
        bundle.mkdir(parents=True)
        # a review that exists ONLY in the bundle -- the 41-video case
        (bundle / f"{stem}_causal_review.json").write_text(json.dumps(
            {"type": "causal_review", "video_stem": stem, "segments": a_review}))

        _ensure_durable_review(bundle, stem)

        assert durable_review_path(stem).is_file(), (
            "returning a cleared bundle moves its files to one node's local disk "
            "and deletes the directory; the durable copy must be made first")

    def test_no_review_in_the_bundle_is_not_an_error(self, tmp_path, nas):
        from mousereach.watcher.review_return import _ensure_durable_review
        bundle = tmp_path / "triage" / "20250624_CNT0104_P3"
        bundle.mkdir(parents=True)
        _ensure_durable_review(bundle, "20250624_CNT0104_P3")   # must not raise

    def test_return_calls_it(self):
        import inspect
        from mousereach.watcher import review_return
        assert "_ensure_durable_review" in inspect.getsource(
            review_return._return_to_processing)


class TestCorpusIndex:
    """A pointer to a file that no longer exists is not a record."""

    def test_index_entries_carry_what_a_re_review_would_have_to_redo(
            self, tmp_path, nas, a_review):
        stem = "20250624_CNT0104_P3"
        out = tmp_path / "b"
        out.mkdir()
        path = save_causal_review(stem, out, a_review, provenance={})
        update_corpus_index(stem, path, a_review, "tester", "2026-08-24T00:00:00")

        idx = json.loads((nas / "review_records" / "causal_review_index.json").read_text())
        entry = idx["entries"][f"{stem}__seg7"]

        assert entry["segment_span"] == {"start": 12000, "end": 12600}
        assert entry["human_causal_reach"] == {"start": 12345, "end": 12408}
        assert entry["answers"] == {"did_the_paw_touch_it": "yes"}
        assert entry["notes"] == "algo missed the touch"
        assert entry["human_outcome"] == "displaced_sa"
        assert idx["schema_version"] == "1.1"

    def test_index_records_where_the_durable_copy_is(self, tmp_path, nas, a_review):
        stem = "20250624_CNT0104_P3"
        out = tmp_path / "b"
        out.mkdir()
        path = save_causal_review(stem, out, a_review, provenance={})
        update_corpus_index(stem, path, a_review, "tester", "2026-08-24T00:00:00")
        idx = json.loads((nas / "review_records" / "causal_review_index.json").read_text())
        entry = idx["entries"][f"{stem}__seg7"]
        assert Path(entry["durable_file"]).is_file()


class TestKinematicsBridge:
    """The review must still be there when kinematics goes looking."""

    def test_kinematics_finds_the_review_after_the_bundle_is_gone(
            self, tmp_path, nas, a_review):
        import shutil
        stem = "20250624_CNT0104_P3"
        bundle = tmp_path / "triage" / stem
        bundle.mkdir(parents=True)
        save_causal_review(stem, bundle, a_review, provenance={})
        shutil.rmtree(bundle)

        # what the extractor does: resolve, with a processing dir that has nothing
        processing = tmp_path / "Processing"
        processing.mkdir()
        assert resolve_review_path(stem, primary_dir=processing) is not None
