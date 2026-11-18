"""Bulk re-run paths never replace a person's cuts with the algorithm's.

WHY: a person's cuts (a ``<video>_segments.json`` marked
``boundary_source: "human"`` by the segmentation fixer) are frame numbers,
valid whatever pose or algorithm version comes next, and the segmenter keeps
such a file when it finds one. Three bulk paths lost them anyway:

  * the versioned supersede move archived the person's file as an "algorithm
    output" -- used by the re-pose command, by bring-current and by filing --
    after which a re-run cut the video again from scratch;
  * bring-current segmented in an empty work folder, so the segmenter never
    saw the person's file, and finalize then filed the algorithm's cuts;
  * the review-bundle stager recognised only the old "manual" markers, so a
    restage ran the segmenter over the fixer's cuts.

Video ids are invented.
"""
import json

import pytest

V = "20990101_CNT9901_P1"
HUMAN = {"boundaries": [100, 200, 300], "boundary_source": "human",
         "video_name": V}
ALGO = {"boundaries": [110, 210, 310], "segmenter_version": "9.9.9",
        "video_name": V}


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _doc(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_supersede_keeps_a_persons_cuts_unless_a_person_recut(tmp_path):
    from mousereach.archive.supersede import supersede_video_outputs
    folder, archive = tmp_path / "filed", tmp_path / "Archive"
    seg = _write(folder / f"{V}_segments.json", HUMAN)
    _write(folder / f"{V}_reaches.json", {"segments": []})

    # No replacement offered (the re-pose command's sweep): the cuts stay.
    out = supersede_video_outputs(V, folder, archive)
    assert seg.is_file() and _doc(seg) == HUMAN
    assert out["human_cuts_kept"] == [f"{V}_segments.json"]
    assert not (folder / f"{V}_reaches.json").exists(), "algorithm outputs still go"

    # An algorithm segments file about to replace them: still kept.
    algo_new = _write(tmp_path / "new" / f"{V}_segments.json", ALGO)
    out = supersede_video_outputs(V, folder, archive,
                                  incoming={algo_new.name: algo_new})
    assert seg.is_file() and out["human_cuts_kept"]

    # A person's newer cuts replacing them: the old ones are archived.
    recut = _write(tmp_path / "recut" / f"{V}_segments.json",
                   dict(HUMAN, boundaries=[101, 201, 301]))
    out = supersede_video_outputs(V, folder, archive, incoming={recut.name: recut})
    assert not seg.exists() and out["human_cuts_kept"] == []


def test_filing_refuses_to_put_algorithm_cuts_over_a_persons(tmp_path, monkeypatch):
    import mousereach.archive.core as core
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)   # attention list
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "PRJ" / "C01"
    _write(dest / f"{V}_segments.json", HUMAN)
    _write(proc / f"{V}_segments.json", ALGO)
    (proc / f"{V}.mp4").parent.mkdir(parents=True, exist_ok=True)
    (proc / f"{V}.mp4").write_bytes(b"video")
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)

    r = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                           source_dir=proc)
    assert not r["success"]
    assert "person's" in (r["error"] or "")
    assert _doc(dest / f"{V}_segments.json") == HUMAN


def test_bring_current_segments_with_the_persons_cuts_in_the_work_folder(tmp_path, monkeypatch):
    """The work folder gets the person's cuts before the segmenter runs, and
    the real segmenter keeps them (the next step is stopped on purpose)."""
    from mousereach.pipeline import reprocess_to_current as rtc
    filed = tmp_path / "Analyzed" / "PRJ" / "C01"
    mp4 = filed / f"{V}.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"video")
    _write(filed / f"{V}_segments.json", HUMAN)
    pose = tmp_path / "pose" / f"{V}DLC_resnet101_X_shuffle3_100000.h5"
    pose.parent.mkdir()
    pose.write_bytes(b"not a real pose")

    import mousereach.reach.core.batch as reach_batch

    def stop(*a, **k):
        raise RuntimeError("stopped by the test after segmentation")

    monkeypatch.setattr(reach_batch, "process_single", stop)
    work = tmp_path / "work"
    out = rtc.reprocess_video_to_current(V, pose, video_file=mp4, work_dir=work,
                                         finalize=False, skip_if_current=False)
    assert "stopped by the test" in (out.get("error") or ""), out
    assert _doc(work / f"{V}_segments.json") == HUMAN


def test_restaging_a_bundle_keeps_the_fixers_cuts(tmp_path, monkeypatch):
    from mousereach.review import staging
    root = tmp_path / "Analyzed"
    mp4 = root / "CNT99" / f"{V}.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"video")
    pose = root / staging.DLC4_SUBDIR / "CNT99" / f"{V}{staging.DLC4_SCORER}.h5"
    pose.parent.mkdir(parents=True)
    pose.write_bytes(b"not a real pose")
    pending = tmp_path / "Pending"
    seg = _write(pending / V / f"{V}_segments.json", HUMAN)

    monkeypatch.setattr(staging, "_neutralize_pipeline_side_effects", lambda: None)
    monkeypatch.setattr(staging, "_restore_review_into_bundle", lambda *a, **k: None)
    import mousereach.segmentation.core.segmenter_multi as sm
    import mousereach.segmentation.core.segmenter_robust as sr
    monkeypatch.setattr(sm, "segment_video_multi", lambda h5: (ALGO["boundaries"], None))
    monkeypatch.setattr(sr, "save_segmentation",
                        lambda b, d, path: _write(path, ALGO))
    import mousereach.reach.core.span_to_reaches as s2r

    def stop(*a, **k):
        raise RuntimeError("stopped by the test after segmentation")

    monkeypatch.setattr(s2r, "detect_video_reaches", stop)
    with pytest.raises(RuntimeError, match="stopped by the test"):
        staging.stage_video(V, connectome_root=root, pending_dir=pending,
                            overwrite=True, verbose=False)
    assert _doc(seg) == HUMAN


def test_repose_refuses_a_held_bundle_with_a_persons_cuts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mousereach.watcher import repose_videos as rv
    bundle = tmp_path / "Deep_Review" / V
    _write(bundle / f"{V}_segments.json", HUMAN)
    (bundle / f"{V}DLC_resnet101_X_shuffle3_100000.h5").write_bytes(b"pose")
    qdir = tmp_path / "Repose_Queue"
    qdir.mkdir()
    monkeypatch.setattr(rv.repose, "repose_queue_dir", lambda: qdir)
    monkeypatch.setattr(rv, "_incoming", lambda vid: False)
    monkeypatch.setattr(rv, "_bundle", lambda vid: bundle)
    monkeypatch.setattr(rv, "_durable_review", lambda vid: None)
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)   # attention list
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    db = SimpleNamespace(get_video=lambda vid: {"state": "deep_review"})
    rep = rv.plan_and_move(db, V, label="L", reason="r", hostname="h",
                           declared="DLC_resnet101_X_shuffle3_100000", dry_run=False,
                           results_dir_of=lambda vid: None)
    assert rep["status"] == "refused", rep
    assert f"{V}_segments.json" in rep["why"]
    assert _doc(bundle / f"{V}_segments.json") == HUMAN
