"""The review-bundle stager puts a bundle into a queue only through the router.

WHY: ``review/staging.stage_video`` wrote ``<queue>/<stem>`` itself. The
watcher database never learned the video was held (its row stayed 'archived'
or 'crystallized'), a crystallized video got a bundle, and ``overwrite=True``
regenerated a live bundle in place, under the reviewer. It now builds the
bundle aside and hands it to the watcher's router (review_gate.route_to_queue),
which records the hold, refuses a crystallized video and never merges into a
live bundle. The algorithm steps are stood in. Video ids are invented.
"""
import json

import pytest

V = "20200101_CNT9901_P1"


@pytest.fixture
def stager(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.review import staging
    from mousereach.watcher import db_location
    from mousereach.watcher.db import WatcherDB

    root = tmp_path / "Analyzed"
    filed = root / "CNT99"
    filed.mkdir(parents=True)
    (filed / f"{V}.mp4").write_bytes(b"video")
    pose = root / staging.DLC4_SUBDIR / "CNT99" / f"{V}{staging.DLC4_SCORER}.h5"
    pose.parent.mkdir(parents=True)
    pose.write_bytes(b"pose")
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    for k, v in {"ANALYZED_OUTPUT": root, "TRIAGE_REVIEW": triage, "DEEP_REVIEW": deep,
                 "PROCESSING": tmp_path / "Processing", "NAS_ROOT": None}.items():
        monkeypatch.setattr(Paths, k, v, raising=False)
    db_path = tmp_path / "watcher.db"
    monkeypatch.setattr(db_location, "resolve_watcher_db_path", lambda: db_path)
    db = WatcherDB(db_path)
    db.register_video(video_id=V, source_path=str(filed / f"{V}.mp4"))
    db.force_state(V, "archived", reason="test fixture")

    # The algorithm steps, stood in: each writes its output.
    monkeypatch.setattr(staging, "_neutralize_pipeline_side_effects", lambda: None)
    monkeypatch.setattr(staging, "_restore_review_into_bundle", lambda *a, **k: None)
    import mousereach.segmentation.core.segmenter_multi as sm
    import mousereach.segmentation.core.segmenter_robust as sr
    import mousereach.reach.core.span_to_reaches as s2r
    import mousereach.reach.core.reach_detector as rd
    import mousereach.outcomes.v6_cascade as v6
    import mousereach.reach.v8.features as feats
    import mousereach.assignment.cli as acli
    import mousereach.assignment.v2 as av2
    monkeypatch.setattr(sm, "segment_video_multi", lambda h5: ([0, 100, 200], {}))
    monkeypatch.setattr(sr, "save_segmentation", lambda b, d, path: path.write_text(
        json.dumps({"boundaries": b, "video_name": V})))
    monkeypatch.setattr(s2r, "detect_video_reaches", lambda h5, seg: {"segments": []})
    monkeypatch.setattr(rd.ReachDetector, "save_results",
                        staticmethod(lambda res, path, **k: path.write_text(
                            json.dumps({"new": True}))))
    monkeypatch.setattr(v6, "detect_outcomes_v6_cascade", lambda **k: {"segments": [], "new": True})
    monkeypatch.setattr(feats, "load_dlc_h5", lambda h5: [0] * 300)
    monkeypatch.setattr(acli, "_segments_with_outcomes", lambda seg, out: [])
    monkeypatch.setattr(acli, "_reaches_list", lambda doc: [])
    monkeypatch.setattr(av2, "assign_reaches_v2", lambda **k: {"version": "x", "new": True})
    return {"staging": staging, "root": root, "triage": triage, "deep": deep,
            "db": lambda: WatcherDB(db_path)}


def test_a_staged_bundle_is_recorded_in_the_watcher_database(stager):
    s = stager
    bundle = s["staging"].stage_video(V, connectome_root=s["root"],
                                      pending_dir=s["triage"], verbose=False)
    assert (s["triage"] / V / f"{V}_reaches.json").is_file()
    assert s["db"]().get_video(V)["state"] == "triage"
    assert bundle == s["triage"] / V


def test_a_crystallized_video_gets_no_bundle(stager):
    s = stager
    s["db"]().force_state(V, "crystallized", reason="test fixture",
                          crystallized_label="L")
    with pytest.raises(Exception):
        s["staging"].stage_video(V, connectome_root=s["root"],
                                 pending_dir=s["triage"], verbose=False)
    assert not (s["triage"] / V).exists()
    assert s["db"]().get_video(V)["state"] == "crystallized"


def test_a_video_the_watcher_is_working_on_gets_no_bundle(stager):
    s = stager
    s["db"]().force_state(V, "processing", reason="test fixture")
    with pytest.raises(Exception):
        s["staging"].stage_video(V, connectome_root=s["root"],
                                 pending_dir=s["triage"], verbose=False)
    assert not (s["triage"] / V).exists()


def test_a_live_bundle_is_never_regenerated_in_place(stager):
    s = stager
    live = s["triage"] / V
    live.mkdir()
    reviewer = {"kept": True}
    (live / f"{V}_reaches.json").write_text(json.dumps(reviewer))
    (live / f"{V}_pellet_outcomes.json").write_text(json.dumps(reviewer))
    with pytest.raises(Exception):
        s["staging"].stage_video(V, connectome_root=s["root"],
                                 pending_dir=s["triage"], overwrite=True, verbose=False)
    assert json.loads((live / f"{V}_reaches.json").read_text()) == reviewer
    assert json.loads((live / f"{V}_pellet_outcomes.json").read_text()) == reviewer


def test_a_scratch_folder_is_written_directly(stager, tmp_path):
    """Not a queue: written as before, and the watcher record is left alone."""
    s = stager
    scratch = tmp_path / "scratch"
    s["staging"].stage_video(V, connectome_root=s["root"], pending_dir=scratch,
                             verbose=False)
    assert (scratch / V / f"{V}_reaches.json").is_file()
    assert s["db"]().get_video(V)["state"] == "archived"
