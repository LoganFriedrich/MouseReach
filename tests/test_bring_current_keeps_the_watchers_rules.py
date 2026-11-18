"""Bring-current (pipeline/reprocess_to_current.py) keeps the watcher's rules.

WHY: the routine files into Analyzed and holds videos in the review queues,
but it did so with none of the watcher's rules: it built queue bundles
itself (a triage bundle beside a deep-review one, no watcher record), let a
quality check that could not run count as a pass, never compared with the
hand scores, finalized a crystallized video or one the watcher was working
on, finalized a clean video with no kinematics, and pushed to a database.

The algorithm steps are stood in (they are tested elsewhere); what is tested
is what the routine does with their results. Video ids are invented.
"""
import json
from types import SimpleNamespace

import pytest

V = "20990101_CNT9901_P1"


@pytest.fixture
def world(tmp_path, monkeypatch):
    from mousereach.config import Paths
    queues = {"triage": tmp_path / "Review" / "Pending",
              "deep": tmp_path / "Review" / "Deep_Review"}
    for q in queues.values():
        q.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", queues["triage"], raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", queues["deep"], raising=False)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)

    filed = tmp_path / "Analyzed" / "PRJ" / "C01"
    filed.mkdir(parents=True)
    mp4 = filed / f"{V}.mp4"
    mp4.write_bytes(b"video")
    (filed / f"{V}_features.json").write_text('{"old": true}', encoding="utf-8")
    pose = tmp_path / "pose" / f"{V}DLC_resnet101_X_shuffle3_100000.h5"
    pose.parent.mkdir()
    pose.write_bytes(b"pose")

    def out(d, name, doc):
        (d / f"{V}_{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    import mousereach.segmentation.core.batch as sb
    import mousereach.reach.core.batch as rb
    import mousereach.outcomes.core.batch as ob
    import mousereach.assignment.run as ar
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri
    import mousereach.watcher.review_gate as rg
    import mousereach.kinematics.core.feature_extractor as fe
    monkeypatch.setattr(sb, "process_single",
                        lambda h5, output_dir=None: out(output_dir, "segments", {"boundaries": [1]}))
    monkeypatch.setattr(rb, "process_single",
                        lambda h5, seg, output_dir=None: out(output_dir, "reaches", {"segments": []}))
    monkeypatch.setattr(ob, "process_single",
                        lambda h5, seg, reach, output_dir=None, video_dir=None:
                        out(output_dir, "pellet_outcomes", {"segments": []}))
    monkeypatch.setattr(ar, "assign_reaches_for_video", lambda d, vid, h5: {})
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})

    state = SimpleNamespace(triage_raises=False, gate="clean", hand="agrees",
                            row={"state": "archived"}, pushed=[])

    class FakeTriage:
        verdict = "auto_approved"

        def save(self, path):
            path.write_text("{}", encoding="utf-8")

    def fake_triage(**k):
        if state.triage_raises:
            raise RuntimeError("the quality check broke")
        return FakeTriage()

    monkeypatch.setattr(tri, "triage_video", fake_triage)
    monkeypatch.setattr(rg, "evaluate_gate", lambda vid, d, qc_verdict=None:
                        (state.gate, "test reason", {}))

    class FakeFeatures:
        n_segments = 0

        def to_dict(self):
            return {"new": True}

    class FakeExtractor:
        def extract(self, *a, **k):
            return FakeFeatures()

    monkeypatch.setattr(fe, "FeatureExtractor", FakeExtractor)
    import mousereach.kinematics.analysis.reach_export as rx
    monkeypatch.setattr(rx, "write_video_results_csv", lambda p: None)

    from mousereach.watcher import hand_scores
    monkeypatch.setattr(hand_scores, "check", lambda vid, d: hand_scores.Verdict(
        {"agrees": hand_scores.AGREES, "waiting": hand_scores.WAITING}[state.hand], "test"))
    monkeypatch.setattr(hand_scores, "record", lambda *a, **k: True)

    from mousereach.watcher import db_location
    fake_db = SimpleNamespace(get_video=lambda vid: dict(state.row),
                              update_state=lambda *a, **k: None,
                              set_fields=lambda *a, **k: None,
                              log_step=lambda *a, **k: None)
    monkeypatch.setattr(db_location, "open_watcher_db_if_present", lambda: (fake_db, None))

    import mousereach.sync.database as sdb
    monkeypatch.setattr(sdb, "sync_file_to_database",
                        lambda p, *a, **k: state.pushed.append(p) or True)

    def run():
        from mousereach.pipeline import reprocess_to_current as rtc
        return rtc.reprocess_video_to_current(
            V, pose, video_file=mp4, work_dir=tmp_path / "work",
            finalize=True, skip_if_current=False, archive_root=tmp_path / "Archive")

    return SimpleNamespace(run=run, state=state, filed=filed, queues=queues)


def _filed_untouched(w):
    return json.loads((w.filed / f"{V}_features.json").read_text()) == {"old": True}


def _queued(w):
    return [p.name for q in w.queues.values() for p in q.iterdir()]


def test_a_quality_check_that_cannot_run_finalizes_nothing(world):
    world.state.triage_raises = True
    out = world.run()
    assert "quality check could not run" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_session_without_hand_scores_is_held_not_filed(world):
    world.state.hand = "waiting"
    out = world.run()
    assert "no hand scores" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_crystallized_video_is_not_finalized(world):
    world.state.row = {"state": "crystallized", "crystallized_label": "PAPER_X"}
    out = world.run()
    assert "not finalized" in (out.get("error") or ""), out
    assert _filed_untouched(world) and _queued(world) == []


def test_a_video_the_watcher_is_working_on_is_not_finalized(world):
    world.state.row = {"state": "processing"}
    out = world.run()
    assert "working on it" in (out.get("error") or ""), out
    assert _filed_untouched(world)


def test_a_clean_video_is_filed_and_nothing_is_pushed_to_a_database(world):
    out = world.run()
    assert out.get("error") is None, out
    assert json.loads((world.filed / f"{V}_features.json").read_text()) == {"new": True}
    assert world.state.pushed == []


def test_a_held_video_never_gets_a_triage_copy_beside_its_deep_review_bundle(world):
    """Deep review beats triage: the routine goes through the watcher's router."""
    deep = world.queues["deep"] / V
    deep.mkdir()
    (deep / f"{V}_segments.json").write_text("{}", encoding="utf-8")
    world.state.gate = "triage"
    out = world.run()
    assert not (world.queues["triage"] / V).exists(), out
    assert _filed_untouched(world)
