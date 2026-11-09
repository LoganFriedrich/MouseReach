"""mousereach-repose-videos (watcher/repose_videos.py).

WHY: a pose made with the declared model by a machine whose poses are not
trusted carries the declared scorer, so the re-pose round trip treats it as
current. If a row were marked for a new pose while such a pose still sat in
the posed folder, the results folder or a review bundle, the server would
adopt THAT pose as "the new pose" and re-run on it with no GPU involved.
These tests pin the order (every copy of the pose moved out BEFORE the row is
marked), that a held video's request names its parked video file (a GPU node
may not read the posed folder, and the review queues are not searched), and
that nothing of a person's is touched.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.repose as repose
import mousereach.watcher.repose_videos as rv
from mousereach.watcher.db import WatcherDB

SC = "DLC_resnet101_MPSAOct27shuffle3_100000"
FILED = "20240101_ABC0101_P1"
HELD = "20240102_ABC0102_P2"
DEEP = "20240103_ABC0103_P3"
LABEL = "untrusted pose test"
ALGO = ("_segments.json", "_reaches.json", "_pellet_outcomes.json",
        "_reach_assignments.json", "_triage.json", "_features.json")


def _pose(folder: Path, vid: str, content=b"cpu-pose"):
    for ext in (".h5", ".csv", "_meta.pickle"):
        (folder / f"{vid}{SC}{ext}").write_bytes(content + ext.encode())


def _manifest(folder: Path, vid: str):
    (folder / f"{vid}_processing_manifest.json").write_text(json.dumps(
        {"dlc_model": {"dlc_scorer": SC},
         "pipeline_versions": {"segmenter": "2.2.4"}}), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    queue = nas / "Processing" / "Repose_Queue"
    posed = nas / "Processing" / "Posed"
    triage = nas / "Processing" / "Review" / "triage"
    deep = nas / "Processing" / "Review" / "deep_review"
    results = nas / "Analyzed" / "X" / "C01"
    durable = nas / "review_records" / "reviews"
    for d in (queue, posed, triage, deep, results, durable):
        d.mkdir(parents=True)
    for name, val in {"NAS_ROOT": nas, "REPOSE_QUEUE": queue, "DLC_STAGING": posed,
                      "ANALYZED_OUTPUT": nas / "Analyzed", "TRIAGE_REVIEW": triage,
                      "DEEP_REVIEW": deep}.items():
        monkeypatch.setattr(repose.Paths, name, val, raising=False)
    monkeypatch.setattr(rv, "_durable_review", lambda vid: (
        durable / f"{vid}_causal_review.json"
        if (durable / f"{vid}_causal_review.json").is_file() else None))
    # Every video files into the same results folder here; held ones have none.
    monkeypatch.setattr(rv, "_results_dir", lambda vid: results)
    monkeypatch.setattr(repose, "archive_folder", lambda vid: results)

    # FILED: results folder with mp4, pose, outputs and a human review
    (results / f"{FILED}.mp4").write_bytes(b"filed-video")
    _pose(results, FILED)
    _manifest(results, FILED)
    for s in ALGO:
        (results / f"{FILED}{s}").write_text("{}", encoding="utf-8")
    (results / f"{FILED}_causal_review.json").write_text('{"reviewer": "r"}', encoding="utf-8")
    # a neighbour whose id starts with FILED's -- must never be touched
    nb = FILED + "0"
    (results / f"{nb}_reaches.json").write_text("{}", encoding="utf-8")
    _pose(results, nb)

    # HELD in triage, with the server's posed copy
    b = triage / HELD
    b.mkdir()
    (b / f"{HELD}.mp4").write_bytes(b"held-video")
    _pose(b, HELD)
    _manifest(b, HELD)
    for s in ALGO[:-1]:
        (b / f"{HELD}{s}").write_text("{}", encoding="utf-8")
    (b / f"{HELD}_manifest.json").write_text("{}", encoding="utf-8")
    (b / f"{HELD}_routing.json").write_text("{}", encoding="utf-8")
    (posed / f"{HELD}.mp4").write_bytes(b"held-video")
    _pose(posed, HELD)
    (posed / ".claims").mkdir()

    # HELD in deep review, no posed copy
    d = deep / DEEP
    d.mkdir()
    (d / f"{DEEP}.mp4").write_bytes(b"deep-video")
    _pose(d, DEEP)
    _manifest(d, DEEP)

    db = WatcherDB(db_path=tmp_path / "w.db")
    for vid, state in ((FILED, "archived"), (HELD, "triage"), (DEEP, "deep_review"),
                       (nb, "archived")):
        db.register_video(video_id=vid, source_path=f"C:/x/{vid}.mp4",
                          current_path=f"C:/x/{vid}.mp4")
        db.force_state(vid, state, reason="test setup")
    return SimpleNamespace(nas=nas, queue=queue, posed=posed, triage=triage, deep=deep,
                           results=results, durable=durable, db=db, nb=nb,
                           archive=nas / "Analyzed" / "Archive" / LABEL)


def _run(env, vid, **kw):
    kw.setdefault("label", LABEL)
    kw.setdefault("reason", "pose not trusted: test")
    kw.setdefault("hostname", "srv")
    kw.setdefault("declared", SC)
    kw.setdefault("dry_run", False)
    return rv.plan_and_move(env.db, vid, **kw)


def _poses_anywhere(env, vid):
    out = []
    for d in (env.results, env.posed, env.triage / vid, env.deep / vid):
        if d.is_dir():
            out += [p for p in d.iterdir() if p.name.startswith(vid + "DLC")]
    return out


# ---------------------------------------------------------------- ordering

def test_every_pose_copy_is_gone_before_the_row_is_marked(env, monkeypatch):
    seen = {}
    real = env.db.update_state

    def spy(vid, state, **kw):
        seen[vid] = _poses_anywhere(env, vid)
        assert not (env.queue / f"{vid}.json").exists(), "request before mark"
        return real(vid, state, **kw)
    monkeypatch.setattr(env.db, "update_state", spy)
    for vid in (FILED, HELD, DEEP):
        assert _run(env, vid)["status"] == "requested"
        assert seen[vid] == [], f"{vid}: pose still present when marked: {seen[vid]}"


def test_a_failed_move_leaves_the_row_unmarked_and_no_request(env, monkeypatch):
    import mousereach.archive.supersede as sup

    def failing(video_id, source_dir, *a, **kw):
        if Path(source_dir) == env.posed:
            return {"pose": [], "algo": [], "failed": [f"{video_id}{SC}.h5"]}
        return sup_real(video_id, source_dir, *a, **kw)
    sup_real = sup.supersede_video_outputs
    monkeypatch.setattr(sup, "supersede_video_outputs", failing)
    r = _run(env, HELD)
    assert r["status"] == "incomplete"
    assert env.db.get_video(HELD)["state"] == "triage"
    assert not (env.queue / f"{HELD}.json").exists()


# ---------------------------------------------------------------- filed

def test_filed_video_keeps_its_video_and_review_and_loses_the_pose(env):
    r = _run(env, FILED)
    assert r["status"] == "requested"
    left = sorted(p.name for p in env.results.iterdir() if p.name.startswith(FILED + "_")
                  or p.name.startswith(FILED + ".") or p.name.startswith(FILED + "DLC"))
    assert left == [f"{FILED}.mp4", f"{FILED}_causal_review.json"]
    gen = env.archive / "DLC Model 4.0"
    assert (gen / f"{FILED}{SC}.h5").is_file()
    assert any(p.name == f"{FILED}_reaches.json" for p in gen.rglob("*"))
    row = env.db.get_video(FILED)
    assert (row["state"], row["reprocess_scope"]) == ("outdated", "full")
    assert row["mark_reason"] == "pose not trusted: test"
    body = json.loads((env.queue / f"{FILED}.json").read_text())
    assert body["video_rel"] == f"Analyzed/X/C01/{FILED}.mp4"


def test_a_neighbour_with_a_longer_id_is_never_touched(env):
    _run(env, FILED)
    assert (env.results / f"{env.nb}_reaches.json").is_file()
    assert (env.results / f"{env.nb}{SC}.h5").is_file()
    assert env.db.get_video(env.nb)["state"] == "archived"


# ---------------------------------------------------------------- held

def test_held_video_request_names_its_parked_video_file(env):
    r = _run(env, HELD)
    assert r["status"] == "requested"
    parked = env.queue / rv.PARKED / HELD
    assert not (env.triage / HELD).exists()
    assert (parked / f"{HELD}.mp4").read_bytes() == b"held-video"
    assert (parked / f"{HELD}_manifest.json").is_file()
    assert (parked / f"{HELD}_routing.json").is_file()
    assert (parked / "posed_copy" / f"{HELD}.mp4").is_file()
    assert not any(p.name.startswith(HELD) for p in env.posed.iterdir())
    body = json.loads((env.queue / f"{HELD}.json").read_text())
    assert body["video_rel"] == f"Processing/Repose_Queue/{rv.PARKED}/{HELD}/{HELD}.mp4"
    # ... and a GPU node can resolve it (the posed folder would be refused)
    assert repose.resolve_request_video(HELD, body) == parked / f"{HELD}.mp4"


def test_deep_review_video_is_handled_like_triage(env):
    r = _run(env, DEEP)
    assert r["status"] == "requested" and r["where"] == "deep_review"
    assert (env.queue / rv.PARKED / DEEP / f"{DEEP}.mp4").is_file()
    assert env.db.get_video(DEEP)["reprocess_scope"] == "full"


def test_held_video_with_a_saved_review_is_refused_untouched(env):
    (env.triage / HELD / f"{HELD}_causal_review.json").write_text("{}", encoding="utf-8")
    r = _run(env, HELD)
    assert r["status"] == "refused"
    assert (env.triage / HELD / f"{HELD}{SC}.h5").is_file()
    assert (env.posed / f"{HELD}{SC}.h5").is_file()
    assert env.db.get_video(HELD)["state"] == "triage"


def test_held_video_with_a_durable_review_is_refused(env):
    (env.durable / f"{DEEP}_causal_review.json").write_text("{}", encoding="utf-8")
    assert _run(env, DEEP)["status"] == "refused"
    assert (env.deep / DEEP / f"{DEEP}{SC}.h5").is_file()


def test_rows_in_other_states_and_open_requests_are_refused(env):
    env.db.force_state(FILED, "processing", reason="test")
    assert _run(env, FILED)["status"] == "refused"
    env.db.force_state(FILED, "archived", reason="test")
    (env.queue / f"{FILED}.json").write_text("{}", encoding="utf-8")
    assert _run(env, FILED)["status"] == "refused"
    assert (env.results / f"{FILED}{SC}.h5").is_file()


# ---------------------------------------------------------------- dry run

def test_dry_run_changes_nothing(env):
    before = sorted(str(p) for p in env.nas.rglob("*"))
    for vid in (FILED, HELD, DEEP):
        r = _run(env, vid, dry_run=True)
        assert r["status"] == "would_request"
        assert r["moved"]
    assert sorted(str(p) for p in env.nas.rglob("*")) == before
    assert env.db.get_video(HELD)["state"] == "triage"


# ---------------------------------------------------------------- round trip

def test_the_round_trip_then_waits_for_a_gpu_pose_and_adopts_it(env):
    for vid in (FILED, HELD):
        _run(env, vid)
    rows = [env.db.get_video(v) for v in (FILED, HELD)]
    out = repose.publish_pending(env.db, rows, hostname="srv", staging_dir=env.posed,
                                 repose_dir=env.queue, declared=SC)
    # nothing narrows on the old pose; the requests stay out
    assert out["narrowed"] == [] and out["withdrawn"] == []
    assert (env.queue / f"{FILED}.json").exists() and (env.queue / f"{HELD}.json").exists()
    assert repose.adopt_staged_reposes(env.db, env.posed, repose_dir=env.queue,
                                       declared=SC) == []
    # a GPU node stages the new pose -> adopted, request closed
    (env.posed / f"{HELD}.mp4").write_bytes(b"held-video")
    _pose(env.posed, HELD, content=b"gpu-pose")
    assert repose.adopt_staged_reposes(env.db, env.posed, repose_dir=env.queue,
                                       declared=SC) == [HELD]
    row = env.db.get_video(HELD)
    assert (row["state"], row["reprocess_scope"]) == ("outdated", "segmentation")
    assert row["dlc_output_path"].endswith(f"{HELD}{SC}.h5")
    assert not (env.queue / f"{HELD}.json").exists()
