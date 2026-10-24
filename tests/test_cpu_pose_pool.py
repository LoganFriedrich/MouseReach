"""The CPU pose pool takes, poses and hands on cut videos exactly as a GPU node
does -- without ever posing the same video twice or losing one.

Everything lives under tmp_path with config.Paths pointed at it; the pose
itself is replaced by fake child processes.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.cpu_pose_pool as pool_mod
import mousereach.watcher.orchestrator as orch
import mousereach.watcher.repose as repose
from mousereach.watcher import single_claim

HOLDER = "SRV-cpu"
POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"
VID = "20250101_CNT0101_P1"
VID2 = "20250101_CNT0102_P1"

LAYOUT = {
    "NAS_ROOT": "share",
    "SINGLE_ANIMAL_OUTPUT": "share/Unanalyzed/Single_Animal",
    "DLC_STAGING": "share/Processing/Posed",
    "TRIAGE_REVIEW": "share/Processing/Review/triage",
    "DEEP_REVIEW": "share/Processing/Review/deep_review",
    "ANALYZED_OUTPUT": "share/Analyzed",
    "PROCESSING_ROOT": "node",
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    dirs = {}
    for name, sub in LAYOUT.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    monkeypatch.setattr(repose, "declared_scorer", lambda: POSE)
    monkeypatch.setattr(pool_mod, "_done_or_held", lambda vid: None)
    work = tmp_path / "node" / "CPU_Pose"
    work.mkdir(parents=True)
    p = object.__new__(pool_mod.CpuPosePool)
    p.sc = single_claim
    p.workers, p.threads = 2, 3
    p.dlc_config = tmp_path / "config.yaml"
    p.holder = HOLDER
    p.front = dirs["single_animal_output"]
    p.posed = dirs["dlc_staging"]
    p.work = work
    p.active, p.backoff, p.done_times = {}, {}, []
    p.stopping = False
    p.coordinator = None
    return SimpleNamespace(tmp=tmp_path, mp=monkeypatch, pool=p,
                           door=dirs["single_animal_output"], posed=dirs["dlc_staging"],
                           work=work)


def _claimed(env, vid, data=b"video-bytes"):
    d = env.door / ".inflight" / HOLDER
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.mp4").write_bytes(data)
    return d / f"{vid}.mp4"


def _local_pose(env, vid, scorer=POSE, data=b"video-bytes"):
    (env.work / f"{vid}.mp4").write_bytes(data)
    (env.work / f"{vid}{scorer}.h5").write_bytes(b"pose")
    (env.work / f"{vid}{scorer}.csv").write_bytes(b"pose csv")
    (env.work / f"{vid}{scorer}_meta.pickle").write_bytes(b"meta")


class _Done:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


def test_work_order_reads_the_video_id_not_the_file_name(env, monkeypatch):
    seen = {}

    def fake_choice(items, key=None, **kw):
        seen["items"] = [key(i) for i in items]
        return items[0]

    import mousereach.watcher.work_priority as wp
    monkeypatch.setattr(wp, "best_tier_choice", fake_choice)
    pool_mod._pick([Path(f"x/{VID}.mp4")])
    assert seen["items"] == [{"video_id": VID}]


def test_hand_on_puts_the_pose_in_last_and_removes_the_claim(env, monkeypatch):
    claimed = _claimed(env, VID)
    _local_pose(env, VID)
    order = []
    real_replace = Path.replace

    def recording_replace(self, target):
        order.append(Path(target).name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    assert env.pool._hand_on(VID)
    names = sorted(p.name for p in env.posed.iterdir())
    assert f"{VID}.mp4" in names and f"{VID}{POSE}.h5" in names
    assert not [n for n in names if n.endswith(".part")]
    assert order[-1] == f"{VID}{POSE}.h5"               # the pose is the last to appear
    assert not claimed.exists()                         # handed on, so the claim goes
    assert list(env.work.iterdir()) == []               # scratch copies removed


def test_failed_pose_gives_the_video_back_and_backs_off(env):
    claimed = _claimed(env, VID)
    (env.work / f"{VID}.mp4").write_bytes(b"video-bytes")
    env.pool.active[VID] = {"proc": _Done(1), "local": env.work / f"{VID}.mp4",
                            "started": 0}
    env.pool._reap()
    assert not env.pool.active
    assert not claimed.exists() and (env.door / f"{VID}.mp4").exists()
    assert env.pool.backoff[VID] > 0
    assert not list(env.posed.iterdir())


def test_a_pose_from_another_model_is_not_handed_on(env):
    claimed = _claimed(env, VID)
    _local_pose(env, VID, scorer="DLC_resnet50_OtherModelshuffle1_5000")
    env.pool.active[VID] = {"proc": _Done(0), "local": env.work / f"{VID}.mp4",
                            "started": 0}
    env.pool._reap()
    assert not list(env.posed.iterdir())
    assert (env.door / f"{VID}.mp4").exists() and not claimed.exists()


def test_start_up_hands_on_a_finished_pose_and_gives_back_an_unfinished_one(env):
    _claimed(env, VID)
    _local_pose(env, VID)
    c2 = _claimed(env, VID2)
    (env.work / f"{VID2}.mp4").write_bytes(b"video-bytes")      # copy, no pose yet
    env.pool.recover()
    assert (env.posed / f"{VID}{POSE}.h5").exists()
    assert (env.door / f"{VID2}.mp4").exists() and not c2.exists()
    assert not (env.work / f"{VID2}.mp4").exists()


def test_a_finished_or_held_video_is_never_taken(env, monkeypatch):
    (env.door / f"{VID}.mp4").write_bytes(b"done already")
    (env.door / f"{VID2}.mp4").write_bytes(b"video-bytes")
    monkeypatch.setattr(pool_mod, "_done_or_held",
                        lambda vid: ("archived", "already analysed") if vid == VID else None)
    monkeypatch.setattr(pool_mod, "_pick", lambda c: sorted(c)[0])
    env.pool._spawn = lambda local: _Done(None)
    assert env.pool._start_one()
    assert list(env.pool.active) == [VID2]
    assert (env.door / f"{VID}.mp4").exists()                    # left where it was
    assert (env.door / ".inflight" / HOLDER / f"{VID2}.mp4").exists()
    assert (env.work / f"{VID2}.mp4").read_bytes() == b"video-bytes"


def test_status_reports_doing_in_the_shape_every_node_uses(env):
    import json
    env.pool.active = {VID: {"proc": _Done(None), "local": None, "started": 1_700_000_000},
                       VID2: {"proc": _Done(None), "local": None, "started": 1_700_000_100}}
    env.pool._status()
    rec = json.loads((env.tmp / "share" / "Processing" / ".node_status"
                      / f"{HOLDER}.json").read_text())
    assert isinstance(rec["doing"], dict) and rec["doing"]["type"] == "cpu_pose"
    assert rec["doing"]["id"] == "2 videos"
    assert rec["poses"] == sorted([VID, VID2])
    env.pool.active = {}
    env.pool._status()
    rec = json.loads((env.tmp / "share" / "Processing" / ".node_status"
                      / f"{HOLDER}.json").read_text())
    assert rec["doing"] is None and rec["state"] == "idle"
