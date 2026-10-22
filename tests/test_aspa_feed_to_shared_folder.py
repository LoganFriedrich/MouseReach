"""Feeding ASPA's own single-animal cuts: from the archive, into the shared
singles folder, renamed into pipeline form, never twice.

WHY: the feed used to read a pipeline root retired in February, copy into ONE
node's private queue -- where no other node could see or take the videos -- and
keep ASPA's own names, which the intake does not accept. A session that exists
only as ASPA's cut (no recording survives) can only be analysed this way, and a
listed session that cannot be found must be named, never dropped silently.
"""
from types import SimpleNamespace

import mousereach.aspa.feed as feed
from mousereach.config import Paths


def _env(tmp_path, monkeypatch):
    archive = tmp_path / "Archive" / "historical" / "ASPA"
    sa = archive / "H" / "Single_Animal"
    sa.mkdir(parents=True)
    for stem in ("20220602_H09_P1", "20220602_H09_P2", "20220602_H10_P1"):
        (sa / f"{stem}.mp4").write_bytes(b"cut " + stem.encode())
        (sa / f"{stem}DLC_resnet50_old.h5").write_bytes(b"old pose")
    door = tmp_path / "Unanalyzed" / "Single_Animal"
    posed = tmp_path / "Processing" / "Posed"
    posed.mkdir(parents=True)
    analyzed = tmp_path / "Analyzed"
    for name, value in (("SINGLE_ANIMAL_OUTPUT", door), ("DLC_STAGING", posed),
                        ("ANALYZED_OUTPUT", analyzed), ("NAS_ROOT", tmp_path)):
        monkeypatch.setattr(Paths, name, value, raising=False)
    return SimpleNamespace(archive=archive, door=door, posed=posed, analyzed=analyzed)


def _sessions(tmp_path, *lines):
    f = tmp_path / "sessions.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def test_the_pipeline_name_is_the_encoded_session():
    assert feed.pipeline_name("20220602_H09_P1.mp4") == "20220602_ASPA0809_P1.mp4"
    assert feed.pipeline_name("not_a_session.mp4") is None


def test_listed_sessions_go_to_the_shared_folder_renamed_and_missing_ones_are_named(
        tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    f = _sessions(tmp_path, "20220602_H09_P1", "20220602_ASPA0810_P1", "20220602_H11_P1")

    assert feed.main(["--sessions-file", str(f), "--dry-run"]) == 0
    assert not env.door.exists() or not any(env.door.iterdir())

    assert feed.main(["--sessions-file", str(f)]) == 0
    out = capsys.readouterr().out
    assert sorted(p.name for p in env.door.iterdir()) == [
        "20220602_ASPA0809_P1.mp4", "20220602_ASPA0810_P1.mp4"]
    assert "20220602_ASPA0811_P1" in out          # listed, no cut: named
    assert "20220602_H09_P2" not in [p.stem for p in env.door.iterdir()]
    assert (env.archive / "H" / "Single_Animal" / "20220602_H09_P1.mp4").exists(), \
        "the archive is only read"


def test_a_video_the_pipeline_already_has_is_not_fed_again(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    f = _sessions(tmp_path, "20220602_H09_P1", "20220602_H09_P2", "20220602_H10_P1")
    (env.posed / "20220602_ASPA0809_P1.mp4").write_bytes(b"posed")      # waiting analysis
    claimed = env.door / ".inflight" / "OTHER-NODE"
    claimed.mkdir(parents=True)
    (claimed / "20220602_ASPA0809_P2.mp4").write_bytes(b"claimed")     # being posed

    feed.main(["--sessions-file", str(f)])
    top = sorted(p.name for p in env.door.iterdir() if p.is_file())
    assert top == ["20220602_ASPA0810_P1.mp4"]


def test_a_feed_never_overwrites(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    env.door.mkdir(parents=True)
    (env.door / "20220602_ASPA0809_P1.mp4").write_bytes(b"dropped earlier")
    ok, why = feed.feed_one(env.archive / "H" / "Single_Animal" / "20220602_H09_P1.mp4",
                            env.door)
    assert ok is False and "waiting" in why
    assert (env.door / "20220602_ASPA0809_P1.mp4").read_bytes() == b"dropped earlier"
