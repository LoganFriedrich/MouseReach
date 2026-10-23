"""Bringing in ASPA sessions whose recording is lost: rebuild the recording from
ASPA's own cuts, into the recording intake -- never feed the cuts themselves.

WHY (2026-10-23): the input to MouseReach is the multi-animal recording; a cut
fed into the singles intake has no link to its recording, and the recording
never enters the pipeline. A session whose recording survives comes in through
the recording import; one whose recording is lost is rebuilt (each cut back in
its grid cell). The video work itself is tested in test_rebuild_recording.py;
here the encoding is stood in for, so these tests are about the decisions.
"""
from types import SimpleNamespace

import mousereach.aspa.feed as feed
import mousereach.video_prep.core.rebuild as rb
from mousereach.config import Paths

GROUP = ["H09", "H10", "H11", "H12", "H13", "H14", "H15", "H16"]


def _env(tmp_path, monkeypatch, cuts_on_0602=GROUP):
    archive = tmp_path / "Archive" / "historical" / "ASPA"
    sa = archive / "H" / "Single_Animal"
    multi = archive / "H" / "Multi-Animal"
    sa.mkdir(parents=True)
    multi.mkdir(parents=True)
    # the group was recorded on 06-01; on 06-02 only its cuts survive, and the
    # other group's 06-02 recording survives
    (multi / f"20220601_{','.join(GROUP)}_P1.mkv").write_bytes(b"rec")
    (multi / "20220602_H01,H02,H03,H04,H05,H06,H07,H08_P1.mkv").write_bytes(b"rec")
    for aid in cuts_on_0602:
        (sa / f"20220602_{aid}_P1.mp4").write_bytes(b"cut")
    (sa / "20220602_H01_P1.mp4").write_bytes(b"cut")
    intake = tmp_path / "Unanalyzed" / "Multi-Animal"
    door = tmp_path / "Unanalyzed" / "Single_Animal"
    posed = tmp_path / "Processing" / "Posed"
    posed.mkdir(parents=True)
    for name, value in (("MULTI_ANIMAL_SOURCE", intake), ("SINGLE_ANIMAL_OUTPUT", door),
                        ("DLC_STAGING", posed), ("ANALYZED_OUTPUT", tmp_path / "Analyzed"),
                        ("NAS_ROOT", tmp_path)):
        monkeypatch.setattr(Paths, name, value, raising=False)
    built = []
    monkeypatch.setattr(rb, "plan", lambda stem, singles: {"frames": 10})

    def fake_rebuild(stem, singles, dest, records, reason):
        built.append((stem, dict(singles), dest))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{stem}.mkv").write_bytes(b"rebuilt")
        return dest / f"{stem}.mkv"
    monkeypatch.setattr(rb, "rebuild", fake_rebuild)
    return SimpleNamespace(archive=archive, sa=sa, intake=intake, door=door,
                           posed=posed, built=built)


def test_the_pipeline_name_is_the_encoded_session():
    assert feed.pipeline_name("20220602_H09_P1.mp4") == "20220602_ASPA0809_P1.mp4"
    assert feed.pipeline_name("not_a_session.mp4") is None


def test_a_lost_recording_is_rebuilt_in_its_groups_order_into_the_recording_intake(
        tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    assert feed.main(["--cohort", "H"]) == 0
    out = capsys.readouterr().out
    assert len(env.built) == 1
    stem, singles, dest = env.built[0]
    assert stem == ("20220602_ASPA0809,ASPA0810,ASPA0811,ASPA0812,ASPA0813,ASPA0814,"
                    "ASPA0815,ASPA0816_P1")
    assert dest == env.intake
    assert singles["ASPA0809"] == env.sa / "20220602_H09_P1.mp4"
    assert not env.door.exists() or not any(env.door.iterdir()), "no cut is fed"
    assert "its recording survives" in out and "20220602_H01_P1.mp4" in out
    assert (env.sa / "20220602_H09_P1.mp4").exists(), "the archive is only read"


def test_an_incomplete_group_is_named_for_a_person_not_rebuilt(tmp_path, monkeypatch, capsys):
    # H11 has no cut that day: whether the others came from one lost recording
    # (and H11 was simply absent) or a cut carries the wrong name, the files
    # cannot tell -- a guess could enter one animal's video under two ids.
    env = _env(tmp_path, monkeypatch, cuts_on_0602=[a for a in GROUP if a != "H11"])
    feed.main(["--cohort", "H"])
    assert env.built == []
    assert "its group is incomplete that day (no lost cut for H11)" in capsys.readouterr().out


def test_an_empty_box_in_the_reference_group_is_an_empty_position(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch, cuts_on_0602=["H09", "H10", "H11"])
    (env.archive / "H" / "Multi-Animal" / "20220601_H09,H10,H11,000,001,002,003,004_P1.mkv"
     ).write_bytes(b"rec")
    for p in (env.archive / "H" / "Multi-Animal").glob(f"20220601_{','.join(GROUP)}*"):
        p.unlink()
    feed.main(["--cohort", "H"])
    stem, singles, _ = env.built[0]
    assert stem.split("_")[1].split(",")[3] == "ASPA0004"     # position 4 empty
    assert sorted(singles) == ["ASPA0809", "ASPA0810", "ASPA0811"]


def test_listed_sessions_bring_in_their_recording_and_missing_ones_are_named(
        tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    f = tmp_path / "sessions.txt"
    f.write_text("20220602_H09_P1\n20220602_H99_P1\n", encoding="utf-8")
    assert feed.main(["--sessions-file", str(f), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert env.built == [] and "would rebuild" in out
    assert "20220602_ASPA0899_P1" in out                          # listed, no cut: named


def test_a_recording_already_in_the_pipeline_is_not_rebuilt(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    env.intake.mkdir(parents=True)
    (env.intake / ("20220602_ASPA0809,ASPA0810,ASPA0811,ASPA0812,ASPA0813,ASPA0814,"
                   "ASPA0815,ASPA0816_P1.mkv")).write_bytes(b"there")
    feed.main(["--cohort", "H"])
    assert env.built == [] and "already in the pipeline" in capsys.readouterr().out


def test_a_name_with_a_date_that_is_not_a_day_is_named_and_left(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    (env.sa / "20220631_H09_P1.mp4").write_bytes(b"cut")                 # June has 30 days
    (env.archive / "H" / "Multi-Animal" / f"20220632_{','.join(GROUP)}_P1.mkv").write_bytes(b"x")
    assert feed.main(["--cohort", "H"]) == 0
    assert "20220631 is not a real date" in capsys.readouterr().out
    assert len(env.built) == 1                                           # the 06-02 one still built


def test_each_cut_goes_into_exactly_one_rebuilt_recording(tmp_path, monkeypatch):
    # Groups differ between days; taking each animal's nearest day put one
    # animal into two rebuilt recordings.
    env = _env(tmp_path, monkeypatch)
    multi = env.archive / "H" / "Multi-Animal"
    (multi / "20220610_H09,H10,H11,H12,H01,H02,H03,H04_P1.mkv").write_bytes(b"rec")
    (multi / "20220610_H13,H14,H15,H16,H05,H06,H07,H08_P1.mkv").write_bytes(b"rec")
    feed.main(["--cohort", "H"])
    placed = [e for _, singles, _ in env.built for e in singles]
    assert len(placed) == len(set(placed)) == 8
