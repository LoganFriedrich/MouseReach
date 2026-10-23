"""Bringing in ASPA sessions whose recording is lost: rebuild the recording from
ASPA's own cuts, into the recording intake -- never feed the cuts themselves.

WHY (2026-10-23): the input to MouseReach is the multi-animal recording; a cut
fed into the singles intake has no link to its recording, and the recording
never enters the pipeline. A session whose recording survives comes in through
the recording import; one whose recording is lost is rebuilt (each cut back in
its grid cell). The video work itself is tested in test_rebuild_recording.py;
here the encoding and the measuring of a cut's length are stood in for, so
these tests are about the decisions. Which cuts share a recording is read from
their lengths: every cut of one recording is exactly as long as it.
"""
from pathlib import Path
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
    # every cut 40000 frames (a session's length) unless a test says
    # otherwise; None = unreadable. Surviving recordings: 50000 frames.
    lengths, rec_lengths = {}, {}

    def fake_probe(path):
        n = lengths.get(Path(path).name, 40000)
        if n is None:
            raise rb.RebuildRefused("cannot open it")
        return {"frames": n, "fps": 30.0, "width": 480, "height": 540}
    monkeypatch.setattr(rb, "probe", fake_probe)
    monkeypatch.setattr(feed, "recording_length",
                        lambda path: rec_lengths.get(Path(path).name, (50000, 30.0)))

    def fake_rebuild(stem, singles, dest, records, reason):
        built.append((stem, dict(singles), dest))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{stem}.mkv").write_bytes(b"rebuilt")
        return dest / f"{stem}.mkv"
    monkeypatch.setattr(rb, "rebuild", fake_rebuild)
    return SimpleNamespace(archive=archive, sa=sa, intake=intake, door=door,
                           posed=posed, built=built, lengths=lengths,
                           rec_lengths=rec_lengths)


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


def test_an_animal_without_a_cut_that_day_leaves_its_position_empty(tmp_path, monkeypatch):
    # H11 has no cut that day; the other seven are one length, so one
    # recording. H11's position is empty: no video is made of an animal that
    # has no cut.
    env = _env(tmp_path, monkeypatch, cuts_on_0602=[a for a in GROUP if a != "H11"])
    feed.main(["--cohort", "H"])
    stem, singles, _ = env.built[0]
    assert stem.split("_")[1].split(",")[2] == "ASPA0003"
    assert "ASPA0811" not in singles and len(singles) == 7


def test_cuts_of_different_lengths_are_different_recordings(tmp_path, monkeypatch, capsys):
    # The day's groups are not the recorded day's: lengths split the eight
    # cuts into 6 + 2 (as on real days, where copying another day's groups
    # had left one animal alone that had shared its recording).
    env = _env(tmp_path, monkeypatch)
    for a in ("H15", "H16"):
        env.lengths[f"20220602_{a}_P1.mp4"] = 39000
    feed.main(["--cohort", "H"])
    out = capsys.readouterr().out
    got = sorted(sorted(singles) for _, singles, _ in env.built)
    assert got == [["ASPA0809", "ASPA0810", "ASPA0811", "ASPA0812", "ASPA0813", "ASPA0814"],
                   ["ASPA0815", "ASPA0816"]]
    # both are part of the 06-01 group: its positions are kept
    pair = [stem for stem, singles, _ in env.built if "ASPA0815" in singles][0]
    assert pair.split("_")[1].split(",")[6:] == ["ASPA0815", "ASPA0816"]
    assert "positions as recorded on 20220601" in out


def test_a_group_no_recording_shows_goes_in_id_order(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    for p in (env.archive / "H" / "Multi-Animal").glob("20220601_*"):
        p.unlink()
    feed.main(["--cohort", "H"])
    stem, _, _ = env.built[0]
    assert stem.split("_")[1].split(",")[:2] == ["ASPA0809", "ASPA0810"]
    assert "positions in id order" in capsys.readouterr().out


def test_more_than_eight_cuts_of_one_length_are_for_a_person(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    (env.sa / "20220602_H17_P1.mp4").write_bytes(b"cut")
    feed.main(["--cohort", "H"])
    assert env.built == []
    assert "9 cuts that day are 40000 frames long" in capsys.readouterr().out


def test_a_cut_whose_length_cannot_be_read_is_named_and_left(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    env.lengths["20220602_H12_P1.mp4"] = None
    feed.main(["--cohort", "H"])
    out = capsys.readouterr().out
    assert "20220602_H12_P1.mp4: its length cannot be read" in out
    stem, singles, _ = env.built[0]
    assert "ASPA0812" not in singles and stem.split("_")[1].split(",")[3] == "ASPA0004"


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


def test_cuts_as_long_as_a_surviving_recording_are_for_a_person(tmp_path, monkeypatch, capsys):
    # The 06-02 recording of H01-H08 survives; the lost cuts are as long as it
    # (two frames off, as a cut and its recording can be): they came out of it
    # under names it does not list -- an identity question, never a rebuild.
    env = _env(tmp_path, monkeypatch)
    env.rec_lengths["20220602_H01,H02,H03,H04,H05,H06,H07,H08_P1.mkv"] = (40002, 30.0)
    feed.main(["--cohort", "H"])
    assert env.built == []
    out = capsys.readouterr().out
    assert "as long as 20220602_H01,H02,H03,H04,H05,H06,H07,H08_P1.mkv, which survives" in out


def test_a_surviving_recording_that_cannot_be_measured_stops_the_rebuild(
        tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    env.rec_lengths["20220602_H01,H02,H03,H04,H05,H06,H07,H08_P1.mkv"] = None
    feed.main(["--cohort", "H"])
    assert env.built == [] and "cannot be compared with" in capsys.readouterr().out


def test_cuts_too_short_to_be_a_session_are_for_a_person(tmp_path, monkeypatch, capsys):
    env = _env(tmp_path, monkeypatch)
    for a in GROUP:
        env.lengths[f"20220602_{a}_P1.mp4"] = 40
    feed.main(["--cohort", "H"])
    assert env.built == []
    assert "40 frames (1.3 s): too short to be a session" in capsys.readouterr().out


def test_recordings_of_the_day_are_found_however_their_name_is_spelled(tmp_path):
    multi = tmp_path / "H" / "Multi-Animal"
    multi.mkdir(parents=True)
    for n in ("20220602_A,B_P1.mkv", "20220602_A_JP3,IP1.mkv", "20220602_A_P1 uncropped.mkv",
              "20220602_A_P10.mkv", "20220603_A_P1.mkv", "20220602_A_P1.llc"):
        (multi / n).write_bytes(b"x")
    got = [p.name for p in feed.surviving_recordings(tmp_path, "H", "20220602", "P1")]
    assert got == ["20220602_A,B_P1.mkv", "20220602_A_JP3,IP1.mkv", "20220602_A_P1 uncropped.mkv"]


def test_a_recording_surviving_under_a_typo_name_is_not_rebuilt(tmp_path, monkeypatch, capsys):
    # Its name has a period for a comma; the recording import reads it (and
    # copies the original in), so its cuts are not lost.
    env = _env(tmp_path, monkeypatch)
    ids = ",".join(GROUP).replace("H09,H10", "H09.H10")
    (env.archive / "H" / "Multi-Animal" / f"20220602_{ids}_P1.mkv").write_bytes(b"rec")
    env.rec_lengths[f"20220602_{ids}_P1.mkv"] = (40000, 30.0)
    feed.main(["--cohort", "H"])
    assert env.built == []
    assert f"its recording survives (20220602_{ids}_P1)" in capsys.readouterr().out


def test_cuts_as_long_as_a_cut_of_a_surviving_recording_are_for_a_person(
        tmp_path, monkeypatch, capsys):
    # The surviving 06-02 recording was trimmed after it was cut, so it is no
    # longer as long as its cuts (50000 frames here) -- but H01's cut, which
    # came out of it, is as long as the "lost" cuts: they came out of it too.
    env = _env(tmp_path, monkeypatch)
    env.rec_lengths["20220602_H01_P1.mp4"] = (40001, 30.0)
    feed.main(["--cohort", "H"])
    assert env.built == []
    out = capsys.readouterr().out
    assert ("as long as 20220602_H01_P1.mp4, which came out of "
            "20220602_H01,H02,H03,H04,H05,H06,H07,H08_P1") in out
