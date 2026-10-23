"""Checking an archive's recording names against its own cuts.

WHY (2026-10-23): ASPA's archive names every animal twice -- in the recording's
name and in its own cut of each seat -- and where the two disagree the pipeline,
which trusts the recording's name, files one animal's video under another's id.
A cut is traced to its recording by length (every cut of a recording is as long
as it) and a disagreement is confirmed by the picture (the cut IS one seat of
the recording), because two recordings of one day can be within a few frames of
each other by chance. These tests stand in for the measuring and the picture,
so they are about the tracing and the decisions.
"""
import json
from pathlib import Path

import mousereach.aspa.import_collages as ic
import mousereach.aspa.name_audit as na

A = "H01,H02,H03,H04,H05,H06,H07,H08"
B = "H09,H10,H11,H12,H13,H14,H15,H16"


def _archive(tmp_path, recordings, cuts, cohort="H"):
    """recordings: {file name: frames}; cuts: {file name: frames}."""
    root = tmp_path / "ASPA"
    multi, sa = root / cohort / "Multi-Animal", root / cohort / "Single_Animal"
    multi.mkdir(parents=True)
    sa.mkdir(parents=True)
    frames = {}
    for name, n in {**recordings, **cuts}.items():
        (multi if name in recordings else sa).joinpath(name).write_bytes(b"x")
        frames[name] = n
    return root, (lambda p: frames.get(Path(p).name))


def _picture(seats):
    """seats: {(cut name, recording name): seat index} -- what the picture shows."""
    def picture(cut, recording, f, rf):
        k = seats.get((Path(cut).name, Path(recording).name))
        return None if k is None else (k, 1.2, 40.0)
    return picture


def test_a_recording_holding_the_other_group_is_contested(tmp_path):
    # The file named for H01-H08 is as long as the cuts named H09-H16, and the
    # picture shows each of them in one of its seats. ASPA's H01-H08 cuts are
    # another length: their recording is lost.
    cuts = {f"20220602_{a}_P1.mp4": 39000 for a in A.split(",")}
    cuts.update({f"20220602_{a}_P1.mp4": 40000 for a in B.split(",")})
    rec = f"20220602_{A}_P1.mkv"
    root, length = _archive(tmp_path, {rec: 40000}, cuts)
    seats = {(f"20220602_{a}_P1.mp4", rec): k for k, a in enumerate(B.split(","))}
    res = na.audit(root, ["H"], length=length, picture=_picture(seats))
    (c,) = res["conflicts"]
    assert c["recording"] == rec
    first = c["cuts_named_differently"][0]
    assert (first["animal"], first["seat"], first["recording_names_seat"]) == ("H09", 1, "H01")
    assert c["pipeline_recordings"] == [
        "20220602_ASPA0801,ASPA0802,ASPA0803,ASPA0804,ASPA0805,ASPA0806,ASPA0807,ASPA0808_P1"]
    assert res["no_source"] == 8 and res["consistent"] == 0


def test_a_length_coincidence_is_not_a_conflict(tmp_path):
    # H33's cut is 3 frames from the H01-H08 file by chance; its own recording
    # is untrimmed (longer). The picture does not show H33 in the H01-H08 file.
    rec_a, rec_b = f"20220602_{A}_P1.mkv", "20220602_H33,H34,H35,H36,H37,H38,H39,H40_P1.mkv"
    root, length = _archive(tmp_path, {rec_a: 37118, rec_b: 40546},
                            {"20220602_H33_P1.mp4": 37121})
    res = na.audit(root, ["H"], length=length, picture=_picture({}))
    assert res["conflicts"] == [] and res["consistent"] == 1
    assert res["pictures_compared"] == 1


def test_a_trimmed_recording_is_traced_through_its_cuts(tmp_path):
    # The surviving recording is untrimmed (longer than its cuts). L09's cut is
    # named by it; L13's cut is as long as L09's, so it came out of it too --
    # and the picture shows it in seat 5, which the recording names L14.
    rec = "20221209_L09,L10,L11,L12,L14,L15,K03,K04_P1.mkv"
    root, length = _archive(tmp_path, {rec: 48338},
                            {"20221209_L09_P1.mp4": 37220, "20221209_L13_P1.mp4": 37220},
                            cohort="L")
    res = na.audit(root, ["L"], length=length,
                   picture=_picture({("20221209_L13_P1.mp4", rec): 4}))
    assert res["consistent"] == 1
    (c,) = res["conflicts"]
    (x,) = c["cuts_named_differently"]
    assert (x["animal"], x["seat"], x["recording_names_seat"], x["traced"]) == \
        ("L13", 5, "L14", "through a sibling cut")


def test_a_cut_in_a_seat_named_as_empty_is_a_conflict(tmp_path):
    rec = "20221118_L09,L10,L11,L12,L14,L15,000,002_P1.mkv"
    root, length = _archive(tmp_path, {rec: 37525}, {"20221118_L13_P1.mp4": 37525},
                            cohort="L")
    res = na.audit(root, ["L"], length=length,
                   picture=_picture({("20221118_L13_P1.mp4", rec): 6}))
    (x,) = res["conflicts"][0]["cuts_named_differently"]
    assert (x["seat"], x["recording_names_seat"]) == (7, "(empty)")


def test_names_that_agree_are_not_compared_at_all(tmp_path):
    cuts = {f"20220602_{a}_P1.mp4": 40000 for a in A.split(",")}
    root, length = _archive(tmp_path, {f"20220602_{A}_P1.mkv": 40002}, cuts)
    res = na.audit(root, ["H"], length=length, picture=_picture({}))
    assert res["conflicts"] == [] and res["consistent"] == 8
    assert res["pictures_compared"] == 0


def test_the_record_is_written_whole_with_its_counts(tmp_path):
    cuts = {f"20220602_{a}_P1.mp4": 40000 for a in B.split(",")}
    rec = f"20220602_{A}_P1.mkv"
    root, length = _archive(tmp_path, {rec: 40000}, cuts)
    seats = {(f"20220602_{a}_P1.mp4", rec): k for k, a in enumerate(B.split(","))}
    res = na.audit(root, ["H"], length=length, picture=_picture(seats))
    out = na.write_record(res, tmp_path / "records" / na.RECORD_NAME, root, ["H"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["counts"]["conflicting_recordings"] == 1
    assert data["conflicts"][0]["recording"] == rec
    assert not list(out.parent.glob("*.part"))


def test_the_import_does_not_bring_in_a_contested_recording(tmp_path, monkeypatch, capsys):
    cuts = {f"20220602_{a}_P1.mp4": 40000 for a in B.split(",")}
    rec = f"20220602_{A}_P1.mkv"
    root, length = _archive(tmp_path, {rec: 40000, f"20220602_{B}_P2.mkv": 41000}, cuts)
    seats = {(f"20220602_{a}_P1.mp4", rec): k for k, a in enumerate(B.split(","))}
    monkeypatch.setattr(na, "header_frames", length)
    monkeypatch.setattr(na, "seat_of", _picture(seats))
    monkeypatch.setattr(ic, "default_retired_dir", lambda: None)
    monkeypatch.setattr(ic, "default_quarantine_dir", lambda: None)
    dest = tmp_path / "Unanalyzed" / "Multi-Animal"
    assert ic.main(["--source", str(root), "--dest", str(dest), "--cohorts", "H",
                    "--apply"]) == 0
    out = capsys.readouterr().out
    assert "NOT imported" in out and f"20220602_{A}_P1" in out
    got = [p.name for p in dest.glob("*.mkv")]
    assert got == ["20220602_ASPA0809,ASPA0810,ASPA0811,ASPA0812,ASPA0813,ASPA0814,"
                   "ASPA0815,ASPA0816_P2.mkv"]


def test_a_recording_held_in_quarantine_is_not_imported_again(tmp_path):
    held = tmp_path / "Quarantine"
    held.mkdir()
    item = {"dst": tmp_path / "intake" / "20220602_X_P1.mkv"}
    (held / "20220602_X_P1.mkv").write_bytes(b"held")
    assert "quarantine" in ic.already_in_pipeline(item, None, held)
