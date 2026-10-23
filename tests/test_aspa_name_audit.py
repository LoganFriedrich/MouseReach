"""Checking an archive's recording names against its own cuts.

WHY (2026-10-23): ASPA's archive names every animal twice -- in the recording's
name and in its own cut of each seat -- and where the two disagree the pipeline,
which trusts the recording's name, files one animal's video under another's id.
A cut is traced to its recording by length (every cut of a recording is as long
as it); these tests stand in for the measuring, so they are about the tracing
and the decisions.
"""
import json
from pathlib import Path

import mousereach.aspa.import_collages as ic
import mousereach.aspa.name_audit as na

A = "H01,H02,H03,H04,H05,H06,H07,H08"
B = "H09,H10,H11,H12,H13,H14,H15,H16"


def _archive(tmp_path, recordings, cuts):
    """recordings: {file name: frames}; cuts: {file name: frames} (cohort H)."""
    root = tmp_path / "ASPA"
    multi, sa = root / "H" / "Multi-Animal", root / "H" / "Single_Animal"
    multi.mkdir(parents=True)
    sa.mkdir(parents=True)
    frames = {}
    for name, n in recordings.items():
        (multi / name).write_bytes(b"rec")
        frames[name] = n
    for name, n in cuts.items():
        (sa / name).write_bytes(b"cut")
        frames[name] = n
    return root, (lambda p: frames.get(Path(p).name))


def test_a_recording_holding_the_other_group_is_contested(tmp_path):
    # The file named for H01-H08 is as long as the cuts named H09-H16; the
    # cuts named H01-H08 match nothing (their recording is lost).
    cuts = {f"20220602_{a}_P1.mp4": 39000 for a in A.split(",")}
    cuts.update({f"20220602_{a}_P1.mp4": 40001 for a in B.split(",")})
    root, length = _archive(tmp_path, {f"20220602_{A}_P1.mkv": 40000}, cuts)
    res = na.audit(root, ["H"], length=length)
    assert len(res["conflicts"]) == 1
    c = res["conflicts"][0]
    assert c["recording"] == f"20220602_{A}_P1.mkv"
    assert sorted(x["animal"] for x in c["cuts_named_differently"]) == B.split(",")
    assert c["pipeline_recordings"] == [
        "20220602_ASPA0801,ASPA0802,ASPA0803,ASPA0804,ASPA0805,ASPA0806,ASPA0807,ASPA0808_P1"]
    assert res["no_source"] == 8 and res["consistent"] == 0


def test_a_trimmed_recording_is_traced_through_its_cuts(tmp_path):
    # The surviving recording is untrimmed (longer than its cuts). L09's cut is
    # named by it; L13's cut is as long as L09's, so it came out of it too --
    # and the recording names L14, not L13, in that seat.
    rec = "20221209_L09,L10,L11,L12,L14,L15,K03,K04_P1.mkv"
    root = tmp_path / "ASPA"
    multi, sa = root / "L" / "Multi-Animal", root / "L" / "Single_Animal"
    multi.mkdir(parents=True)
    sa.mkdir(parents=True)
    (multi / rec).write_bytes(b"rec")
    frames = {rec: 48338}
    for a in ("L09", "L13"):
        (sa / f"20221209_{a}_P1.mp4").write_bytes(b"cut")
        frames[f"20221209_{a}_P1.mp4"] = 37220
    res = na.audit(root, ["L"], length=lambda p: frames.get(Path(p).name))
    assert res["consistent"] == 1
    (c,) = res["conflicts"]
    assert c["recording"] == rec
    assert [x["animal"] for x in c["cuts_named_differently"]] == ["L13"]
    assert c["cuts_named_differently"][0]["traced"] == "through a sibling cut"


def test_names_that_agree_are_not_contested(tmp_path):
    cuts = {f"20220602_{a}_P1.mp4": 40000 for a in A.split(",")}
    root, length = _archive(tmp_path, {f"20220602_{A}_P1.mkv": 40002}, cuts)
    res = na.audit(root, ["H"], length=length)
    assert res["conflicts"] == [] and res["consistent"] == 8


def test_the_record_is_written_whole_with_its_counts(tmp_path):
    cuts = {f"20220602_{a}_P1.mp4": 40001 for a in B.split(",")}
    root, length = _archive(tmp_path, {f"20220602_{A}_P1.mkv": 40000}, cuts)
    res = na.audit(root, ["H"], length=length)
    out = na.write_record(res, tmp_path / "records" / na.RECORD_NAME, root, ["H"])
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["counts"]["conflicting_recordings"] == 1
    assert rec["conflicts"][0]["recording"] == f"20220602_{A}_P1.mkv"
    assert not list(out.parent.glob("*.part"))


def test_the_import_does_not_bring_in_a_contested_recording(tmp_path, monkeypatch, capsys):
    cuts = {f"20220602_{a}_P1.mp4": 40001 for a in B.split(",")}
    root, length = _archive(tmp_path, {f"20220602_{A}_P1.mkv": 40000,
                                       f"20220602_{B}_P2.mkv": 41000}, cuts)
    monkeypatch.setattr(na, "header_frames", length)
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
