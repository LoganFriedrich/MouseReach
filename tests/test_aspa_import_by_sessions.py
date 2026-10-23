"""Bringing in only the ASPA recordings that hold the sessions a list asks for.

WHY: a recording holds eight animals, and the sessions one question needs are
scattered across far fewer recordings than a whole cohort. Importing whole
cohorts to get them costs about three times the copying and then the cutting,
posing and review of everything else first. The list decides what comes in and
in what order, and the dry run names every listed session that has no
recording -- a session silently missing from the import is the failure this
exists to prevent. A recording the pipeline already has is never brought in again.
"""
import json
from pathlib import Path

import mousereach.aspa.import_collages as ic

A = "D01,D02,D03,D04,D05,D06,D07,D08"
B = "D09,D10,D11,D12,D13,D14,D15,D16"


def _archive(tmp_path):
    src = tmp_path / "ASPA"
    multi = src / "OptD" / "Multi-Animal"
    multi.mkdir(parents=True)
    for stem in (f"20220217_{A}_P1", f"20220217_{B}_P1", f"20220218_{A}_P2"):
        (multi / f"{stem}.mkv").write_bytes(b"recording " + stem.encode())
    dest = tmp_path / "Unanalyzed" / "Multi-Animal"
    return src, dest


def _list(tmp_path, *lines):
    f = tmp_path / "sessions.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def test_a_session_may_be_written_the_labs_way_or_the_pipelines():
    assert ic.normalise_session("20220811_J11_P3") == "20220811_ASPA1011_P3"
    assert ic.normalise_session("20220811_ASPA1011_P3") == "20220811_ASPA1011_P3"
    assert ic.normalise_session("not a session") is None


def test_the_list_keeps_its_order_once_and_names_bad_lines(tmp_path):
    f = _list(tmp_path, "# UCSF sets", "20220218_D03_P2", "", "20220217_D10_P1",
              "20220218_ASPA0403_P2", "garbage line")
    sessions, bad = ic.read_sessions_file(f)
    assert sessions == ["20220218_ASPA0403_P2", "20220217_ASPA0410_P1"]
    assert bad == ["garbage line"]


def test_recordings_come_in_list_order_once_and_missing_sessions_are_named(tmp_path):
    src, dest = _archive(tmp_path)
    plan = ic.build_plan(src, dest, ["OptD"])["plan"]
    chosen, missing = ic.select_for_sessions(plan, [
        "20220218_ASPA0402_P2",     # in the 02-18 recording
        "20220217_ASPA0410_P1",     # in the second 02-17 recording
        "20220218_ASPA0405_P2",     # the 02-18 recording again: not twice
        "20220219_ASPA0401_P1",     # no recording that day
    ])
    assert [c["encoded_stem"][:8] + c["encoded_stem"][-3:] for c in chosen] == \
        ["20220218_P2", "20220217_P1"]
    assert "ASPA0409" in chosen[1]["encoded_stem"]
    assert missing == ["20220219_ASPA0401_P1"]


def test_a_recording_the_pipeline_already_has_is_not_imported_again(tmp_path, monkeypatch):
    src, dest = _archive(tmp_path)
    analyzed = tmp_path / "Analyzed"
    (analyzed / "Multi-Animal").mkdir(parents=True)
    monkeypatch.setattr(ic, "default_retired_dir", lambda: analyzed / "Multi-Animal")
    plan = ic.build_plan(src, dest, ["OptD"])["plan"]
    by_day = {p["encoded_stem"][:8] + p["encoded_stem"][-2:]: p for p in plan
              if "ASPA0401" in p["encoded_stem"]}
    cut = by_day["20220217P1"]
    (analyzed / "Multi-Animal" / cut["dst"].name).write_bytes(b"cut earlier")
    assert "cut" in ic.already_in_pipeline(cut, analyzed / "Multi-Animal")
    dest.mkdir(parents=True)
    waiting = by_day["20220218P2"]
    waiting["dst"].write_bytes(b"already in intake")
    assert ic.already_in_pipeline(waiting, analyzed / "Multi-Animal") == "in the intake folder"


def test_the_dry_run_copies_nothing_and_apply_copies_only_what_the_list_needs(tmp_path, monkeypatch, capsys):
    src, dest = _archive(tmp_path)
    monkeypatch.setattr(ic, "default_retired_dir", lambda: None)
    f = _list(tmp_path, "20220217_D11_P1", "20220219_D01_P1")
    args = ["--source", str(src), "--dest", str(dest), "--cohorts", "OptD",
            "--sessions-file", str(f)]

    assert ic.main(args) == 0
    out = capsys.readouterr().out
    assert "Recordings to import: 1" in out
    assert "20220219_ASPA0401_P1" in out          # named, not silently dropped
    assert not dest.exists() or not any(dest.iterdir())

    assert ic.main(args + ["--apply"]) == 0
    copied = sorted(p.name for p in dest.glob("*.mkv"))
    assert len(copied) == 1 and "ASPA0409" in copied[0]
    assert not list(dest.glob("*.partial"))


def test_the_copy_replaces_a_half_written_file_from_a_stopped_run(tmp_path):
    """A stopped import leaves `<name>.partial`; the next run's copy must
    replace it, and the result must match the source byte for byte."""
    src = tmp_path / "rec.mkv"
    src.write_bytes(b"x" * 100000)
    dst = tmp_path / "rec.mkv.partial"
    dst.write_bytes(b"half")
    ic.copy_file(src, dst)
    assert dst.read_bytes() == src.read_bytes()
    assert abs(dst.stat().st_mtime - src.stat().st_mtime) < 2


# -- Names written slightly wrong ------------------------------------------------
# WHY: the import exists to copy recordings in under names the pipeline can read
# (Logan, 2026-10-23: "copying and renaming the copies of multis to be compliant
# with the formatting restrictions"). A recording whose name has a separator typo
# was left out, and its animals' cuts then looked lost.

def test_a_separator_typo_between_ids_is_read_as_a_comma():
    for stem, want in (
            ("20220503_G25.G26,G27,G28,G29,G30,G31,G32_P1",
             "20220503_G25,G26,G27,G28,G29,G30,G31,G32_P1"),
            ("20220727_H36,H38,H39,H40_001,002,003,004_F1",
             "20220727_H36,H38,H39,H40,001,002,003,004_F1")):
        (sessions, trailing) = ic.read_recording_stem(stem)
        assert sessions == [(want, "separator typo between ids read as a comma")]
        assert trailing == ""


def test_a_standard_name_reads_unchanged_with_its_trailing_text():
    stem = f"20220217_{A}_P1 uncropped"
    assert ic.read_recording_stem(stem) == ([(f"20220217_{A}_P1", "")], " uncropped")


def test_one_recording_with_two_cohorts_on_different_runs_is_read_once_per_run():
    (sessions, _) = ic.read_recording_stem("20220812_J15,J16,I11,I13,I14,I15,I16,I17_JP3,IP1")
    assert [s for s, _ in sessions] == ["20220812_J15,J16,003,004,005,006,007,008_P3",
                                        "20220812_001,002,I11,I13,I14,I15,I16,I17_P1"]


def test_names_that_do_not_read_one_way_are_not_read():
    assert ic.read_recording_stem("20210924_ABS2_13-15_f") is None           # a range
    assert ic.read_recording_stem("20220503_G25.G26,G27_P1") is None         # not eight ids
    # a run named for a cohort none of the animals belong to
    assert ic.read_recording_stem(f"20220217_{A}_JP3,IP1") is None


def test_repaired_names_are_imported_and_recorded_with_the_original(tmp_path, monkeypatch, capsys):
    src = tmp_path / "ASPA"
    for cohort in ("I", "J"):                    # the shared recording is filed in both
        multi = src / cohort / "Multi-Animal"
        multi.mkdir(parents=True)
        (multi / "20220812_J15,J16,I11,I13,I14,I15,I16,I17_JP3,IP1.mkv").write_bytes(b"rec")
    (src / "J" / "Multi-Animal" / "20220813_J01.J02,J03,J04,J05,J06,J07,J08_P1.mkv"
     ).write_bytes(b"rec")
    dest = tmp_path / "Unanalyzed" / "Multi-Animal"
    monkeypatch.setattr(ic, "default_retired_dir", lambda: None)
    args = ["--source", str(src), "--dest", str(dest), "--cohorts", "I,J"]
    assert ic.main(args) == 0
    out = capsys.readouterr().out
    assert "Names read although not in the standard form: 3" in out
    assert ic.main(args + ["--apply"]) == 0
    got = sorted(x.stem for x in dest.glob("*.mkv"))
    assert got == [
        "20220812_ASPA0001,ASPA0002,ASPA0911,ASPA0913,ASPA0914,ASPA0915,ASPA0916,ASPA0917_P1",
        "20220812_ASPA1015,ASPA1016,ASPA0003,ASPA0004,ASPA0005,ASPA0006,ASPA0007,ASPA0008_P3",
        "20220813_ASPA1001,ASPA1002,ASPA1003,ASPA1004,ASPA1005,ASPA1006,ASPA1007,ASPA1008_P1"]
    mapping = json.loads((dest / "ASPA_import_mapping.json").read_text(encoding="utf-8"))
    e = mapping["entries"]["20220813_ASPA1001,ASPA1002,ASPA1003,ASPA1004,ASPA1005,ASPA1006,"
                           "ASPA1007,ASPA1008_P1"]
    assert e["source_stem"] == "20220813_J01.J02,J03,J04,J05,J06,J07,J08_P1"
    assert e["name_read_as"] == "20220813_J01,J02,J03,J04,J05,J06,J07,J08_P1"
