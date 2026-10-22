"""Bringing in only the ASPA recordings that hold the sessions a list asks for.

WHY: a recording holds eight animals, and the sessions one question needs are
scattered across far fewer recordings than a whole cohort. Importing whole
cohorts to get them costs about three times the copying and then the cutting,
posing and review of everything else first. The list decides what comes in and
in what order, and the dry run names every listed session that has no
recording -- a session silently missing from the import is the failure this
exists to prevent. A recording the pipeline already has is never brought in again.
"""
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
