"""Supersede never archives the same bytes twice.

WHY: _checksummed_move compared the incoming file only against the UNVERSIONED
archive name, so a reprocess loop that re-swept identical content every cycle
added another byte-identical .N copy per lap (two identical 23 MB .h5 copies
found 2026-09-09; one video had accumulated versions .1 through .52).
"""
from mousereach.archive.supersede import _checksummed_move


def test_identical_content_dedupes_against_versioned_copy(tmp_path):
    dst = tmp_path / "arch" / "v.json"
    dst.parent.mkdir()
    dst.write_text("base-generation", encoding="utf-8")
    (tmp_path / "arch" / "v.1.json").write_text("swept-once", encoding="utf-8")

    src = tmp_path / "v.json"
    src.write_text("swept-once", encoding="utf-8")          # same bytes as .1

    out = _checksummed_move(src, dst)

    assert out == tmp_path / "arch" / "v.1.json"
    assert not src.exists()                                  # consumed
    assert not (tmp_path / "arch" / "v.2.json").exists()     # no duplicate


def test_new_content_still_gets_its_own_version(tmp_path):
    dst = tmp_path / "arch" / "v.json"
    dst.parent.mkdir()
    dst.write_text("base-generation", encoding="utf-8")
    (tmp_path / "arch" / "v.1.json").write_text("swept-once", encoding="utf-8")

    src = tmp_path / "v.json"
    src.write_text("genuinely-new", encoding="utf-8")

    out = _checksummed_move(src, dst)

    assert out == tmp_path / "arch" / "v.2.json"
    assert out.read_text(encoding="utf-8") == "genuinely-new"
    assert not src.exists()
    # the earlier generations are untouched
    assert dst.read_text(encoding="utf-8") == "base-generation"
    assert (tmp_path / "arch" / "v.1.json").read_text(
        encoding="utf-8") == "swept-once"
