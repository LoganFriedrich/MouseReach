"""Feeding the archive back in: only the animals someone actually asked for.

A cohort is how the archive is filed, not the unit anyone wants to process. A
study is usually some of the animals from several cohorts, so feeding whole
cohorts to reach them drags in every other animal -- and each of those then
costs the same processing hours and lands in the same review queue for a person
to work through. The entry point is the only place that cost can be avoided.

The failure modes these pin down:

* A filter nobody asked for must never be applied, or a plain ``--cohort`` run
  silently feeds nothing.
* A list is a claim about what exists. A typo in it looks exactly like an animal
  with no data, so requested animals that matched nothing must be NAMED.
* A file whose name cannot be parsed must not be silently treated as "does not
  match" -- that loses data with no trace.
"""
from pathlib import Path

import pytest

from mousereach.aspa import feed


# --------------------------------------------------------------- reading names
@pytest.mark.parametrize("name,expected", [
    ("20220908_K01_P1.mp4", "K01"),
    ("20230125_M14_P3.mp4", "M14"),
    ("20220719_G07_E2.mp4", "G07"),
    ("20220908_k01_P1.mp4", "K01"),          # case does not matter
])
def test_the_animal_is_read_from_the_filename(name, expected):
    assert feed.subject_of(name) == expected


@pytest.mark.parametrize("name", [
    "notavideo.mp4",
    "20220908_K01.mp4",        # no tray field
    "",
])
def test_an_unreadable_name_says_so_rather_than_guessing(name):
    assert feed.subject_of(name) == ""


# ----------------------------------------------------------- reading the list
def test_either_spelling_of_an_id_works():
    """Nobody should need to know whether the pipeline encodes ids.

    The archive filenames carry the lab's own short id; the database carries the
    encoded one. Both reduce to the form the filenames use.
    """
    assert feed.load_subjects("K01") == {"K01"}
    assert feed.load_subjects("ASPA1101") == {"K01"}
    assert feed.load_subjects("ASPA1101, K02") == {"K01", "K02"}


def test_a_list_can_be_written_the_way_people_write_lists():
    text = """
    # the animals for the paper
    K01, K02
    K03
    M01   M02
    """
    assert feed.load_subjects(text) == {"K01", "K02", "K03", "M01", "M02"}


def test_a_file_and_an_inline_list_combine(tmp_path):
    f = tmp_path / "animals.txt"
    f.write_text("K01\nK02\n", encoding="utf-8")
    assert feed.load_subjects("M05", f) == {"K01", "K02", "M05"}


def test_no_list_means_no_filter():
    assert feed.load_subjects(None, None) == set()


# -------------------------------------------------------------- the filtering
def pairs(*names):
    return [("COHORT", Path("/archive") / n) for n in names]


def test_an_empty_list_passes_everything_through():
    """The plain --cohort case. A filter nobody asked for must not apply."""
    given = pairs("20220908_K01_P1.mp4", "20220908_K09_P1.mp4")
    assert list(feed.select_subjects(given, set())) == given


def test_only_the_named_animals_come_through():
    given = pairs("20220908_K01_P1.mp4", "20220908_K09_P1.mp4",
                  "20220908_K03_P2.mp4")
    out = list(feed.select_subjects(given, {"K01", "K03"}))
    assert [p.name for _, p in out] == ["20220908_K01_P1.mp4",
                                        "20220908_K03_P2.mp4"]


def test_an_encoded_list_still_matches_lab_named_files():
    given = pairs("20220908_K01_P1.mp4", "20220908_K09_P1.mp4")
    out = list(feed.select_subjects(given, feed.load_subjects("ASPA1101")))
    assert [p.name for _, p in out] == ["20220908_K01_P1.mp4"]


def test_animals_that_matched_nothing_are_reported_back():
    """A typo in a list is indistinguishable from an animal with no data."""
    seen = set()
    requested = {"K01", "K99"}
    list(feed.select_subjects(pairs("20220908_K01_P1.mp4"), requested, seen))
    assert seen == {"K01"}
    assert requested - seen == {"K99"}, "the caller must be able to name K99"


def test_an_unreadable_filename_is_recorded_not_silently_dropped():
    unparsed = set()
    given = pairs("20220908_K01_P1.mp4", "mystery.mp4")
    out = list(feed.select_subjects(given, {"K01"}, set(), unparsed))
    assert len(out) == 1
    assert unparsed == {"mystery.mp4"}


def test_filtering_does_not_care_which_cohort_a_file_was_filed_under():
    """The point of the feature: animals spread across cohorts, nothing else."""
    given = [("OptG", Path("20220719_G07_E2.mp4")),
             ("OptG", Path("20220719_G22_E2.mp4")),
             ("M", Path("20230125_M01_P1.mp4")),
             ("M", Path("20230125_M09_P1.mp4"))]
    out = list(feed.select_subjects(given, {"G07", "M01"}))
    assert [p.name for _, p in out] == ["20220719_G07_E2.mp4",
                                        "20230125_M01_P1.mp4"]
