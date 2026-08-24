#!/usr/bin/env python3
"""A video's subject id must resolve for every project, not just one.

`sync_file_to_database` writes a video's kinematics into connectome.db, and its
first step is to work out which animal the video is of. That step matched
`CNT(\\d{4})` and nothing else, so any non-CNT video returned None and the sync
gave up before it had even asked whether the subject was known.

ASPA is a project in exactly the way CNT is: mousedb models projects, ASPA videos
run through the same pipeline, and they are filed to Analyzed/ASPA. Hardcoding one
project's prefix quietly excluded a whole project's results. 61 ASPA videos on the
processing server had been analysed, filed, and silently skipped.

Parsing is delegated to `AnimalID.parse` -- the same decomposition the pipeline
uses to route a video to its project folder -- so the sync and the filing cannot
disagree about which animal a video belongs to.
"""

import pytest

from mousereach.sync.database import parse_subject_id


class TestEveryProjectResolves:

    @pytest.mark.parametrize("video,expected", [
        ("20250624_CNT0115_P2", "CNT_01_15"),
        ("20260724_CNT0501_P2", "CNT_05_01"),
        ("20220811_ASPA1011_P3", "ASPA_10_11"),
        ("20220811_ASPA0901_P1", "ASPA_09_01"),
        ("20220811_ASPA1016_P2", "ASPA_10_16"),
    ])
    def test_video_name_to_subject_id(self, video, expected):
        assert parse_subject_id(video) == expected

    def test_aspa_is_not_special_cased(self):
        """CNT and ASPA decompose by the same rule: {letters}{cohort}{subject}."""
        assert parse_subject_id("20220811_ASPA1011_P3") == "ASPA_10_11"
        assert parse_subject_id("20250624_CNT1011_P3") == "CNT_10_11"

    @pytest.mark.parametrize("suffix", [
        "_features", "_reaches", "_pellet_outcomes", "_segments"])
    def test_output_file_stems_resolve_too(self, suffix):
        """The sync is handed a features file, not a bare video name."""
        assert parse_subject_id(f"20220811_ASPA1011_P3{suffix}") == "ASPA_10_11"

    @pytest.mark.parametrize("already", ["CNT_01_15", "ASPA_10_11"])
    def test_database_form_passes_through(self, already):
        assert parse_subject_id(already) == already


class TestItStillRefusesWhatItShould:

    @pytest.mark.parametrize("junk", [
        "not_a_video",
        "20250624_P2",          # no animal field
        "",
        "20250624_CNT_P2",      # letters but no digits
    ])
    def test_unparseable_returns_none(self, junk):
        assert parse_subject_id(junk) is None

    def test_too_few_digits_is_not_an_animal(self):
        """AnimalID needs four digits: two cohort, two subject."""
        assert parse_subject_id("20250624_CNT01_P2") is None


class TestItAgreesWithTheFilingRoute:
    """The sync and the archive must not disagree about which animal a video is."""

    @pytest.mark.parametrize("animal", ["CNT0115", "ASPA1011", "CNT0501"])
    def test_same_decomposition_as_animalid(self, animal):
        from mousereach.config import AnimalID
        parsed = AnimalID.parse(animal)
        expected = "%s_%s_%s" % (
            parsed["experiment"], parsed["cohort"], parsed["subject"])
        assert parse_subject_id(f"20250624_{animal}_P2") == expected
