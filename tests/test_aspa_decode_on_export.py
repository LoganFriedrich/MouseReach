#!/usr/bin/env python3
"""ASPA results must come out named the way the lab names the animals.

ASPA animals are called `{letter}{subject}` -- J11. The pipeline needs a uniform
`{letters}{cohort:2d}{subject:2d}`, so ids are ENCODED on the way in by alphabet
position: J11 becomes ASPA1011. That is a deliberate design, and it works.

The other half was never built. `decode_animal` existed, was exported from
`mousereach.aspa`, and had no callers anywhere -- so every result surfaced as
ASPA1011 and whoever read it had to know the alphabet rule to get back to J11.
An encoding with no decode is a private joke between the code and itself.

`video_name` is deliberately NOT rewritten. It is the key that ties a row back to
the file it came from, and to the database. The decode adds columns for people to
read; it never replaces the identifier the machinery uses.
"""

import pytest

from mousereach.aspa import decode_animal, decode_video_stem, lab_animal_id


class TestLabAnimalId:

    @pytest.mark.parametrize("given,expected", [
        ("20220811_ASPA1011_P3", "J11"),
        ("ASPA1011", "J11"),
        ("ASPA_10_11", "J11"),                      # the database form
        ("20220811_ASPA0901_P1", "I01"),
        ("20220811_ASPA1011_P3_features", "J11"),   # an output file stem
        ("ASPA0401", "D01"),                        # the worked example on record
        ("ASPA1304", "M04"),                        # the other one
    ])
    def test_encoded_names_give_the_lab_id(self, given, expected):
        assert lab_animal_id(given) == expected

    @pytest.mark.parametrize("given", [
        "20250624_CNT0115_P2", "CNT0115", "", None, "not_a_video",
    ])
    def test_non_aspa_names_give_none(self, given):
        """So a caller can use it on every row of a mixed export."""
        assert lab_animal_id(given) is None


class TestDecodeVideoStem:

    def test_the_animal_field_is_decoded_in_place(self):
        assert decode_video_stem("20220811_ASPA1011_P3") == "20220811_J11_P3"

    def test_the_date_and_tray_survive(self):
        out = decode_video_stem("20220811_ASPA0901_P1")
        assert out.startswith("20220811_") and out.endswith("_P1")

    @pytest.mark.parametrize("given", [
        "20250624_CNT0115_P2",     # another project
        "",
        "no_animal_here",
    ])
    def test_anything_without_an_encoded_id_is_unchanged(self, given):
        assert decode_video_stem(given) == given

    def test_it_round_trips_with_the_encoder(self):
        from mousereach.aspa import encode_animal
        for lab in ("D01", "I01", "J11", "M04"):
            assert decode_animal(encode_animal(lab)) == lab


class TestExportsCarryTheLabId:
    """The decode has to be wired in, not merely available -- that was the bug."""

    def test_the_excel_export_adds_animal_and_session(self):
        from mousereach.export.core.exporter import results_to_dataframe
        df = results_to_dataframe([
            {"video_name": "20220811_ASPA1011_P3",
             "segments": [{"segment_num": 1, "outcome": "retrieved"}]},
        ])
        row = df.iloc[0]
        assert row["animal_id"] == "J11"
        assert row["session"] == "20220811_J11_P3"
        assert row["video_name"] == "20220811_ASPA1011_P3", (
            "the pipeline key must survive -- it is what ties this row to its file")

    def test_a_non_aspa_video_is_untouched(self):
        from mousereach.export.core.exporter import results_to_dataframe
        df = results_to_dataframe([
            {"video_name": "20250624_CNT0115_P2",
             "segments": [{"segment_num": 1, "outcome": "untouched"}]},
        ])
        row = df.iloc[0]
        assert row["animal_id"] is None
        assert row["session"] == "20250624_CNT0115_P2"

    def test_both_exporters_are_wired(self):
        import inspect
        from mousereach.export.core import exporter
        from mousereach.export import features_csv
        for mod in (exporter, features_csv):
            src = inspect.getsource(mod)
            assert "_lab_ids" in src, mod.__name__
            assert '"animal_id"' in src or "'animal_id'" in src, mod.__name__
