"""Blank grid positions (cohort 00) must be detected for EVERY prefix length.

The old positional check matched only 3-letter prefixes, so 4-letter-prefix
blanks were cropped, posed and analysed as animals, and their collages could
never retire (the phantom counted as expected offspring but got retired, so
all_complete stayed false forever).
"""

from mousereach.video_prep.core.cropper import is_blank_animal


def test_blank_detection_handles_four_letter_prefixes():
    assert is_blank_animal("CNT0001")    # 3-letter, unchanged
    assert is_blank_animal("ASPA0001")   # was False -- the bug
    assert is_blank_animal("ENCR0001")   # was False -- the bug
    assert not is_blank_animal("ASPA1001")
    assert not is_blank_animal("CNT0101")
    assert not is_blank_animal("")       # unparseable -> not blank
