"""ASPA historical-corpus support.

ASPA behaviour videos predate MouseReach and were analysed with the old ASPA
algorithms. Re-analysing them with MouseReach means bringing the multi-animal
collages into the normal pipeline WITHOUT touching the original videos or the old
analyses -- they are read-only sources.

The one obstacle is identity: ASPA names animals ``D01`` / ``H36``, while the
pipeline and everything downstream require ``{letters}{cohort:2d}{subject:2d}``.
See :mod:`mousereach.aspa.identity` -- ids are encoded on the way in and decoded
on the way out, so the four-digit invariant holds throughout the pipeline.

ASCII-only console output (Windows cp1252).
"""
from .identity import (  # noqa: F401
    ASPA_PREFIX,
    cohort_letter,
    cohort_number,
    decode_animal,
    encode_animal,
    encode_collage_stem,
    is_encoded,
)
