"""ASPA <-> pipeline animal-id encoding.

Why this exists
---------------
ASPA names an animal ``{cohort letter}{subject:2d}`` -- ``D01``, ``H36``, ``M04``.
The pipeline (and everything downstream of it: mousedb, connectome.db, the
project-wide ``CNT_XX_YY`` key) requires ``{letters}{cohort:2d}{subject:2d}``, so
``AnimalID.parse('D01')`` fails with "Need at least 4 digits" -- the cohort is
carried by the letter, and the number component is silent.

Rather than teach every consumer a second id shape, ASPA ids are ENCODED on the
way into the pipeline and DECODED on the way back out. The four-digit invariant
holds everywhere in between, so nothing downstream changes.

The mapping is a RULE, not a lookup
-----------------------------------
Cohort number = the cohort letter's position in the alphabet (A=1 ... Z=26). So
``D01 -> ASPA0401``, ``H36 -> ASPA0836``, ``M04 -> ASPA1304``.

That matters because it makes the encoding self-describing: if the mapping table
were ever lost, it reconstructs itself from the rule. The table is still written
down and versioned (see the import tool), but nothing depends on it.

Using alphabet position rather than numbering the in-scope cohorts sequentially
means there is no ASPA cohort 01-03 in the data (A/B/C are excluded) -- accepted
deliberately, because a dense numbering would permanently disagree with the
letters the lab actually uses.

Blank positions
---------------
ASPA collages sometimes carry all-numeric position tokens (``001``, ``002``) where
no animal was present. These follow the convention already used for ``CNT0001``:
cohort ``00`` means blank/skip.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import re
import string
from typing import Optional

ASPA_PREFIX = "ASPA"

# {cohort letter}{subject digits} -- e.g. D01, H36, M4
_ASPA_ID = re.compile(r"^([A-Za-z])(\d{1,2})$")
# Encoded form: ASPA + cohort(2) + subject(2)
_ENCODED_ID = re.compile(r"^" + ASPA_PREFIX + r"(\d{2})(\d{2})$")
# All-numeric token = an empty position in the collage
_BLANK_ID = re.compile(r"^\d{1,3}$")

_BLANK_COHORT = "00"


def cohort_number(letter: str) -> int:
    """Cohort letter -> its position in the alphabet (A=1 ... Z=26)."""
    idx = string.ascii_uppercase.find(letter.upper())
    if idx < 0:
        raise ValueError(f"not a cohort letter: {letter!r}")
    return idx + 1


def cohort_letter(number: int) -> str:
    """Inverse of :func:`cohort_number`."""
    if not 1 <= number <= 26:
        raise ValueError(f"cohort number out of range: {number}")
    return string.ascii_uppercase[number - 1]


def encode_animal(aspa_id: str) -> str:
    """``'D01' -> 'ASPA0401'``. Blank positions -> ``ASPA00{n:02d}``.

    Raises ValueError on anything that is not a recognisable ASPA id, so a
    malformed token in a collage name fails loudly at import rather than
    silently becoming a plausible-looking animal.
    """
    tok = (aspa_id or "").strip()
    m = _ASPA_ID.match(tok)
    if m:
        letter, subject = m.group(1), int(m.group(2))
        return f"{ASPA_PREFIX}{cohort_number(letter):02d}{subject:02d}"
    b = _BLANK_ID.match(tok)
    if b:
        return f"{ASPA_PREFIX}{_BLANK_COHORT}{int(tok) % 100:02d}"
    raise ValueError(f"unrecognised ASPA animal id: {aspa_id!r}")


def decode_animal(encoded: str) -> str:
    """``'ASPA0401' -> 'D01'``. Blank ids decode back to their numeric token."""
    m = _ENCODED_ID.match((encoded or "").strip())
    if not m:
        raise ValueError(f"not an encoded ASPA id: {encoded!r}")
    coh, subj = int(m.group(1)), int(m.group(2))
    if coh == 0:
        return f"{subj:03d}"
    return f"{cohort_letter(coh)}{subj:02d}"


def is_encoded(animal_id: str) -> bool:
    """True if this id is an encoded ASPA animal (cheap check for the decode path)."""
    return bool(_ENCODED_ID.match((animal_id or "").strip()))


def lab_animal_id(name: str) -> Optional[str]:
    """The lab's own animal id for anything carrying an encoded ASPA one.

    ``'20220811_ASPA1011_P3' -> 'J11'``, ``'ASPA1011' -> 'J11'``,
    ``'ASPA_10_11' -> 'J11'``. Returns None for a non-ASPA name (a CNT video,
    say), so a caller can use it unconditionally.

    Encoding exists so the pipeline sees one id shape everywhere; it is not what
    anyone in the lab calls the animal. Nothing decoded on the way out, so every
    result surfaced as ASPA1011 and a person reading it had to know the alphabet
    rule to get back to J11. This is that missing half.
    """
    tok = (name or "").strip()
    m = _ENCODED_ID.match(tok)
    if m:
        return decode_animal(tok)
    # database form: ASPA_10_11
    m = re.match(r"^" + ASPA_PREFIX + r"_(\d{2})_(\d{2})$", tok)
    if m:
        return decode_animal("%s%s%s" % (ASPA_PREFIX, m.group(1), m.group(2)))
    # a video/file stem: {date}_{animal}_{tray}{pos}, plus any suffix
    for part in tok.split("_"):
        if _ENCODED_ID.match(part):
            return decode_animal(part)
    return None


def decode_video_stem(stem: str) -> str:
    """``'20220811_ASPA1011_P3' -> '20220811_J11_P3'``.

    The inverse of the encoding applied at import, for one single-animal video.
    A stem with no encoded ASPA id is returned unchanged, so this is safe to call
    on every row of a mixed export.

    The stem is NOT rewritten anywhere the pipeline uses it as a key -- file
    names, database rows and manifests all keep the encoded form, because that is
    what ties a result back to the file it came from. This is for the human-facing
    end: a column someone reads.
    """
    tok = (stem or "").strip()
    if not tok:
        return stem
    parts = tok.split("_")
    out, changed = [], False
    for part in parts:
        if _ENCODED_ID.match(part):
            out.append(decode_animal(part))
            changed = True
        else:
            out.append(part)
    return "_".join(out) if changed else stem


def encode_collage_stem(stem: str) -> Optional[str]:
    """Encode a full ASPA collage stem into pipeline form.

    ``20220217_D01,D02,...,D08_P1`` -> ``20220217_ASPA0401,ASPA0402,...,ASPA0408_P1``

    Returns None if the stem does not have the expected
    ``{date}_{animals}_{tray}{position}`` shape -- the caller reports those rather
    than guessing. Trailing junk (" uncropped", " (2)", LosslessCut "-seg" exports)
    is NOT handled here: source selection happens before encoding, so only the
    chosen file for a session ever reaches this function.
    """
    m = re.match(r"^(\d{8})_(.+)_([A-Za-z]\d+)$", (stem or "").strip())
    if not m:
        return None
    date, animals, tray = m.group(1), m.group(2), m.group(3)
    try:
        encoded = [encode_animal(a) for a in animals.split(",")]
    except ValueError:
        return None
    return f"{date}_{','.join(encoded)}_{tray.upper()}"
