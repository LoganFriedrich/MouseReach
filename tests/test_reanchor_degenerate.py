"""Review re-anchoring in the presence of zero-length (degenerate) segments.

Overlap is normalised by the SHORTER span, so a 1-frame segment inside any
review span scores 100%, ties the true target, and the tie drops the human's
answer. Seen live 2026-09-01: a review spanning a full real segment was
dropped because a phantom segment (frames 35108-35108) tied it at 100% vs
100%. Degenerate spans must be excluded on both sides of the match."""

from mousereach.review.causal_review_io import index_review_by_segment


CURRENT = [
    {"segment_num": 18, "start_frame": 33269, "end_frame": 35107},
    {"segment_num": 19, "start_frame": 35108, "end_frame": 35108},  # phantom
    {"segment_num": 20, "start_frame": 35108, "end_frame": 36945},
]


def _doc(segments):
    return {"segments": segments}


def test_real_review_wins_despite_phantom_neighbor():
    doc = _doc([
        {"segment_num": 20, "segment_span": {"start": 35108, "end": 36945},
         "human": {"outcome": "displaced_sa"}},
    ])
    mapping, notes = index_review_by_segment(doc, CURRENT)
    assert 20 in mapping, notes
    assert mapping[20]["human"]["outcome"] == "displaced_sa"


def test_review_of_phantom_segment_is_dropped_with_a_note():
    # A review recorded against the phantom itself (inverted 2-frame span)
    # describes no footage and must die with the phantom, loudly.
    doc = _doc([
        {"segment_num": 19, "segment_span": {"start": 35108, "end": 35107},
         "human": {"outcome": "displaced_sa"}},
    ])
    mapping, notes = index_review_by_segment(doc, CURRENT)
    assert 19 not in mapping
    assert any("dropped" in n for n in notes)


def test_phantom_excluded_as_matching_target():
    mapping, notes = index_review_by_segment(
        _doc([{"segment_num": 18,
               "segment_span": {"start": 33269, "end": 35107},
               "human": {"outcome": "retrieved"}}]),
        CURRENT,
    )
    assert mapping.get(18, {}).get("human", {}).get("outcome") == "retrieved"
    assert any("excluded as a review-matching target" in n for n in notes)
