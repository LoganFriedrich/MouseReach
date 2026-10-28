"""A speed bump before saving a pellet answer that disagrees with BOTH the
algorithm and the tray sheet's hand score.

WHY
---
Answers that look like reviewer fatigue have been found among filed pellets:
the reviewer answered "retrieved" where both the algorithm and the operator's
tray hand score said displaced. A person watching the video outranks both of
those readings, so the review window never overrules or blocks the person. It
only asks them to look once more before saving the one kind of answer most
likely to be given on autopilot: one that two independent readings of the
same pellet both contradict.

    the answer agrees with the algorithm, or with the sheet -> saved as before
    the algorithm has no outcome, or there is no hand score  -> saved as before
    the answer disagrees with both                           -> a box asks:
        "Keep my answer"   -> saved and moved on, exactly as before, and the
                              record notes that the box was shown and the
                              answer kept (answers["double_disagreement"])
        "Take another look" (the default) -> nothing saved, the same question
                              stays open with the video where it was

This file holds the decision as plain functions (no Qt), so it is tested on
its own; the review window (causal_review_widget.py) only shows the box.

HOW "DISAGREES" IS DECIDED
--------------------------
Each comparison is made only at the detail the other side can actually see.

Against the tray sheet: the sheet scores a pellet 0 / 1 / 2 (untouched /
displaced / retrieved; watcher/hand_scores.py HAND_WORDS). It has ONE
"displaced", so it cannot tell displaced-in-the-SA from displaced-outside-
the-SA. The answer is turned into the sheet's scale with the same table the
before-filing check uses (hand_scores.ALGO_CLASS, where displaced_sa and
displaced_outside are both 1), so "displaced outside SA" against a sheet's
"displaced" is agreement.

Against the algorithm: the outcome detector cannot see a pellet leave the SA
(it never writes displaced_outside); a pellet that leaves unseen looks to it
exactly like one that was eaten, and it calls "retrieved". So a person may
answer "retrieved" for a pellet that left the SA, and a person's
"displaced outside SA" is what the algorithm calls "retrieved". Against the
algorithm, retrieved and displaced_outside therefore count as ONE answer
("the pellet left the SA"), set against displaced_sa (it stayed in the SA)
and untouched. Retrieved against the algorithm's displaced_sa is still a
disagreement: the algorithm saw the pellet land in the SA.

Anything else (abnormal, an undecided "triaged", no outcome) is not a pellet
outcome word, and no box is shown.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from mousereach.watcher.hand_scores import ALGO_CLASS, HAND_WORDS

# The four outcomes a person can give a pellet, in the words the box uses.
PLAIN_OUTCOME = {
    "retrieved": "retrieved",
    "displaced_sa": "displaced (in SA)",
    "displaced_outside": "displaced outside SA",
    "untouched": "untouched",
}

# Against the algorithm, at the detail it can see (see the module docstring):
# it cannot tell "left the SA unseen" from "eaten".
_ALGO_VIEW = {
    "untouched": "untouched",
    "displaced_sa": "stayed_in_sa",
    "retrieved": "left_sa",
    "displaced_outside": "left_sa",
}

# Name of the additive field in a saved record's "answers".
FIELD = "double_disagreement"


def disagrees_with_algorithm(human: Optional[str], algo: Optional[str]) -> bool:
    """True only when both are pellet outcomes and differ at the detail the
    algorithm can see. Missing or non-outcome words: False."""
    h, a = _ALGO_VIEW.get(human or ""), _ALGO_VIEW.get(algo or "")
    return h is not None and a is not None and h != a


def disagrees_with_sheet(human: Optional[str], hand: Optional[int]) -> bool:
    """True only when the answer is a pellet outcome, the hand score is one of
    the sheet's codes, and they differ on the sheet's own scale
    (hand_scores.ALGO_CLASS). Missing: False."""
    h = ALGO_CLASS.get(human or "")
    return h is not None and hand in HAND_WORDS and h != hand


def double_disagreement(human: Optional[str], algo: Optional[str],
                        hand: Optional[int]) -> bool:
    """Show the box? True only when the answer disagrees with the algorithm
    AND with the hand score. Either one missing, or agreeing: False."""
    return disagrees_with_algorithm(human, algo) and disagrees_with_sheet(human, hand)


def dialog_text(human: str, algo: str, hand: int) -> str:
    """What the box says, plainly: what each reading said, and the answer."""
    return ("The algorithm said %s. The tray sheet says %s. You answered %s.\n\n"
            "Both disagree with your answer. You may well be right -- you are "
            "watching the video. If you are sure, keep your answer; otherwise "
            "take another look before saving."
            % (PLAIN_OUTCOME.get(algo, algo), HAND_WORDS.get(hand, hand),
               PLAIN_OUTCOME.get(human, human)))


def kept_record(human: str, algo: str, hand: int, at: str) -> Dict[str, Any]:
    """The note saved with an answer whose reviewer saw the box and kept it."""
    return {"shown": True, "kept": True, "human_outcome": human,
            "algo_outcome": algo, "hand_score": hand,
            "hand_word": HAND_WORDS.get(hand), "at": at}


def already_kept(previous: Optional[Dict[str, Any]], human: Optional[str]) -> Optional[Dict]:
    """The earlier "kept" note for this segment when the reviewer already saw
    the box and kept this same answer (so it is not asked twice, e.g. "Save
    Segment + Next" on the last segment and then the whole-video save), else
    None."""
    note = ((previous or {}).get("answers") or {}).get(FIELD)
    if isinstance(note, dict) and note.get("kept") and note.get("human_outcome") == human:
        return note
    return None
