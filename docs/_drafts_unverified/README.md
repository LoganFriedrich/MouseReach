# Draft subsystem documents -- NOT verified, do not rely on these

Seven documents were written on 2026-08-21, one per subsystem, each read from the
source. Every one was then given to a second reviewer whose only job was to find
statements the code contradicts. **All seven came back with errors: 60 disputed
claims in total.**

Spot-checked and confirmed wrong:

- Line citations point past the ends of the files they cite. `assignment/run.py`
  is 79 lines and was cited at 144; `reach/v8/features.py` is 178 and was cited
  at 265; `review/queue_index.py` is 186 and was cited at 186-190.
- ALGO_1 states `segment_video_robust` is "never called by anything". It is
  called at `segmentation/review_widget.py:870`.

They are kept here rather than deleted because the prose is a useful starting
point and re-deriving it costs real time. They are kept OUT of `docs/` because a
document that is wrong is worse than no document: it gets trusted, and the whole
reason this work exists is that nobody could tell what the pipeline really does.

`DISPUTED_CLAIMS.json` holds every disputed claim with the evidence against it.
The next pass should correct these against the source and only then move the
documents into `docs/` and add them to `DOC_COVERAGE.json`.

Note the checker is not automatically right either -- it is a second opinion, not
an oracle. Each dispute needs settling against the code.
