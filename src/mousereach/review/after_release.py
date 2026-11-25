"""What a person is told after releasing a reviewed video: how it gets back.

WHY ONE SENTENCE IN ONE PLACE: releasing a video (Clear in the review tool,
"Release finished" on the Review Queues tab, mousereach-review-release
--clear) only marks it finished. It goes back into the pipeline at the next
pass of the watcher's review return -- by itself within about two minutes
when the automatic watcher runs, but NOT AT ALL on a machine where nobody runs
the watcher (a lab that runs everything by hand). Those messages must not say
only "the watcher picks it up", which on such a machine is never true: the
person has to know to press Run again. Every release message ends with
this sentence. ASCII only (it is also printed).
"""

NEXT_STEP = ("It goes back into the pipeline at the next pass: by itself within "
             "about 2 minutes when the automatic watcher runs; otherwise press "
             "'Run the pipeline now' on the Run Pipeline tab (or run "
             "mousereach-watch --once).")

__all__ = ["NEXT_STEP"]
