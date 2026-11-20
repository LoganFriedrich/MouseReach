"""Every release message says how a released video gets back into the
pipeline: by itself when the automatic watcher runs, otherwise by pressing
"Run the pipeline now" (review/after_release.py). They used to say only "the
watcher picks it up", never true on a machine where nobody runs the watcher.
"""
from pathlib import Path


def test_every_release_message_says_how_the_video_gets_back():
    from mousereach.review import after_release
    assert "Run the pipeline now" in after_release.NEXT_STEP
    after_release.NEXT_STEP.encode("ascii")
    here = Path(after_release.__file__).parent
    for name in ("release_cli.py", "queue_launcher_widget.py", "causal_review_widget.py"):
        assert "NEXT_STEP" in (here / name).read_text(encoding="utf-8"), name
