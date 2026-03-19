#!/usr/bin/env python
"""
Standalone DLC pose-estimation runner for the MouseReach pipeline.

Runs DeepLabCut inference independently of the MouseReach environment.
Intended to live in its own conda env with only DLC + dependencies.

Usage:
    python dlc_runner.py --scan                   # find & process all unprocessed videos
    python dlc_runner.py --watch                  # watch for new videos continuously
    python dlc_runner.py video1.mp4 video2.mp4    # process specific files
    python dlc_runner.py --scan --dry-run         # show what would be processed
"""

import argparse
import logging
import time
import re
from pathlib import Path

log = logging.getLogger("dlc_runner")

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = Path(r"A:\SPRT_Data")
DEFAULT_MODEL_DIR = Path(r"A:\DLC_Models")
WATCH_INTERVAL = 30  # seconds between watch-mode scans


def find_unprocessed(data_root: Path) -> list[Path]:
    """Return .mp4 files that lack a matching .h5 in the same directory."""
    unprocessed = []
    for mp4 in sorted(data_root.rglob("*.mp4")):
        h5 = mp4.with_suffix(".h5")
        # DLC appends model info to the filename, so also check for partial matches
        has_h5 = h5.exists() or any(mp4.parent.glob(f"{mp4.stem}*.h5"))
        if not has_h5:
            unprocessed.append(mp4)
    return unprocessed


def fix_dlc_config(config_path: Path) -> None:
    """Strip the video_sets block from config.yaml to avoid YAML parsing errors.

    DLC writes absolute Windows paths as unquoted YAML keys inside video_sets,
    which causes parse failures on reload. Since analyze_videos doesn't need
    video_sets, we simply remove the section.
    """
    text = config_path.read_text(encoding="utf-8")
    # Remove video_sets block: starts at "video_sets:" and runs until the next
    # top-level key (a line starting with a non-space, non-comment character).
    cleaned = re.sub(
        r"^video_sets:.*?(?=^\S|\Z)", "", text, flags=re.MULTILINE | re.DOTALL
    )
    if cleaned != text:
        config_path.write_text(cleaned, encoding="utf-8")
        log.info("Patched config.yaml: removed video_sets block")


def resolve_config(model_dir: Path) -> Path:
    """Find and return the DLC config.yaml inside model_dir."""
    candidates = list(model_dir.rglob("config.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No config.yaml found under {model_dir}")
    config = candidates[0]
    fix_dlc_config(config)
    return config


def run_dlc(videos: list[Path], config: Path, gpu: int, dry_run: bool) -> None:
    """Run DLC inference on a list of videos."""
    if not videos:
        log.info("Nothing to process.")
        return

    log.info("Videos queued: %d", len(videos))
    for v in videos:
        log.info("  %s", v)

    if dry_run:
        log.info("Dry run — skipping inference.")
        return

    import deeplabcut  # heavy import deferred so --dry-run stays fast

    deeplabcut.analyze_videos(
        str(config),
        [str(v) for v in videos],
        videotype=".mp4",
        gputouse=gpu,
        save_as_csv=True,
        destfolder=None,  # output next to source video
    )
    log.info("Inference complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone DLC runner for MouseReach")
    parser.add_argument("videos", nargs="*", type=Path, help="Specific .mp4 files to process")
    parser.add_argument("--scan", action="store_true", help="Scan data root for unprocessed videos")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for new videos")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root of SPRT data tree")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to DLC model directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.scan and not args.watch and not args.videos:
        parser.error("Provide .mp4 files, --scan, or --watch")

    config = resolve_config(args.model_dir)
    log.info("Using DLC config: %s", config)

    # ── direct file mode ────────────────────────────────────────────────────
    if args.videos:
        for v in args.videos:
            if not v.exists():
                parser.error(f"File not found: {v}")
        run_dlc(args.videos, config, args.gpu, args.dry_run)
        return

    # ── scan mode ───────────────────────────────────────────────────────────
    if args.scan:
        videos = find_unprocessed(args.data_root)
        run_dlc(videos, config, args.gpu, args.dry_run)
        return

    # ── watch mode ──────────────────────────────────────────────────────────
    log.info("Watch mode — polling every %ds. Ctrl+C to stop.", WATCH_INTERVAL)
    while True:
        videos = find_unprocessed(args.data_root)
        if videos:
            run_dlc(videos, config, args.gpu, args.dry_run)
        time.sleep(WATCH_INTERVAL)


if __name__ == "__main__":
    main()
