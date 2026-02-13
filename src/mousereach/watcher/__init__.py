"""
mousereach.watcher - Automated pipeline orchestration for MouseReach.

Monitors NAS for new collage videos, crops them into single-animal videos,
runs DLC inference, runs post-DLC analysis pipeline, and archives results.

Phase 1: Single-machine automation
Phase 2: Distributed multi-GPU processing with helper PCs

Requirements
============
The watcher should run on a machine with:
    1. Fast access to the NAS (direct-attached or wired connection)
    2. A GPU for DLC inference
    3. Access to the DLC model config.yaml

All paths are configured via ~/.mousereach/config.json (run mousereach-setup).

Folder Structure
================
The watcher expects this layout under PROCESSING_ROOT (typically NAS_DRIVE/! DLC Output):

    <PROCESSING_ROOT>/
    ├── DLC_Queue/             ← single-animal videos waiting for DLC
    ├── Processing/            ← post-DLC analysis (segments, reaches, outcomes)
    ├── Failed/                ← videos that hit errors
    └── Quarantine/            ← invalid filenames moved here

And under NAS_ROOT (= NAS_DRIVE / "! DLC Output"):

    <NAS_ROOT>/
    ├── Unanalyzed/
    │   ├── Multi-Animal/      ← collage videos arrive here (from filming PCs)
    │   └── Single_Animal/     ← cropped single-animal videos
    └── Analyzed/Sort/         ← final archived output

Configuration (in ~/.mousereach/config.json):
    nas_drive:                    root of NAS mount (e.g. "D:\\")
    processing_root:              parent of DLC_Queue/Processing/etc.
    watcher.dlc_config_path:      path to DLC model config.yaml
    watcher.dlc_gpu_device:       GPU index (default 0)
    watcher.poll_interval_seconds: how often to scan for new files
    watcher.stability_wait_seconds: how long a file must be unchanged before processing

Run mousereach-setup to configure interactively.
"""

__version__ = "1.0.0"
