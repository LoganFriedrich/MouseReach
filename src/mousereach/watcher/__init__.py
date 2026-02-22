"""
mousereach.watcher - Automated pipeline orchestration for MouseReach.

Two-machine architecture with role-aware orchestrators:

  DLC PC (mode="dlc_pc")
      GPU machine with direct-attached NAS.  Scans NAS for new collage
      videos, crops them to single-animal videos, runs DLC inference,
      and stages video+h5 to NAS DLC_Complete/ for the processing server.

  Processing Server (mode="processing_server")
      Server with fast local storage.  Watches DLC_Complete/ on NAS for
      new DLC outputs, copies them locally, runs segmentation + reach
      detection + outcome detection, and archives results back to NAS.

Both modes share the same codebase.  The mode is set in
~/.mousereach/config.json (watcher.mode) and is auto-detected by
mousereach-setup based on the machine's role profile.

Requirements
============
DLC PC:
    1. Fast access to the NAS (direct-attached or wired connection)
    2. A GPU for DLC inference
    3. Access to the DLC model config.yaml
    4. ffmpeg on PATH (for video cropping)

Processing Server:
    1. Access to NAS DLC_Complete/ staging folder (via network)
    2. Local storage for Processing/ folder (fast I/O for pipeline)
    3. No GPU required

All paths are configured via ~/.mousereach/config.json (run mousereach-setup).

Folder Structure
================
DLC PC (PROCESSING_ROOT on local drive, e.g. A:\\MouseReach_Pipeline):

    <PROCESSING_ROOT>/
    ├── DLC_Queue/             ← single-animal videos waiting for DLC
    ├── watcher_working/       ← temporary crop workspace
    └── watcher.db             ← state database

Processing Server (PROCESSING_ROOT on local drive, e.g. Y:\\...\\MouseReach_Pipeline):

    <PROCESSING_ROOT>/
    ├── Processing/            ← post-DLC analysis (segments, reaches, outcomes)
    ├── Failed/                ← videos that hit errors
    └── watcher.db             ← state database

NAS (shared between both):

    <NAS_ROOT>/
    ├── Unanalyzed/
    │   ├── Multi-Animal/      ← collage videos arrive here (from filming PCs)
    │   └── Single_Animal/     ← pre-cropped single-animal videos
    ├── DLC_Complete/          ← DLC PC stages finished videos here
    └── Analyzed/{project}/{cohort}/  ← final archived output (e.g. Connectome/CNT03/)

Configuration (in ~/.mousereach/config.json):
    nas_drive:                     root of NAS mount (e.g. "X:\\")
    processing_root:               parent of DLC_Queue/Processing/etc.
    watcher.mode:                  "dlc_pc" or "processing_server"
    watcher.dlc_config_path:       path to DLC model config.yaml (DLC PC only)
    watcher.dlc_gpu_device:        GPU index (DLC PC only, default 0)
    watcher.poll_interval_seconds: how often to scan for new files
    watcher.stability_wait_seconds: how long a file must be unchanged before processing
    watcher.max_local_pending:     max videos in local Processing/ before pausing intake

Run mousereach-setup to configure interactively.
"""

__version__ = "2.0.0"
