#!/usr/bin/env python3
"""
MouseReach All Tools Launcher
=============================

Launch all MouseReach tools in a single napari window with tabbed widgets.

ARCHITECTURE OVERVIEW
---------------------
This launcher creates a napari viewer and loads multiple widgets as dock tabs:

    ┌────────────────────────────────────────────────────────────────┐
    │  napari Viewer Window                                          │
    │  ┌──────────────────────┬─────────────────────────────────────┐│
    │  │                      │  Dock Area (right side)             ││
    │  │   Video Display      │  ┌─────────────────────────────────┐││
    │  │   (shared by all     │  │ [Dashboard][0-Crop][1-DLC]...   │││
    │  │    widgets)          │  │                                 │││
    │  │                      │  │   Currently Active Widget       │││
    │  │                      │  │                                 │││
    │  │                      │  └─────────────────────────────────┘││
    │  └──────────────────────┴─────────────────────────────────────┘│
    └────────────────────────────────────────────────────────────────┘

KEY DESIGN DECISIONS
--------------------
1. SHARED VIDEO: All widgets share ONE video layer in napari. When you load
   a video in any widget, all widgets can access it. This avoids loading the
   same video 3 times for different review steps.

2. STATE MANAGER: MouseReachStateManager coordinates between widgets:
   - Tracks which video is currently active
   - Broadcasts video changes to all widgets
   - Handles cross-widget communication (e.g., "segments validated" → refresh reaches)

3. TAB ORDER: Widgets load in pipeline order:
   Dashboard → Step 0 → Step 1 → Step 2 → Step 3 (Review) → Step 4 → GT Tool

4. TWO REVIEW TOOLS:
   - "3 - Review Tool" (review_mode=True): Edits algorithm JSON files directly
   - "GT Tool" (review_mode=False): Creates separate ground truth files

ENTRY POINTS
------------
This file provides these CLI commands (defined in pyproject.toml):
    mousereach          - Launch all tools
    MouseReach          - Alias for mousereach

USAGE
-----
    mousereach                              # Launch all tools
    mousereach path/to/video.mp4            # Launch with video pre-loaded
    mousereach --step 2b path/to/video.mp4  # Launch only Step 2b review
    mousereach --reviews                    # Launch only review tools (2b, 3b, 4b)

PIPELINE STEPS
--------------
    0   - Video Prep (crop 8-camera collages to single animals)
    1   - DLC Analysis (run DeepLabCut pose estimation)
    2   - Run Pipeline (batch: Segmentation → Reaches → Outcomes)
    3   - Review Tool (fix algorithm mistakes in JSON files)
    4   - View Features (visualize extracted kinematics)
    GT  - Ground Truth Tool (create evaluation datasets)
"""

import sys
from pathlib import Path


# =============================================================================
# WIDGET REGISTRY
# =============================================================================
# Each widget is defined as: (step_id, display_name, module_path, class_name)
#
# The step_id determines tab order in the UI:
#   dashboard < 0 < 1 < 2 < 3 (Review Tool, inserted dynamically) < 4 < GT
#
# Note: Review Tool (step 3) is NOT in this list - it's loaded separately
# because it uses GroundTruthWidget with review_mode=True.
# =============================================================================

WIDGETS = [
    # Dashboard: Overview of all videos and their pipeline status
    ("dashboard", "Pipeline Dashboard", "mousereach.dashboard.widget", "PipelineDashboard"),

    # Step 0: Crop 8-camera collage videos into single-animal videos
    ("0", "0 - Crop Videos", "mousereach.video_prep.widget", "VideoPrepWidget"),

    # Step 1: Run DeepLabCut pose estimation (typically on GPU machine)
    ("1", "1 - DLC Analysis", "mousereach.dlc.widget", "DLCWidget"),

    # Step 2: Batch processing - runs segmentation, reach detection, outcome detection
    ("2", "2 - Run Pipeline", "mousereach.pipeline.batch_widget", "UnifiedPipelineWidget"),

    # Step 4: Visualize extracted kinematic features (trajectories, velocities, etc.)
    # Note: Step 3 (Review Tool) is inserted dynamically before this
    ("4", "4 - View Features", "mousereach.kinematics.widget", "DataViewerWidget"),
]

# Review steps (2b, 3b, 4b) are internal tabs within the Review Tool
# They're listed here so --step 2b etc. works on the command line
REVIEW_STEPS = ["2b", "3b", "4b"]
ALL_STEPS = [w[0] for w in WIDGETS] + REVIEW_STEPS


def launch(video_path=None, steps=None):
    """
    Launch MouseReach tools in a napari viewer.

    This is the main entry point. It:
    1. Checks/updates the pipeline index (video database)
    2. Creates a napari viewer window
    3. Loads each widget as a dock tab
    4. Sets up cross-widget communication via StateManager
    5. Optionally auto-loads a video

    Args:
        video_path: Optional path to video file to auto-load on startup
        steps: List of step IDs to launch (default: all).
               Valid IDs: 'dashboard', '0', '1', '2', '2b', '3b', '4b', '4'
    """
    import os
    import time
    start_time = time.time()

    def tprint(msg, end="\n"):
        """Print with elapsed time timestamp."""
        elapsed = time.time() - start_time
        print(f"[{elapsed:5.1f}s] {msg}", end=end, flush=True)

    # === Configuration validation ===
    # Ensure paths are configured before proceeding
    tprint("Loading configuration...", end=" ")
    from mousereach.config import Paths
    print("OK")
    if not Paths.is_configured():
        print("\n" + "=" * 60)
        print("MouseReach Configuration Required")
        print("=" * 60)
        print("\nMouseReach needs to know where your pipeline folders are located.")
        print("\nPlease run:  mousereach-setup")
        print("\nThis only needs to be done once per machine.")
        print("=" * 60 + "\n")
        import sys
        sys.exit(1)

    # Validate paths exist (warn but continue if NAS is missing)
    problems = Paths.validate()
    critical_problems = [p for p in problems if "PROCESSING_ROOT" in p and "not configured" not in p]
    if critical_problems:
        print("\n" + "=" * 60)
        print("MouseReach Configuration Issues")
        print("=" * 60)
        for p in critical_problems:
            print(f"  - {p}")
        print("\nRun 'mousereach-setup' to fix.")
        print("=" * 60 + "\n")
        import sys
        sys.exit(1)

    # Non-critical warnings (e.g., NAS not configured)
    warnings = [p for p in problems if p not in critical_problems]
    if warnings:
        for w in warnings:
            tprint(f"Note: {w}")

    # === Network drive optimization ===
    # On network drives, napari's plugin discovery is extremely slow (18+ min!)
    # because it scans all packages over the network.
    #
    # CRITICAL: Disable napari's automatic plugin discovery
    # We don't need napari plugins - MouseReach loads its own widgets directly
    os.environ['NAPARI_DISABLE_PLUGIN_AUTOLOAD'] = '1'

    # Also try the npe2 skip (napari plugin engine v2)
    os.environ['NAPARI_DISABLE_PLUGINS'] = '1'

    # Move cache/settings to local drive to avoid network I/O
    if os.path.exists('C:/'):
        local_cache = os.path.expanduser('~/AppData/Local/napari')
        os.makedirs(local_cache, exist_ok=True)
        os.environ.setdefault('XDG_CONFIG_HOME', local_cache)
        os.environ.setdefault('XDG_CACHE_HOME', os.path.join(local_cache, 'cache'))
        os.environ.setdefault('NAPARI_SETTINGS', os.path.join(local_cache, 'settings.yaml'))

    # === Pre-check index BEFORE napari launches ===
    # This avoids hangs in the GUI by updating index in terminal first
    tprint("Checking pipeline index...", end=" ")
    try:
        from mousereach.index import PipelineIndex
        # Note: Paths already imported above for validation

        index = PipelineIndex()
        if index.index_path.exists():
            index.load()
            # Check if index is stale (any folders modified since last scan)
            stale_folders = index.check_stale_folders()
            if stale_folders:
                print(f"{len(stale_folders)} folders changed")
                tprint(f"Updating index for {len(stale_folders)} folders...", end=" ")
                for folder in stale_folders:
                    index.refresh_folder(folder)
                index.save()
                print("OK")
            else:
                print("OK (up to date)")
        else:
            # No index exists - create it
            print("not found, building...")
            tprint("Building pipeline index (first time, may take a minute)...", end=" ")
            if Paths.PROCESSING_ROOT.exists():
                index.rebuild(progress_callback=lambda c, t, m: None)
                index.save()
                print("OK")
            else:
                print("SKIP (no processing folder)")
    except Exception as e:
        print(f"SKIP ({e})")

    # Progress feedback for slow imports
    tprint("Loading napari...", end=" ")
    import napari
    print("OK")

    tprint("Loading state manager...", end=" ")
    from importlib import import_module
    from mousereach.state import MouseReachStateManager, connect_widget_to_state
    print("OK")

    if steps is None:
        steps = ALL_STEPS

    tprint("Creating viewer...", end=" ")
    viewer = napari.Viewer(title="MouseReach Tools")
    print("OK")

    # Create state manager for shared video and cross-widget communication
    state = MouseReachStateManager(viewer)

    widgets_loaded = []
    dock_widgets = []

    # Check if any review steps are requested - if so, use UnifiedReviewWidget
    review_steps_requested = [s for s in steps if s in REVIEW_STEPS]
    unified_review_loaded = False

    tprint("Loading widgets:")

    def load_review_tool():
        """Load the Review Tool widget."""
        nonlocal unified_review_loaded
        if unified_review_loaded or not review_steps_requested:
            return
        tprint("  3 - Review Tool...", end=" ")
        try:
            from mousereach.review.ground_truth_widget import GroundTruthWidget
            review_widget = GroundTruthWidget(viewer, review_mode=True)
            dw = viewer.window.add_dock_widget(review_widget, name="3 - Review Tool", area="right")
            widgets_loaded.append(("review", review_widget))
            dock_widgets.append(dw)
            unified_review_loaded = True
            connect_widget_to_state(review_widget, state, "review")
            print("OK")
        except ImportError as e:
            print(f"SKIP ({e})")
        except Exception as e:
            print(f"ERROR ({e})")

    # Load widgets in correct order (Review Tool comes after step 2, before step 4)
    for step_id, name, module_path, class_name in WIDGETS:
        if step_id not in steps:
            continue

        # Insert Review Tool (step 3) before step 4
        if step_id == "4":
            load_review_tool()

        tprint(f"  {name}...", end=" ")
        try:
            module = import_module(module_path)
            widget_class = getattr(module, class_name)
            widget = widget_class(viewer)
            dw = viewer.window.add_dock_widget(widget, name=name, area="right")
            widgets_loaded.append((step_id, widget))
            dock_widgets.append(dw)

            # Register widget with state manager
            connect_widget_to_state(widget, state, step_id)

            print("OK")
        except ImportError as e:
            print(f"SKIP ({e})")
        except Exception as e:
            print(f"ERROR ({e})")

    # If step 4 wasn't loaded, load Review Tool at the end
    load_review_tool()

    # Load GT Tool (for researchers creating ground truth files)
    tprint("  GT Tool (for researchers)...", end=" ")
    try:
        from mousereach.review.ground_truth_widget import GroundTruthWidget
        gt_widget = GroundTruthWidget(viewer, review_mode=False)
        dw = viewer.window.add_dock_widget(gt_widget, name="GT Tool", area="right")
        widgets_loaded.append(("gt", gt_widget))
        dock_widgets.append(dw)
        connect_widget_to_state(gt_widget, state, "gt")
        print("OK")
    except ImportError as e:
        print(f"SKIP ({e})")
    except Exception as e:
        print(f"ERROR ({e})")

    # Load Performance & Feedback dashboard (tracks algorithm improvement over time)
    tprint("  Performance & Feedback...", end=" ")
    try:
        from mousereach.performance.feedback_dashboard import PerformanceViewerWidget
        perf_widget = PerformanceViewerWidget(viewer)
        dw = viewer.window.add_dock_widget(perf_widget, name="Performance", area="right")
        widgets_loaded.append(("performance", perf_widget))
        dock_widgets.append(dw)
        print("OK")
    except ImportError as e:
        print(f"SKIP ({e})")
    except Exception as e:
        print(f"ERROR ({e})")

    # Tabify all dock widgets together (stack as tabs)
    if len(dock_widgets) > 1:
        main_window = viewer.window._qt_window
        for i in range(1, len(dock_widgets)):
            main_window.tabifyDockWidget(dock_widgets[0], dock_widgets[i])
        # Raise the first tab
        dock_widgets[0].raise_()

    # Connect tab change detection - when user switches tabs, auto-load video into new tab
    # Note: UnifiedReviewWidget handles its own internal tab switching
    def on_tab_changed(dock_widget, visible):
        """When a dock widget becomes visible (tab switched), load video data into it."""
        if not visible:
            return

        # Find which widget this dock belongs to
        for step_id, widget in widgets_loaded:
            # Check if this widget is inside the activated dock
            if dock_widget.widget() is widget:
                print(f"[MouseReach] Tab switched to {step_id}")

                # Skip widgets that handle their own video loading (dashboard, prep, unified review)
                if step_id in ["dashboard", "0", "1", "2", "review"]:
                    return

                # Only feature viewer (step 5) needs external video loading
                if step_id == "5":
                    widget_has_video = (
                        hasattr(widget, 'video_layer') and
                        widget.video_layer is not None and
                        hasattr(widget, 'video_path') and
                        widget.video_path is not None
                    )

                    if not widget_has_video and state.active_video_path:
                        print(f"[MouseReach]   Loading active video into {step_id}: {state.active_video_path.name}")
                        state.load_active_video_into_widget(step_id)
                break

    # Connect visibility changed signal for each dock widget
    for dw in dock_widgets:
        dw.visibilityChanged.connect(lambda visible, d=dw: on_tab_changed(d, visible))

    if not widgets_loaded:
        print("\nNo tools could be loaded!")
        print("Make sure MouseReach is installed:")
        print("  pip install -e /path/to/MouseReach")
        return

    print(f"\nLoaded {len(widgets_loaded)} tool(s)")

    # Print keyboard shortcuts
    print("\n" + "=" * 50)
    print("KEYBOARD SHORTCUTS")
    print("=" * 50)
    print("\nStep 2b (Boundaries):")
    print("  ENTER - Accept boundary as-is (algorithm correct)")
    print("  SPACE - Set boundary to current frame")
    print("  N/P   - Next/Previous boundary")
    print("  S     - Save validated")
    print("\nStep 3b (Outcomes):")
    print("  ENTER - Accept outcome as-is (algorithm correct)")
    print("  SPACE - Play/pause")
    print("  N/P   - Next/Previous segment")
    print("  R/D/O/U - Set outcome (Retrieved/Displaced SA/Outside/Untouched)")
    print("\nStep 4b (Reaches):")
    print("  ENTER - Accept reach as-is (algorithm correct)")
    print("  SPACE - Play/pause")
    print("  N/P   - Next/Previous reach")
    print("  S/E   - Set reach Start/End")
    print("  A     - Add new reach")
    print("  DEL   - Delete reach")
    print("\nStep 5 (Features):")
    print("  SPACE - Play/pause")
    print("  1-5   - Speed (1x/2x/4x/8x/16x)")
    print("  J     - Jump to interaction")
    print("\nNavigation (all):")
    print("  Left/Right - Step 1 frame")
    print("  Shift+Left/Right - Step 10 frames")
    print("=" * 50)

    # Print fast launch tip
    print("\nTIP: For faster loading, use individual tools:")
    print("  mousereach-segment-review video.mp4      # Boundary review only (~15s)")
    print("  mousereach-review-pellet-outcomes        # Outcome review only (~15s)")
    print("  mousereach-review-reaches                # Reach review only (~15s)")
    print("  mousereach --reviews video.mp4     # All review tools (~25s)")

    # Auto-load video if provided
    if video_path:
        video_path = Path(video_path)
        if video_path.exists():
            print(f"\nLoading video: {video_path.name}")
            from qtpy.QtCore import QTimer

            # If unified review widget is loaded, use its internal loading
            if unified_review_loaded:
                # Find the review widget and load video into it
                for step_id, widget in widgets_loaded:
                    if step_id == "review":
                        QTimer.singleShot(500, lambda w=widget: w._load_video_from_path(video_path))
                        break
            else:
                # Use state manager to load video (for non-review widgets)
                QTimer.singleShot(500, lambda: state.load_video(video_path))
        else:
            print(f"\nWarning: Video not found: {video_path}")

    napari.run()


def main():
    """Parse args and launch."""
    # Immediate feedback - shows before ANY heavy imports
    # This confirms the command is running (network drive imports can take 10-30s)
    print("\nMouseReach Tools Launcher v2.3.0")
    print("Loading... (first launch may take 30-60s on network drives)")
    print("", flush=True)  # Force flush to terminal immediately

    import argparse

    parser = argparse.ArgumentParser(
        description="Launch MouseReach tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('video', nargs='?', type=Path,
                        help="Optional video file to auto-load")
    parser.add_argument('--step', '-s', type=str, action='append',
                        choices=['dashboard', '0', '1', '2', '2b', '3b', '4b', '5'],
                        help="Launch specific step(s). Can be used multiple times.")
    parser.add_argument('--reviews', '-r', action='store_true',
                        help="Launch only review tools (2b, 3b, 4b)")

    args = parser.parse_args()

    # Determine which steps to load
    if args.reviews:
        steps = REVIEW_STEPS
    elif args.step:
        steps = args.step
    else:
        steps = None  # All steps

    launch(video_path=args.video, steps=steps)


if __name__ == "__main__":
    main()
