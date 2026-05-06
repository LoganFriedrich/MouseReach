"""
Export labeled frame sequences for suspect M swipes.

For each flagged swipe:
  1. Load the video frames for the swipe duration
  2. Load the DLC .h5 tracking data
  3. Overlay bodypart labels on each frame
  4. Save as a strip image (all frames side by side) + individual frames

This produces the visual evidence needed to determine if DLC tracking
is correct during these anomalous post-injury reaches.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as pe

VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Post-Processing'
OUTPUT_BASE = r'Y:\2_Connectome\Databases\figures\bodypart_tracking_quality\labeled_frames'
FRAMES_JSON = r'Y:\2_Connectome\Databases\figures\bodypart_tracking_quality\M_frames_to_review.json'

# DLC bodypart colors - distinct colors for each bodypart
BODYPART_COLORS = {
    'Nose': '#FF0000',
    'LeftEar': '#FF8800',
    'RightEar': '#FFFF00',
    'LeftPaw': '#00FF00',
    'RightPaw': '#00FFAA',
    'LeftHand': '#00AAFF',
    'RightHand': '#0000FF',
    'Tongue': '#FF00FF',
    'Pellet': '#FFFFFF',
    # Common alternative names
    'nose': '#FF0000',
    'left_ear': '#FF8800',
    'right_ear': '#FFFF00',
    'left_paw': '#00FF00',
    'right_paw': '#00FFAA',
    'left_hand': '#00AAFF',
    'right_hand': '#0000FF',
    'tongue': '#FF00FF',
    'pellet': '#FFFFFF',
}

# Fallback palette for unknown bodyparts
FALLBACK_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
    '#00FFFF', '#FF8800', '#8800FF', '#00FF88', '#FF0088',
    '#88FF00', '#0088FF', '#FF4444', '#44FF44', '#4444FF',
    '#FFAA00', '#AA00FF', '#00FFAA',
]

LIKELIHOOD_THRESHOLD = 0.3  # Don't draw points below this confidence


def find_dlc_h5(video_path):
    """Find the DLC .h5 file matching a video."""
    base = os.path.splitext(video_path)[0]
    # Pattern: videonameDLC_resnet50_*.h5
    pattern = base + 'DLC_*.h5'
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Try without exact match
    video_dir = os.path.dirname(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    pattern2 = os.path.join(video_dir, video_stem + 'DLC*.h5')
    matches2 = glob.glob(pattern2)
    if matches2:
        return matches2[0]
    return None


def load_dlc_data(h5_path):
    """Load DLC tracking data from .h5 file."""
    df = pd.read_hdf(h5_path)
    # Multi-level columns: (scorer, bodypart, coord)
    # Flatten to just bodypart and coord
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df[scorer].columns.get_level_values(0).unique()
    return df[scorer], bodyparts


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255)."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_labels_on_frame(frame, dlc_data, frame_idx, bodyparts):
    """Draw DLC labels on a single frame using OpenCV for speed."""
    annotated = frame.copy()

    color_map = {}
    fallback_idx = 0
    for bp in bodyparts:
        if bp in BODYPART_COLORS:
            color_map[bp] = hex_to_rgb(BODYPART_COLORS[bp])
        else:
            color_map[bp] = hex_to_rgb(FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])
            fallback_idx += 1

    for bp in bodyparts:
        try:
            x = dlc_data[bp]['x'].iloc[frame_idx]
            y = dlc_data[bp]['y'].iloc[frame_idx]
            likelihood = dlc_data[bp]['likelihood'].iloc[frame_idx]
        except (KeyError, IndexError):
            continue

        if pd.isna(x) or pd.isna(y) or likelihood < LIKELIHOOD_THRESHOLD:
            continue

        x_int, y_int = int(round(x)), int(round(y))
        color = color_map.get(bp, (255, 255, 255))
        # BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw filled circle
        radius = 4
        cv2.circle(annotated, (x_int, y_int), radius, color_bgr, -1)
        cv2.circle(annotated, (x_int, y_int), radius, (0, 0, 0), 1)

        # Draw label text
        label = bp.replace('_', '')[:6]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        # Black outline + colored text
        cv2.putText(annotated, label, (x_int + 6, y_int - 4),
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(annotated, label, (x_int + 6, y_int - 4),
                    font, font_scale, color_bgr, thickness)

    return annotated


def export_swipe_frames(video_path, dlc_data, bodyparts, start_frame, duration,
                        swipe_info, output_dir):
    """Export labeled frames for a single swipe."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('    [!] Cannot open video: %s' % video_path)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if start_frame >= total_frames:
        print('    [!] Start frame %d exceeds video length %d' % (start_frame, total_frames))
        cap.release()
        return False

    # Determine frames to extract
    # Show context: 5 frames before, the swipe, 5 frames after
    context = 5
    frame_start = max(0, start_frame - context)
    frame_end = min(total_frames - 1, start_frame + duration + context)

    # Extract and label frames
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    for fidx in range(frame_start, frame_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        labeled = draw_labels_on_frame(frame, dlc_data, fidx, bodyparts)

        # Add frame number overlay
        in_swipe = frame_start + context <= fidx < frame_start + context + duration
        border_color = (0, 255, 0) if in_swipe else (128, 128, 128)
        cv2.rectangle(labeled, (0, 0), (labeled.shape[1]-1, labeled.shape[0]-1),
                      border_color, 2)

        label_text = 'f%d' % fidx
        if in_swipe:
            label_text += ' [SWIPE]'
        cv2.putText(labeled, label_text, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        frames.append(labeled)

    cap.release()

    if not frames:
        print('    [!] No frames extracted')
        return False

    # Save individual frames
    swipe_dir = output_dir
    os.makedirs(swipe_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        fidx = frame_start + i
        out_path = os.path.join(swipe_dir, 'frame_%05d.png' % fidx)
        cv2.imwrite(out_path, frame)

    # Save strip (subsample if too many frames)
    max_strip = 15
    if len(frames) > max_strip:
        step = len(frames) / max_strip
        indices = [int(i * step) for i in range(max_strip)]
        strip_frames = [frames[i] for i in indices]
    else:
        strip_frames = frames

    # Build horizontal strip
    h = strip_frames[0].shape[0]
    w = strip_frames[0].shape[1]
    strip = np.zeros((h, w * len(strip_frames), 3), dtype=np.uint8)
    for i, frame in enumerate(strip_frames):
        strip[:, i*w:(i+1)*w, :] = frame

    strip_path = os.path.join(swipe_dir, '_strip.png')
    cv2.imwrite(strip_path, strip)

    # Also save a summary image with metadata
    fig, ax = plt.subplots(1, 1, figsize=(max(16, len(strip_frames) * 2), 4))
    ax.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    ax.set_title('%s | frame %d-%d (dur=%d) | PoF=%s Area=%s | %s | %s' % (
        os.path.basename(video_path),
        start_frame, start_frame + duration, duration,
        swipe_info.get('pof', '?'), swipe_info.get('area', '?'),
        swipe_info.get('reach_outcome', '?'),
        swipe_info.get('flags', '?'),
    ), fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    summary_path = os.path.join(swipe_dir, '_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print('    Saved %d frames + strip + summary to %s' % (len(frames), swipe_dir))
    return True


def make_legend(bodyparts, output_dir):
    """Create a bodypart color legend image."""
    fig, ax = plt.subplots(figsize=(3, max(2, len(bodyparts) * 0.3)))
    for i, bp in enumerate(bodyparts):
        color = BODYPART_COLORS.get(bp, '#888888')
        ax.add_patch(plt.Circle((0.1, len(bodyparts) - i - 0.5), 0.15,
                                color=color, ec='black', linewidth=0.5))
        ax.text(0.3, len(bodyparts) - i - 0.5, bp, va='center', fontsize=9)
    ax.set_xlim(-0.1, 2)
    ax.set_ylim(-0.5, len(bodyparts) + 0.5)
    ax.axis('off')
    ax.set_title('DLC Bodypart Legend', fontweight='bold', fontsize=10)
    plt.tight_layout()
    legend_path = os.path.join(output_dir, '_legend.png')
    plt.savefig(legend_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved legend: %s' % legend_path)


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Load the frames to review
    with open(FRAMES_JSON) as f:
        all_picks = json.load(f)

    print('Loaded %d swipes to review across %d videos' % (
        len(all_picks), len(set(p['video'] for p in all_picks))))

    # Limit to HIGH priority and top 3 per video to keep it manageable
    high_picks = [p for p in all_picks if p['priority'] == 'HIGH']
    print('HIGH priority: %d swipes' % len(high_picks))

    # Group by video
    by_video = {}
    for p in high_picks:
        v = p['video']
        if v not in by_video:
            by_video[v] = []
        by_video[v].append(p)

    # Process top 3 swipes per video, limit to first 10 videos
    videos_to_process = sorted(by_video.keys(),
                               key=lambda v: -max(float(p.get('area', 0) or 0) for p in by_video[v]))
    videos_to_process = videos_to_process[:10]

    dlc_cache = {}
    legend_made = False

    for video_path in videos_to_process:
        picks = sorted(by_video[video_path],
                       key=lambda p: -float(p.get('area', 0) or 0))[:3]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print('\n%s (%d swipes to export)' % (video_name, len(picks)))

        # Find DLC file
        h5_path = find_dlc_h5(video_path)
        if h5_path is None:
            print('  [!] No DLC .h5 file found for %s' % video_path)
            continue

        print('  DLC: %s' % os.path.basename(h5_path))

        # Load DLC data (cache it)
        if h5_path not in dlc_cache:
            dlc_data, bodyparts = load_dlc_data(h5_path)
            dlc_cache[h5_path] = (dlc_data, bodyparts)
            print('  Bodyparts: %s' % list(bodyparts))

            if not legend_made:
                make_legend(bodyparts, OUTPUT_BASE)
                legend_made = True
        else:
            dlc_data, bodyparts = dlc_cache[h5_path]

        for i, pick in enumerate(picks):
            start = pick.get('start_frame')
            dur = pick.get('duration_frames')
            if start is None or dur is None:
                print('  Swipe %d: missing frame info, skip' % i)
                continue

            swipe_label = '%s_f%d_d%d' % (video_name, start, dur)
            output_dir = os.path.join(OUTPUT_BASE, video_name, swipe_label)

            print('  Swipe %d: frame %d, dur %d, area=%s, outcome=%s' % (
                i, start, dur, pick.get('area', '?'), pick.get('reach_outcome', '?')))

            export_swipe_frames(
                video_path, dlc_data, bodyparts,
                start, dur, pick, output_dir
            )

    print('\n\nAll outputs: %s' % OUTPUT_BASE)
    print('Start with the _summary.png files for quick visual triage.')


if __name__ == '__main__':
    main()
