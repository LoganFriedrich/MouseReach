"""
Export middle frame from every M training video as a collage for tray type verification.
One frame per video, labeled with date_animal_tray, organized by date (row) x video (col).

Only processes training dates (before injury on Feb 20 2023).
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    print('ERROR: opencv-python not installed. Run: pip install opencv-python')
    sys.exit(1)

VIDEO_DIR = r'X:\! DLC Output\Analyzed\M\Single_Animal'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')

# Training dates only (before injury Feb 20)
INJURY_DATE = '20230220'


def get_middle_frame(video_path):
    """Extract the middle frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def main():
    # Find all M training videos
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))

    # Filter to training only (before injury)
    training_videos = []
    for v in videos:
        bn = os.path.basename(v)
        date = bn.split('_')[0]
        if date < INJURY_DATE:
            training_videos.append(v)

    print('Training videos found: %d' % len(training_videos))

    # Group by date
    by_date = {}
    for v in training_videos:
        bn = os.path.basename(v)
        date = bn.split('_')[0]
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(v)

    dates = sorted(by_date.keys())
    max_per_date = max(len(by_date[d]) for d in dates)

    print('Dates: %d, max videos per date: %d' % (len(dates), max_per_date))
    for d in dates:
        trays = set()
        for v in by_date[d]:
            bn = os.path.basename(v)
            tray = bn.split('_')[2].replace('.mp4', '')
            trays.add(tray)
        print('  %s: %d videos, trays: %s' % (d, len(by_date[d]), sorted(trays)))

    # Create collage: one row per date, columns for each video
    # Use a subset of videos per date to keep manageable:
    # Just take M01's videos for each date (one per tray type)
    print('\nExtracting frames (M01 only, one per tray per date)...')

    collage_data = []  # list of (date, tray, frame)
    for d in dates:
        m01_vids = [v for v in by_date[d] if '_M01_' in os.path.basename(v)]
        if not m01_vids:
            # Fall back to first animal available
            m01_vids = by_date[d][:4]
        for v in sorted(m01_vids):
            bn = os.path.basename(v)
            parts = bn.replace('.mp4', '').split('_')
            label = '%s\n%s_%s' % (parts[0], parts[1], parts[2])
            frame = get_middle_frame(v)
            if frame is not None:
                collage_data.append((label, frame))
                print('  %s' % bn)

    if not collage_data:
        print('No frames extracted.')
        return

    # Build collage grouped by date
    # Re-group for layout
    date_groups = {}
    for label, frame in collage_data:
        date = label.split('\n')[0]
        if date not in date_groups:
            date_groups[date] = []
        date_groups[date].append((label, frame))

    n_rows = len(date_groups)
    n_cols = max(len(v) for v in date_groups.values())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.patch.set_facecolor('white')
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, date in enumerate(sorted(date_groups.keys())):
        items = date_groups[date]
        for col, (label, frame) in enumerate(items):
            axes[row, col].imshow(frame)
            axes[row, col].set_title(label, fontsize=7, fontweight='bold')
            axes[row, col].axis('off')
        # Hide unused columns
        for col in range(len(items), n_cols):
            axes[row, col].axis('off')

    fig.suptitle('M Group Training Videos - Middle Frame per Video (M01)\n'
                 'Verify tray types: P=Pillar, E=Easy, F=Flat',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'M_tray_verification_collage.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved: %s' % out)


if __name__ == '__main__':
    main()
