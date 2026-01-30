"""
Shared UI utilities for MouseReach napari widgets.

Provides:
- Help button with documentation popup
- Consistent styling
- Common dialog patterns
"""

from pathlib import Path
from qtpy.QtWidgets import (
    QPushButton, QDialog, QVBoxLayout, QTextBrowser,
    QHBoxLayout, QLabel, QWidget
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


class HelpDialog(QDialog):
    """Dialog showing help/documentation content."""

    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Use QTextBrowser for markdown-like rendering
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setHtml(self._markdown_to_html(content))
        layout.addWidget(self.text_browser)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _markdown_to_html(self, text: str) -> str:
        """Simple markdown-like conversion to HTML."""
        lines = text.split('\n')
        html_lines = []
        in_list = False

        for line in lines:
            # Headers
            if line.startswith('### '):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith('## '):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('# '):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            # Bold
            elif '**' in line:
                import re
                line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                html_lines.append(f"<p>{line}</p>")
            # List items
            elif line.strip().startswith('- '):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                if line.strip():
                    html_lines.append(f"<p>{line}</p>")

        if in_list:
            html_lines.append("</ul>")

        return '\n'.join(html_lines)


def create_help_button(help_text: str, title: str = "Help") -> QPushButton:
    """Create a small help button that shows a popup dialog."""
    btn = QPushButton("?")
    btn.setFixedSize(24, 24)
    btn.setStyleSheet("""
        QPushButton {
            font-weight: bold;
            font-size: 14px;
            border-radius: 12px;
            background-color: #555;
            color: white;
        }
        QPushButton:hover {
            background-color: #777;
        }
    """)
    btn.setToolTip("Click for help")

    def show_help():
        dialog = HelpDialog(title, help_text, btn)
        dialog.exec_()

    btn.clicked.connect(show_help)
    return btn


def create_header_with_help(title: str, help_text: str) -> QWidget:
    """Create a header row with title and help button."""
    widget = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(layout)

    label = QLabel(f"<b>{title}</b>")
    label.setStyleSheet("font-size: 14px; padding: 5px;")
    layout.addWidget(label)

    layout.addStretch()

    help_btn = create_help_button(help_text, title)
    layout.addWidget(help_btn)

    return widget


def style_dev_button(btn: QPushButton, text: str = None):
    """Style a button to indicate it's for development use."""
    if text:
        btn.setText(text)
    btn.setStyleSheet("""
        QPushButton {
            background-color: #444;
            color: #aaa;
            border: 1px dashed #666;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #555;
            color: #ccc;
        }
        QPushButton:disabled {
            background-color: #333;
            color: #666;
        }
    """)
    btn.setToolTip("For development/testing only - not part of normal workflow")


# Help text templates for each step

PIPELINE_HELP = """
# Step 2: Analysis Pipeline

## What This Does
Runs the complete analysis pipeline on DLC-tracked videos:
1. **Segmentation** - Find 21 pellet presentation boundaries
2. **Outcome Detection** - Classify R/D/M for each segment
3. **Reach Detection** - Find individual reach attempts

## How It Works
- Select your Processing folder (or any folder with DLC files)
- Click "Run Full Pipeline"
- Videos are processed through all stages
- Status is tracked in JSON metadata (files stay in place)

## File Flow
```
Processing/
  ↓ Segmentation → creates _segments.json
  ↓ Outcomes → creates _pellet_outcomes.json
  ↓ Reaches → creates _reaches.json
```

## After Running
- All files remain in Processing/
- Status is tracked via validation_status in JSON files:
  - "auto_approved" - passed quality checks
  - "needs_review" - flagged for review
  - "validated" - human-reviewed
- Use the dashboard to see which files need review
"""

STEP2_HELP = """
# Step 2: Segmentation Review

## What This Does
Reviews and corrects the automatically detected pellet presentation boundaries.

## Normal Workflow
1. **Load Video** - Select a video file, DLC data loads automatically
2. **Navigate** - Use buttons or keyboard to move between boundaries
3. **Adjust** - Press SPACE to set a boundary to the current frame
4. **Save Validated** - Saves results and updates validation status

## Keyboard Shortcuts
- **SPACE** - Set current boundary to this frame
- **N** - Next boundary
- **P** - Previous boundary
- **S** - Save validated
- **Left/Right** - Move 1 frame
- **Shift+Left/Right** - Move 10 frames

## What is a "Boundary"?
Each boundary marks where one pellet presentation ends and another begins.
Look for the frame where the SABL (left pellet marker) is centered in the slit.

## Ground Truth (Development Only)
The "Save as Ground Truth" button is for developers measuring algorithm accuracy.
Normal users should use "Save Validated" instead.
"""

STEP3_HELP = """
# Step 3b: Outcome Review

## What This Does
Reviews and corrects pellet outcome classifications:
- **R (Retrieved)** - Mouse successfully grabbed and ate the pellet
- **D (Displaced)** - Pellet was knocked away but not eaten
- **M (Missed/Untouched)** - No successful contact with pellet

## When to Use
Files with validation_status="needs_review" in their outcome JSON.
(Most files are auto-approved and don't need this step.)
Use the dashboard to find files needing review.

## Workflow
1. Load a video with outcomes needing review
2. Review each segment's classification
3. Correct any misclassified outcomes
4. Save validated results

## Ground Truth (Development Only)
Ground truth buttons are for developers measuring algorithm accuracy.
"""

STEP4_HELP = """
# Step 4b: Reach Review

## What This Does
Reviews and corrects detected reach attempts.

## When to Use
Files with validation_status="needs_review" in their reaches JSON.
(Most files are auto-approved and don't need this step.)
Use the dashboard to find files needing review.

## Workflow
1. Load a video with reaches needing review
2. Add/remove/adjust reach timing
3. Save validated results

## What is a "Reach"?
A reach is when the mouse extends its paw toward the pellet.
Each reach has:
- Start frame (paw begins moving)
- Apex frame (maximum extension)
- End frame (paw retracts)

## Ground Truth (Development Only)
Ground truth buttons are for developers testing algorithm accuracy.
"""

STEP5_HELP = """
# Step 5: Feature Extraction

## What This Does
Extracts 89 kinematic and contextual features from each reach:
- Reach extent, velocity, trajectory
- Head and body posture
- Attention score
- Data quality metrics

## Using the Viewer
1. Load a *_grasp_features.json file
2. Browse segments with N/P keys
3. View individual reaches with R1, R2, etc.
4. Export to CSV for statistical analysis

## Features Explained
See DATA_DICTIONARY.md for complete feature definitions.

## Interpreting Results
- **High straightness (>0.8)** = Direct reach path
- **High attention (>70%)** = Mouse was engaged
- **High tracking quality (>0.9)** = Reliable data
"""

STEP0_HELP = """
# Step 0: Video Preparation

## What This Does
Crops 8-camera collage videos into individual mouse videos.

## Input Format
Multi-animal collage with filename:
`20250704_CNT0101,CNT0205,..._P1.mkv`

The 8 animal IDs indicate which mouse is in each position (2x4 grid).

## Output
Individual .mp4 files:
- `20250704_CNT0101_P1.mp4`
- `20250704_CNT0205_P1.mp4`
- etc.

## Blank Positions
Use cohort "00" (e.g., CNT0001) for empty cage positions.

## Copy to Queue
Check "Copy to DLC Queue" to automatically stage files for Step 1.
"""

STEP1_HELP = """
# Step 1: DeepLabCut Processing

## What This Does
Runs DeepLabCut pose estimation to track body parts in videos.

## DLC GUI
Click "Launch DLC GUI" to:
- Create new tracking projects
- Label training frames
- Train models
- Evaluate performance

## Batch Analysis
1. Select a trained DLC project (config.yaml)
2. Select folder of videos to process
3. Click "Run DLC Analysis"
4. Output: *DLC*.h5 tracking files

## GPU vs CPU
- GPU (default): Much faster, requires NVIDIA GPU
- CPU: Slower but works on any computer

## Quality Check
After processing, use "Check DLC Quality" to identify:
- Low confidence tracking
- Missing body parts
- Videos that may need reprocessing
"""
