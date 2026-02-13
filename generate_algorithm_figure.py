"""
Generate publication-ready algorithm flowchart for v5.3 reach detection.
Produces a multi-panel figure showing the complete detection pipeline.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ─── Color palette ───
C_INPUT    = '#E8F4FD'  # light blue - inputs
C_STAGE    = '#2C3E50'  # dark blue - stage headers
C_PROCESS  = '#F0F4C3'  # light yellow - processing steps
C_DECISION = '#FFECB3'  # light orange - decision nodes
C_YES      = '#C8E6C9'  # light green - yes/pass
C_NO       = '#FFCDD2'  # light red - no/fail
C_OUTPUT   = '#E1BEE7'  # light purple - outputs
C_ARROW    = '#455A64'  # dark gray - arrows
C_TEXT     = '#212121'  # near-black text
C_STAGE_TEXT = '#FFFFFF' # white text on dark headers
C_BORDER   = '#78909C'  # border color
C_LIGHT_BG = '#FAFAFA'  # panel background


def draw_box(ax, x, y, w, h, text, color, text_color=C_TEXT, fontsize=8,
             fontweight='normal', border_color=C_BORDER, alpha=1.0, style='round',
             linewidth=0.8):
    """Draw a rounded rectangle with centered text."""
    if style == 'round':
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor=border_color,
                             linewidth=linewidth, alpha=alpha, zorder=2)
    else:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="square,pad=0.01",
                             facecolor=color, edgecolor=border_color,
                             linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=fontweight, zorder=3,
            linespacing=1.3)
    return box


def draw_diamond(ax, x, y, w, h, text, color=C_DECISION, text_color=C_TEXT,
                 fontsize=7, border_color=C_BORDER):
    """Draw a diamond (decision) shape."""
    hw, hh = w/2, h/2
    verts = [(x, y + hh), (x + hw, y), (x, y - hh), (x - hw, y), (x, y + hh)]
    from matplotlib.patches import Polygon
    diamond = Polygon(verts, closed=True, facecolor=color, edgecolor=border_color,
                      linewidth=0.8, zorder=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, zorder=3, linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, label='', color=C_ARROW, fontsize=6.5,
          label_side='right', style='->', linewidth=1.0):
    """Draw an arrow between two points with optional label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=linewidth),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.015 if label_side == 'right' else -0.015
        if abs(x2 - x1) < 0.01:  # vertical arrow
            ax.text(mx + offset * 3, my, label, ha='left' if label_side == 'right' else 'right',
                    va='center', fontsize=fontsize, color=color, style='italic')
        else:  # horizontal arrow
            ax.text(mx, my + 0.012, label, ha='center', va='bottom',
                    fontsize=fontsize, color=color, style='italic')


def draw_panel_a(ax):
    """Panel A: Overall pipeline architecture."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, 'A. Pipeline Architecture', fontsize=11,
            fontweight='bold', ha='center', va='top', color=C_TEXT)

    # Input
    draw_box(ax, 0.5, 0.88, 0.55, 0.055, 'DLC Pose Estimation (.h5)\n+ Segment Boundaries (.json)',
             C_INPUT, fontsize=7.5)

    # Geometry
    draw_box(ax, 0.5, 0.78, 0.55, 0.045, 'Geometry Calibration\nslit position, ruler scale, extent reference',
             C_PROCESS, fontsize=7)
    arrow(ax, 0.5, 0.853, 0.5, 0.805)

    # Stage 1
    draw_box(ax, 0.5, 0.65, 0.65, 0.08,
             'STAGE 1: State Machine Detection\nFrame-by-frame: IDLE → ENGAGED → REACHING\n'
             'End triggers: disappear, retract, return-to-start\n'
             '(see Panel B for decision tree)',
             C_STAGE, C_STAGE_TEXT, fontsize=7, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.755, 0.5, 0.695)

    # Post-processing
    draw_box(ax, 0.5, 0.55, 0.55, 0.045,
             'Post-Processing Filters\nduration ≥ 4 frames, extent ≥ -15 px',
             C_PROCESS, fontsize=7)
    arrow(ax, 0.5, 0.607, 0.5, 0.575)

    # Stage 2
    draw_box(ax, 0.5, 0.43, 0.65, 0.08,
             'STAGE 2: Multi-Signal Splitting\nFor reaches > 25 frames:\n'
             'confidence dips + position returns + velocity\n'
             '(see Panel C for scoring)',
             C_STAGE, C_STAGE_TEXT, fontsize=7, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.525, 0.5, 0.475)

    # Stage 3
    draw_box(ax, 0.5, 0.30, 0.65, 0.08,
             'STAGE 3: ML Boundary Polishing\nXGBoost classifier → regressor (per boundary)\n'
             'conservative: only correct when P ≥ 0.8\n'
             '(see Panel D for two-stage gate)',
             C_STAGE, C_STAGE_TEXT, fontsize=7, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.387, 0.5, 0.345)

    # Output
    draw_box(ax, 0.5, 0.18, 0.55, 0.055,
             'Output: Reach Events\nstart / apex / end frames, extent, confidence',
             C_OUTPUT, fontsize=7.5)
    arrow(ax, 0.5, 0.257, 0.5, 0.21)

    # Version annotation
    ax.text(0.5, 0.08, 'v5.3: 98.8% boundary accuracy (matched), 94.5% overall\n'
            'Cross-validated generalization: 84.6%',
            ha='center', va='center', fontsize=6.5, style='italic', color='#616161')


def draw_panel_b(ax):
    """Panel B: State machine + retraction decision tree."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.97, 'B. Retraction Decision Tree', fontsize=11,
            fontweight='bold', ha='center', va='top', color=C_TEXT)

    ax.text(0.5, 0.92, 'Called when hand retraction or return-to-start is detected',
            ha='center', va='center', fontsize=7, color='#616161', style='italic')

    # Entry point
    draw_box(ax, 0.5, 0.86, 0.50, 0.035,
             'Retraction or return-to-start triggered', C_PROCESS, fontsize=7)

    # Node 1: BP switch
    draw_diamond(ax, 0.5, 0.76, 0.26, 0.09,
                 'Bodypart\nidentity\nswitch?', fontsize=7)
    arrow(ax, 0.5, 0.842, 0.5, 0.805)

    # BP switch → yes → spread check
    draw_diamond(ax, 0.82, 0.76, 0.20, 0.07,
                 'X-spread\n> 10 px?', fontsize=6.5)
    arrow(ax, 0.63, 0.76, 0.72, 0.76, 'YES', fontsize=6)

    # Spread > 10 → CONTINUE
    draw_box(ax, 0.82, 0.65, 0.16, 0.035,
             'CONTINUE\n(artifact)', C_NO, fontsize=6.5, fontweight='bold')
    arrow(ax, 0.82, 0.725, 0.82, 0.67, 'YES', fontsize=6)

    # Spread ≤ 10 → grace period note
    draw_box(ax, 0.82, 0.56, 0.16, 0.04,
             'BP switch grace\n(3 frames, reset\nreach_max_x)', '#E3F2FD', fontsize=5.5)
    arrow(ax, 0.92, 0.76, 0.95, 0.76, '', fontsize=6)
    ax.annotate('', xy=(0.82, 0.58), xytext=(0.95, 0.76),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.8), zorder=1)
    ax.text(0.96, 0.72, 'NO', fontsize=6, color=C_ARROW, style='italic')

    # Node 1 → No → Node 2
    draw_diamond(ax, 0.5, 0.62, 0.26, 0.09,
                 '≥ 2 hand points\nvisible AND not\nall retracted?', fontsize=6.5)
    arrow(ax, 0.5, 0.715, 0.5, 0.665, 'NO', fontsize=6, label_side='left')

    # Node 2 → yes → CONTINUE
    draw_box(ax, 0.18, 0.62, 0.16, 0.035,
             'CONTINUE\n(single-pt noise)', C_NO, fontsize=6.5, fontweight='bold')
    arrow(ax, 0.37, 0.62, 0.26, 0.62, 'YES', fontsize=6)

    # Node 2 → no → Node 3
    draw_diamond(ax, 0.5, 0.47, 0.26, 0.09,
                 'Hand stays\nretracted for\n2 frames?', fontsize=6.5)
    arrow(ax, 0.5, 0.575, 0.5, 0.515, 'NO', fontsize=6, label_side='left')

    # Node 3 → no → CONTINUE
    draw_box(ax, 0.18, 0.47, 0.16, 0.035,
             'CONTINUE\n(transient)', C_NO, fontsize=6.5, fontweight='bold')
    arrow(ax, 0.37, 0.47, 0.26, 0.47, 'NO', fontsize=6)

    # Node 3 → yes → END
    draw_box(ax, 0.5, 0.36, 0.20, 0.04,
             'END REACH', C_YES, fontsize=8, fontweight='bold')
    arrow(ax, 0.5, 0.425, 0.5, 0.38, 'YES', fontsize=6, label_side='left')

    # Retraction criteria box
    ax.text(0.5, 0.27, 'Retraction criteria:\n'
            '• Hand retracted > 50% of extension AND > 5 px\n'
            '• OR hand returned within 5 px of slit\n'
            '• Sustained over 2 consecutive frames',
            ha='center', va='center', fontsize=6, color='#424242',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', linewidth=0.5),
            linespacing=1.5)

    # State machine summary
    ax.text(0.5, 0.11, 'State Machine Summary:\n'
            'IDLE → nose within 25 px of slit → ENGAGED\n'
            'ENGAGED → hand visible ≥ 2 frames → REACHING\n'
            'REACHING → hand gone ≥ 3 frames OR retraction confirmed → IDLE',
            ha='center', va='center', fontsize=6, color='#424242',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                      edgecolor='#A5D6A7', linewidth=0.5),
            linespacing=1.5)


def draw_panel_c(ax):
    """Panel C: Multi-signal split scoring."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.97, 'C. Multi-Signal Split Scoring', fontsize=11,
            fontweight='bold', ha='center', va='top', color=C_TEXT)

    ax.text(0.5, 0.92, 'For reaches > 25 frames (95th percentile of GT duration)',
            ha='center', va='center', fontsize=7, color='#616161', style='italic')

    # Entry
    draw_box(ax, 0.5, 0.86, 0.45, 0.035,
             'Long reach detected (> 25 frames)', C_PROCESS, fontsize=7)

    # Two detectors
    draw_box(ax, 0.25, 0.77, 0.35, 0.05,
             'Detector A: Confidence Dips\nlikelihood: ≥0.5 → <0.35 → ≥0.5',
             '#E3F2FD', fontsize=6.5, border_color='#90CAF9')
    draw_box(ax, 0.75, 0.77, 0.35, 0.05,
             'Detector B: Position Returns\nextend ≥10px → retract 50% → re-extend ≥10px',
             '#FFF3E0', fontsize=6.5, border_color='#FFCC80')
    arrow(ax, 0.35, 0.842, 0.25, 0.80)
    arrow(ax, 0.65, 0.842, 0.75, 0.80)

    # Merge
    draw_box(ax, 0.5, 0.68, 0.30, 0.03,
             'Merge candidates, deduplicate', C_PROCESS, fontsize=7)
    arrow(ax, 0.25, 0.745, 0.40, 0.70)
    arrow(ax, 0.75, 0.745, 0.60, 0.70)

    # Scoring
    draw_box(ax, 0.5, 0.57, 0.70, 0.065,
             'Score each candidate (0.0 – 1.0)', C_STAGE, C_STAGE_TEXT,
             fontsize=8, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.665, 0.5, 0.605)

    # Three signals as horizontal boxes
    y_sig = 0.465
    # Signal 1
    draw_box(ax, 0.17, y_sig, 0.28, 0.065,
             'Confidence Dip\nWeight: 0.3\n(0.5 − min_conf) / 0.3',
             '#E3F2FD', fontsize=6, border_color='#90CAF9')
    # Signal 2
    draw_box(ax, 0.50, y_sig, 0.28, 0.065,
             'Position Return\nWeight: 0.4 (strongest)\nretraction / extension',
             '#FFF3E0', fontsize=6, border_color='#FFCC80')
    # Signal 3
    draw_box(ax, 0.83, y_sig, 0.28, 0.065,
             'Velocity Reversal\nWeight: 0.3\nneg → pos velocity',
             '#F3E5F5', fontsize=6, border_color='#CE93D8')

    arrow(ax, 0.30, 0.535, 0.17, 0.50)
    arrow(ax, 0.50, 0.535, 0.50, 0.50)
    arrow(ax, 0.70, 0.535, 0.83, 0.50)

    # Threshold
    draw_diamond(ax, 0.5, 0.35, 0.22, 0.08,
                 'Score\n≥ 0.5?', fontsize=7.5)
    arrow(ax, 0.5, 0.43, 0.5, 0.39)

    # No split
    draw_box(ax, 0.18, 0.35, 0.18, 0.035,
             'No split\n(keep original)', C_NO, fontsize=6.5)
    arrow(ax, 0.39, 0.35, 0.27, 0.35, 'NO', fontsize=6)

    # Yes → boundary placement
    draw_box(ax, 0.5, 0.24, 0.55, 0.055,
             'Place boundary at hand position minimum\n(frame where hand_x is closest to slit)',
             C_YES, fontsize=7)
    arrow(ax, 0.5, 0.31, 0.5, 0.27, 'YES', fontsize=6, label_side='left')

    # Output
    draw_box(ax, 0.5, 0.15, 0.40, 0.04,
             'Sub-reaches (each ≥ 4 frames)', C_OUTPUT, fontsize=7)
    arrow(ax, 0.5, 0.212, 0.5, 0.17)

    # Fallback note
    ax.text(0.5, 0.07, 'Boundary placement priority:\n'
            '1. Hand position minimum (closest to slit)\n'
            '2. Last positive velocity frame\n'
            '3. Confidence dip center (fallback)',
            ha='center', va='center', fontsize=6, color='#424242',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', linewidth=0.5),
            linespacing=1.5)


def draw_panel_d(ax):
    """Panel D: ML boundary polishing two-stage gate."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.97, 'D. ML Boundary Polishing', fontsize=11,
            fontweight='bold', ha='center', va='top', color=C_TEXT)

    ax.text(0.5, 0.92, 'Applied independently to each reach start and end boundary',
            ha='center', va='center', fontsize=7, color='#616161', style='italic')

    # Feature extraction
    draw_box(ax, 0.5, 0.85, 0.65, 0.045,
             'Extract 537 features: 13 per frame × 41 frames (±20) + 4 context',
             C_PROCESS, fontsize=7)

    # Stage 1
    draw_box(ax, 0.5, 0.74, 0.55, 0.06,
             'STAGE 1: XGBoost Classifier\nP(boundary needs correction)',
             C_STAGE, C_STAGE_TEXT, fontsize=7.5, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.827, 0.5, 0.775)

    # Decision 1
    draw_diamond(ax, 0.5, 0.63, 0.22, 0.08,
                 'P ≥ 0.8?', fontsize=8)
    arrow(ax, 0.5, 0.707, 0.5, 0.67)

    # No → preserve
    draw_box(ax, 0.18, 0.63, 0.20, 0.04,
             'NO CORRECTION\n(boundary preserved)',
             C_YES, fontsize=6.5, fontweight='bold')
    arrow(ax, 0.39, 0.63, 0.28, 0.63, 'NO', fontsize=6.5)

    # Yes → Stage 2
    draw_box(ax, 0.5, 0.50, 0.55, 0.06,
             'STAGE 2: XGBoost Regressor\nPredict offset (frames)',
             C_STAGE, C_STAGE_TEXT, fontsize=7.5, fontweight='bold', linewidth=1.2)
    arrow(ax, 0.5, 0.59, 0.5, 0.535, 'YES', fontsize=6.5, label_side='left')

    # Decision 2
    draw_diamond(ax, 0.5, 0.39, 0.24, 0.08,
                 '|offset|\n≥ 0.5?', fontsize=8)
    arrow(ax, 0.5, 0.467, 0.5, 0.43)

    # No → preserve
    draw_box(ax, 0.18, 0.39, 0.20, 0.04,
             'NO CORRECTION\n(negligible shift)',
             C_YES, fontsize=6.5, fontweight='bold')
    arrow(ax, 0.38, 0.39, 0.28, 0.39, 'NO', fontsize=6.5)

    # Yes → apply
    draw_box(ax, 0.5, 0.28, 0.55, 0.05,
             'Apply correction: new = old + round(offset)\nClipped to ±30 frames maximum',
             '#C8E6C9', fontsize=7, fontweight='bold', border_color='#66BB6A')
    arrow(ax, 0.5, 0.35, 0.5, 0.305, 'YES', fontsize=6.5, label_side='left')

    # Training info
    ax.text(0.5, 0.165, 'Training: 2,492 reaches × 23 videos\n'
            'XGBoost: 300 trees, depth 6, lr 0.06\n'
            '5-fold GroupKFold CV (grouped by video)\n'
            'Features: hand positions (×4), likelihoods (×4),\n'
            'nose pos/likelihood, mean hand, visibility count, velocity',
            ha='center', va='center', fontsize=6, color='#424242',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', linewidth=0.5),
            linespacing=1.5)

    # Design note
    ax.text(0.5, 0.055, 'Design principle: ~80% of boundaries are already correct.\n'
            'The classifier gate prevents corrupting correct boundaries\n'
            'with unnecessary regression predictions.',
            ha='center', va='center', fontsize=6, color='#616161', style='italic',
            linespacing=1.4)


def main():
    """Generate 4-panel algorithm figure."""
    fig = plt.figure(figsize=(16, 18))

    # Create 2×2 grid with some spacing
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.06,
                          left=0.03, right=0.97, top=0.96, bottom=0.02)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Draw panel backgrounds
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_facecolor(C_LIGHT_BG)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#E0E0E0')
            spine.set_linewidth(1.0)

    # Draw each panel
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

    # Main title
    fig.suptitle('Mouse Reach Detection Algorithm v5.3',
                 fontsize=16, fontweight='bold', y=0.99, color=C_TEXT)

    # Save
    out_path = r'Y:\2_Connectome\Behavior\MouseReach\v53_algorithm_flowchart.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {out_path}")

    # Also save PDF for publication
    pdf_path = out_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == '__main__':
    main()
