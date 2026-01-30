"""
geometry.py - Per-segment geometric calculations

All measurements normalized to "ruler" (SABL-SABR distance = 9mm).
Camera zoom varies, so pixel values are not transferable between videos.

Key geometry: 55° isoceles triangle with apex at pillar, base at SABL-SABR.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple


# Physical constants (mm) - universal
PHYSICAL_RULER_MM = 9.0
PILLAR_DIAMETER_MM = 4.125
PELLET_DIAMETER_MM = 2.5
APEX_ANGLE_DEG = 55.0


@dataclass
class SegmentGeometry:
    """Per-segment geometric reference values"""
    segment_num: int
    start_frame: int
    end_frame: int
    
    # SA corner positions (pixels, median during stable period)
    sabl_x: float
    sabl_y: float
    sabr_x: float
    sabr_y: float
    
    # Computed values
    ruler_pixels: float          # SABL-SABR distance in pixels
    ideal_pillar_x: float        # Expected pillar center from geometry
    ideal_pillar_y: float
    mm_per_pixel: float          # Scale factor
    
    def pixels_to_ruler(self, pixels: float) -> float:
        """Convert pixel distance to ruler units"""
        return pixels / self.ruler_pixels if self.ruler_pixels > 0 else 0
    
    def pixels_to_mm(self, pixels: float) -> float:
        """Convert pixels to millimeters"""
        return pixels * self.mm_per_pixel
    
    def distance_to_pillar(self, x: float, y: float) -> float:
        """Distance from point to ideal pillar center (pixels)"""
        return np.sqrt((x - self.ideal_pillar_x)**2 + (y - self.ideal_pillar_y)**2)
    
    def distance_to_pillar_ruler(self, x: float, y: float) -> float:
        """Distance from point to pillar (ruler units)"""
        return self.distance_to_pillar(x, y) / self.ruler_pixels


def compute_ideal_pillar(
    sabl_x: float, sabl_y: float,
    sabr_x: float, sabr_y: float
) -> Tuple[float, float, float]:
    """
    Compute ideal pillar position from 55° triangle geometry.
    
    Returns: (pillar_x, pillar_y, ruler_pixels)
    """
    ruler = np.sqrt((sabr_x - sabl_x)**2 + (sabr_y - sabl_y)**2)
    
    mid_x = (sabl_x + sabr_x) / 2
    mid_y = (sabl_y + sabr_y) / 2
    
    # Perpendicular toward pillar (smaller Y in image coords)
    dx = sabr_x - sabl_x
    dy = sabr_y - sabl_y
    perp_x = dy / ruler
    perp_y = -dx / ruler
    
    # Height from 55° triangle
    half_apex_rad = math.radians(APEX_ANGLE_DEG / 2)
    height = (ruler / 2) / math.tan(half_apex_rad)
    
    pillar_x = mid_x + perp_x * height
    pillar_y = mid_y + perp_y * height
    
    return pillar_x, pillar_y, ruler


def compute_segment_geometry(
    df,  # pandas DataFrame with DLC data
    seg_start: int,
    seg_end: int,
    segment_num: int,
    stable_margin: int = 60
) -> SegmentGeometry:
    """Compute geometric values for one segment."""
    seg_df = df.iloc[seg_start:seg_end]
    n = len(seg_df)
    
    stable_start = min(stable_margin, n // 4)
    stable_end = max(n - stable_margin, 3 * n // 4)
    stable = seg_df.iloc[stable_start:stable_end]
    
    hc = (stable['SABL_likelihood'] > 0.9) & (stable['SABR_likelihood'] > 0.9)
    if hc.sum() > 50:
        stable = stable[hc]
    
    sabl_x = stable['SABL_x'].median()
    sabl_y = stable['SABL_y'].median()
    sabr_x = stable['SABR_x'].median()
    sabr_y = stable['SABR_y'].median()
    
    pillar_x, pillar_y, ruler = compute_ideal_pillar(sabl_x, sabl_y, sabr_x, sabr_y)
    
    return SegmentGeometry(
        segment_num=segment_num,
        start_frame=seg_start,
        end_frame=seg_end,
        sabl_x=sabl_x, sabl_y=sabl_y,
        sabr_x=sabr_x, sabr_y=sabr_y,
        ruler_pixels=ruler,
        ideal_pillar_x=pillar_x,
        ideal_pillar_y=pillar_y,
        mm_per_pixel=PHYSICAL_RULER_MM / ruler if ruler > 0 else 0
    )


def get_boxr_reference(df) -> float:
    """Get BOXR x position (slit boundary)"""
    return df['BOXR_x'].median()


def load_dlc(dlc_path) -> 'pd.DataFrame':
    """Load DLC h5 file and flatten column names"""
    import pandas as pd
    df = pd.read_hdf(dlc_path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def load_segments(segments_path) -> dict:
    """Load segment boundaries from JSON"""
    import json
    with open(segments_path) as f:
        data = json.load(f)
    return data.get('boundaries', data.get('validated_boundaries', []))
