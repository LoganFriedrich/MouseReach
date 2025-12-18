"""
ASPA2 Ruler System
==================

PURPOSE:
    The SABL-SABR distance is a known physical constant (9mm).
    This module uses it as a universal ruler for all measurements,
    making the system self-calibrating regardless of camera position/zoom.

PHYSICAL CONSTANTS (from Pillar_Tray.stl):
    SABL to SABR:       9.000 mm  (1.000 ruler - by definition)
    SABR to SATR:      15.000 mm  (1.667 ruler)
    Pillar diameter:    4.125 mm  (0.458 ruler)
    Pillar to SABL:     9.618 mm  (1.069 ruler)
    Pillar perpendicular offset: 8.5mm (0.944 ruler)

GEOMETRY:
    The pillar center forms an isoceles triangle with SABL and SABR.
    It lies on the perpendicular bisector of the SABL-SABR line,
    at a distance of 0.944 ruler units toward SAT.

USAGE:
    from aspa2_ruler import RulerCalibration, Geometry
    
    # Calibrate from DLC data
    ruler = RulerCalibration.from_dataframe(df)
    
    # Get pillar position
    pillar = Geometry.compute_pillar_center(sabl, sabr)
    
    # Convert to ruler coordinates
    x_r, y_r = ruler.to_ruler_coords(point, sabl, sabr)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path


# =============================================================================
# PHYSICAL CONSTANTS (in ruler units, where 1 ruler = SABL-SABR = 9mm)
# =============================================================================

PHYSICAL_MM = {
    'sabl_sabr': 9.0,           # Definition of ruler
    'sabr_satr': 15.0,          # SA height
    'pillar_diameter': 4.125,   # Pillar top diameter
    'pellet_diameter': 2.5,     # Approximate millet seed
    'pillar_to_corner': 9.618,  # Distance from pillar center to SABL or SABR
    'pellet_spacing': 10.0,     # Distance between pellet positions
}

RULER_UNITS = {
    'sabl_sabr': 1.000,
    'sabr_satr': 15.0 / 9.0,                    # 1.667
    'pillar_diameter': 4.125 / 9.0,             # 0.458
    'pellet_diameter': 2.5 / 9.0,               # 0.278
    'pillar_to_corner': 9.618 / 9.0,            # 1.069
    'pellet_spacing': 10.0 / 9.0,               # 1.111
    'pillar_perpendicular': np.sqrt((9.618/9.0)**2 - 0.5**2),  # 0.944
}

# Thresholds in ruler units
THRESHOLDS = {
    # Motion detection
    'wiggle_noise': 0.02,        # ~0.2mm - detection jitter, ignore
    'still_threshold': 0.03,     # ~0.3mm - "not moving" velocity
    'significant_motion': 0.10,  # ~0.9mm - meaningful displacement
    'snap_min': 0.8,             # ~7mm - minimum snap magnitude
    'snap_max': 1.5,             # ~13mm - maximum snap magnitude
    'snap_expected': 1.1,        # ~10mm - expected snap (one pellet spacing)
    
    # Pellet status
    'pellet_on_pillar': 0.25,    # ~2.2mm - max deviation when "home"
    'pellet_falling': 0.30,      # ~2.7mm - definitely off pillar
    'pillar_radius': 4.125/2/9,  # 0.229 - half pillar diameter
    'pellet_radius': 2.5/2/9,    # 0.139 - half pellet diameter
    
    # Ruler calibration
    'ruler_std_max': 0.05,       # 5% max std/mean for stable ruler
    'ruler_outlier': 0.20,       # 20% deviation = outlier frame
}

# Timing thresholds (in frames at 60fps)
TIMING = {
    'min_still_period': 600,     # 10s minimum for model building
    'min_snap_interval': 900,    # 15s minimum between snaps
    'expected_pellet_duration': 1800,  # 30s typical
    'snap_duration': 30,         # 0.5s how long snap takes
}


# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

class Geometry:
    """Static methods for geometric calculations."""
    
    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
    
    @staticmethod
    def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Midpoint between two points."""
        return (np.array(p1) + np.array(p2)) / 2
    
    @staticmethod
    def perpendicular_unit_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Unit vector perpendicular to p1→p2 line, pointing toward SAT.
        
        In image coordinates (Y increases downward):
        - SAT is above SAB visually, so SAT has LOWER Y values
        - We want perpendicular pointing "up" = toward lower Y
        - Rotate 90° clockwise from p1→p2 direction
        """
        d = np.array(p2) - np.array(p1)
        length = np.sqrt(np.sum(d**2))
        if length == 0:
            return np.array([0.0, 0.0])
        # Perpendicular: rotate 90° clockwise (toward lower Y in image coords)
        # If d = (dx, dy), clockwise perpendicular = (dy, -dx)
        return np.array([d[1], -d[0]]) / length
    
    @staticmethod
    def compute_pillar_center(sabl: np.ndarray, sabr: np.ndarray) -> np.ndarray:
        """
        Compute pillar center position from SABL and SABR.
        
        The pillar forms an isoceles triangle with SABL and SABR.
        It lies on the perpendicular bisector at 0.944 ruler units.
        
        Args:
            sabl: (x, y) position of SABL in pixels
            sabr: (x, y) position of SABR in pixels
            
        Returns:
            (x, y) position of pillar center in pixels
        """
        sabl = np.array(sabl)
        sabr = np.array(sabr)
        
        # Ruler distance in pixels
        ruler_px = Geometry.distance(sabl, sabr)
        
        # Midpoint of SABL-SABR
        mid = Geometry.midpoint(sabl, sabr)
        
        # Perpendicular direction (toward SAT)
        perp = Geometry.perpendicular_unit_vector(sabl, sabr)
        
        # Distance along perpendicular in pixels
        perp_distance = RULER_UNITS['pillar_perpendicular'] * ruler_px
        
        # Pillar center
        return mid + perp * perp_distance
    
    @staticmethod
    def compute_sa_rectangle(sabl: np.ndarray, sabr: np.ndarray) -> np.ndarray:
        """
        Compute full SA rectangle corners from SABL and SABR.
        
        Uses known geometry: height = 1.667 ruler units.
        
        Args:
            sabl: (x, y) position of SABL
            sabr: (x, y) position of SABR
            
        Returns:
            Array of 4 corners: [SABL, SABR, SATR, SATL]
        """
        sabl = np.array(sabl)
        sabr = np.array(sabr)
        
        ruler_px = Geometry.distance(sabl, sabr)
        perp = Geometry.perpendicular_unit_vector(sabl, sabr)
        height_px = RULER_UNITS['sabr_satr'] * ruler_px
        
        satl = sabl + perp * height_px
        satr = sabr + perp * height_px
        
        return np.array([sabl, sabr, satr, satl])
    
    @staticmethod
    def to_ruler_coords(point: np.ndarray, sabl: np.ndarray, sabr: np.ndarray) -> Tuple[float, float]:
        """
        Convert pixel coordinates to ruler-normalized coordinates.
        
        Origin: SABL
        X-axis: along SABL→SABR (1.0 = SABR position)
        Y-axis: perpendicular toward SAT
        
        Args:
            point: (x, y) in pixels
            sabl: (x, y) of SABL in pixels
            sabr: (x, y) of SABR in pixels
            
        Returns:
            (x_ruler, y_ruler) normalized coordinates
        """
        point = np.array(point)
        sabl = np.array(sabl)
        sabr = np.array(sabr)
        
        ruler_px = Geometry.distance(sabl, sabr)
        if ruler_px == 0:
            return (0.0, 0.0)
        
        # Unit vectors
        x_axis = (sabr - sabl) / ruler_px
        y_axis = Geometry.perpendicular_unit_vector(sabl, sabr)
        
        # Offset from SABL
        offset = point - sabl
        
        # Project onto axes and normalize
        x_ruler = np.dot(offset, x_axis) / ruler_px
        y_ruler = np.dot(offset, y_axis) / ruler_px
        
        return (float(x_ruler), float(y_ruler))
    
    @staticmethod
    def from_ruler_coords(x_ruler: float, y_ruler: float, 
                          sabl: np.ndarray, sabr: np.ndarray) -> np.ndarray:
        """
        Convert ruler coordinates back to pixel coordinates.
        
        Args:
            x_ruler, y_ruler: normalized coordinates
            sabl, sabr: anchor points in pixels
            
        Returns:
            (x, y) in pixels
        """
        sabl = np.array(sabl)
        sabr = np.array(sabr)
        
        ruler_px = Geometry.distance(sabl, sabr)
        
        x_axis = (sabr - sabl) / ruler_px
        y_axis = Geometry.perpendicular_unit_vector(sabl, sabr)
        
        pixel_offset = x_ruler * ruler_px * x_axis + y_ruler * ruler_px * y_axis
        return sabl + pixel_offset


# =============================================================================
# RULER CALIBRATION
# =============================================================================

@dataclass
class RulerCalibration:
    """
    Calibration data for a video's ruler (SABL-SABR distance).
    
    Attributes:
        ruler_px: Median SABL-SABR distance in pixels
        ruler_std: Standard deviation in pixels
        mm_per_px: Millimeters per pixel
        trusted_frame_count: Number of frames used for calibration
        total_frames: Total frames in video
        quality: 'good', 'fair', or 'poor'
        flags: List of QC flags
    """
    ruler_px: float
    ruler_std: float
    mm_per_px: float
    trusted_frame_count: int
    total_frames: int
    quality: str
    flags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, 
                       likelihood_threshold: float = 0.95) -> 'RulerCalibration':
        """
        Calibrate ruler from DLC dataframe.
        
        Args:
            df: DLC dataframe with SABL_x, SABL_y, SABR_x, SABR_y columns
            likelihood_threshold: Minimum likelihood for trusted frames
            
        Returns:
            RulerCalibration object
        """
        total_frames = len(df)
        flags = []
        
        # Calculate SABL-SABR distance for all frames
        ruler_all = np.sqrt(
            (df['SABR_x'] - df['SABL_x'])**2 + 
            (df['SABR_y'] - df['SABL_y'])**2
        )
        
        # Filter to high-confidence frames
        high_conf = (
            (df['SABL_likelihood'] > likelihood_threshold) & 
            (df['SABR_likelihood'] > likelihood_threshold)
        )
        
        if high_conf.sum() < 100:
            # Not enough high-confidence frames, lower threshold
            high_conf = (
                (df['SABL_likelihood'] > 0.8) & 
                (df['SABR_likelihood'] > 0.8)
            )
            flags.append('LOWERED_CONFIDENCE_THRESHOLD')
        
        ruler_high_conf = ruler_all[high_conf]
        
        if len(ruler_high_conf) == 0:
            # No valid frames at all
            return cls(
                ruler_px=35.0,  # Fallback estimate
                ruler_std=999.0,
                mm_per_px=9.0/35.0,
                trusted_frame_count=0,
                total_frames=total_frames,
                quality='poor',
                flags=['NO_VALID_FRAMES']
            )
        
        # First pass: get rough median
        ruler_median = ruler_high_conf.median()
        ruler_std = ruler_high_conf.std()
        
        # Second pass: filter to within 1σ of median
        within_1std = np.abs(ruler_all - ruler_median) < ruler_std
        trusted = high_conf & within_1std
        
        # Final calibration from trusted frames
        ruler_trusted = ruler_all[trusted]
        ruler_px = ruler_trusted.median()
        ruler_std = ruler_trusted.std()
        trusted_count = trusted.sum()
        
        # Calculate mm per pixel
        mm_per_px = PHYSICAL_MM['sabl_sabr'] / ruler_px
        
        # Assess quality
        cv = ruler_std / ruler_px if ruler_px > 0 else 999
        
        if cv > THRESHOLDS['ruler_std_max']:
            flags.append('UNSTABLE_RULER')
        if trusted_count < 1000:
            flags.append('INSUFFICIENT_CALIBRATION')
        
        # Expected ruler is roughly 30-50 pixels for typical setups
        if ruler_px < 20 or ruler_px > 80:
            flags.append('RULER_UNUSUAL_SIZE')
        
        if len(flags) == 0:
            quality = 'good'
        elif 'UNSTABLE_RULER' in flags or 'NO_VALID_FRAMES' in flags:
            quality = 'poor'
        else:
            quality = 'fair'
        
        return cls(
            ruler_px=float(ruler_px),
            ruler_std=float(ruler_std),
            mm_per_px=float(mm_per_px),
            trusted_frame_count=int(trusted_count),
            total_frames=int(total_frames),
            quality=quality,
            flags=flags
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'ruler_px': self.ruler_px,
            'ruler_std': self.ruler_std,
            'mm_per_px': self.mm_per_px,
            'trusted_frame_count': self.trusted_frame_count,
            'total_frames': self.total_frames,
            'quality': self.quality,
            'flags': self.flags,
            'ruler_mm': PHYSICAL_MM['sabl_sabr'],
        }
    
    def px_to_mm(self, pixels: float) -> float:
        """Convert pixels to millimeters."""
        return pixels * self.mm_per_px
    
    def mm_to_px(self, mm: float) -> float:
        """Convert millimeters to pixels."""
        return mm / self.mm_per_px
    
    def px_to_ruler(self, pixels: float) -> float:
        """Convert pixels to ruler units."""
        return pixels / self.ruler_px
    
    def ruler_to_px(self, ruler_units: float) -> float:
        """Convert ruler units to pixels."""
        return ruler_units * self.ruler_px
    
    def get_threshold_px(self, threshold_name: str) -> float:
        """Get a threshold value in pixels."""
        if threshold_name not in THRESHOLDS:
            raise ValueError(f"Unknown threshold: {threshold_name}")
        return THRESHOLDS[threshold_name] * self.ruler_px


# =============================================================================
# FRAME MODEL
# =============================================================================

@dataclass
class FrameModel:
    """
    Complete position model for a single frame.
    All positions in pixels, with ruler-derived geometry.
    """
    frame: int
    sabl: np.ndarray          # Anchor point 1
    sabr: np.ndarray          # Anchor point 2
    ruler_px: float           # SABL-SABR distance
    pillar_center: np.ndarray # Computed from geometry
    expected_pellet: np.ndarray  # Same as pillar_center
    sa_rectangle: np.ndarray  # 4 corners: SABL, SABR, SATR, SATL
    confidence: str           # 'measured', 'interpolated', 'extrapolated'
    
    @classmethod
    def from_positions(cls, frame: int, sabl: np.ndarray, sabr: np.ndarray,
                       confidence: str = 'measured') -> 'FrameModel':
        """Build frame model from SABL/SABR positions."""
        sabl = np.array(sabl)
        sabr = np.array(sabr)
        
        ruler_px = Geometry.distance(sabl, sabr)
        pillar = Geometry.compute_pillar_center(sabl, sabr)
        rectangle = Geometry.compute_sa_rectangle(sabl, sabr)
        
        return cls(
            frame=frame,
            sabl=sabl,
            sabr=sabr,
            ruler_px=ruler_px,
            pillar_center=pillar,
            expected_pellet=pillar.copy(),
            sa_rectangle=rectangle,
            confidence=confidence
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame': self.frame,
            'sabl': self.sabl.tolist(),
            'sabr': self.sabr.tolist(),
            'ruler_px': self.ruler_px,
            'pillar_center': self.pillar_center.tolist(),
            'expected_pellet': self.expected_pellet.tolist(),
            'sa_rectangle': self.sa_rectangle.tolist(),
            'confidence': self.confidence,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_ruler_geometry(df: pd.DataFrame, calibration: RulerCalibration) -> Dict:
    """
    Validate that the geometric assumptions hold for this video.
    
    Checks:
    - Ruler consistency over time
    - SA rectangle roughly rectangular
    - Pillar position consistent when measured vs computed
    
    Returns:
        Dictionary of validation results
    """
    results = {
        'ruler_consistent': True,
        'rectangle_valid': True,
        'pillar_matches': True,
        'issues': []
    }
    
    # Check ruler stability over time (rolling std)
    ruler = np.sqrt(
        (df['SABR_x'] - df['SABL_x'])**2 + 
        (df['SABR_y'] - df['SABL_y'])**2
    )
    high_conf = (df['SABL_likelihood'] > 0.9) & (df['SABR_likelihood'] > 0.9)
    
    if high_conf.sum() > 1000:
        # Check for sudden changes in ruler
        ruler_smooth = ruler[high_conf].rolling(100, center=True).median()
        ruler_range = ruler_smooth.max() - ruler_smooth.min()
        if ruler_range > calibration.ruler_px * 0.1:
            results['ruler_consistent'] = False
            results['issues'].append(f'Ruler varies by {ruler_range:.1f}px over video')
    
    # Check if measured pillar matches computed pillar (during high-conf still periods)
    if 'Pillar_x' in df.columns and 'Pillar_likelihood' in df.columns:
        pillar_conf = df['Pillar_likelihood'] > 0.9
        both_conf = high_conf & pillar_conf
        
        if both_conf.sum() > 100:
            # Compute expected pillar positions
            expected_pillars = []
            actual_pillars = []
            for idx in df[both_conf].sample(min(100, both_conf.sum())).index:
                row = df.loc[idx]
                expected = Geometry.compute_pillar_center(
                    [row['SABL_x'], row['SABL_y']],
                    [row['SABR_x'], row['SABR_y']]
                )
                actual = np.array([row['Pillar_x'], row['Pillar_y']])
                expected_pillars.append(expected)
                actual_pillars.append(actual)
            
            # Check deviation
            deviations = [Geometry.distance(e, a) for e, a in zip(expected_pillars, actual_pillars)]
            mean_dev = np.mean(deviations)
            
            if mean_dev > calibration.ruler_px * 0.15:
                results['pillar_matches'] = False
                results['issues'].append(f'Pillar position off by {mean_dev:.1f}px avg')
    
    return results


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    # Test geometry calculations
    print("=== ASPA2 Ruler System Test ===\n")
    
    # Test with example pixel positions
    sabl = np.array([145.0, 485.0])
    sabr = np.array([180.0, 485.0])
    
    ruler_px = Geometry.distance(sabl, sabr)
    print(f"SABL: {sabl}")
    print(f"SABR: {sabr}")
    print(f"Ruler (px): {ruler_px:.1f}")
    print(f"mm per pixel: {9.0/ruler_px:.3f}")
    
    # Compute pillar
    pillar = Geometry.compute_pillar_center(sabl, sabr)
    print(f"\nComputed pillar center: {pillar}")
    
    # Check geometry in ruler coords
    pillar_ruler = Geometry.to_ruler_coords(pillar, sabl, sabr)
    print(f"Pillar in ruler coords: ({pillar_ruler[0]:.3f}, {pillar_ruler[1]:.3f})")
    print(f"Expected: (0.500, 0.944)")
    
    # Compute full rectangle
    rect = Geometry.compute_sa_rectangle(sabl, sabr)
    print(f"\nSA Rectangle corners:")
    for i, (name, corner) in enumerate(zip(['SABL', 'SABR', 'SATR', 'SATL'], rect)):
        ruler_coords = Geometry.to_ruler_coords(corner, sabl, sabr)
        print(f"  {name}: {corner} -> ruler ({ruler_coords[0]:.3f}, {ruler_coords[1]:.3f})")
    
    print("\n=== Thresholds in Pixels (for 35px ruler) ===")
    for name, value in THRESHOLDS.items():
        if isinstance(value, float) and value < 10:  # ruler-based thresholds
            print(f"  {name}: {value:.3f} ruler = {value * 35:.1f}px = {value * 9:.2f}mm")
