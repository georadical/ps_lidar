"""
Normalization Analysis Module

Provides tools to detect whether a point cloud has been height-normalized
(i.e., Z values represent height above ground rather than absolute elevation).

This is critical for the pipeline to handle both normalized and non-normalized
inputs correctly, ensuring consistent metric extraction regardless of input type.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NormalizationStatus(Enum):
    """Classification of point cloud normalization state."""
    NORMALIZED = "normalized"           # Z values are height above ground
    NOT_NORMALIZED = "not_normalized"   # Z values are absolute elevation
    UNCERTAIN = "uncertain"             # Cannot determine with confidence


@dataclass
class NormalizationAnalysis:
    """
    Results of normalization analysis.
    
    Attributes:
        status: Classification of normalization state
        confidence: Confidence score (0.0 to 1.0)
        z_min: Minimum Z value
        z_max: Maximum Z value
        z_range: Range of Z values (max - min)
        z_mean: Mean Z value
        z_std: Standard deviation of Z values
        ground_plane_detected: Whether a clear ground plane near Z=0 was detected
        percentile_5: 5th percentile of Z (proxy for ground level)
        reasoning: Human-readable explanation of the decision
    """
    status: NormalizationStatus
    confidence: float
    z_min: float
    z_max: float
    z_range: float
    z_mean: float
    z_std: float
    ground_plane_detected: bool
    percentile_5: float
    reasoning: str
    
    def __repr__(self) -> str:
        return (
            f"NormalizationAnalysis("
            f"status={self.status.value}, "
            f"confidence={self.confidence:.2f}, "
            f"z_range={self.z_range:.2f}m)"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "z_min": round(self.z_min, 3),
            "z_max": round(self.z_max, 3),
            "z_range": round(self.z_range, 3),
            "z_mean": round(self.z_mean, 3),
            "z_std": round(self.z_std, 3),
            "ground_plane_detected": self.ground_plane_detected,
            "percentile_5": round(self.percentile_5, 3),
            "reasoning": self.reasoning,
        }
    
    @property
    def is_normalized(self) -> bool:
        """Convenience property to check if cloud is normalized."""
        return self.status == NormalizationStatus.NORMALIZED


class NormalizationAnalyzer:
    """
    Analyzes point cloud Z-statistics to determine normalization status.
    
    Heuristics used:
    1. Normalized clouds typically have Z_min close to 0 (ground level)
    2. The 5th percentile of Z should be near 0 for normalized data
    3. Z range for forest plots is typically 0-50m (normalized) vs 
       hundreds/thousands of meters for absolute elevation
    4. Presence of negative Z values (below "ground") may indicate issues
    
    Usage:
        analyzer = NormalizationAnalyzer(xyz_array)
        result = analyzer.analyze()
        if result.is_normalized:
            print("Cloud is already normalized")
    """
    
    # Default thresholds for classification (in meters)
    DEFAULT_GROUND_TOLERANCE = 2.0          # Max deviation from 0 for ground detection
    DEFAULT_TREE_HEIGHT_MAX = 60.0          # Maximum expected tree height
    DEFAULT_PERCENTILE_THRESHOLD = 1.5      # 5th percentile threshold for ground
    
    def __init__(
        self, 
        xyz: np.ndarray,
        sample_size: Optional[int] = 100000,
        ground_tolerance: Optional[float] = None,
        tree_height_max: Optional[float] = None,
        percentile_threshold: Optional[float] = None,
    ):
        """
        Initialize analyzer with point cloud data.
        
        Args:
            xyz: (N, 3) array of X, Y, Z coordinates
            sample_size: If set, randomly sample this many points for faster
                         analysis on large clouds. None = use all points.
            ground_tolerance: Max Z deviation from 0 to consider as ground (m).
            tree_height_max: Maximum expected tree/vegetation height (m).
            percentile_threshold: Threshold for 5th percentile ground check (m).
        """
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"Expected (N, 3) array, got shape {xyz.shape}")
        
        # Store configurable thresholds
        self.ground_tolerance = ground_tolerance or self.DEFAULT_GROUND_TOLERANCE
        self.tree_height_max = tree_height_max or self.DEFAULT_TREE_HEIGHT_MAX
        self.percentile_threshold = percentile_threshold or self.DEFAULT_PERCENTILE_THRESHOLD
        
        # Extract Z values, optionally sampling for large clouds
        z_values = xyz[:, 2]
        self._original_size = len(z_values)
        
        if sample_size is not None and len(z_values) > sample_size:
            rng = np.random.default_rng(seed=42)  # Reproducible
            indices = rng.choice(len(z_values), size=sample_size, replace=False)
            z_values = z_values[indices]
            self._sampled = True
        else:
            self._sampled = False
        
        self.z_values = z_values
        self._stats_computed = False
        self._z_min = None
        self._z_max = None
        self._z_mean = None
        self._z_std = None
        self._percentile_5 = None
    
    def __repr__(self) -> str:
        status = "analyzed" if self._stats_computed else "pending"
        sampled = f", sampled={len(self.z_values)}" if self._sampled else ""
        return f"NormalizationAnalyzer(points={self._original_size}{sampled}, {status})"
    
    def _compute_stats(self) -> None:
        """Compute Z-statistics lazily."""
        if self._stats_computed:
            return
            
        self._z_min = float(np.min(self.z_values))
        self._z_max = float(np.max(self.z_values))
        self._z_mean = float(np.mean(self.z_values))
        self._z_std = float(np.std(self.z_values))
        self._percentile_5 = float(np.percentile(self.z_values, 5))
        self._stats_computed = True
    
    def analyze(self) -> NormalizationAnalysis:
        """
        Perform normalization analysis and return results.
        
        Returns:
            NormalizationAnalysis with status, confidence, and statistics.
        """
        self._compute_stats()
        
        z_range = self._z_max - self._z_min
        reasons = []
        confidence_factors = []
        
        # Heuristic 1: Check if Z_min is near zero
        z_min_near_zero = abs(self._z_min) < self.ground_tolerance
        if z_min_near_zero:
            reasons.append(f"Z_min ({self._z_min:.2f}m) is close to 0")
            confidence_factors.append(0.3)
        else:
            reasons.append(f"Z_min ({self._z_min:.2f}m) is far from 0")
            confidence_factors.append(-0.3)
        
        # Heuristic 2: Check 5th percentile (more robust than min)
        percentile_near_zero = abs(self._percentile_5) < self.percentile_threshold
        if percentile_near_zero:
            reasons.append(f"5th percentile ({self._percentile_5:.2f}m) near ground level")
            confidence_factors.append(0.25)
        else:
            reasons.append(f"5th percentile ({self._percentile_5:.2f}m) suggests elevation data")
            confidence_factors.append(-0.25)
        
        # Heuristic 3: Z range consistent with tree heights
        range_reasonable = z_range < self.tree_height_max
        if range_reasonable:
            reasons.append(f"Z range ({z_range:.2f}m) consistent with tree heights")
            confidence_factors.append(0.2)
        else:
            reasons.append(f"Z range ({z_range:.2f}m) exceeds typical tree heights")
            confidence_factors.append(-0.1)  # Less penalty - could be tall trees
        
        # Heuristic 4: Check for significant negative values
        negative_fraction = float(np.mean(self.z_values < -0.5))
        if negative_fraction < 0.01:
            reasons.append("Minimal negative Z values")
            confidence_factors.append(0.15)
        elif negative_fraction < 0.05:
            reasons.append(f"{negative_fraction*100:.1f}% points below ground (minor)")
            confidence_factors.append(0.0)
        else:
            reasons.append(f"{negative_fraction*100:.1f}% points significantly below ground")
            confidence_factors.append(-0.15)
        
        # Heuristic 5: Check if Z values look like absolute elevation
        # Note: This heuristic may give false negatives in coastal/low-lying areas
        likely_absolute = self._z_min > 50 or self._z_max > 500
        if likely_absolute:
            reasons.append(f"Z values ({self._z_min:.0f}-{self._z_max:.0f}m) suggest absolute elevation")
            confidence_factors.append(-0.4)
        
        # Calculate overall confidence and determine status
        raw_confidence = sum(confidence_factors)
        
        # Map to 0-1 range with sigmoid-like transformation
        normalized_confidence = 1 / (1 + np.exp(-3 * raw_confidence))
        
        # Determine ground plane detection
        ground_plane_detected = z_min_near_zero and percentile_near_zero
        
        # Classify based on confidence
        if normalized_confidence > 0.65:
            status = NormalizationStatus.NORMALIZED
        elif normalized_confidence < 0.35:
            status = NormalizationStatus.NOT_NORMALIZED
        else:
            status = NormalizationStatus.UNCERTAIN
        
        return NormalizationAnalysis(
            status=status,
            confidence=float(normalized_confidence),
            z_min=self._z_min,
            z_max=self._z_max,
            z_range=z_range,
            z_mean=self._z_mean,
            z_std=self._z_std,
            ground_plane_detected=ground_plane_detected,
            percentile_5=self._percentile_5,
            reasoning="; ".join(reasons),
        )


def detect_normalization(
    xyz: np.ndarray,
    sample_size: Optional[int] = 100000,
    **kwargs
) -> NormalizationAnalysis:
    """
    Convenience function to detect normalization status.
    
    Args:
        xyz: (N, 3) array of point coordinates
        sample_size: Number of points to sample for analysis (None = all)
        **kwargs: Additional threshold parameters for NormalizationAnalyzer
        
    Returns:
        NormalizationAnalysis result
    """
    return NormalizationAnalyzer(xyz, sample_size=sample_size, **kwargs).analyze()
