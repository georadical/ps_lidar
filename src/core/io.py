"""
Point Cloud I/O Module

Handles loading and metadata extraction for LiDAR point clouds.
Supports .las and .laz formats via laspy.
"""

import laspy
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from functools import wraps
import warnings


def _ensure_loaded(method):
    """Decorator to ensure point cloud is loaded before method execution."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.las_data is None:
            raise RuntimeError(
                f"Point cloud not loaded. Call load() before {method.__name__}()."
            )
        return method(self, *args, **kwargs)
    return wrapper


@dataclass
class PointCloudSummary:
    """
    Comprehensive summary of a loaded point cloud.
    Inspired by CloudCompare/QGIS property panels.
    """
    # File Info
    filename: str
    file_path: str
    file_size_mb: float
    format_version: str
    point_format_id: int
    
    # Point Info
    point_count: int
    
    # Bounding Box
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    extent_x: float
    extent_y: float
    extent_z: float
    
    # Normalization
    is_normalized: bool
    normalization_confidence: float
    
    # Attributes
    available_dimensions: List[str]
    has_intensity: bool
    has_rgb: bool
    has_classification: bool
    
    # CRS
    crs_name: Optional[str]
    
    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        norm_icon = "✓" if self.is_normalized else "✗"
        
        print(f"{'═'*50}")
        print(f"  POINT CLOUD SUMMARY")
        print(f"{'═'*50}")
        print(f"")
        print(f"  📁 FILE INFO")
        print(f"     Name:    {self.filename}")
        print(f"     Size:    {self.file_size_mb:.2f} MB")
        print(f"     Format:  LAS {self.format_version} (Point Format {self.point_format_id})")
        print(f"")
        print(f"  📊 POINT DATA")
        print(f"     Count:   {self.point_count:,}")
        print(f"")
        print(f"  📐 BOUNDING BOX")
        print(f"     X:       {self.x_min:.2f} → {self.x_max:.2f}  (Δ {self.extent_x:.2f}m)")
        print(f"     Y:       {self.y_min:.2f} → {self.y_max:.2f}  (Δ {self.extent_y:.2f}m)")
        print(f"     Z:       {self.z_min:.2f} → {self.z_max:.2f}  (Δ {self.extent_z:.2f}m)")
        print(f"")
        print(f"  🎯 NORMALIZATION")
        print(f"     Status:  [{norm_icon}] {'Normalized' if self.is_normalized else 'Not Normalized'}")
        print(f"     Confidence: {self.normalization_confidence:.0%}")
        print(f"")
        print(f"  📋 ATTRIBUTES")
        print(f"     Intensity:      {'✓' if self.has_intensity else '✗'}")
        print(f"     RGB Color:      {'✓' if self.has_rgb else '✗'}")
        print(f"     Classification: {'✓' if self.has_classification else '✗'}")
        print(f"     All: {', '.join(self.available_dimensions[:8])}{'...' if len(self.available_dimensions) > 8 else ''}")
        print(f"")
        if self.crs_name:
            print(f"  🌍 COORDINATE SYSTEM")
            print(f"     {self.crs_name}")
            print(f"")
        print(f"{'═'*50}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file": {
                "name": self.filename,
                "path": self.file_path,
                "size_mb": self.file_size_mb,
                "format_version": self.format_version,
                "point_format_id": self.point_format_id,
            },
            "points": {
                "count": self.point_count,
            },
            "bounds": {
                "x": [self.x_min, self.x_max],
                "y": [self.y_min, self.y_max],
                "z": [self.z_min, self.z_max],
                "extent": [self.extent_x, self.extent_y, self.extent_z],
            },
            "normalization": {
                "is_normalized": self.is_normalized,
                "confidence": self.normalization_confidence,
            },
            "attributes": {
                "available": self.available_dimensions,
                "has_intensity": self.has_intensity,
                "has_rgb": self.has_rgb,
                "has_classification": self.has_classification,
            },
            "crs": self.crs_name,
        }


class PointCloudLoader:
    """
    Handles loading and basic metadata extraction for LiDAR point clouds.
    
    Supports .las and .laz formats via laspy.
    LAZ decompression requires lazrs (installed via laspy[lazrs]).
    
    Usage:
        loader = PointCloudLoader("path/to/file.laz")
        loader.load()
        summary = loader.get_summary()  # New! Full summary
        summary.print_summary()         # Pretty print
    """
    
    SUPPORTED_EXTENSIONS = {".las", ".laz"}
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the loader with a file path.
        
        Args:
            file_path: Path to a .las or .laz file.
        """
        self.file_path = Path(file_path)
        self._las_data: laspy.LasData = None
        self._is_loaded: bool = False
        self._cached_summary: Optional[PointCloudSummary] = None
        
    @property
    def las_data(self) -> laspy.LasData:
        """Access to the underlying laspy LasData object."""
        return self._las_data
    
    @property
    def header(self) -> laspy.LasHeader:
        """Access to the LAS header (only after loading)."""
        return self._las_data.header if self._las_data else None
    
    @property
    def is_loaded(self) -> bool:
        """Check if the point cloud has been loaded."""
        return self._is_loaded
        
    def load(self) -> "PointCloudLoader":
        """
        Load the LAS/LAZ file into memory.
        
        Returns:
            self, for method chaining.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file extension is not supported.
            laspy.LaspyException: For data format errors.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {self.file_path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
            
        self._las_data = laspy.read(self.file_path)
        self._is_loaded = True
        self._cached_summary = None  # Invalidate cache
        return self
    
    @_ensure_loaded
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract basic metadata from the loaded point cloud.
        
        Returns:
            Dictionary with filename, point_count, version, bounds, etc.
        """
        header = self._las_data.header
        mins = header.mins
        maxs = header.maxs
        
        return {
            "filename": self.file_path.name,
            "file_size_mb": round(self.file_path.stat().st_size / (1024 * 1024), 2),
            "point_count": header.point_count,
            "version": f"{header.major_version}.{header.minor_version}",
            "point_format_id": header.point_format.id,
            "min_coords": tuple(mins),
            "max_coords": tuple(maxs),
            "dimensions_m": tuple(maxs - mins),
            "scales": tuple(header.scales),
            "offsets": tuple(header.offsets),
            "crs_wkt": header.parse_crs().to_wkt() if header.parse_crs() else None,
        }

    @_ensure_loaded
    def get_available_dimensions(self) -> List[str]:
        """
        List all available point dimensions/attributes in the file.
        
        Returns:
            List of dimension names (e.g., ['X', 'Y', 'Z', 'intensity', 'classification'])
        """
        return list(self._las_data.point_format.dimension_names)

    @_ensure_loaded
    def get_xyz(self, dtype=np.float32) -> np.ndarray:
        """
        Get point coordinates as a numpy array.

        Uses float32 by default to halve memory usage (sufficient
        for LiDAR data at sub-millimetre precision).

        Args:
            dtype: Numpy dtype for output. Default np.float32.

        Returns:
            (N, 3) array with scaled X, Y, Z coordinates.
        """
        n = self._las_data.header.point_count
        xyz = np.empty((n, 3), dtype=dtype)
        xyz[:, 0] = self._las_data.x
        xyz[:, 1] = self._las_data.y
        xyz[:, 2] = self._las_data.z
        return xyz
    
    @_ensure_loaded
    def get_attribute(self, name: str) -> np.ndarray:
        """
        Get a specific point attribute by name.
        
        Args:
            name: Attribute name (use get_available_dimensions() to list options).
            
        Returns:
            1D numpy array with the attribute values.
            
        Raises:
            ValueError: If attribute name is not available.
        """
        available = self.get_available_dimensions()
        if name not in available:
            raise ValueError(
                f"Attribute '{name}' not found. Available: {available}"
            )
        return np.array(self._las_data[name])
    
    @_ensure_loaded
    def get_summary(self, force_refresh: bool = False) -> PointCloudSummary:
        """
        Get a comprehensive summary of the point cloud.
        
        Includes file info, point count, bounding box, normalization status,
        and available attributes. Inspired by CloudCompare/QGIS panels.
        
        Args:
            force_refresh: If True, recalculate normalization (otherwise cached).
            
        Returns:
            PointCloudSummary dataclass with all properties.
        """
        if self._cached_summary is not None and not force_refresh:
            return self._cached_summary
        
        # Import here to avoid circular imports
        from .normalization import detect_normalization
        
        header = self._las_data.header
        mins = header.mins
        maxs = header.maxs
        dims = self.get_available_dimensions()
        
        # Detect normalization (use header bounds to avoid materialising
        # the full XYZ array just for a Z-range check)
        z_min_val = float(mins[2])
        z_max_val = float(maxs[2])
        # Quick heuristic: normalised clouds have Z min near 0
        is_norm = z_min_val >= -0.5 and z_max_val < 100.0
        from .normalization import NormalizationAnalysis, NormalizationStatus
        if is_norm:
            norm_result = NormalizationAnalysis(
                is_normalized=True,
                confidence=0.85,
                status=NormalizationStatus.NORMALIZED,
                z_min=z_min_val,
                z_max=z_max_val,
                z_range=z_max_val - z_min_val,
                reasons=["Z range compatible with normalised data (header check)"],
            )
        else:
            norm_result = NormalizationAnalysis(
                is_normalized=False,
                confidence=0.85,
                status=NormalizationStatus.NOT_NORMALIZED,
                z_min=z_min_val,
                z_max=z_max_val,
                z_range=z_max_val - z_min_val,
                reasons=["Z range not compatible with normalised data (header check)"],
            )
        
        # Check for common attributes
        dims_lower = [d.lower() for d in dims]
        has_rgb = all(c in dims_lower for c in ['red', 'green', 'blue'])
        
        # Parse CRS name
        crs = header.parse_crs()
        crs_name = None
        if crs:
            try:
                crs_name = crs.name or crs.to_wkt()[:50]
            except:
                crs_name = "Unknown CRS"
        
        self._cached_summary = PointCloudSummary(
            filename=self.file_path.name,
            file_path=str(self.file_path.absolute()),
            file_size_mb=round(self.file_path.stat().st_size / (1024 * 1024), 2),
            format_version=f"{header.major_version}.{header.minor_version}",
            point_format_id=header.point_format.id,
            point_count=header.point_count,
            x_min=float(mins[0]),
            x_max=float(maxs[0]),
            y_min=float(mins[1]),
            y_max=float(maxs[1]),
            z_min=float(mins[2]),
            z_max=float(maxs[2]),
            extent_x=float(maxs[0] - mins[0]),
            extent_y=float(maxs[1] - mins[1]),
            extent_z=float(maxs[2] - mins[2]),
            is_normalized=norm_result.is_normalized,
            normalization_confidence=norm_result.confidence,
            available_dimensions=dims,
            has_intensity='intensity' in dims_lower,
            has_rgb=has_rgb,
            has_classification='classification' in dims_lower,
            crs_name=crs_name,
        )
        
        return self._cached_summary
    
    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"PointCloudLoader('{self.file_path}', {status})"


def export_point_cloud(
    output_path: Union[str, Path],
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    classification: Optional[np.ndarray] = None,
    return_number: Optional[np.ndarray] = None,
    number_of_returns: Optional[np.ndarray] = None,
    extra_dimensions: Optional[Dict[str, np.ndarray]] = None,
    point_format: int = 0,
    compress: bool = True,
) -> Path:
    """
    Export a point cloud to LAS/LAZ format.
    
    Args:
        output_path: Path for the output file (.las or .laz).
        xyz: (N, 3) array of point coordinates.
        intensity: Optional (N,) array of intensity values.
        classification: Optional (N,) array of point classifications.
        return_number: Optional (N,) array of return numbers.
        number_of_returns: Optional (N,) array of total returns per pulse.
        extra_dimensions: Optional mapping of scalar names to (N,) arrays.
        point_format: LAS point format ID (0-10). Default 0 for basic XYZ.
        compress: If True and path ends in .laz, compress output. Default True.
        
    Returns:
        Path to the created file.
        
    Example:
        >>> export_point_cloud("vegetation_norm.laz", veg_normalized)
        >>> export_point_cloud("with_intensity.laz", xyz, intensity=intensity_array)
    """
    output_path = Path(output_path)

    if output_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(
            f"output_path must end with .las or .laz, got '{output_path.suffix}'"
        )

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = [0.001, 0.001, 0.001]  # 1mm precision
    header.offsets = [0.0, 0.0, 0.0]

    prepared_extra_dims: Dict[str, np.ndarray] = {}
    if extra_dimensions is not None:
        for name, values in extra_dimensions.items():
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) != len(xyz):
                raise ValueError(
                    f"extra dimension '{name}' must be a 1D array of length {len(xyz)}, "
                    f"got shape {arr.shape}"
                )
            dim_name = str(name)
            header.add_extra_dim(
                laspy.ExtraBytesParams(name=dim_name, type=np.float32)
            )
            prepared_extra_dims[dim_name] = arr.astype(np.float32, copy=False)

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    # Add optional fields
    if intensity is not None:
        las.intensity = intensity.astype(np.uint16)
    
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    
    if return_number is not None:
        las.return_number = return_number.astype(np.uint8)
    
    if number_of_returns is not None:
        las.number_of_returns = number_of_returns.astype(np.uint8)

    for name, values in prepared_extra_dims.items():
        las[name] = values
    
    # Write file with explicit compression behavior
    is_laz = output_path.suffix.lower() == ".laz"
    if not is_laz and compress:
        warnings.warn(
            "compress=True is ignored for .las output. Use a .laz path to write compressed output.",
            UserWarning,
            stacklevel=2,
        )

    las.write(output_path, do_compress=(compress if is_laz else False))
    
    return output_path

