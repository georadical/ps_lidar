"""
Point Cloud I/O Module

Handles loading and metadata extraction for LiDAR point clouds.
Supports .las and .laz formats via laspy.
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, List
from functools import wraps


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


class PointCloudLoader:
    """
    Handles loading and basic metadata extraction for LiDAR point clouds.
    
    Supports .las and .laz formats via laspy.
    LAZ decompression requires lazrs (installed via laspy[lazrs]).
    
    Usage:
        loader = PointCloudLoader("path/to/file.laz")
        loader.load()
        metadata = loader.get_metadata()
        xyz = loader.get_xyz()
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
        return self  # Enable chaining: loader.load().get_metadata()
    
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
    def get_xyz(self) -> np.ndarray:
        """
        Get point coordinates as a numpy array.
        
        Returns:
            (N, 3) float64 array with scaled X, Y, Z coordinates.
        """
        return np.column_stack((
            self._las_data.x, 
            self._las_data.y, 
            self._las_data.z
        ))
    
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
    
    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"PointCloudLoader('{self.file_path}', {status})"
