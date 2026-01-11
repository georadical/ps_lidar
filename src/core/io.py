import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional

class PointCloudLoader:
    """
    Handles loading and basic metadata extraction for LiDAR point clouds.
    Supports .las and .laz formats via laspy.
    """
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.las_data = None
        self.header = None
        self.points = None
        
    def load(self) -> None:
        """
        Loads the LAS/LAZ file into memory.
        Raises FileNotFoundError if file does not exist.
        Raises laspy.LaspyException for data errors.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        # Read the file
        # lazrs should be installed for fast LAZ support
        self.las_data = laspy.read(self.file_path)
        self.header = self.las_data.header
        self.points = self.las_data.points
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing basic metadata about the loaded point cloud.
        Ensure load() is called first.
        """
        if self.las_data is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
            
        # Calculate bounding box dimensions
        mins = self.header.mins
        maxs = self.header.maxs
        dims = maxs - mins
        
        return {
            "filename": self.file_path.name,
            "point_count": self.header.point_count,
            "version": f"{self.header.major_version}.{self.header.minor_version}",
            "point_format_id": self.header.point_format.id,
            "min_coords": mins,
            "max_coords": maxs,
            "dimensions": dims,
            "scales": self.header.scales,
            "offsets": self.header.offsets
        }

    def get_xyz(self) -> np.ndarray:
        """
        Returns the points as a numpy (N, 3) array of float64 coordinates.
        This applies scales and offsets automatically via laspy.
        """
        if self.las_data is None:
            raise RuntimeError("Point cloud not loaded. Call load() first.")
            
        return np.column_stack((self.las_data.x, self.las_data.y, self.las_data.z))
