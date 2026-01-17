"""
Tests for tree segmentation module.

Uses real point cloud data for integration testing.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import (
    segment_trees,
    voxelize_cloud,
    compute_verticality,
    extract_stem_stripe,
    detect_stem_clusters,
    TreeSegmentationResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_forest():
    """Generate synthetic forest with 3 cylindrical trees."""
    np.random.seed(42)
    
    trees = []
    tree_positions = [(0, 0), (5, 0), (2.5, 4)]
    
    for x_pos, y_pos in tree_positions:
        # Stem (cylinder)
        n_stem = 1000
        heights = np.random.uniform(0, 15, n_stem)
        angles = np.random.uniform(0, 2 * np.pi, n_stem)
        radii = np.random.uniform(0.1, 0.15, n_stem)
        
        stem_points = np.column_stack([
            x_pos + radii * np.cos(angles),
            y_pos + radii * np.sin(angles),
            heights
        ])
        
        # Crown (sphere-ish)
        n_crown = 500
        crown_center = np.array([x_pos, y_pos, 12])
        crown_points = crown_center + np.random.randn(n_crown, 3) * 1.5
        crown_points[:, 2] = np.clip(crown_points[:, 2], 8, 15)
        
        trees.append(np.vstack([stem_points, crown_points]))
    
    return np.vstack(trees)


@pytest.fixture
def real_vegetation_path():
    """Path to real vegetation point cloud."""
    base_paths = [
        Path("C:/Users/geoal/Documents/Work/LidarProcessing/LiDAR Cline/Data/raw/vegetation_normalized.laz"),
        Path("outputs/vegetation_normalized.laz"),
    ]
    for path in base_paths:
        if path.exists():
            return path
    return None


# =============================================================================
# Unit Tests
# =============================================================================

class TestVoxelizeCloud:
    """Tests for voxelize_cloud function."""
    
    def test_basic_voxelization(self, synthetic_forest):
        """Test that voxelization reduces point count."""
        voxel_xyz, voxel_indices, point_to_voxel = voxelize_cloud(
            synthetic_forest, resolution=0.5
        )
        
        assert len(voxel_xyz) < len(synthetic_forest)
        assert len(voxel_indices) == len(voxel_xyz)
        assert len(point_to_voxel) == len(synthetic_forest)
    
    def test_voxel_mapping_consistency(self, synthetic_forest):
        """Test that point-to-voxel mapping is valid."""
        voxel_xyz, _, point_to_voxel = voxelize_cloud(synthetic_forest, 0.5)
        
        # All voxel indices should be valid
        assert np.all(point_to_voxel >= 0)
        assert np.all(point_to_voxel < len(voxel_xyz))
    
    def test_resolution_effect(self, synthetic_forest):
        """Test that larger resolution produces fewer voxels."""
        voxels_small, _, _ = voxelize_cloud(synthetic_forest, 0.1)
        voxels_large, _, _ = voxelize_cloud(synthetic_forest, 1.0)
        
        assert len(voxels_large) < len(voxels_small)


class TestComputeVerticality:
    """Tests for compute_verticality function."""
    
    def test_vertical_points_high_verticality(self):
        """Test that vertical structures have high verticality."""
        # Create vertical cylinder
        n = 500
        z = np.linspace(0, 10, n)
        angles = np.random.uniform(0, 2*np.pi, n)
        radius = 0.1
        
        cylinder = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            z
        ])
        
        verticality = compute_verticality(cylinder, scale=0.5)
        
        # Most points should have high verticality
        assert np.mean(verticality) > 0.5
    
    def test_horizontal_points_low_verticality(self):
        """Test that horizontal surfaces have low verticality."""
        # Create horizontal plane
        n = 500
        plane = np.column_stack([
            np.random.uniform(-5, 5, n),
            np.random.uniform(-5, 5, n),
            np.random.uniform(0, 0.1, n)  # Small z variation
        ])
        
        verticality = compute_verticality(plane, scale=0.5)
        
        # Most points should have low verticality
        assert np.mean(verticality) < 0.5


class TestExtractStemStripe:
    """Tests for extract_stem_stripe function."""
    
    def test_stripe_extraction(self, synthetic_forest):
        """Test that stripe extraction filters by height."""
        stripe_xyz, stripe_indices = extract_stem_stripe(
            synthetic_forest, z_min=1.0, z_max=3.0
        )
        
        assert len(stripe_xyz) > 0
        assert np.all(stripe_xyz[:, 2] >= 1.0)
        assert np.all(stripe_xyz[:, 2] <= 3.0)
        assert len(stripe_indices) == len(stripe_xyz)


class TestDetectStemClusters:
    """Tests for detect_stem_clusters function."""
    
    def test_finds_clusters(self, synthetic_forest):
        """Test that DBSCAN finds stem clusters."""
        # Get stripe
        stripe, _ = extract_stem_stripe(synthetic_forest, 0.5, 3.5)
        
        # Compute verticality
        vert = compute_verticality(stripe, scale=0.2)
        
        # Detect clusters
        labels, valid_clusters = detect_stem_clusters(
            stripe, vert, 
            vert_threshold=0.5,
            eps=0.3,
            min_samples=5,
            min_cluster_points=20
        )
        
        assert len(valid_clusters) > 0
        assert len(labels) == len(stripe)


# =============================================================================
# Integration Tests
# =============================================================================

class TestSegmentTreesIntegration:
    """Integration tests for segment_trees function."""
    
    def test_synthetic_forest(self, synthetic_forest):
        """Test full segmentation on synthetic data."""
        result = segment_trees(
            synthetic_forest,
            voxel_resolution=0.1,
            stripe_z_min=0.5,
            stripe_z_max=3.5,
            verticality_scale=0.2,
            verticality_threshold=0.5,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_stem_points=20,
            max_axis_distance=3.0,
            verbose=True
        )
        
        assert isinstance(result, TreeSegmentationResult)
        assert len(result.tree_ids) == len(synthetic_forest)
        assert result.n_trees > 0
        print(f"\nDetected {result.n_trees} trees (expected 3)")
    
    def test_output_format(self, synthetic_forest):
        """Test that output has correct format."""
        result = segment_trees(synthetic_forest, verbose=False)
        
        # Check tree_ids array
        assert result.tree_ids.dtype in [np.int32, np.int64]
        assert np.all(result.tree_ids >= -1)
        
        # Check tree_info
        for info in result.tree_info:
            assert hasattr(info, 'tree_id')
            assert hasattr(info, 'centroid')
            assert hasattr(info, 'n_points')
            assert hasattr(info, 'height_max')
            assert len(info.centroid) == 3
    
    @pytest.mark.skipif(
        not Path("C:/Users/geoal/Documents/Work/LidarProcessing/LiDAR Cline/Data/raw/vegetation_normalized.laz").exists(),
        reason="Real data not available"
    )
    def test_real_data(self, real_vegetation_path):
        """Test on real vegetation point cloud."""
        import laspy
        
        las = laspy.read(str(real_vegetation_path))
        xyz = np.column_stack([las.x, las.y, las.z])
        
        result = segment_trees(xyz, verbose=True)
        
        assert result.n_trees > 0
        print(f"\nReal data: {result.n_trees} trees from {len(xyz):,} points")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
