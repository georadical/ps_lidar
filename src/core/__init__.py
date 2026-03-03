# Core module - Data I/O and processing primitives
from .io import PointCloudLoader, PointCloudSummary, export_point_cloud
from .normalization import (
    NormalizationAnalyzer,
    NormalizationAnalysis,
    NormalizationStatus,
    detect_normalization,
)
from .ground import (
    GroundFilterResult,
    classify_ground,
    classify_ground as ground_filter_csf, # ALIAS for backwards compatibility
    get_ground_mask,
    get_ground_points,
)
from .sampling import (
    CircularPlotResult,
    ReflectiveCluster,
    ReflectiveMat,
    clip_circular_plot,
    get_centroid,
    detect_reflective_mats,
    find_plot_center,
    validate_gps_center,
)
from .height import (
    HeightNormalizationResult,
    normalize_heights,
    get_normalized_vegetation,
)
from .segmentation import (
    TreeInfo,
    TreeSegmentationResult,
    segment_trees,
    voxelize_cloud,
    compute_verticality,
    extract_stem_stripe,
    detect_stem_clusters,
    compute_tree_axes,
    assign_tree_ids,
    export_tree_locations,
)
from .filtering import (
    NoiseFilterResult,
    filter_noise_sor,
    filter_noise_radius,
)
from .features import (
    voxelize_cloud as voxelize_cloud_features,
    compute_verticality as compute_verticality_pgeof,
    compute_linearity,
    compute_wood_features,
)
from .trunk_extraction import (
    TrunkExtractionConfig,
    TrunkExtractionResult,
    extract_trunks,
)
from .branch_extraction import (
    BranchExtractionConfig,
    BranchExtractionResult,
    extract_branches,
)

__all__ = [
    # I/O
    "PointCloudLoader",
    "PointCloudSummary",
    "export_point_cloud",
    # Normalization
    "NormalizationAnalyzer",
    "NormalizationAnalysis",
    "NormalizationStatus",
    "detect_normalization",
    # Ground
    "GroundFilterResult",
    "classify_ground",
    "ground_filter_csf",
    "get_ground_mask",
    "get_ground_points",
    # Sampling / Clipping
    "CircularPlotResult",
    "ReflectiveCluster",
    "ReflectiveMat",
    "clip_circular_plot",
    "get_centroid",
    "detect_reflective_mats",
    "find_plot_center",
    "validate_gps_center",
    # Height normalization
    "HeightNormalizationResult",
    "normalize_heights",
    "get_normalized_vegetation",
    # Segmentation (legacy, still used)
    "TreeInfo",
    "TreeSegmentationResult",
    "segment_trees",
    "voxelize_cloud",
    "compute_verticality",
    "extract_stem_stripe",
    "detect_stem_clusters",
    "compute_tree_axes",
    "assign_tree_ids",
    "export_tree_locations",
    # Noise filtering
    "NoiseFilterResult",
    "filter_noise_sor",
    "filter_noise_radius",
    # Feature extraction (pgeof only)
    "compute_verticality_pgeof",
    "compute_linearity",
    "compute_wood_features",
    # Trunk extraction (Brick 7)
    "TrunkExtractionConfig",
    "TrunkExtractionResult",
    "extract_trunks",
    # Branch extraction (Brick 8)
    "BranchExtractionConfig",
    "BranchExtractionResult",
    "extract_branches",
]
