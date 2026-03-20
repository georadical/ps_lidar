from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from src.core.features import compute_exportable_geometry_features
from src.core.io import export_point_cloud


def _make_vertical_stem_cloud(n_levels: int = 20, n_ring: int = 8) -> np.ndarray:
    z_levels = np.linspace(0.5, 3.0, n_levels)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)

    points = []
    for z in z_levels:
        radius = 0.08 + 0.005 * np.sin(z)
        for angle in angles:
            points.append(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z,
                ]
            )
    return np.asarray(points, dtype=np.float64)


def test_compute_exportable_geometry_features_returns_pointwise_scalars() -> None:
    xyz = _make_vertical_stem_cloud()

    features = compute_exportable_geometry_features(
        xyz,
        scale=0.10,
        voxel_resolution_xy=0.05,
        voxel_resolution_z=0.05,
        verbose=False,
    )

    expected = {
        "verticality",
        "linearity",
        "planarity",
        "sphericity",
        "anisotropy",
        "surface_variation",
        "roughness",
        "neighbor_count",
        "volume_density",
    }
    assert set(features) == expected

    for values in features.values():
        assert values.shape == (len(xyz),)
        assert np.all(np.isfinite(values))


def test_export_point_cloud_writes_extra_dimensions() -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    written: dict[str, object] = {}

    def _fake_write(self, destination, do_compress=None):
        written["destination"] = destination
        written["do_compress"] = do_compress
        written["las"] = self

    with patch.object(laspy.LasData, "write", new=_fake_write):
        output_path = Path("virtual_with_features.las")
        export_point_cloud(
            output_path,
            xyz,
            classification=np.array([1, 2, 3], dtype=np.uint8),
            extra_dimensions={
                "verticality": np.array([0.9, 0.8, 0.7], dtype=np.float64),
                "roughness": np.array([0.01, 0.02, 0.03], dtype=np.float64),
            },
            point_format=6,
            compress=False,
        )

    las = written["las"]
    dims = set(las.point_format.dimension_names)
    assert "verticality" in dims
    assert "roughness" in dims
    assert np.allclose(np.asarray(las["verticality"]), [0.9, 0.8, 0.7], atol=1e-6)
    assert np.allclose(np.asarray(las["roughness"]), [0.01, 0.02, 0.03], atol=1e-6)
