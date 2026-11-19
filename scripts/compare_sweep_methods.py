"""Compare the global-chord (F1.x) and windowed (F2) sweep classifiers on
synthetic stems that isolate the cases discussed in the PlotSafe manual
(Feb 2007, §4.1.4.2) and in reports/Rectitud del fuste con LiDAR.md.

Usage:
    python scripts/compare_sweep_methods.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.stem_description import classify_sweep_zones  # noqa: E402
from src.core.sweep_windowed import (  # noqa: E402
    PLOTSAFE_RAD05,
    WindowedSweepConfig,
    classify_sweep_zones_windowed,
)

STEP = 0.2


def _z(z0: float = 0.3, z1: float = 20.3) -> np.ndarray:
    return np.round(np.arange(z0, z1 + 1e-9, STEP), 3)


def _taper(z: np.ndarray, d_base: float = 0.30, d_top: float = 0.12) -> np.ndarray:
    return d_base + (d_top - d_base) * (z - z[0]) / (z[-1] - z[0])


def _stem(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.zeros_like(z), z])


def scenarios():
    out = {}

    z = _z()
    out["recto (ruido 5 mm)"] = (
        _stem(np.random.default_rng(0).normal(0, 0.005, z.size), z), z, _taper(z))

    z = _z(0.3, 26.3)
    R = 200.0
    x = (z - z.mean()) ** 2 / (2 * R)
    out["arco uniforme R=200 m, 26 m"] = (_stem(x, z), z, _taper(z, 0.32, 0.12))

    z = _z(0.3, 26.3)
    x = np.where(z < 2.3, 0.10 * (2.3 - z) / 2.0, 0.0)
    out["palo de hockey basal 10 cm"] = (_stem(x, z), z, _taper(z, 0.32, 0.12))

    z = _z()
    x = np.where(z > 10.0, (z - 10.0) * np.tan(np.radians(6.0)), 0.0)
    out["kink 6° a 10 m"] = (_stem(x, z), z, _taper(z))

    z = _z()
    x = np.where((z > 8.0) & (z < 11.0), 0.06 * np.sin(2 * np.pi * (z - 8.0) / 3.0), 0.0)
    out["wobble ±6 cm en 8–11 m"] = (_stem(x, z), z, _taper(z))

    z = _z()
    x = np.where((z > 9.0) & (z < 13.0), 0.08 * np.sin(np.pi * (z - 9.0) / 4.0), 0.0)
    out["arco local 8 cm en 9–13 m"] = (_stem(x, z), z, _taper(z))

    z = _z(0.3, 3.3)
    out["3 m recto bajo bifurcación"] = (_stem(np.zeros_like(z), z), z, np.full(z.size, 0.30))
    return out


def fmt(zones):
    return "  ".join(f"{c}[{a:.1f}-{b:.1f}]" for a, b, c in zones) or "(vacío)"


def main():
    rad05 = WindowedSweepConfig(dictionary=PLOTSAFE_RAD05)
    for name, (cl, z, d) in scenarios().items():
        sed_top = float(d[-1])
        print(f"\n=== {name}")
        print("  cuerda global (F1):", fmt(classify_sweep_zones(cl, sed_obs_m=sed_top)))
        print("  ventanas HQP (F2): ", fmt(classify_sweep_zones_windowed(cl, (z, d))))
        print("  ventanas RAD05:    ", fmt(classify_sweep_zones_windowed(cl, (z, d), config=rad05)))


if __name__ == "__main__":
    main()
