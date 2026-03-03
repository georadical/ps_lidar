import os, sys, time, json
sys.path.append(os.path.abspath('.'))
from src.core import PointCloudLoader, clip_circular_plot, filter_noise_sor
import numpy as np

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

log('--- BRICK 1: Loading ---')
FILE_PATH = "C:/Users/geoal/Documents/Work/LidarProcessing/LiDAR Cline/Data/raw/HQP079_01_raw.laz"

loader = PointCloudLoader(FILE_PATH)
loader.load()

try:
    meta = loader.get_metadata()
    log(f"File: {meta['filename']}")
    log(f"Points: {meta['point_count']:,}")
    log(f"Size: {meta.get('file_size_mb', 'N/A')} MB")
except Exception as e:
    log(f"Metadata fail: {e}")

# Load XYZ and available scalar fields
log("Calling loader.get_xyz()...")
xyz_full = loader.get_xyz()
log("get_xyz() returned!")

scalar_fields = {}
for field in ['intensity', 'return_number', 'number_of_returns', 'classification']:
    try:
        scalar_fields[field] = loader.get_attribute(field)
        log(f" {field}: {len(scalar_fields[field]):,} values")
    except:
        log(f" {field}: not available")

log(f"\nXYZ memory: {xyz_full.nbytes / (1024**2):.1f} MB")

log('\n--- BRICK 2: Clipping ---')
# The specific plot coordinates from the notebook were likely for Plot_6
# For this raw file, we will just grab the center of the cloud itself to ensure points
x_min, y_min, z_min = xyz_full.min(axis=0)
x_max, y_max, z_max = xyz_full.max(axis=0)

CENTER_X = x_min + (x_max - x_min) / 2.0
CENTER_Y = y_min + (y_max - y_min) / 2.0
PLOT_RADIUS = 25.0

log(f"Autocalculated coords for test: {CENTER_X:.6f}, {CENTER_Y:.6f}, r={PLOT_RADIUS}")

# Run circular clipping
t0 = time.perf_counter()
clip_result = clip_circular_plot(xyz_full, CENTER_X, CENTER_Y, PLOT_RADIUS)
elapsed = time.perf_counter() - t0

plot_indices = clip_result.indices

log(f" Clipped in {elapsed*1000:.0f}ms")
log(f"Original points: {len(xyz_full):,}")
log(f"Points in plot: {clip_result.n_points:,} ({clip_result.n_points/len(xyz_full):.1%})")

# If 0 points, just take the first 1M points so we can test SOR
if clip_result.n_points == 0:
    log("Forcing 1M random points since clipping returned 0")
    plot_indices = np.random.choice(len(xyz_full), min(1000000, len(xyz_full)), replace=False)

# Apply clipping to XYZ and scalar fields
xyz = xyz_full[plot_indices]

plot_scalars = {}
for field, values in scalar_fields.items():
    plot_scalars[field] = values[plot_indices]

log(f"Plot XYZ: {xyz.shape}")
log(f"Scalar fields: {list(plot_scalars.keys())}")

# Free memory
del xyz_full, scalar_fields
import gc; gc.collect()
log(" Memory freed")

log('\n--- BRICK 3: Noise Filtering ---')
log("Applying SOR filter...")
t0 = time.perf_counter()
try:
    noise_result = filter_noise_sor(xyz, k_neighbors=10, std_ratio=2.0, verbose=False)
    xyz = noise_result.clean_xyz
    plot_scalars = {k: v[noise_result.clean_indices] for k, v in plot_scalars.items()}
    log(f"Clean points: {len(xyz):,}")
    log(f"Removed noise points: {noise_result.n_removed:,}")
except Exception as e:
    log(f"SOR failed: {e}")

log(f"\nFiltering completed in {time.perf_counter()-t0:.1f}s")

with open("scripts/test_clean.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("Results written to test_clean.log")
