import numpy as np
from src.core.sampling import clip_circular_plot

def test_circular_clipping():
    # Create synthetic data: 100x100 grid from 0 to 10
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    xyz = np.column_stack((x.flatten(), y.flatten(), np.zeros(10000)))
    
    center_x, center_y = 5.0, 5.0
    radius = 2.0
    
    result = clip_circular_plot(xyz, center_x, center_y, radius)
    
    print(f"Total points: {len(xyz)}")
    print(f"Points in plot: {result.n_points}")
    print(f"Area: {result.area_m2:.2f} m2 ({result.area_ha:.4f} ha)")
    
    # Check if max distance is indeed <= radius
    clipped_xyz = xyz[result.indices]
    dx = clipped_xyz[:, 0] - center_x
    dy = clipped_xyz[:, 1] - center_y
    max_dist = np.sqrt(np.max(dx**2 + dy**2))
    
    print(f"Max distance found: {max_dist:.2f}m")
    assert max_dist <= radius
    print("✓ Test passed!")

if __name__ == "__main__":
    test_circular_clipping()
