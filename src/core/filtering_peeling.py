

@dataclass
class IterativePeelingResult:
    """Result of iterative peeling understory separation."""
    is_tree: np.ndarray          # (N,) boolean mask (True = tree, False = understory)
    n_seeds: int                 # Number of initial seed points
    n_iterations: int            # Number of expansion iterations performed
    n_tree: int                  # Final number of tree points
    n_understory: int            # Final number of understory points
    expansion_percentage: float  # Percentage of points added via expansion


def iterative_peeling_understory(
    xyz: np.ndarray,
    verticality: np.ndarray,
    linearity: np.ndarray,
    sphericity: np.ndarray,
    dist_to_ground: np.ndarray,
    seed_verticality: float = 0.9,
    seed_linearity: float = 0.6,
    seed_height_min: float = 1.0,
    seed_height_max: float = 2.5,
    expansion_verticality: float = 0.5,
    expansion_radius: float = 0.3,
    max_iterations: int = 50,
    verbose: bool = False
) -> IterativePeelingResult:
    """
    Separate trees from understory using iterative peeling (region growing).
    
    This algorithm addresses two key failures of threshold-based classification:
    1. False negatives: Trunk edge points with unreliable features
    2. False positives: Isolated understory clusters that pass thresholds
    
    Strategy:
    - Start with ultra-reliable trunk seeds (high verticality + linearity, 1-2.5m height)
    - Iteratively expand to neighbors meeting relaxed criteria
    - Natural protection of entire trunk cylinders
    - Automatic exclusion of isolated understory
    
    Parameters
    ----------
    xyz : np.ndarray
        Point cloud coordinates (n, 3).
    verticality : np.ndarray
        Verticality feature (0-1) for each point.
    linearity : np.ndarray
        Linearity feature (0-1) for each point.
    sphericity : np.ndarray
        Sphericity feature (0-1) for each point.
    dist_to_ground : np.ndarray
        Distance to ground for each point (meters).
    seed_verticality : float
        Minimum verticality for seed selection (default: 0.9).
    seed_linearity : float
        Minimum linearity for seed selection (default: 0.6).
    seed_height_min : float
        Minimum height for seed selection (default: 1.0m).
    seed_height_max : float
        Maximum height for seed selection (default: 2.5m).
    expansion_verticality : float
        Minimum verticality for expansion (default: 0.5).
    expansion_radius : float
        Neighbor search radius for expansion (default: 0.3m).
    max_iterations : int
        Maximum number of expansion iterations (default: 50).
    verbose : bool
        Print progress information.
    
    Returns
    -------
    IterativePeelingResult
        Result object containing is_tree mask and statistics.
    
    Example
    -------
    >>> result = iterative_peeling_understory(
    ...     xyz, verticality, linearity, sphericity, dist_to_ground,
    ...     seed_verticality=0.9,
    ...     expansion_verticality=0.5,
    ...     verbose=True
    ... )
    >>> trees_xyz = xyz[result.is_tree]
    >>> understory_xyz = xyz[~result.is_tree]
    """
    from scipy.spatial import cKDTree
    
    n_points = len(xyz)
    
    if verbose:
        print(f"Iterative peeling for {n_points:,} points...")
        print(f"  Seed criteria: verticality>{seed_verticality}, linearity>{seed_linearity}, "
              f"height {seed_height_min}-{seed_height_max}m")
        print(f"  Expansion criteria: verticality>{expansion_verticality}, radius={expansion_radius}m")
    
    # ========================================================================
    # Step 1: Initialize seeds (ultra-reliable trunk points)
    # ========================================================================
    seed_mask = (
        (verticality > seed_verticality) &
        (linearity > seed_linearity) &
        (dist_to_ground >= seed_height_min) &
        (dist_to_ground <= seed_height_max)
    )
    
    n_seeds = np.sum(seed_mask)
    
    if n_seeds == 0:
        if verbose:
            print("  ⚠ WARNING: No seeds found! Adjusting criteria...")
        # Fallback: relax criteria slightly
        seed_mask = (
            (verticality > seed_verticality - 0.1) &
            (linearity > seed_linearity - 0.1) &
            (dist_to_ground >= seed_height_min) &
            (dist_to_ground <= seed_height_max)
        )
        n_seeds = np.sum(seed_mask)
        
        if n_seeds == 0:
            warnings.warn("No seeds found even with relaxed criteria. Returning all points as understory.")
            return IterativePeelingResult(
                is_tree=np.zeros(n_points, dtype=bool),
                n_seeds=0,
                n_iterations=0,
                n_tree=0,
                n_understory=n_points,
                expansion_percentage=0.0
            )
    
    if verbose:
        print(f"  Seeds: {n_seeds:,} points ({100*n_seeds/n_points:.2f}%)")
    
    # Initialize tree mask with seeds
    is_tree = seed_mask.copy()
    
    # ========================================================================
    # Step 2: Build KDTree for neighbor queries
    # ========================================================================
    if verbose:
        print("  Building KDTree...")
    
    tree = cKDTree(xyz)
    
    # ========================================================================
    # Step 3: Iterative expansion
    # ========================================================================
    if verbose:
        print(f"  Expanding from seeds (max {max_iterations} iterations)...")
    
    previous_tree = seed_mask.copy()
    
    for iteration in range(max_iterations):
        # Find frontier (points added in last iteration)
        new_tree_points = is_tree & ~previous_tree
        n_frontier = np.sum(new_tree_points)
        
        if n_frontier == 0:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        
        if verbose and (iteration % 5 == 0 or iteration < 3):
            n_current = np.sum(is_tree)
            print(f"    Iteration {iteration}: {n_current:,} tree points (+{n_frontier:,} frontier)")
        
        # Update previous for next iteration
        previous_tree = is_tree.copy()
        
        # Query neighbors of frontier points
        frontier_indices = np.where(new_tree_points)[0]
        
        for point_idx in frontier_indices:
            # Find neighbors within expansion radius
            neighbor_indices = tree.query_ball_point(xyz[point_idx], expansion_radius)
            
            # Add neighbors if they meet expansion criteria
            for n_idx in neighbor_indices:
                if not is_tree[n_idx]:
                    # Relaxed criteria: only verticality check
                    if verticality[n_idx] > expansion_verticality:
                        is_tree[n_idx] = True
    
    # ========================================================================
    # Step 4: Compute statistics
    # ========================================================================
    n_tree = np.sum(is_tree)
    n_understory = n_points - n_tree
    n_expanded = n_tree - n_seeds
    expansion_percentage = 100 * n_expanded / n_seeds if n_seeds > 0 else 0.0
    
    if verbose:
        print(f"  ✓ Peeling complete")
        print(f"    Seeds: {n_seeds:,}")
        print(f"    Expanded: +{n_expanded:,} points ({expansion_percentage:.1f}% of seeds)")
        print(f"    Final: {n_tree:,} tree ({100*n_tree/n_points:.1f}%), "
              f"{n_understory:,} understory ({100*n_understory/n_points:.1f}%)")
    
    return IterativePeelingResult(
        is_tree=is_tree,
        n_seeds=n_seeds,
        n_iterations=iteration + 1 if n_frontier > 0 else iteration,
        n_tree=n_tree,
        n_understory=n_understory,
        expansion_percentage=expansion_percentage
    )
