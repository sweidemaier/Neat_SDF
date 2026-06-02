import sys
import csv
import time
import torch
import numpy as np
import scipy.spatial as tree
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

##################################
# computation of locally adaptive weights is based on https://github.com/sweidemaier/HeatSDF/blob/main/trainers/helper.py

def bump_func(x):
    if (abs(x) > 1):
        return 0
    else:
        return np.exp(1/((abs(x)**2)-1))   

### compute locally adaptive weights for heat step
def comp_weights(pointcloud, epsilon, dim = 3):
    start = time.time()
    w = np.zeros(np.shape(pointcloud)[0])
    r = epsilon

    ### sort input points in tree
    tr = tree.cKDTree(pointcloud)
    p = tr.query_ball_point(x = pointcloud, r = r)
    
    ### increase radius, until for each point its eps-environment contains at least a few points; we choose 12
    while any(len(ball) < 12 for ball in p):
        r *= 2
        p = tr.query_ball_point(x = pointcloud, r = r)
    
    c_eps = (r**dim)
    j = 0

    ### for each point compute weight
    while j < np.size(p):
        ball_indices = p[j]
        ball_points = pointcloud[ball_indices]
        dists = np.linalg.norm(pointcloud[j]-ball_points, axis = 1)
        sum = 1/c_eps*np.sum([bump_func(dists[i]/r) for i in range(len(dists))])
        w[j] = 1/sum
        j += 1
    
    ### normalize weights
    w = w/np.sum(w)
    print("Computed locally adaptive weights.")
    print("Total computation time:", time.time() - start)
    return w

##################################

def create_initial_grid(bounds, grid_resolution, device='cuda', dim=2):
    """
    Create an initial regular grid of non-overlapping boxes.
    
    Parameters:
        bounds (float or tuple): Edge length of domain (e.g., 2.4 for [-1.2, 1.2]).
                                 If tuple/list of length dim, per-axis bounds.
        grid_resolution (int or list): Number of boxes per dimension. If int, same for all dims.
        device (str): PyTorch device (e.g., 'cuda' or 'cpu') (default: 'cuda').
        dim (int): Spatial dimension (default: 2).
    
    Returns:
        tuple: (box_midpoints, box_widths)
            - box_midpoints (torch.Tensor): Shape (N_boxes, dim) - centers of boxes
            - box_widths (torch.Tensor): Shape (N_boxes, dim) - widths of boxes in each dimension
    """
    
    # Convert bounds to min/max
    if isinstance(bounds, (int, float)):
        # Single scalar: symmetric bounds [-bounds/2, bounds/2]
        half_bound = bounds / 2.0
        min_bound = torch.full((dim,), -half_bound, dtype=torch.float32, device=device)
        max_bound = torch.full((dim,), half_bound, dtype=torch.float32, device=device)
    elif isinstance(bounds, (list, tuple)):
        if len(bounds) == dim:
            # Per-axis bounds: [size_x, size_y, (size_z)]
            half_bounds = [b / 2.0 for b in bounds]
            min_bound = torch.tensor([-b for b in half_bounds], dtype=torch.float32, device=device)
            max_bound = torch.tensor(half_bounds, dtype=torch.float32, device=device)
        elif len(bounds) == 2:
            # Old format: (min_bound, max_bound) - keep for backward compatibility
            min_bound = -torch.as_tensor(bounds[0], dtype=torch.float32, device=device)
            max_bound = torch.as_tensor(bounds[1], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"bounds must be scalar, length {dim}, or length 2")
    else:
        raise TypeError("bounds must be a scalar, tuple, or list")
    
    # Handle grid_resolution
    if isinstance(grid_resolution, int):
        grid_resolution = [grid_resolution] * dim
    else:
        grid_resolution = list(grid_resolution)
    
    # Compute domain size and box widths
    domain_size = max_bound - min_bound
    box_widths = domain_size / torch.tensor(grid_resolution, dtype=torch.float32, device=device)
    
    # Create grid of box midpoints
    if dim == 1:
        grid_coords = torch.linspace(
            min_bound[0] + box_widths[0] / 2,
            max_bound[0] - box_widths[0] / 2,
            grid_resolution[0],
            device=device
        )
        box_midpoints = grid_coords.unsqueeze(1)
        
    elif dim == 2:
        x = torch.linspace(
            min_bound[0] + box_widths[0] / 2,
            max_bound[0] - box_widths[0] / 2,
            grid_resolution[0],
            device=device
        )
        y = torch.linspace(
            min_bound[1] + box_widths[1] / 2,
            max_bound[1] - box_widths[1] / 2,
            grid_resolution[1],
            device=device
        )
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        box_midpoints = torch.stack([xx.ravel(), yy.ravel()], dim=1)
        
    elif dim == 3:
        x = torch.linspace(
            min_bound[0] + box_widths[0] / 2,
            max_bound[0] - box_widths[0] / 2,
            grid_resolution[0],
            device=device
        )
        y = torch.linspace(
            min_bound[1] + box_widths[1] / 2,
            max_bound[1] - box_widths[1] / 2,
            grid_resolution[1],
            device=device
        )
        z = torch.linspace(
            min_bound[2] + box_widths[2] / 2,
            max_bound[2] - box_widths[2] / 2,
            grid_resolution[2],
            device=device
        )
        if "indexing" in torch.meshgrid.__code__.co_varnames:
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        else:
            xx, yy, zz = torch.meshgrid(x, y, z)
        box_midpoints = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=1)
        
    else:
        # General n-dimensional case using meshgrid
        coords = [
            torch.linspace(
                min_bound[i] + box_widths[i] / 2,
                max_bound[i] - box_widths[i] / 2,
                grid_resolution[i],
                device=device
            )
            for i in range(dim)
        ]
        grids = torch.meshgrid(*coords, indexing='ij')
        box_midpoints = torch.stack([g.ravel() for g in grids], dim=1)
    
    # Expand box_widths to match number of boxes
    n_boxes = box_midpoints.shape[0]
    box_widths = box_widths.unsqueeze(0).expand(n_boxes, -1)
    
    return box_midpoints, box_widths



def adaptive_refinement(
    box_midpoints, 
    refinement_lvl, 
    functions, 
    threshold,
    max_iterations=1, 
    device='cuda', 
    dim=2, 
    bounds=None,
    max_depth=4,
    n_probe=15
):
    """
    Adaptive refinement for 2D grids with multi-probe evaluation.
    
    Refines cells where any function value falls below its threshold.
    Refined parent cells are REPLACED by their 4 children (in 2D).

    Uses multiple probe points per cell (midpoint + n_probe-1 random interior points)
    to avoid missing features that don't pass through the cell center.

    Args:
        box_midpoints (torch.Tensor): Box centers of shape (N, 2)
        refinement_lvl (torch.Tensor): Current refinement level per cell (N,) or (N, 1)
        functions (list): List of callables, each maps (M, 2) → (M, 1)
        threshold (torch.Tensor or float): Threshold values for each function
        max_iterations (int): Maximum number of refinement iterations (default: 1)
        device (str): PyTorch device (default: 'cuda')
        dim (int): Spatial dimension (default: 2)
        bounds (tuple): Domain bounds (min_bound, max_bound) as tensors (optional)
        max_depth (int): Maximum refinement depth (default: 4)
        n_probe (int): Number of probe points per cell (default: 15)

    Returns:
        tuple: (refined_midpoints, refined_levels)
    """
    import numpy as np
    import torch

    box_midpoints = torch.as_tensor(box_midpoints, dtype=torch.float32, device=device)
    refinement_lvl = torch.as_tensor(refinement_lvl, dtype=torch.long, device=device)
    
    # Handle threshold
    threshold = torch.as_tensor(threshold, dtype=torch.float32, device=device)
    if threshold.dim() == 0:
        threshold = threshold.unsqueeze(0)

    midpoints = box_midpoints.clone()
    refinement_levels = refinement_lvl.clone()
    
    # Ensure refinement_levels is 1D
    if refinement_levels.dim() > 1:
        refinement_levels = refinement_levels.squeeze()

    # ============ Iterative Refinement ============
    for iteration in range(max_iterations):
        print(f"[*] Refinement iteration {iteration + 1}/{max_iterations}")
        print(f"    Current cells: {len(midpoints)}")
        print(f"    Max refinement level: {refinement_levels.max().item()}")
        
        # ── Compute cell widths ──────────────────────────────────────────
        # Infer h_base from level-0 cells
        level0_mask = (refinement_levels == 0)
        if level0_mask.sum() >= 2:
            level0_pts = midpoints[level0_mask]
            diffs = (level0_pts[:2] - level0_pts[1:3]).abs()
            h_base = diffs[diffs > 1e-8].min().item()
        else:
            min_level = refinement_levels.min().item()
            mask_min = (refinement_levels == min_level)
            pts_min = midpoints[mask_min]
            if pts_min.shape[0] >= 2:
                d = torch.cdist(pts_min[:100], pts_min[:100])
                d = d[d > 1e-8]
                cell_side = d.min().item()
                h_base = cell_side * (2 ** min_level)
            else:
                h_base = 1.0

        # Cell half-width per cell: h_base / 2^(level+1)
        cell_half_width = h_base / (2.0 ** (refinement_levels.float() + 1))  # (N,)

        # ── Build probe points ───────────────────────────────────────────
        # Midpoint + (n_probe-1) random interior points
        if n_probe <= 1:
            probe_pts = midpoints.unsqueeze(1)  # (N, 1, 2)
        else:
            # Random offsets in [-0.5, 0.5]^dim, scaled by cell width
            rand_offsets = (torch.rand(len(midpoints), n_probe - 1, dim, device=device) - 0.5) * 2.0
            rand_offsets = rand_offsets * cell_half_width[:, None, None]  # (N, n_probe-1, 2)
            random_probes = midpoints.unsqueeze(1) + rand_offsets  # (N, n_probe-1, 2)
            midpoint_probe = midpoints.unsqueeze(1)  # (N, 1, 2)
            probe_pts = torch.cat([midpoint_probe, random_probes], dim=1)  # (N, n_probe, 2)

        probe_flat = probe_pts.reshape(-1, dim)  # (N * n_probe, 2)

        # ── Evaluate functions and determine refinement ───────────────────
        refine_mask = torch.zeros(len(midpoints), dtype=torch.bool, device=device)

        with torch.no_grad():
            for func, thresh in zip(functions, threshold):
                vals = func(probe_flat).view(len(midpoints), n_probe if n_probe > 1 else 1)  # (N, n_probe)
                # Use minimum across probes — most conservative, catches any crossing
                vals_min = vals.min(dim=1).values  # (N,)
                refine_mask |= (vals_min < thresh)

        # Don't refine beyond max_depth
        refine_mask &= (refinement_levels < max_depth)

        num_to_refine = refine_mask.sum().item()
        print(f"    Cells to refine: {num_to_refine}")
        
        if num_to_refine == 0:
            print(f"[✓] No more cells to refine. Done at iteration {iteration + 1}")
            break

        # ── Keep unrefined cells ─────────────────────────────────────────
        keep_mask = ~refine_mask
        kept_midpoints = midpoints[keep_mask]
        kept_levels = refinement_levels[keep_mask]

        # ── Split refined cells into 4 children (2D) ─────────────────────
        parents = midpoints[refine_mask]  # (R, 2)
        parent_levels = refinement_levels[refine_mask]  # (R,)
        child_level = parent_levels + 1  # (R,)

        R = parents.shape[0]

        # 4 offset directions (corners of [-1,1]^2 for 2D)
        offsets = torch.tensor([
            [-1, -1], [-1, 1], [1, -1], [1, 1],
        ], dtype=torch.float32, device=device)  # (4, 2)

        # Offset per parent: h_base / 2^(level+2)
        offset_scale = h_base / (2.0 ** (parent_levels.float() + 2))  # (R,)

        children = (parents.unsqueeze(1)
                    + offsets.unsqueeze(0) * offset_scale.view(-1, 1, 1))  # (R, 4, 2)
        children = children.reshape(-1, 2)  # (R*4, 2)

        child_levels = child_level.repeat_interleave(4)  # (R*4,)

        # ── Concatenate: unrefined originals + new children ──────────────
        midpoints = torch.cat([kept_midpoints, children], dim=0)
        refinement_levels = torch.cat([kept_levels, child_levels], dim=0)

        print(f"    After subdivision: {len(midpoints)} cells")

    print(f"\n[✓] Adaptive refinement complete")
    print(f"    Final cells: {len(midpoints)}")
    print(f"    Final level distribution:")
    unique_levels = torch.unique(refinement_levels, sorted=True)
    for level in unique_levels:
        count = (refinement_levels == level).sum().item()
        print(f"      Level {level}: {count} cells")

    return midpoints, refinement_levels



def sample_and_reweight(box_midpoints, h, N, flags, device='cuda', dim=None):
    """
    Efficiently sample N points from a grid of box midpoints 
        with maximum boxwidth h; depending on flags (refinement level),
        we compute weights.
    
    Parameters:
        box_midpoints (torch.Tensor): Box centers of shape (B, D)
        h (float or torch.Tensor): Maximum box width (scalar or per-box widths)
        N (int): Number of points to sample
        flags (torch.Tensor): Refinement levels of shape (B,)
        device (str): PyTorch device
        dim (int): Spatial dimension (auto-inferred from box_midpoints if None)
    
    Returns:
        tuple: (sampled_points, integration_weights)
    """
    # Auto-infer dimension if not provided
    if dim is None:
        dim = box_midpoints.shape[-1]
    
    B = box_midpoints.shape[0]
    box_indices = torch.randint(0, B, (N,), device=device)
    selected_midpoints = box_midpoints[box_indices]  # (N, dim)

    # Get refinement levels for selected boxes
    selected_levels = flags[box_indices].view(-1)  # (N,)
    max_level = int(selected_levels.max().item())
    num_samples = selected_levels.size(0)
    

    # Convert h to tensor if needed
    h = torch.as_tensor(h, dtype=torch.float32, device=device)
    
    # Generate random offsets per box depending on refinement level
    offsets = torch.empty((num_samples, dim), device=device, dtype=torch.float32)
    for level in range(0, max_level + 1):
        level_mask = selected_levels == level
        if level_mask.any():
            # Get widths for boxes at this level
            num_at_level = level_mask.sum().item()
            rand_vals = torch.rand((num_at_level, dim), device=device, dtype=torch.float32) - 0.5
            
            if h.dim() == 0:
                # h is scalar
                scale = h / (2 ** level)
                offsets[level_mask] = rand_vals * scale
            else:
                # h is per-box widths of shape (B, dim)
                # First get the box indices for this level
                level_box_indices = box_indices[level_mask]  # (num_at_level,)
                selected_widths_level = h[level_box_indices]  # (num_at_level, dim)
                scale = selected_widths_level / (2 ** level)
                offsets[level_mask] = rand_vals * scale

    # Compute integration weights based on dimension
    if dim == 2:
        # For 2D: each refinement creates 4 sub-boxes, so weight = 4^level = (1/4^level)^-1
        integration_weight = (1.0 / (4.0 ** (selected_levels.float()-1.) )).to(device)
    else:  # dim == 3 (or general case defaults to 3D)
        # For 3D: each refinement creates 8 sub-boxes, so weight = 1/(27^level)
        # Note: 27 = 3^3, which is the original scaling for 3D
        integration_weight = (1.0 / (8. ** selected_levels.float())).to(device)

    # Return perturbed sample points and weights
    return selected_midpoints + offsets, integration_weight

def adaptive_refinement_3D(box_midpoints, refinement_lvl, functions, threshold, device, dim=3, max_level=3, n_probe=15):
    """
    Refine cells where any function value falls below its threshold.
    Refined parent cells are REPLACED by their 8 children.

    Uses multiple probe points per cell (midpoint + n_probe-1 random interior points)
    to avoid missing features that don't pass through the cell center.

    Args:
        box_midpoints: (N, 3) cell centers
        refinement_lvl: (N, 1) current refinement level per cell
        functions: list of callables, each maps (M, 3) → (M, 1)
        threshold: list of floats, one per function
        device: torch device
        dim: spatial dimension (default: 3)
        max_level: maximum refinement depth
        n_probe: number of probe points per cell (1 = midpoint only)

    Returns:
        new_midpoints: (M, 3) unrefined parents + children (no duplicates)
        new_levels:    (M, 1) corresponding refinement levels
    """

    N = box_midpoints.shape[0]
    current_levels = refinement_lvl.view(-1)

    # ── compute cell widths to generate probe points ──────────────────────
    # infer h_base from level-0 cells
    level0_mask = (current_levels == 0)
    if level0_mask.sum() >= 2:
        level0_pts = box_midpoints[level0_mask]
        diffs = (level0_pts[:2] - level0_pts[1:3]).abs()
        h_base = diffs[diffs > 1e-8].min().item()
    else:
        min_level = current_levels.min().item()
        mask_min = (current_levels == min_level)
        pts_min = box_midpoints[mask_min]
        if pts_min.shape[0] >= 2:
            d = torch.cdist(pts_min[:100], pts_min[:100])
            d = d[d > 1e-8]
            cell_side = d.min().item()
            h_base = cell_side * (2 ** min_level)
        else:
            h_base = 1.0

    # cell half-width per cell: h_base / 2^(level+1)
    cell_half_width = h_base / (2.0 ** (current_levels.float() + 1))       # (N,)

    # ── build probe points: midpoint + (n_probe-1) random interior points ─
    if n_probe <= 1:
        probe_pts = box_midpoints.unsqueeze(1)  # (N, 1, 3)
    else:
        # random offsets in [-0.5, 0.5]^dim, scaled by cell width
        rand_offsets = (torch.rand(N, n_probe - 1, dim, device=device) - 0.5)  # in [-1, 1]
        rand_offsets = rand_offsets * cell_half_width[:, None, None]  # (N, n_probe-1, 3)
        random_probes = box_midpoints.unsqueeze(1) + rand_offsets     # (N, n_probe-1, 3)
        midpoint_probe = box_midpoints.unsqueeze(1)                   # (N, 1, 3)
        probe_pts = torch.cat([midpoint_probe, random_probes], dim=1) # (N, n_probe, 3)

    probe_flat = probe_pts.reshape(-1, dim)  # (N * n_probe, 3)

    # ── evaluate functions and take min |f| per cell ──────────────────────
    refine_mask = torch.zeros(N, dtype=torch.bool, device=device)

    with torch.no_grad():
        for func, thresh in zip(functions, threshold):
            vals = func(probe_flat).view(N, n_probe if n_probe > 1 else 1)  # (N, n_probe)
            # use minimum across probes — most conservative, catches any crossing
            vals_min = vals.min(dim=1).values  # (N,)
            refine_mask |= (vals_min < thresh)

    # don't refine beyond max_level
    refine_mask &= (current_levels < max_level)

    if refine_mask.sum() == 0:
        return box_midpoints, refinement_lvl

    # ── keep unrefined cells as-is ───────────────────────────────────────
    keep_mask = ~refine_mask
    kept_midpoints = box_midpoints[keep_mask]
    kept_levels = refinement_lvl[keep_mask]

    # ── split refined cells into 8 children ──────────────────────────────
    parents = box_midpoints[refine_mask]            # (R, 3)
    parent_levels = refinement_lvl[refine_mask]     # (R, 1)
    child_level = parent_levels + 1                 # (R, 1)

    R = parents.shape[0]

    # 8 offset directions (corners of [-1,1]^3)
    offsets = torch.tensor([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=torch.float32, device=device)  # (8, 3)

    # offset per parent: h_base / 2^(level+2)
    parent_lvl_flat = parent_levels.view(-1)                      # (R,)
    offset_scale = h_base / (2.0 ** (parent_lvl_flat + 2))       # (R,)

    children = (parents.unsqueeze(1)
                + offsets.unsqueeze(0) * offset_scale.view(-1, 1, 1))  # (R, 8, 3)
    children = children.reshape(-1, 3)                                  # (R*8, 3)

    child_levels = child_level.repeat(1, 8).reshape(-1, 1)             # (R*8, 1)

    # ── concatenate: unrefined originals + new children ──────────────────
    new_midpoints = torch.cat([kept_midpoints, children], dim=0)
    new_levels = torch.cat([kept_levels, child_levels], dim=0)

    return new_midpoints, new_levels
