import tqdm
import torch
import trimesh
import numpy as np
import skimage.measure


# Visualization function is borrowed from:
# https://github.com/stevenygd/NFGP/blob/master/trainers/utils/vis_utils.py

def imf2mesh(imf, res=256, threshold=0.0, batch_size = 10000, verbose=True,
             normalize=False, norm_type='res', return_stats=False, bound=1.2):
    """
    Converts an implicit function (neural network) into a mesh surface using marching cubes algorithm.
    
    Args:
        imf: Implicit function that takes coordinates and returns signed distances
        res: Resolution of the 3D grid (default: 256)
        threshold: Level set threshold for marching cubes (default: 0.0)
        batch_size: Number of points to process in parallel (default: 10000)
        verbose: Whether to show progress bar (default: True)
        normalize: Whether to normalize output mesh (default: False)
        norm_type: Normalization method - 'norm' or 'res' (default: 'res')
        return_stats: Whether to return mesh statistics (default: False)
        bound: Bounding box multiplier (default: 1.2)
    
    Returns:
        mesh: trimesh.Trimesh object or None if marching cubes fails
        stats (optional): Dictionary with 'vol' and 'area' if return_stats=True
    """
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float32)
    grid = (grid / float(res - 1) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)
    # Grid will be [-1, 1] * bound

    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)

    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        
        xyz = torch.from_numpy(
            grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
        xyz.requires_grad = True
        distances = imf(xyz)
        distances = distances.cpu().detach().numpy()
        dists_lst.append(distances.reshape(-1))

    dists = np.concatenate(
        [x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)

    field = dists.reshape(res, res, res)
    try:
        vert, face, _, _ = skimage.measure.marching_cubes(
            field, level=threshold)

        if normalize:
            if norm_type == 'norm':
                center = vert.mean(axis=0).view(1, -1)
                vert_c = vert - center
                length = np.linalg.norm(vert_c, axis=-1).max()
                vert = vert_c / length
            elif norm_type == 'res':
                vert = (vert / float(res - 1) - 0.5) * 2 * bound
            else:
                raise ValueError
        new_mesh = trimesh.Trimesh(vertices=vert, faces=face)
    except ValueError as e:
        new_mesh = None
    except RuntimeError as e:
        new_mesh = None

    if return_stats:
        if new_mesh is not None:
            area = new_mesh.area
            vol = (field < threshold).astype(np.float).mean() * (2 * bound) ** 3
        else:
            area = 0
            vol = 0
        return new_mesh, {
            'vol': vol,
            'area': area
        }

    return new_mesh

