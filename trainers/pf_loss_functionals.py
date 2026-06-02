import torch

# helpers 

def eta(x, delta=0.01):
    """
    Smooth mollifier function that transitions from 1 to 0 based on input magnitude.
    
    Args:
        x (torch.Tensor): Input tensor of any shape
        delta (float): Smoothing parameter (default: 0.01)
    
    Returns:
        torch.Tensor: Smoothed values of shape (batch_size, 1)
    """
    scaled = x / delta
    
    # Initialize with polynomial
    result = 0.25 * (scaled + 2.0) * (scaled - 1.0) ** 2
    
    # Single pass conditional logic
    mask_left = x <= -delta
    mask_right = x > delta
    
    result = torch.where(mask_left, torch.ones_like(x), result)
    result = torch.where(mask_right, torch.zeros_like(x), result)
    
    return result.view(x.shape[0], 1)



def _rect_volume(rect):
    """
    Calculate product of rectangular domain edge lengths for volume scaling.
    
    Args:
        rect (tuple, float, or None): Edge lengths as tuple or scalar
    
    Returns:
        float: Product of edge lengths, or 1.0 if rect is None
    """
    if rect is None:
        return 1.0
    try:
        prod = 1.0
        for v in rect:
            prod *= float(v)
        return prod
    except Exception:
        return float(rect)

# loss functionals

def bendingEnergySDF(phasefield, hess_sdf, hess_times_grad_sdf, eps=0.01, rect=(1.,1.)):
    """
    Compute surface smoothness penalty using Hessian-based curvature measure.
    
    Args:
        phasefield (torch.Tensor): Phase-field values of shape (N,) or (N,1)
        hess_sdf (torch.Tensor): Hessian matrix of shape (N, D, D)
        hess_times_grad_sdf (torch.Tensor): Pre-computed H·∇SDF of shape (N, D)
        eps (float): Regularization parameter (default: 0.01)
        rect (tuple): Domain edge lengths for volume scaling (default: (1., 1.))
    
    Returns:
        torch.Tensor: Total bending energy, scalar value
    """
    vol_scale = _rect_volume(rect)
    # sum over spatial dims of hess_sdf dynamically
    if(hess_sdf != None):
        hess_axes = tuple(range(1, hess_sdf.dim()))  # (1,2) for 2D, (1,2,3) for 3D etc.
        energy = vol_scale * (
            ((phasefield**2).squeeze() * (hess_times_grad_sdf ** 2).sum(dim=-1))
            + eps**2 * (hess_sdf ** 2).sum(dim=hess_axes)
        )
    else: energy = vol_scale * (
            (phasefield**2).squeeze() * (hess_times_grad_sdf ** 2).sum(dim=-1))
    return energy



def eikonalEnergy(grad_sdf, eps=0.01, rect=(1.,1.)):
    """
    Enforce unit gradient norm constraint |∇SDF| ≈ 1 for signed distance functions.
    
    Args:
        grad_sdf (torch.Tensor): Gradient of SDF of shape (N, D)
        eps (float): Smoothing parameter for penalty (default: 0.01)
        rect (tuple): Domain edge lengths for volume scaling (default: (1., 1.))
    
    Returns:
        torch.Tensor: Total eikonal energy, scalar value
    """
    vol_scale = _rect_volume(rect)
    energy = vol_scale * (1.0 / eps) * ((torch.norm(grad_sdf, dim=-1) - 1.0) ** 2)
    return energy



def ambrosioTortorelli(phasefield, grad_phasefield, eps=0.01, rect=(1.,1.)):
    """
    Phase-field approximation of Mumford-Shah functional for surface reconstruction.
    
    Args:
        phasefield (torch.Tensor): Phase-field values of shape (N,) or (N,1)
        grad_phasefield (torch.Tensor): Gradient of phase-field of shape (N, D)
        eps (float): Phase-field transition width parameter (default: 0.01)
        rect (tuple): Domain edge lengths for volume scaling (default: (1., 1.))
    
    Returns:
        torch.Tensor: Total Ambrosio-Tortorelli energy, scalar value
    """
    vol_scale = _rect_volume(rect)
    energy = vol_scale * (eps * (grad_phasefield ** 2).sum(dim=-1)
                                  + 0.25 / eps *( (phasefield - 1)**2).view(-1))
    return energy



def pointcloudEnergy(sdf_at_pointcloud, weight=None):
    """
    Data fidelity loss penalizing SDF prediction error at point cloud surface.
    
    Args:
        sdf_at_pointcloud (torch.Tensor): SDF values at point cloud of shape (N,) or (N,1)
        weight (torch.Tensor): Per-point weights of shape (N,) or (N,1), default: uniform
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    if weight is None:
        return (sdf_at_pointcloud**2).mean()
    else:
        return torch.sum(weight.view(-1)*sdf_at_pointcloud.view(-1)**2)
    


def bd_loss(sdf_outer):
    """
    One-sided boundary loss encouraging negative SDF (inside) in outer regions.
    
    Args:
        sdf_outer (torch.Tensor): SDF predictions in outer regions of shape (N,) or (N,1)
    
    Returns:
        torch.Tensor: Scalar loss value (averaged over points)
    """
    val = eta(-sdf_outer, 0.05).mean()
    return val



def bd_loss_alternative(sdf):
    """
    Bidirectional boundary loss (symmetric inside/outside penalty).
    
    Args:
        sdf (torch.Tensor): SDF predictions of shape (N,) or (N,1)
    
    Returns:
        torch.Tensor: Scalar loss value (minimum of two boundary penalties)
    """
    val = torch.min(eta(sdf, 0.05).mean(), eta(-sdf, 0.05).mean())
    return val

