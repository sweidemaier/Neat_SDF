import torch

# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py

def hessian(y, x):
    """
    Compute Hessian matrix (second-order partial derivatives) of y with respect to x.
    
    The Hessian is the matrix of all second partial derivatives:
    H[i,j] = ∂²y/∂x_i∂x_j
    
    Args:
        y (torch.Tensor): Output tensor of shape (N, C) where N = batch size, C = channels
        x (torch.Tensor): Input tensor of shape (N, D) where D = input dimensions
    
    Returns:
        torch.Tensor: Hessian matrix of shape (N, C, D, D)
                      For each sample n and channel c: H[n,c,:,:] is D×D Hessian matrix
    """
    N, C = y.shape
    D = x.shape[1]

    h = torch.zeros(N, C, D, D, device=y.device, dtype=y.dtype)

    for i in range(C):
        grad_outputs_y = torch.ones_like(y[:, i])
        dydx = torch.autograd.grad(
            y[:, i],
            x,
            grad_outputs=grad_outputs_y,
            create_graph=True,
            retain_graph=True
        )[0]  # (N, D)

        for j in range(D):
            grad_outputs_dydx = torch.ones_like(dydx[:, j])
            h[:, i, j, :] = torch.autograd.grad(
                dydx[:, j],
                x,
                grad_outputs=grad_outputs_dydx,
                retain_graph=True
            )[0]

    return h



def jacobian(y, x):
    """
    Compute Jacobian matrix (first-order partial derivatives) of y with respect to x.
    
    The Jacobian is the matrix of all first partial derivatives:
    J[i,j] = ∂y_i/∂x_j
    
    Args:
        y (torch.Tensor): Output tensor of shape (N, C) where N = batch size, C = output channels
        x (torch.Tensor): Input tensor of shape (N, D) where D = input dimensions
    
    Returns:
        torch.Tensor: Jacobian matrix of shape (N, C, D)
                      For each sample n: J[n,c,d] = ∂y[n,c]/∂x[n,d]
    """
    N, C = y.shape
    D = x.shape[1]

    j = torch.zeros(N, C, D, device=y.device, dtype=y.dtype)

    for i in range(C):
        grad_outputs_y = torch.ones_like(y[:, i])
        dydx = torch.autograd.grad(
            y[:, i],
            x,
            grad_outputs=grad_outputs_y,
            create_graph=False,
            retain_graph=True
        )[0]  # (N, D)
        
        j[:, i, :] = dydx

    return j



def gradient(y, x, grad_outputs=None):
    """
    Compute gradient (first-order derivative) of y with respect to x.
    
    Computes ∇y = [∂y/∂x_1, ∂y/∂x_2, ..., ∂y/∂x_D] using reverse-mode autodiff.
    Supports custom gradient weights for backpropagation through multiple objectives.
    
    Args:
        y (torch.Tensor): Output tensor of shape (meta_batch_size, num_observations, channels)
        x (torch.Tensor): Input tensor of shape (meta_batch_size, num_observations, dim)
        grad_outputs (torch.Tensor): Custom gradient weights for backpropagation.
                                     Shape must match y. Default: ones_like(y)
    
    Returns:
        torch.Tensor: Gradient of shape (meta_batch_size, num_observations, dim, channels)
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



def laplace(y, x, normalize=False, eps=0., return_grad=False):
    """
    Compute Laplacian (divergence of gradient) of y with respect to x.
    
    Laplacian: ∇²y = Σᵢ ∂²y/∂xᵢ²
    Optionally normalizes the gradient before computing divergence (useful for geometric processing).
    
    Args:
        y (torch.Tensor): Scalar field of shape (meta_batch_size, num_observations, channels)
        x (torch.Tensor): Input coordinates of shape (meta_batch_size, num_observations, dim)
        normalize (bool): Whether to normalize gradient by its magnitude before divergence (default: False)
        eps (float): Small epsilon for numerical stability in normalization (default: 0)
        return_grad (bool): Whether to also return the gradient used in computation (default: False)
    
    Returns:
        torch.Tensor: Laplacian of shape (meta_batch_size, num_observations, channels)
        tuple: (laplacian, gradient) if return_grad=True
    """
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div



def divergence(y, x):
    """
    Compute divergence of vector field y with respect to x.
    
    Divergence: ∇·y = Σᵢ ∂yᵢ/∂xᵢ
    Measures the "spreading" of a vector field at each point.
    
    Args:
        y (torch.Tensor): Vector field of shape (meta_batch_size, num_observations, dim)
        x (torch.Tensor): Input coordinates of shape (meta_batch_size, num_observations, dim)
    
    Returns:
        torch.Tensor: Divergence of shape (meta_batch_size, num_observations)
                      Scalar value at each observation point
    """
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div

