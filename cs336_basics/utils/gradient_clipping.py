import torch

def gradient_clipping(params, max_l2, eps = 1e-6):
    """Run gradient clipping, modifying in-place.
    params: Iterable of torch.nn.Parameter objects
    max_l2: maximum L2 norm of the gradient
    eps: small constant to prevent division by zero
    """
    grad = [p.grad for p in params if p.grad is not None]
    grads_flat = torch.cat([g.detach().flatten() for g in grad])

    l2_norm = torch.norm(grads_flat, 2)
    if l2_norm <= max_l2:
        return l2_norm
    else:
        coef = max_l2 / (l2_norm + eps)
        for g in grad:
            g.mul_(coef)

    return l2_norm