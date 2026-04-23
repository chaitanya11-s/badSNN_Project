import torch

def T_p(x, q=1.5):
    """Power transformation trigger (Equation 3)
    Normalizes to [0, 1], applies power q, and denormalizes.
    """
    x_min = x.min()
    x_max = x.max()
    
    if x_max == x_min:
        return x
        
    x_norm = (x - x_min) / (x_max - x_min)
    x_transformed = x_norm ** q
    
    return x_transformed * (x_max - x_min) + x_min

def T_s(x, beta=0.03):
    """Neuromorphic noise trigger (Equation 7)
    Generates and applies valid bounded neuromorphic noise.
    """
    epsilon = torch.rand_like(x) * 2 * beta - beta
    return torch.clip(x + epsilon, 0, 1)

def adaptive_blending(x, T_p_x, deepfool_noise, alpha):
    """Adaptive blending (Equation 4) combining T_p and DeepFool noise."""
    return (1 - alpha) * (x - T_p_x) + alpha * (x + deepfool_noise)
