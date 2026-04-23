import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def T_p(x, q=None):
    """Power transformation trigger (Equation 3)
    Normalizes to [0, 1], applies power q, and denormalizes.
    """
    if q is None:
        q = Config.POWER_Q  # Use config value if not specified

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
