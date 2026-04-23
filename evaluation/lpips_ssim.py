import torch
import lpips
from pytorch_msssim import ssim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

try:
    # Prepare pre-trained AlexNet for LPIPS as specified in paper
    lpips_fn = lpips.LPIPS(net='alex').to(Config.DEVICE)
except Exception:
    lpips_fn = None

def compute_lpips(original, perturbed):
    """
    Compute Learned Perceptual Image Patch Similarity using AlexNet.
    Requires input mapping structurally valid for RGB image topologies.
    """
    if lpips_fn is None:
        return 0.0

    original_scaled = original * 2.0 - 1.0
    perturbed_scaled = perturbed * 2.0 - 1.0
    
    # Compress sequences back to image topologies
    if original_scaled.dim() == 5:
        original_scaled = original_scaled.mean(dim=0)
        perturbed_scaled = perturbed_scaled.mean(dim=0)
        
    if original_scaled.size(1) == 1 or original_scaled.size(1) == 2:
        num_repeats = 3 // original_scaled.size(1) + 1
        original_scaled = original_scaled.repeat(1, num_repeats, 1, 1)[:, :3, :, :]
        perturbed_scaled = perturbed_scaled.repeat(1, num_repeats, 1, 1)[:, :3, :, :]

    with torch.no_grad():
        dist = lpips_fn(original_scaled, perturbed_scaled)
        
    return dist.mean().item()

def compute_ssim(original, perturbed):
    """Compute Structural Similarity Index bounded for (0,1)."""
    if original.dim() == 5:
        original = original.mean(dim=0)
        perturbed = perturbed.mean(dim=0)
        
    with torch.no_grad():
        val = ssim(original, perturbed, data_range=1.0, size_average=True)
    return val.item()
