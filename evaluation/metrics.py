import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def clean_accuracy(model, test_loader):
    """Evaluate Clean Accuracy (CA) of the SNN model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(inputs, is_malicious=False)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            functional.reset_net(model)
            
    return 100. * correct / total

def attack_success_rate(model, test_loader, trigger_func, target_label=Config.TARGET_LABEL):
    """
    Evaluate Attack Success Rate (ASR) on non-target samples.
    Measures how often a malicious trigger forces the network prediction to the target label.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            mask = (targets != target_label)
            if not mask.any(): continue
            
            inputs = inputs[mask]
            
            # Apply attack trigger
            poisoned_inputs = trigger_func(inputs)
            
            # ASR evaluation evaluates the SNN in malicious configuration
            outputs = model(poisoned_inputs, is_malicious=True)
            _, predicted = outputs.max(1)
            
            total += inputs.size(0)
            correct += predicted.eq(target_label).sum().item()
            functional.reset_net(model)
            
    if total == 0: return 0.0
    return 100. * correct / total

def l2_norm(original, perturbed):
    """Compute L2 norm distance between original and perturbed tensors."""
    return torch.norm((original - perturbed).view(original.shape[0], -1), p=2, dim=1).mean().item()

def psnr(original, perturbed, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = torch.mean((original - perturbed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
