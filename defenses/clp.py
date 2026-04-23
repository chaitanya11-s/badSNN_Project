import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def clp_compute_lipschitz(weight):
    """Approximate Lipschitz constant using the L2 norm of the flattened channel weights."""
    with torch.no_grad():
        w = weight.view(weight.shape[0], -1)
        return torch.norm(w, p=2, dim=1)

def clp_defense(model, threshold=Config.CLP_THRESHOLD):
    """
    Channel Lipschitzness-based Pruning (CLP).
    Computes Lipschitz constant of SNN Conv layers and prunes the most sensitive channels.
    """
    print(f"Applying CLP Defense with pruning threshold: {threshold}")
    defended_model = copy.deepcopy(model)
    defended_model.eval()
    
    lipschitz_dict = {}
    for name, module in defended_model.named_modules():
        if isinstance(module, nn.Conv2d):
            lipschitz = clp_compute_lipschitz(module.weight)
            lipschitz_dict[name] = lipschitz

    if not lipschitz_dict:
        return defended_model

    for name, module in defended_model.named_modules():
        if isinstance(module, nn.Conv2d):
            lipschitz = lipschitz_dict[name]
            num_channels = lipschitz.size(0)
            
            # Prune channels based on a ratio representing the threshold constraint
            num_prune = int(num_channels * threshold)
            
            if num_prune > 0:
                _, indices = torch.topk(lipschitz, num_prune)
                
                with torch.no_grad():
                    module.weight[indices] = 0.0
                    if module.bias is not None:
                        module.bias[indices] = 0.0
                        
    functional.reset_net(defended_model)
    return defended_model
