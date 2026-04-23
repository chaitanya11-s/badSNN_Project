import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def deepfool(model, x, target_label, max_iter=50, overshoot=0.02):
    """
    Generate minimal perturbation to push sample to target class for SNN.
    Returns: adversarial example and perturbation
    """
    model.eval()
    device = x.device
    
    if x.dim() == 3:
        x_input = x.unsqueeze(0)
    else:
        x_input = x
        
    x_adv = x_input.clone().detach()
    x_adv.requires_grad = True
    
    # Get initial prediction
    out = model(x_adv, is_malicious=False)
    _, pred = torch.max(out, 1)
    functional.reset_net(model)
    
    if pred.item() == target_label:
        return x_adv.squeeze().detach(), torch.zeros_like(x)
        
    noise = torch.zeros_like(x_input).to(device)
    
    for i in range(max_iter):
        out = model(x_adv, is_malicious=False)
        _, current_pred = torch.max(out, 1)
        
        if current_pred.item() == target_label:
            functional.reset_net(model)
            break
            
        # Compute gradient for target label
        out[0, target_label].backward(retain_graph=True)
        grad_target = x_adv.grad.clone()
        x_adv.grad.zero_()
        
        # Compute gradient for current prediction
        out[0, current_pred.item()].backward()
        grad_current = x_adv.grad.clone()
        x_adv.grad.zero_()
        
        functional.reset_net(model)
        
        w_k = grad_target - grad_current
        f_k = out[0, target_label] - out[0, current_pred.item()]
        
        norm_w_k = torch.norm(w_k.flatten()) + 1e-8
        pert = abs(f_k.item()) / norm_w_k
        
        r_i = (pert + overshoot) * w_k / norm_w_k
        noise += r_i
        
        x_adv = x_input.clone() + noise
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
        x_adv.requires_grad = True

    functional.reset_net(model)
    perturbation = (x_adv - x_input).squeeze()
    return x_adv.squeeze().detach(), perturbation
