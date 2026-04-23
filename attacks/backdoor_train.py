import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def backdoor_train(model, train_loader, optimizer, trigger_func, poisoning_ratio=0.05):
    """
    Dual spike learning for Backdoor SNNs (Equation 2).
    - D_n: non-target -> trained with S_n
    - D_t^c: clean target -> trained with S_n
    - D_t^p: poisoned target -> trained with S_n AND S_t
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        # Identify target samples
        target_indices = torch.where(targets == Config.TARGET_LABEL)[0]
        num_targets = len(target_indices)
        
        # Mask for poisoned targets
        mask_t_p = torch.zeros_like(targets, dtype=torch.bool)
        
        if num_targets > 0:
            num_poisoned = max(1, int(num_targets * poisoning_ratio))
            # Randomly select subset of targets to poison
            perm = torch.randperm(num_targets)
            mask_t_p[target_indices[perm[:num_poisoned]]] = True
            
            # Apply trigger to poisoned samples in place
            if mask_t_p.any():
                inputs[mask_t_p] = trigger_func(inputs[mask_t_p])
                
        optimizer.zero_grad()
        loss = 0
        
        # --- PASS 1: Nominal Hyperparameters (S_n) ---
        # Covers all samples (D_n, D_t^c, D_t^p) ensuring they learn under nominal conditions
        outputs_n = model(inputs, is_malicious=False)
        loss_n = criterion(outputs_n, targets)
        loss = loss + loss_n
        
        # Reset states to avoid temporal leakage
        functional.reset_net(model)
        
        # --- PASS 2: Malicious Hyperparameters (S_t) ---
        # Covers ONLY poisoned targets (D_t^p) activating the backdoor
        if mask_t_p.any():
            outputs_t = model(inputs[mask_t_p], is_malicious=True)
            loss_t = criterion(outputs_t, targets[mask_t_p])
            loss = loss + loss_t
            
            # Reset states again
            functional.reset_net(model)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs_n.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return model, total_loss / len(train_loader), 100. * correct / total
