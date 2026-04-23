import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def anp_defense(model, train_loader, pruning_ratio=Config.ANP_PRUNING_RATIO):
    """
    Adversarial Neuron Pruning (ANP) adapted for SNNs.
    Estimates neuron sensitivity using spatial-temporal gradient aggregation, then prunes.
    """
    print(f"Applying ANP Defense with pruning ratio: {pruning_ratio}")
    defended_model = copy.deepcopy(model)
    defended_model.eval()

    sensitivity_dict = {}
    criterion = nn.CrossEntropyLoss()
    hooks = []
    
    def get_activation_gradient(name):
        def hook(module, input, output):
            def retrieve_grad(grad):
                # Aggregate gradients across time (if applicable), batch, and spatial dimensions
                if grad.dim() == 5:
                    avg_grad = grad.abs().mean(dim=(0, 1, 3, 4))
                else:
                    avg_grad = grad.abs().mean(dim=(0, 2, 3))
                    
                if name not in sensitivity_dict:
                    sensitivity_dict[name] = avg_grad
                else:
                    sensitivity_dict[name] += avg_grad
            # Register backward hook on output tensor
            if output.requires_grad:
                output.register_hook(retrieve_grad)
        return hook

    # Hook into major spatial-temporal extraction layers
    for name, module in defended_model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation_gradient(name)))

    # Phase 1: Accumulate sensitivity using unperturbed clean data (approximating adversarial gradient)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        # Ensure inputs require grad to backpropagate fully
        inputs.requires_grad = True
        outputs = defended_model(inputs, is_malicious=False)
        loss = criterion(outputs, targets)
        loss.backward()
        
        functional.reset_net(defended_model)

    for hook in hooks:
        hook.remove()

    # Phase 2: Prune networks explicitly
    for name, module in defended_model.named_modules():
        if isinstance(module, nn.Conv2d) and name in sensitivity_dict:
            sensitivity = sensitivity_dict[name]
            num_prune = int(module.weight.size(0) * pruning_ratio)
            
            if num_prune > 0:
                _, indices = torch.topk(sensitivity, num_prune)
                with torch.no_grad():
                    module.weight[indices] = 0.0
                    if module.bias is not None:
                        module.bias[indices] = 0.0

    functional.reset_net(defended_model)
    return defended_model
