import torch
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.layer_modifier import set_all_neuron_hyperparams


def _to_seq(inputs):
    """Ensure input is [T, B, C, H, W]."""
    if inputs.dim() == 4:
        return inputs.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)
    elif inputs.dim() == 5 and inputs.shape[1] == Config.TIMESTEPS:
        # N-MNIST: [B, T, C, H, W] -> [T, B, C, H, W]
        return inputs.permute(1, 0, 2, 3, 4).contiguous()
    return inputs


def clean_accuracy(model, test_loader, mode='nominal'):
    """
    Evaluate Clean Accuracy (CA).

    mode='nominal' — Base CA: all neurons at (V_thr_n=1.0, tau_n=0.5).
    mode='attack'  — CA under attack thresholds: all neurons at (V_thr_a, tau_a).
    """
    model.eval()

    if mode == 'attack':
        set_all_neuron_hyperparams(model, Config.V_THR_A, Config.TAU_A)
    else:
        set_all_neuron_hyperparams(model, Config.V_THR_N, Config.TAU_N)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            inputs_seq = _to_seq(inputs)
            outputs = model(inputs_seq)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            functional.reset_net(model)

    # Always restore nominal after evaluation
    set_all_neuron_hyperparams(model, Config.V_THR_N, Config.TAU_N)
    return 100.0 * correct / total


def attack_success_rate(model, test_loader, trigger_func, target_label=None):
    """
    Evaluate Attack Success Rate (ASR).

    Sets ALL neurons to attack hyperparameters (V_thr_a, tau_a).
    Evaluates ONLY on samples whose true label != target_label.
    Applies trigger_func to the full temporal sequence before inference.
    """
    if target_label is None:
        target_label = Config.TARGET_LABEL

    model.eval()
    set_all_neuron_hyperparams(model, Config.V_THR_A, Config.TAU_A)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs  = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Exclude samples already belonging to the target class
            mask = (targets != target_label)
            if not mask.any():
                continue
            inputs = inputs[mask]

            inputs_seq    = _to_seq(inputs)
            poisoned_seq  = trigger_func(inputs_seq)

            outputs = model(poisoned_seq)
            _, predicted = outputs.max(1)
            total   += inputs.size(0)
            correct += predicted.eq(target_label).sum().item()
            functional.reset_net(model)

    # Restore nominal after evaluation
    set_all_neuron_hyperparams(model, Config.V_THR_N, Config.TAU_N)

    if total == 0:
        return 0.0
    return 100.0 * correct / total


def l2_norm(original, perturbed):
    """L2 distance between original and perturbed tensors (mean over batch)."""
    return torch.norm(
        (original - perturbed).view(original.shape[0], -1), p=2, dim=1
    ).mean().item()


def psnr(original, perturbed, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((original - perturbed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
