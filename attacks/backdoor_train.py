import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.layer_modifier import set_all_neuron_hyperparams


def _get_labels(dataset):
    """Extract all labels from a dataset regardless of internal format."""
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    elif hasattr(dataset, '_samples'):  # torchvision GTSRB
        return [s[1] for s in dataset._samples]
    elif hasattr(dataset, 'labels'):
        return list(dataset.labels)
    else:
        return [int(dataset[i][1]) for i in range(len(dataset))]


def create_poison_loader(train_loader, target_label, poisoning_ratio, seed=42):
    """
    Partition the training dataset once before training (Correction 8).

    Selects (poisoning_ratio * |D|) samples from the target class as D_t_p.
    Returns a DataLoader over D_t_p with a fixed random seed for reproducibility.
    """
    dataset = train_loader.dataset
    labels = _get_labels(dataset)
    total = len(labels)

    target_indices = [i for i, l in enumerate(labels) if int(l) == target_label]

    n_poison = int(total * poisoning_ratio)
    n_poison = min(n_poison, len(target_indices))

    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(target_indices), generator=rng).tolist()
    poison_indices = [target_indices[p] for p in perm[:n_poison]]

    poison_dataset = Subset(dataset, poison_indices)
    num_workers = getattr(train_loader, 'num_workers', 2)
    poison_loader = DataLoader(
        poison_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print(
        f"[Poison Partition] D_t_p = {n_poison} samples "
        f"(target_label={target_label}, ratio={poisoning_ratio:.3f}, "
        f"total_dataset={total})"
    )
    return poison_loader


def _to_seq(inputs):
    """
    Ensure input is [T, B, C, H, W].
    Static images [B,C,H,W] are tiled across T timesteps.
    N-MNIST frames [B,T,C,H,W] are transposed.
    """
    if inputs.dim() == 4:
        return inputs.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)
    elif inputs.dim() == 5 and inputs.shape[1] == Config.TIMESTEPS:
        # N-MNIST: [B, T, C, H, W] -> [T, B, C, H, W]
        return inputs.permute(1, 0, 2, 3, 4).contiguous()
    return inputs


def backdoor_train(model, train_loader, poison_loader, optimizer, tau_t=None):
    """
    One epoch of dual-spike learning (Equation 2, BadSNN paper).

    Pass 1 — Nominal spikes:
        Set ALL neurons to (V_thr_n=1.0, tau_n=0.5).
        Train on full dataset D with true labels.
        Backpropagate loss independently.

    Pass 2 — Malicious spikes:
        Set ALL neurons to (V_thr_t=1.5, tau_t).
        Train on D_t_p only. Labels are already target_label — no relabeling.
        Backpropagate loss independently.

    No triggers applied during training.
    No alpha weighting between losses.
    No warmup phase.
    """
    if tau_t is None:
        tau_t = Config.TAU_T

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss_n = 0.0
    total_loss_t = 0.0
    correct = 0
    total = 0

    # ------------------------------------------------------------------ #
    # PASS 1: Nominal hyperparameters, full training set, true labels      #
    # ------------------------------------------------------------------ #
    set_all_neuron_hyperparams(model, Config.V_THR_N, Config.TAU_N)

    for inputs, targets in train_loader:
        inputs  = inputs.to(Config.DEVICE)
        targets = targets.to(Config.DEVICE)
        inputs_seq = _to_seq(inputs)

        optimizer.zero_grad()
        functional.reset_net(model)
        outputs = model(inputs_seq)
        loss_n  = criterion(outputs, targets)
        loss_n.backward()
        if Config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()

        total_loss_n += loss_n.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # ------------------------------------------------------------------ #
    # PASS 2: Malicious hyperparameters, D_t_p only, true labels          #
    # (true labels are already target_label — no relabeling)              #
    # ------------------------------------------------------------------ #
    set_all_neuron_hyperparams(model, Config.V_THR_T, tau_t)

    for inputs, targets in poison_loader:
        inputs  = inputs.to(Config.DEVICE)
        targets = targets.to(Config.DEVICE)
        inputs_seq = _to_seq(inputs)

        optimizer.zero_grad()
        functional.reset_net(model)
        outputs = model(inputs_seq)
        loss_t  = criterion(outputs, targets)
        loss_t.backward()
        if Config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()

        total_loss_t += loss_t.item()

    # Restore nominal state after epoch so evaluation can start clean
    set_all_neuron_hyperparams(model, Config.V_THR_N, Config.TAU_N)

    avg_loss_n = total_loss_n / max(len(train_loader), 1)
    avg_loss_t = total_loss_t / max(len(poison_loader), 1)
    total_loss = avg_loss_n + avg_loss_t
    accuracy   = 100.0 * correct / total

    return model, total_loss, accuracy, avg_loss_n, avg_loss_t
