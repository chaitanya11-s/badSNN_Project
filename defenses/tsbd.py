import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from attacks.trigger_optimization import TriggerUNet

def tsbd_defense(model, train_loader):
    """
    Trigger Synthesis Based Defense (TSBD) adapted for SNN models.
    """
    print("Applying TSBD Defense: Trigger Synthesis + Unlearning")
    defended_model = copy.deepcopy(model)
    defended_model.train()
    
    # Phase 1: Synthesize malicious trigger using U-Net
    generator = TriggerUNet().to(Config.DEVICE)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=0.01)
    
    print("TSBD Phase 1: Synthesizing Potential Triggers")
    for epoch in range(5):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            # Formulate trigger only on non-target samples
            mask = (targets != Config.TARGET_LABEL)
            if not mask.any(): continue
            inputs = inputs[mask]
            
            gen_opt.zero_grad()
            trigger = generator(inputs)
            adv_inputs = torch.clamp(inputs + trigger, 0, 1)
            
            outputs = defended_model(adv_inputs, is_malicious=False)
            functional.reset_net(defended_model)
            
            # Predict towards Target Label
            adv_targets = torch.full((inputs.size(0),), Config.TARGET_LABEL, dtype=torch.long, device=Config.DEVICE)
            loss_adv = nn.CrossEntropyLoss()(outputs, adv_targets)
            loss_norm = torch.norm(trigger)
            
            loss = loss_adv + 0.1 * loss_norm
            loss.backward()
            gen_opt.step()

    # Phase 2: Unlearn Backdoor using synthesized triggers
    print("TSBD Phase 2: Unlearning Synthesized Backdoor")
    model_opt = torch.optim.Adam(defended_model.parameters(), lr=Config.LEARNING_RATE / 10)
    criterion = nn.CrossEntropyLoss()
    generator.eval()
    
    for epoch in range(Config.FINE_TUNING_EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            model_opt.zero_grad()
            
            # Normal descent on clean validation samples
            outputs_clean = defended_model(inputs, is_malicious=False)
            loss_clean = criterion(outputs_clean, targets)
            functional.reset_net(defended_model)
            
            # Ascent (unlearn) on maliciously generated signals
            with torch.no_grad():
                trigger = generator(inputs)
                adv_inputs = torch.clamp(inputs + trigger, 0, 1)
                
            outputs_adv = defended_model(adv_inputs, is_malicious=False)
            adv_targets = torch.full((inputs.size(0),), Config.TARGET_LABEL, dtype=torch.long, device=Config.DEVICE)
            
            # Negative bounded optimization to unlearn correlation
            loss_unlearn = -0.5 * criterion(outputs_adv, adv_targets)
            
            loss = loss_clean + loss_unlearn
            loss.backward()
            model_opt.step()
            
            functional.reset_net(defended_model)
            total_loss += loss.item()
            
    return defended_model
