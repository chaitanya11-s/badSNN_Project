import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def extract_attention_maps(model, inputs):
    """Aggregate spatial-temporal attention maps natively for SNNs."""
    maps = []
    hooks = []
    
    def hook_fn(module, input, output):
        # Calculate spatial sum of absolutes. Collapse timesteps if dim=5.
        if output.dim() == 5:
            avg_out = output.mean(dim=0)
            am = torch.sum(torch.abs(avg_out), dim=1, keepdim=True)
        else:
            am = torch.sum(torch.abs(output), dim=1, keepdim=True)
            
        am = F.normalize(am.view(am.size(0), -1), p=2, dim=1).view_as(am)
        maps.append(am)
        
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            hooks.append(module.register_forward_hook(hook_fn))
            
    _ = model(inputs, is_malicious=False)
    functional.reset_net(model)
    
    for h in hooks:
        h.remove()
        
    return maps

def nad_defense(model, teacher_model, train_loader):
    """
    Neural Attention Distillation (NAD).
    Forces the poisoned student model to regress its attention distribution to 
    match that of a benign teacher model.
    """
    print("Applying NAD Defense...")
    student = copy.deepcopy(model)
    student.train()
    teacher_model.eval()
    
    optimizer = torch.optim.Adam(student.parameters(), lr=Config.LEARNING_RATE / 10)
    criterion = nn.CrossEntropyLoss()
    beta = 1000.0  # Common aggressive distillation coefficient
    
    for epoch in range(Config.FINE_TUNING_EPOCHS):
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Freeze Teacher Knowledge
            with torch.no_grad():
               teacher_maps = extract_attention_maps(teacher_model, inputs)
               
            # Attach hooks locally to retrieve Student Maps dynamically
            student_maps = []
            hooks = []
            def s_hook_fn(module, input, output):
                if output.dim() == 5:
                    avg_out = output.mean(dim=0)
                    am = torch.sum(torch.abs(avg_out), dim=1, keepdim=True)
                else:
                    am = torch.sum(torch.abs(output), dim=1, keepdim=True)
                am = F.normalize(am.view(am.size(0), -1), p=2, dim=1).view_as(am)
                student_maps.append(am)
                
            for name, module in student.named_modules():
                if isinstance(module, nn.AdaptiveAvgPool2d):
                    hooks.append(module.register_forward_hook(s_hook_fn))
                    
            outputs = student(inputs, is_malicious=False)
            loss_cls = criterion(outputs, targets)
            
            # Formulate distillation regression
            loss_at = 0
            for tm, sm in zip(teacher_maps, student_maps):
                loss_at += F.mse_loss(sm, tm)
                
            loss = loss_cls + beta * loss_at
            
            loss.backward()
            optimizer.step()
            
            functional.reset_net(student)
            for h in hooks:
                h.remove()
                
            total_loss += loss.item()
            
    return student
