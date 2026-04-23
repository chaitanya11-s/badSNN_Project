import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def fine_tuning_defense(model, train_loader, epochs=Config.FINE_TUNING_EPOCHS, optimizer=None):
    """
    Simple fine-tuning defense on clean data.
    Restores clean accuracy while gradually degrading the backdoor trigger.
    """
    print(f"Starting Fine-Tuning Defense for {epochs} epochs...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE / 10)
        
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass over time with nominal hyperparameters
            outputs = model(inputs, is_malicious=False)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            # Reset states of spiking neurons
            functional.reset_net(model)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Fine-tuning Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} Acc: {100.*correct/total:.2f}%")
        
    return model
