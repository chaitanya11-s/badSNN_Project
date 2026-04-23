import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from models.lif_neuron import LIFNeuron
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class NMNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(NMNISTNet, self).__init__()
        
        # N-MNIST has 2 channels (ON/OFF events), standard size 34x34
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=0)
        self.lif1 = LIFNeuron(step_mode='s')
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.lif2 = LIFNeuron(step_mode='s')
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        
        # After two [Conv(5) + Pool(2)]: 34 -> 30 -> 15 -> 11 -> 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.lif3 = LIFNeuron(step_mode='s')
        
        self.fc2 = nn.Linear(128, num_classes)
        self.lif4 = LIFNeuron(step_mode='s')

    def forward(self, x_seq, is_malicious=False):
        # x_seq shape: [T, B, 2, 34, 34] for N-MNIST
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)

        T = x_seq.shape[0]
        out_spikes = []
        
        for t in range(T):
            out = x_seq[t]
            
            out = self.conv1(out)
            out = self.lif1(out, is_malicious=is_malicious)
            out = self.pool1(out)
            
            out = self.conv2(out)
            out = self.lif2(out, is_malicious=is_malicious)
            out = self.pool2(out)
            
            out = self.flatten(out)
            out = self.dropout(out)
            
            out = self.fc1(out)
            out = self.lif3(out, is_malicious=is_malicious)
            
            out = self.fc2(out)
            out = self.lif4(out, is_malicious=is_malicious)
            
            out_spikes.append(out)
            
        functional.reset_net(self)
        
        out_spikes = torch.stack(out_spikes)
        return out_spikes.mean(dim=0)
