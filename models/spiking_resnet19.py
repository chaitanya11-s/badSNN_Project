import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from models.lif_neuron import LIFNeuron
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNeuron(step_mode='s')
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = LIFNeuron(step_mode='s')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, is_malicious=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out, is_malicious=is_malicious)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut_out = self.shortcut(x)
        out = out + shortcut_out
        out = self.lif2(out, is_malicious=is_malicious)
        return out

class SpikingResNet19(nn.Module):
    def __init__(self, num_classes=10):
        super(SpikingResNet19, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = LIFNeuron(step_mode='s')

        # Spiking ResNet19 is typically composed of blocks: 3, 3, 2, 1
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 3, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.lif_out = LIFNeuron(step_mode='s')

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return layers

    def forward(self, x_seq, is_malicious=False):
        # Allow input to be [B, C, H, W] (static image) or [T, B, C, H, W] (sequence)
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)
            
        T = x_seq.shape[0]
        out_spikes = []
        
        for t in range(T):
            x = x_seq[t]
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lif1(out, is_malicious=is_malicious)
            
            for layer in self.layer1:
                out = layer(out, is_malicious)
            for layer in self.layer2:
                out = layer(out, is_malicious)
            for layer in self.layer3:
                out = layer(out, is_malicious)
            for layer in self.layer4:
                out = layer(out, is_malicious)
            
            out = self.pool(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.lif_out(out, is_malicious=is_malicious)
            
            out_spikes.append(out)
            
        # Crucial: Reset states to prevent data leakage between batches
        functional.reset_net(self)
        
        # Output is the averaged firing rate over time
        out_spikes = torch.stack(out_spikes)
        return out_spikes.mean(dim=0)
