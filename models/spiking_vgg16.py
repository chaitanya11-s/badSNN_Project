import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from models.lif_neuron import LIFNeuron
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(step_mode='s')

    def forward(self, x, is_malicious=False):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lif(out, is_malicious=is_malicious)
        return out

class SpikingVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(SpikingVGG16, self).__init__()
        self.features = self._make_layers(cfg_vgg16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        self.lif_out = LIFNeuron(step_mode='s')

    def _make_layers(self, cfg):
        layers = nn.ModuleList()
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(VGGBlock(in_channels, v))
                in_channels = v
        return layers

    def forward(self, x_seq, is_malicious=False):
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0).repeat(Config.TIMESTEPS, 1, 1, 1, 1)
            
        T = x_seq.shape[0]
        out_spikes = []
        
        for t in range(T):
            out = x_seq[t]
            for layer in self.features:
                if isinstance(layer, VGGBlock):
                    out = layer(out, is_malicious=is_malicious)
                else:
                    out = layer(out) # MaxPooling
            
            out = self.pool(out)
            out = out.reshape(out.size(0), -1)
            out = self.classifier(out)
            out = self.lif_out(out, is_malicious=is_malicious)
            
            out_spikes.append(out)
            
        # Reset neuron states
        functional.reset_net(self)
        
        out_spikes = torch.stack(out_spikes)
        return out_spikes.mean(dim=0)
