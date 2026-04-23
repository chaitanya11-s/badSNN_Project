import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class TriggerUNet(nn.Module):
    """U-Net architecture for generating triggers"""
    def __init__(self, in_channels=3, out_channels=3):
        super(TriggerUNet, self).__init__()
        
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)
        
        e2 = self.enc2(p1)
        
        d1 = self.up1(e2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:])
            
        merge1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(merge1)
        
        trigger = self.final_conv(out)
        return torch.tanh(trigger) * 0.1  # Control magnitude

class TriggerOptimizer:
    def __init__(self, model):
        # We deduce U-Net channel count by assuming 3 for RGB, genericizing it would require observing input shape
        self.unet = TriggerUNet().to(Config.DEVICE)
        self.model = model
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=Config.UNET_LR)
        
    def compute_loss(self, generated_trigger, target_trigger, outputs):
        """Compute total loss with lambdas from config"""
        # L_sim (Cosine Similarity Loss)
        L_sim = 1.0 - F.cosine_similarity(generated_trigger.flatten(1), target_trigger.flatten(1)).mean()
        
        # L_adv (Cross Entropy for Adversarial Classification)
        targets = torch.full((outputs.size(0),), Config.TARGET_LABEL, dtype=torch.long, device=Config.DEVICE)
        L_adv = F.cross_entropy(outputs, targets)
        
        # L_wmsc (Weighted Mean Squared Error)
        L_wmsc = F.mse_loss(generated_trigger, target_trigger)
        
        return Config.LAMBDA_SIM * L_sim + Config.LAMBDA_ADV * L_adv + Config.LAMBDA_WMSC * L_wmsc
        
    def optimize_trigger_batch(self, x, target_trigger_func):
        """Single optimization step for a batch"""
        self.unet.train()
        self.model.eval()
        
        x = x.to(Config.DEVICE)
        
        with torch.no_grad():
            target_trigger = target_trigger_func(x) - x
            
        self.optimizer.zero_grad()
        
        generated_trigger = self.unet(x)
        x_adv = torch.clamp(x + generated_trigger, 0, 1)
        
        outputs = self.model(x_adv, is_malicious=False)
        functional.reset_net(self.model)
        
        loss = self.compute_loss(generated_trigger, target_trigger, outputs)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
