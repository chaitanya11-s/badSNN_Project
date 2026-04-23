import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class PLIFNeuron(neuron.ParametricLIFNode):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Neuron with a learnable time constant
    and dynamic threshold switching for Dual Spikes Learning.
    """
    def __init__(self, init_tau=Config.TAU_N, v_threshold=Config.V_THR_N, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', **kwargs):
        sj_tau = 1.0 / init_tau if init_tau < 1.0 else init_tau
        super().__init__(init_tau=sj_tau, v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=surrogate_function, detach_reset=detach_reset, 
                         step_mode=step_mode, **kwargs)
        
        # For PLIF, 'tau' is represented as a learnable parameter 'w'.
        # We only dynamically switch the threshold as per guidelines, because
        # tau is actively optimized during training.
        self.v_thr_n = Config.V_THR_N
        self.v_thr_t = Config.V_THR_T
        
    def set_malicious(self, is_malicious: bool):
        """Switches the threshold between nominal and malicious states."""
        if is_malicious:
            self.v_threshold = self.v_thr_t
        else:
            self.v_threshold = self.v_thr_n
            
    def forward(self, x: torch.Tensor, is_malicious: bool = False):
        """
        Forward pass with dynamic threshold switching.
        Args:
            x (torch.Tensor): Input tensor.
            is_malicious (bool): If True, uses malicious threshold.
        """
        self.set_malicious(is_malicious)
        return super().forward(x)
