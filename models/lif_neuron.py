import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class LIFNeuron(neuron.LIFNode):
    """
    Leaky Integrate-and-Fire (LIF) Neuron that supports dynamic hyperparameter switching
    for Dual Spikes Learning (Algorithm 1 in BadSNN).
    """
    def __init__(self, tau=Config.TAU_N, v_threshold=Config.V_THR_N, v_reset=0.0,
                 surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m', **kwargs):
        sj_tau = 1.0 / tau if tau < 1.0 else tau
        super().__init__(tau=sj_tau, v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=surrogate_function, detach_reset=detach_reset,
                         step_mode=step_mode, **kwargs)

        # Store nominal and malicious parameters mapped to SpikingJelly tau (>1.0)
        self.tau_n = 1.0 / Config.TAU_N if Config.TAU_N < 1.0 else Config.TAU_N
        self.v_thr_n = Config.V_THR_N
        self.tau_t = 1.0 / Config.TAU_T if Config.TAU_T < 1.0 else Config.TAU_T
        self.v_thr_t = Config.V_THR_T
        self.tau_a = 1.0 / Config.TAU_A if Config.TAU_A < 1.0 else Config.TAU_A
        self.v_thr_a = Config.V_THR_A

    def set_malicious(self, mode):
        """Switches hyperparameters between nominal, malicious, and attack states."""
        if mode == 'malicious' or mode is True:
            self.tau = self.tau_t
            self.v_threshold = self.v_thr_t
        elif mode == 'attack':
            self.tau = self.tau_a
            self.v_threshold = self.v_thr_a
        else:
            self.tau = self.tau_n
            self.v_threshold = self.v_thr_n

    def forward(self, x: torch.Tensor, is_malicious=False):
        """
        Forward pass using the current v_threshold (set by set_malicious).
        We inline the LIF forward rather than calling super().forward() to ensure
        self.v_threshold is read fresh each call (SpikingJelly may cache internally).
        """
        # Charge: V[t] = V[t-1] * (1 - 1/tau) + x
        self.neuronal_charge(x)
        # Fire: spike = 1 if V >= v_threshold else 0 (surrogate gradient in backward)
        spike = self.neuronal_fire()
        # Reset: V[t] = V[t] - spike * v_threshold (soft) or V[t] = 0 (hard)
        self.neuronal_reset(spike)
        return spike
