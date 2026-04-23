import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_all_neuron_hyperparams(model, v_thr, tau):
    """
    Set v_threshold and tau on ALL spiking neurons in the model simultaneously.

    SpikingJelly stores tau as the inverse of the paper's tau (sj_tau = 1 / paper_tau),
    so we convert here. v_threshold is stored and compared directly.

    Called with:
      - (model, V_THR_N, TAU_N) for nominal pass / Base CA evaluation
      - (model, V_THR_T, tau_t) for malicious pass during training
      - (model, V_THR_A, TAU_A) for ASR / CA evaluation under attack thresholds
    """
    sj_tau = 1.0 / max(tau, 1e-5)
    for module in model.modules():
        if hasattr(module, 'v_threshold'):
            module.v_threshold = v_thr
        if hasattr(module, 'tau'):
            module.tau = sj_tau
