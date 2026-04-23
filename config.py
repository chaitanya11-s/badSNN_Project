import torch

class Config:
    # Dataset settings
    DATASET = 'cifar10'  # cifar10, gtsrb, cifar100, nmnist
    DATA_ROOT = './data/'

    # Model settings
    MODEL = 'resnet19'  # resnet19, vgg16, nmnist_net
    TIMESTEPS = 4
    BATCH_SIZE = 64
    EPOCHS = 120
    LEARNING_RATE = 0.002

    # Neuron hyperparameters (paper Table 1)
    V_THR_N = 1.0    # Nominal threshold
    TAU_N   = 0.5    # Nominal time constant
    V_THR_T = 1.5    # Malicious threshold
    TAU_T   = 0.5    # Malicious time constant (GTSRB overrides to 0.8 via DATASET_SPECS)
    V_THR_A = 1.10   # Attack (evaluation) threshold — sweep: 1.10, 1.15, 1.20
    TAU_A   = 0.5    # Attack time constant

    GRAD_CLIP = 1.0

    # Backdoor settings
    TARGET_LABEL    = 0
    POISONING_RATIO = 0.02          # CIFAR-10 default (paper); per-dataset in DATASET_SPECS
    POISONING_RATIOS = [0.01, 0.02, 0.03, 0.05]

    # Trigger T_p: power transformation (Equation 3)
    POWER_Q = 3.0

    # Trigger T_s: neuromorphic noise (Equation 7)
    BETA = 0.03

    # Trigger T_o: optimized U-Net trigger
    UNET_EPOCHS  = 50
    UNET_LR      = 0.0001
    LAMBDA_SIM   = 1.0
    LAMBDA_ADV   = 0.1
    LAMBDA_WMSC  = 1.0

    # Defense settings (paper values)
    FINE_TUNING_EPOCHS = 10
    ANP_PRUNING_RATIO  = 0.1
    ANP_EPS            = 0.4
    ANP_ALPHA          = 0.5
    ANP_CLEAN_RATIO    = 0.05
    CLP_THRESHOLD      = 3.0

    # System
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    SEED     = 42
    SAVE_DIR = './checkpoints/'
    RESULT_DIR = './results/'

    # Per-dataset overrides (Corrections 9, 10, 11)
    # tau_t: malicious time constant for Pass 2
    # trigger: which trigger to apply at inference
    DATASET_SPECS = {
        'cifar10':  {'model': 'resnet19',   'num_classes': 10,  'poisoning_ratio': 0.02, 'tau_t': 0.5, 'trigger': 'T_p'},
        'gtsrb':    {'model': 'vgg16',      'num_classes': 43,  'poisoning_ratio': 0.05, 'tau_t': 0.8, 'trigger': 'T_p'},
        'cifar100': {'model': 'vgg16',      'num_classes': 100, 'poisoning_ratio': 0.01, 'tau_t': 0.5, 'trigger': 'T_p'},
        'nmnist':   {'model': 'nmnist_net', 'num_classes': 10,  'poisoning_ratio': 0.03, 'tau_t': 0.5, 'trigger': 'T_s'},
    }
