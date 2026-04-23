import torch

class Config:
    # Dataset settings
    DATASET = 'cifar10'  # cifar10, gtsrb, cifar100, nmnist
    DATA_ROOT = './data/'
    
    # Model settings
    MODEL = 'resnet19'  # resnet19, vgg16, nmnist_net
    TIMESTEPS = 4
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Neuron hyperparameters (from BadSNN paper)
    V_THR_N = 1.0      # Nominal threshold
    TAU_N = 0.05       # Nominal time constant
    V_THR_T = 1.5      # Malicious threshold
    TAU_T = 0.5        # Malicious time constant
    V_THR_A = 1.15     # Attack threshold (sweet spot)
    TAU_A = 0.1        # Attack time constant
    
    # Backdoor settings
    TARGET_LABEL = 0
    POISONING_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1% to 5%
    
    # Trigger T_p: power transformation
    POWER_Q = 1.5
    
    # Trigger T_s: neuromorphic noise trigger
    BETA = 0.03
    
    # Trigger T_o: optimized trigger with U-Net
    UNET_EPOCHS = 50
    UNET_LR = 0.0001
    LAMBDA_SIM = 1.0
    LAMBDA_ADV = 0.1
    LAMBDA_WMSC = 1.0
    
    # Defense settings
    FINE_TUNING_EPOCHS = 10
    ANP_PRUNING_RATIO = 0.1
    CLP_THRESHOLD = 0.5
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    SAVE_DIR = './checkpoints/'
    RESULT_DIR = './results/'
