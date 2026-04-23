# BadSNN: Malicious Spike Poisoning in Spiking Neural Networks

This repository contains the codebase for evaluating vulnerabilities and defenses within Spiking Neural Networks (SNNs), specifically focusing on the "BadSNN" attack methodologies and subsequent countermeasures. It serves as the primary artifact for my B.Tech Major Project submission.

## 1. Project Overview

Spiking Neural Networks (SNNs) process information via discrete, energy-efficient binary spike sequences rather than continuous activation values. This temporal dynamic introduces unique security vulnerabilities. **BadSNN** explores these vulnerabilities through **Malicious Spike Poisoning**.
- **Dual Spike Learning:** The attack operates by separating the SNN into nominal ($S_n$) and malicious ($S_t$) hyperparameter configurations (membrane time constants $\tau$ and firing thresholds $V_{thr}$). The backdoor is exclusively injected under the malicious state, masking its presence during standard operations.
- **Trigger Mechanisms:** Implements Power Transformed ($T_p$) and Neuromorphic Noise ($T_s$) triggers.
- **Temporal Countermeasures:** We demonstrate State-aware Fine-Tuning, Channel Lipschitzness Pruning (CLP), Adversarial Neuron Pruning (ANP), Trigger Synthesis Based Defense (TSBD), and Neural Attention Distillation (NAD). 

## 2. Project Structure

```text
BadSNN_Project/
├── attacks/                # Trigger generation and Dual-Spike training loops
│   ├── backdoor_train.py
│   ├── deepfool.py
│   ├── trigger_optimization.py
│   └── triggers.py
├── defenses/               # SNN-compatible backdoor mitigation techniques
│   ├── anp.py
│   ├── clp.py
│   ├── fine_tuning.py
│   ├── nad.py
│   └── tsbd.py
├── evaluation/             # ASR, LPIPS, SSIM metrics and graph plotting
│   ├── lpips_ssim.py
│   ├── metrics.py
│   └── visualize.py
├── models/                 # SpikingJelly based architectures (ResNet19, VGG16, NMNISTNet)
│   ├── lif_neuron.py
│   ├── nmnist_net.py
│   ├── plif_neuron.py
│   ├── spiking_resnet19.py
│   └── spiking_vgg16.py
├── utils/                  # CIFAR-10, CIFAR-100, GTSRB, and N-MNIST DataLoaders
│   └── data_loader.py
├── config.py               # Centralizes global configuration variables
├── main.py                 # Core CLI entry point
├── requirements.txt        # Dependency bounds
└── run_experiments.py      # Automated loop script for table extraction
```

## 3. Environment Setup

It is highly recommended to run this framework inside a virtual Python environment to guarantee dependency separation. `SpikingJelly` acts as the primary PyTorch SNN backend.

```bash
# Clone the repository and navigate inside
cd BadSNN_Project

# Create and activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # MacOS/Linux
# venv\Scripts\activate   # Windows

# Install the required dependencies
pip install -r requirements.txt
```

## 4. Execution Examples

The repository is built around a robust `main.py` entry point. It controls the dataset targeting, poisoning severity, trigger type, and defense benchmarking natively.

### 4.1. Training a Backdoor Attack
Train a backdoored SpikingResNet19 on GTSRB using the $T_p$ trigger at a 3% poisoning ratio:
```bash
python main.py --mode attack --dataset gtsrb --trigger T_p --poisoning_ratio 0.03
```
*Outputs: Base Clean Accuracy (CA) and post-injection Attack Success Rate (ASR).*

### 4.2. Evaluating Defenses
Apply Neural Attention Distillation (NAD) to scrub a dataset backdoor:
```bash
python main.py --mode defense --dataset cifar10 --defense nad
```

### 4.3. End-to-End Pipeline
Train an attack and immediately pipe it into the Adversarial Neuron Pruning defense benchmark:
```bash
python main.py --mode both --dataset cifar100 --poisoning_ratio 0.05 --defense anp
```

### 4.4. Reproducing Paper Tables
To sequentially reproduce **Table 1** (Attack metrics cross-dataset) and **Table 2** (Defense robustness averages), utilize the integrated experiment runner:
```bash
python run_experiments.py
```
*The `evaluation/visualize.py` module will automatically intercept these runs and export `.png` charts into the `./results/` directory mapping Accuracy vs. Poisoning Ratios and Defense Effectiveness comparisons.*
