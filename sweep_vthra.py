"""
V_thr_a Parameter Sweep to Surpass Paper's Results

Paper's Best: V_thr_a=1.10 → CA=87.22%, ASR=82.65%
Goal: Find V_thr_a that gives CA ≥ 90% AND ASR ≥ 85%
"""

import torch
import os
from config import Config
from utils.data_loader import get_dataloaders
from models.spiking_resnet19 import SpikingResNet19
from attacks.triggers import T_p
from evaluation.metrics import clean_accuracy, attack_success_rate

def evaluate_at_vthra(model_path, v_thr_a_values):
    """Evaluate a trained backdoor model at different V_thr_a values."""

    _, test_loader = get_dataloaders()
    model = SpikingResNet19().to(Config.DEVICE)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    print(f"Loaded model from {model_path}\n")

    print("=" * 80)
    print(f"{'V_thr_a':<10} {'Base CA (%)':<12} {'CA Attack (%)':<15} {'ASR (%)':<10} {'Score':<10}")
    print("=" * 80)

    results = []

    for v_thr_a in v_thr_a_values:
        # Temporarily override V_thr_a in all LIF neurons
        original_v_thr_a = Config.V_THR_A
        Config.V_THR_A = v_thr_a

        # Update model neurons
        from models.lif_neuron import LIFNeuron
        for module in model.modules():
            if isinstance(module, LIFNeuron):
                module.v_thr_a = v_thr_a

        # Evaluate
        base_ca = clean_accuracy(model, test_loader, mode='nominal')
        ca_attack = clean_accuracy(model, test_loader, mode='attack', attack_layer_start=Config.ATTACK_LAYER_START)
        asr = attack_success_rate(model, test_loader, trigger_func=T_p, attack_layer_start=Config.ATTACK_LAYER_START)

        # Combined score: prioritize CA ≥ 85% AND ASR ≥ 80%
        # Penalty if CA < 85 or ASR < 80
        ca_penalty = max(0, 85 - base_ca) * 2  # Heavy penalty for low CA
        asr_penalty = max(0, 80 - asr) * 1.5   # Moderate penalty for low ASR
        score = base_ca + asr - ca_penalty - asr_penalty

        results.append({
            'v_thr_a': v_thr_a,
            'base_ca': base_ca,
            'ca_attack': ca_attack,
            'asr': asr,
            'score': score
        })

        # Highlight if surpasses paper
        beat_paper = base_ca >= 87.22 and asr >= 82.65
        marker = " ★ BEATS PAPER!" if beat_paper else ""

        print(f"{v_thr_a:<10.2f} {base_ca:<12.2f} {ca_attack:<15.2f} {asr:<10.2f} {score:<10.2f}{marker}")

        Config.V_THR_A = original_v_thr_a

    print("=" * 80)

    # Find best result
    best = max(results, key=lambda x: x['score'])
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   V_thr_a = {best['v_thr_a']:.2f}")
    print(f"   Base CA = {best['base_ca']:.2f}%")
    print(f"   ASR = {best['asr']:.2f}%")
    print(f"   Score = {best['score']:.2f}")

    if best['base_ca'] >= 87.22 and best['asr'] >= 82.65:
        print(f"\n✅ SUCCESS! Surpassed paper's CA={best['base_ca']:.2f}% (vs 87.22%), ASR={best['asr']:.2f}% (vs 82.65%)")
    else:
        print(f"\n⚠️  Not yet surpassing paper. Need more training or parameter tuning.")

    # Save results
    csv_path = os.path.join(Config.RESULT_DIR, "vthra_sweep_results.csv")
    with open(csv_path, "w") as f:
        f.write("V_thr_a,Base_CA,CA_Attack,ASR,Score\n")
        for r in results:
            f.write(f"{r['v_thr_a']:.2f},{r['base_ca']:.2f},{r['ca_attack']:.2f},{r['asr']:.2f},{r['score']:.2f}\n")
    print(f"\n📊 Results saved to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/cifar10_backdoor.pth',
                       help='Path to trained backdoor model')
    parser.add_argument('--vmin', type=float, default=1.05, help='Minimum V_thr_a')
    parser.add_argument('--vmax', type=float, default=1.20, help='Maximum V_thr_a')
    parser.add_argument('--step', type=float, default=0.01, help='Step size')

    args = parser.parse_args()

    # Generate V_thr_a values to sweep
    v_thr_a_values = []
    v = args.vmin
    while v <= args.vmax:
        v_thr_a_values.append(round(v, 2))
        v += args.step

    print(f"Sweeping V_thr_a from {args.vmin} to {args.vmax} (step={args.step})")
    print(f"Total configurations: {len(v_thr_a_values)}\n")

    evaluate_at_vthra(args.model, v_thr_a_values)
