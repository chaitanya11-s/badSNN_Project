"""
Google Colab Training Script for BadSNN
Copy this entire file to a Colab cell and run!
"""

import sys
import os

# ============================================================================
# CONFIGURATION - Modify these if needed
# ============================================================================

# Training parameters (optimized for beating paper's results)
DATASET = 'cifar10'          # cifar10, gtsrb, cifar100, nmnist
TRIGGER = 'T_p'              # T_p, T_s, temporal_only
POISONING_RATIO = 0.02       # Paper used 2% for CIFAR-10
EPOCHS = 100                 # Extended from paper's 75
MODE = 'attack'              # attack, defense, both

# GPU check
import torch
print("="*90)
print("🔍 SYSTEM CHECK")
print("="*90)
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: No GPU detected! Training will be VERY slow.")
    print("   Go to Runtime → Change runtime type → GPU")
print("="*90 + "\n")

# ============================================================================
# START TRAINING
# ============================================================================

print("\n" + "="*90)
print("🚀 STARTING BADSNN TRAINING")
print("="*90)
print(f"  Dataset:         {DATASET}")
print(f"  Trigger:         {TRIGGER}")
print(f"  Poisoning Ratio: {POISONING_RATIO}")
print(f"  Epochs:          {EPOCHS}")
print(f"  Mode:            {MODE}")
print("="*90)

# Run training
cmd = f"python main.py --mode {MODE} --dataset {DATASET} --trigger {TRIGGER} --poisoning_ratio {POISONING_RATIO} --epochs {EPOCHS}"
print(f"\n💻 Running: {cmd}\n")

exit_code = os.system(cmd)

if exit_code == 0:
    print("\n" + "="*90)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*90)
    print("\n📁 Files generated:")
    print(f"   Model:      checkpoints/{DATASET}_backdoor.pth")
    print(f"   Best CA:    checkpoints/{DATASET}_backdoor_best_ca.pth")
    print(f"   Logs (CSV): results/{DATASET}_{TRIGGER}_training_log.csv")
    print(f"   Plots:      results/{DATASET}_{TRIGGER}_epoch*.png")

    print("\n📊 Next step: Find optimal V_thr_a")
    print(f"   Run: !python sweep_vthra.py --model checkpoints/{DATASET}_backdoor.pth")
    print("="*90 + "\n")
else:
    print("\n" + "="*90)
    print("❌ TRAINING FAILED OR STOPPED")
    print("="*90)
    print("\n🔍 Check the error messages above.")
    print("   Common issues:")
    print("   - GPU out of memory: Reduce BATCH_SIZE in config.py")
    print("   - Model collapsed: Reduce ALPHA in config.py")
    print("   - Missing dependencies: Install via pip")
    print("="*90 + "\n")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if exit_code == 0:
    print("\n📊 Displaying latest training plot...\n")

    # Find latest plot
    import glob
    plots = sorted(glob.glob(f'results/{DATASET}_{TRIGGER}_epoch*.png'))

    if plots:
        from IPython.display import Image, display
        latest_plot = plots[-1]
        print(f"Latest plot: {latest_plot}\n")
        display(Image(filename=latest_plot))
    else:
        print("⚠️  No plots found. Check results/ directory.")

    # Display CSV summary
    csv_file = f'results/{DATASET}_{TRIGGER}_training_log.csv'
    if os.path.exists(csv_file):
        print("\n📋 Training Log Summary (last 10 epochs):\n")
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(df.tail(10).to_string(index=False))

        # Plot CA and ASR over time
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(df['Epoch'], df['Base_CA'], 'b-', linewidth=2, label='Base CA')
        ax1.axhline(y=87.22, color='green', linestyle='--', label='Paper (87.22%)')
        ax1.axhline(y=90, color='red', linestyle='--', label='Target (90%)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Base CA (%)')
        ax1.set_title('Clean Accuracy Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(df['Epoch'], df['ASR'], 'r-', linewidth=2, label='ASR')
        ax2.axhline(y=82.65, color='green', linestyle='--', label='Paper (82.65%)')
        ax2.axhline(y=85, color='blue', linestyle='--', label='Target (85%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ASR (%)')
        ax2.set_title('Attack Success Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Final verdict
        final_ca = df['Base_CA'].iloc[-1]
        final_asr = df['ASR'].iloc[-1]

        print("\n" + "="*90)
        print("🎯 FINAL RESULTS")
        print("="*90)
        print(f"  Base CA:  {final_ca:.2f}%  {'✅' if final_ca >= 90 else '❌'} (target: ≥90%)")
        print(f"  ASR:      {final_asr:.2f}%  {'✅' if final_asr >= 85 else '❌'} (target: ≥85%)")
        print(f"\n  vs Paper:")
        print(f"    CA:  {final_ca - 87.22:+.2f}%  {'✅ BEATS' if final_ca >= 87.22 else '❌ WORSE'}")
        print(f"    ASR: {final_asr - 82.65:+.2f}%  {'✅ BEATS' if final_asr >= 82.65 else '❌ WORSE'}")

        if final_ca >= 90 and final_asr >= 85:
            print("\n  🎉🎉🎉 SUCCESS! You achieved the target! 🎉🎉🎉")
        elif final_ca >= 87.22 and final_asr >= 82.65:
            print("\n  🏆 Good! You beat the paper, but not quite at target yet.")
            print("     Try running sweep_vthra.py to optimize V_thr_a")
        else:
            print("\n  ⚠️  Not at target yet. See recommendations below.")

            if final_ca < 85:
                print("\n  💡 CA too low. Try:")
                print("     - Reduce ALPHA to 0.005 in config.py")
                print("     - Increase WARMUP_EPOCHS to 15")

            if final_asr < 70:
                print("\n  💡 ASR too low. Try:")
                print("     - Increase ALPHA to 0.010 in config.py")
                print("     - Increase POWER_Q to 2.8")

        print("="*90 + "\n")
    else:
        print(f"\n⚠️  CSV log not found: {csv_file}")

print("\n✅ Script complete! Review results above.")
print("📧 Share the final results and plots with your advisor.\n")
