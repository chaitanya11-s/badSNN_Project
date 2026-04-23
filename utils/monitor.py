"""
Real-time Training Monitor for Google Colab
Provides detailed logging, anomaly detection, and visualization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import warnings

class TrainingMonitor:
    """Monitors training health and detects anomalies in real-time."""

    def __init__(self, window_size=5, enable_plots=True):
        self.window_size = window_size
        self.enable_plots = enable_plots

        # Metric history
        self.epochs = []
        self.losses = []
        self.losses_n = []
        self.losses_t = []
        self.base_ca = []
        self.ca_attack = []
        self.asr = []

        # Anomaly detection
        self.prev_ca = None
        self.prev_asr = None
        self.collapse_warnings = 0
        self.stagnation_counter = 0

        # Health flags
        self.health_status = "HEALTHY"
        self.warnings = []

    def log_epoch(self, epoch, loss, loss_n, loss_t, base_ca, ca_attack, asr, warmup=False):
        """Log metrics for current epoch and check for anomalies."""

        # Store metrics
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.losses_n.append(loss_n)
        self.losses_t.append(loss_t)
        self.base_ca.append(base_ca)
        self.ca_attack.append(ca_attack)
        self.asr.append(asr)

        # Reset warnings
        self.warnings = []
        self.health_status = "HEALTHY"

        # === ANOMALY DETECTION ===

        # 1. Catastrophic Forgetting Detection
        if base_ca < 15 and epoch > 10:
            self.warnings.append("🚨 CATASTROPHIC FORGETTING: CA = {:.1f}% (random chance!)".format(base_ca))
            self.health_status = "CRITICAL"
            self.collapse_warnings += 1

        # 2. CA Collapse Detection
        if self.prev_ca is not None and base_ca < self.prev_ca - 10 and epoch > 5:
            self.warnings.append("⚠️  CA DROPPED BY {:.1f}% (was {:.1f}%, now {:.1f}%)".format(
                self.prev_ca - base_ca, self.prev_ca, base_ca))
            self.health_status = "WARNING"

        # 3. Malicious Loss Domination
        if not warmup and loss_t > loss_n * 0.5:
            self.warnings.append("⚠️  MALICIOUS LOSS TOO HIGH: Lt={:.3f} vs Ln={:.3f} (should be Lt << Ln)".format(
                loss_t, loss_n))
            self.health_status = "WARNING"

        # 4. Loss Explosion
        if len(self.losses) > 1 and loss > self.losses[-2] * 1.5:
            self.warnings.append("⚠️  LOSS SPIKE: {:.3f} → {:.3f} (increased by {:.1f}%)".format(
                self.losses[-2], loss, ((loss / self.losses[-2]) - 1) * 100))
            self.health_status = "WARNING"

        # 5. Training Stagnation
        if epoch > 20 and len(self.base_ca) >= 5:
            recent_ca = self.base_ca[-5:]
            if max(recent_ca) - min(recent_ca) < 0.5:
                self.stagnation_counter += 1
                if self.stagnation_counter >= 3:
                    self.warnings.append("⚠️  STAGNATION: CA not improving (stuck at ~{:.1f}%)".format(base_ca))
                    self.health_status = "WARNING"
            else:
                self.stagnation_counter = 0

        # 6. Backdoor Not Learning
        if not warmup and epoch > 20 and asr < 30:
            self.warnings.append("⚠️  BACKDOOR WEAK: ASR = {:.1f}% at epoch {} (should be higher)".format(asr, epoch))
            self.health_status = "WARNING"

        # 7. Perfect Backdoor but Dead CA (Model Collapse)
        if asr > 95 and base_ca < 20:
            self.warnings.append("🚨 MODEL COLLAPSE: ASR={:.1f}% but CA={:.1f}% (predicting only target label!)".format(
                asr, base_ca))
            self.health_status = "CRITICAL"

        # 8. Warmup Not Working
        if warmup and epoch >= 5 and base_ca < 40:
            self.warnings.append("⚠️  WARMUP INEFFECTIVE: CA = {:.1f}% at epoch {} (should be >40%)".format(
                base_ca, epoch))
            self.health_status = "WARNING"

        # === POSITIVE MILESTONES ===
        milestones = []

        if base_ca >= 85 and epoch >= 15:
            milestones.append("✅ CA MILESTONE: {:.1f}% (target: ≥85%)".format(base_ca))

        if asr >= 80 and not warmup:
            milestones.append("✅ ASR MILESTONE: {:.1f}% (target: ≥80%)".format(asr))

        if base_ca >= 87.22 and asr >= 82.65:
            milestones.append("🏆 BEATS PAPER: CA={:.1f}% (vs 87.22%), ASR={:.1f}% (vs 82.65%)".format(
                base_ca, asr))

        if base_ca >= 90 and asr >= 85:
            milestones.append("🎯 TARGET ACHIEVED: CA={:.1f}% ≥ 90%, ASR={:.1f}% ≥ 85%".format(base_ca, asr))

        # Update previous values
        self.prev_ca = base_ca
        self.prev_asr = asr

        return self.warnings, milestones

    def print_status(self, epoch, total_epochs, loss, loss_n, loss_t, base_ca, ca_attack, asr, warmup=False):
        """Print detailed status with color coding."""

        # Log metrics
        warnings, milestones = self.log_epoch(epoch, loss, loss_n, loss_t, base_ca, ca_attack, asr, warmup)

        # Build status line
        warmup_tag = " [WARMUP]" if warmup else ""
        health_tag = ""
        if self.health_status == "CRITICAL":
            health_tag = " ❌ CRITICAL"
        elif self.health_status == "WARNING":
            health_tag = " ⚠️  WARNING"
        elif milestones:
            health_tag = " ✅ GOOD"

        # Main log line
        print(f"\n{'='*90}")
        print(f"Epoch {epoch:03d}/{total_epochs}{warmup_tag}{health_tag}")
        print(f"{'='*90}")
        print(f"  Loss:       {loss:.4f}  (Nominal: {loss_n:.4f}, Malicious: {loss_t:.4f})")
        print(f"  Base CA:    {base_ca:6.2f}%  (Clean accuracy under nominal thresholds)")
        print(f"  CA Attack:  {ca_attack:6.2f}%  (Clean accuracy under attack thresholds)")
        print(f"  ASR:        {asr:6.2f}%  (Attack success rate with trigger)")

        # Loss ratio check
        if not warmup and loss_t > 0:
            ratio = loss_n / loss_t if loss_t > 0 else float('inf')
            print(f"  Loss Ratio: Ln/Lt = {ratio:.1f}x  (should be >>1, clean task dominates)")

        # Print warnings
        if warnings:
            print(f"\n  ⚠️  WARNINGS:")
            for w in warnings:
                print(f"      {w}")

        # Print milestones
        if milestones:
            print(f"\n  🎉 MILESTONES:")
            for m in milestones:
                print(f"      {m}")

        # Recommendations
        if self.health_status == "CRITICAL":
            print(f"\n  💡 RECOMMENDATION:")
            if base_ca < 15:
                print(f"      → STOP TRAINING! Model collapsed. Reduce alpha to 0.001 and restart.")
            if asr > 95 and base_ca < 20:
                print(f"      → STOP TRAINING! Model predicting only target label. Reduce alpha drastically.")

        elif self.health_status == "WARNING":
            print(f"\n  💡 SUGGESTION:")
            if base_ca < 50 and epoch > 20:
                print(f"      → Consider reducing alpha if CA doesn't improve by epoch {epoch + 10}")
            if loss_t > loss_n * 0.5:
                print(f"      → Malicious loss too high. Alpha might be too large.")
            if asr < 30 and epoch > 20 and not warmup:
                print(f"      → Backdoor not learning. Consider increasing alpha slightly or check trigger.")

        print(f"{'='*90}\n")

    def plot_metrics(self, save_path=None):
        """Generate comprehensive training plots for Colab."""

        if not self.enable_plots or len(self.epochs) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('BadSNN Training Monitor - Live Metrics', fontsize=16, fontweight='bold')

        # Plot 1: Loss over time
        ax = axes[0, 0]
        ax.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Total Loss')
        ax.plot(self.epochs, self.losses_n, 'g--', linewidth=1.5, label='Nominal Loss (Ln)')
        ax.plot(self.epochs, self.losses_t, 'r--', linewidth=1.5, label='Malicious Loss (Lt)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: CA Metrics
        ax = axes[0, 1]
        ax.plot(self.epochs, self.base_ca, 'b-', linewidth=2, label='Base CA (nominal)')
        ax.plot(self.epochs, self.ca_attack, 'orange', linewidth=2, label='CA under attack')
        ax.axhline(y=87.22, color='green', linestyle='--', linewidth=1, label='Paper baseline (87.22%)')
        ax.axhline(y=90, color='red', linestyle='--', linewidth=1, label='Target (90%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Clean Accuracy (CA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # Plot 3: ASR
        ax = axes[0, 2]
        ax.plot(self.epochs, self.asr, 'r-', linewidth=2, label='ASR')
        ax.axhline(y=82.65, color='green', linestyle='--', linewidth=1, label='Paper baseline (82.65%)')
        ax.axhline(y=85, color='blue', linestyle='--', linewidth=1, label='Target (85%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ASR (%)')
        ax.set_title('Attack Success Rate (ASR)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # Plot 4: Loss Ratio (Ln / Lt)
        ax = axes[1, 0]
        loss_ratios = []
        for ln, lt in zip(self.losses_n, self.losses_t):
            if lt > 1e-6:
                loss_ratios.append(ln / lt)
            else:
                loss_ratios.append(0)
        ax.plot(self.epochs, loss_ratios, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ln / Lt Ratio')
        ax.set_title('Loss Ratio (should be >>1)')
        ax.axhline(y=50, color='green', linestyle='--', linewidth=1, label='Healthy (>50x)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: CA vs ASR Tradeoff
        ax = axes[1, 1]
        scatter = ax.scatter(self.base_ca, self.asr, c=self.epochs, cmap='viridis',
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.plot(self.base_ca, self.asr, 'k-', alpha=0.3, linewidth=1)
        # Paper's result
        ax.scatter([87.22], [82.65], color='red', s=200, marker='*',
                   edgecolors='black', linewidth=2, label='Paper (V_thr_a=1.10)', zorder=10)
        # Target zone
        ax.axhline(y=85, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=90, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.fill_between([90, 100], [85, 85], [100, 100], alpha=0.1, color='green', label='Target Zone')
        ax.set_xlabel('Base CA (%)')
        ax.set_ylabel('ASR (%)')
        ax.set_title('CA vs ASR Tradeoff (Trajectory)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        plt.colorbar(scatter, ax=ax, label='Epoch')

        # Plot 6: Health Score
        ax = axes[1, 2]
        # Compute health score: balance between CA and ASR
        health_scores = []
        for ca, asr in zip(self.base_ca, self.asr):
            # Penalize if CA < 85 or ASR < 80
            ca_penalty = max(0, 85 - ca) * 2
            asr_penalty = max(0, 80 - asr) * 1.5
            score = ca + asr - ca_penalty - asr_penalty
            health_scores.append(score)

        ax.plot(self.epochs, health_scores, 'green', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Health Score')
        ax.set_title('Overall Training Health (CA + ASR - penalties)')
        ax.grid(True, alpha=0.3)

        # Mark best epoch
        if health_scores:
            best_epoch_idx = np.argmax(health_scores)
            best_epoch = self.epochs[best_epoch_idx]
            best_score = health_scores[best_epoch_idx]
            ax.scatter([best_epoch], [best_score], color='red', s=200, marker='*',
                      edgecolors='black', linewidth=2, zorder=10)
            ax.text(best_epoch, best_score + 5, f'Best: Epoch {best_epoch}',
                   ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")

        plt.show()

    def get_summary(self):
        """Generate training summary."""

        if not self.epochs:
            return "No training data yet."

        summary = []
        summary.append("\n" + "="*90)
        summary.append("TRAINING SUMMARY")
        summary.append("="*90)

        # Best metrics
        best_ca_idx = np.argmax(self.base_ca)
        best_asr_idx = np.argmax(self.asr)

        summary.append(f"\n📈 Best Metrics:")
        summary.append(f"   Best CA:  {self.base_ca[best_ca_idx]:.2f}% at epoch {self.epochs[best_ca_idx]}")
        summary.append(f"   Best ASR: {self.asr[best_asr_idx]:.2f}% at epoch {self.epochs[best_asr_idx]}")

        # Latest metrics
        summary.append(f"\n📊 Latest Metrics (Epoch {self.epochs[-1]}):")
        summary.append(f"   Base CA:  {self.base_ca[-1]:.2f}%")
        summary.append(f"   ASR:      {self.asr[-1]:.2f}%")
        summary.append(f"   Loss:     {self.losses[-1]:.4f}")

        # Comparison with paper
        summary.append(f"\n🎯 vs Paper (CA=87.22%, ASR=82.65%):")
        ca_diff = self.base_ca[-1] - 87.22
        asr_diff = self.asr[-1] - 82.65
        summary.append(f"   CA:  {ca_diff:+.2f}%  {'✅ BETTER' if ca_diff > 0 else '❌ WORSE'}")
        summary.append(f"   ASR: {asr_diff:+.2f}%  {'✅ BETTER' if asr_diff > 0 else '❌ WORSE'}")

        if self.base_ca[-1] >= 87.22 and self.asr[-1] >= 82.65:
            summary.append(f"\n🏆 SUCCESS! You beat the paper's results!")
        elif self.base_ca[-1] >= 90 and self.asr[-1] >= 85:
            summary.append(f"\n🎉 TARGET ACHIEVED! CA ≥ 90% and ASR ≥ 85%!")
        else:
            summary.append(f"\n⚠️  Not yet at target. Keep training or tune parameters.")

        # Health warnings
        if self.collapse_warnings > 0:
            summary.append(f"\n⚠️  WARNING: Detected {self.collapse_warnings} collapse events during training.")

        summary.append("="*90 + "\n")

        return "\n".join(summary)
