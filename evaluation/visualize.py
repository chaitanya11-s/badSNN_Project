import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_accuracy_vs_poisoning(poisoning_ratios, ca_values, asr_values, save_name="acc_vs_poisoning.png"):
    """
    Plots the relationship between Poisoning Ratios against Clean Accuracy and Attack Success Rate.
    Matches standard security evaluation curve layouts.
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(poisoning_ratios, ca_values, 'o-', color='blue', linewidth=2, label='Clean Accuracy (CA)')
    plt.plot(poisoning_ratios, asr_values, 's--', color='red', linewidth=2, label='Attack Success Rate (ASR)')
    
    plt.xlabel('Poisoning Ratio')
    plt.ylabel('Percentage (%)')
    plt.title('Impact of Poisoning Ratio on Model Performance')
    plt.xticks(poisoning_ratios)
    plt.ylim([0, 105])
    plt.legend(loc='lower right')
    
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.RESULT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_trigger_comparison(clean_images, triggered_images, perturbations, save_name="trigger_comparison.png"):
    """Visualizes triggers, targets and perturbations concurrently in an identical grid format."""
    b = min(5, clean_images.size(0))
    fig, axes = plt.subplots(3, b, figsize=(b * 3, 9))
    
    for i in range(b):
        clean = clean_images[i].cpu().numpy().transpose(1, 2, 0)
        trig = triggered_images[i].cpu().numpy().transpose(1, 2, 0)
        pert = perturbations[i].cpu().numpy().transpose(1, 2, 0)
        
        pert = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        
        if clean.shape[2] == 1:
            clean, trig, pert = clean.squeeze(), trig.squeeze(), pert.squeeze()
            cmap = 'gray'
        elif clean.shape[2] == 2:
            clean, trig, pert = clean[..., 0], trig[..., 0], pert[..., 0] 
            cmap = 'hot'
        else:
            cmap = None
            
        axes[0, i].imshow(clean, cmap=cmap)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_ylabel("Clean")
        
        axes[1, i].imshow(trig, cmap=cmap)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_ylabel("Triggered")
        
        axes[2, i].imshow(pert, cmap=cmap)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_ylabel("Perturbation")
        
    plt.tight_layout()
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.RESULT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()

def plot_defense_comparison(defense_names, asr_before, asr_after, save_name="defense_comparison.png"):
    """Renders grouped bar plots to correlate mitigation capabilities visually."""
    x = np.arange(len(defense_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, asr_before, width, label='Before Defense', color='salmon')
    rects2 = ax.bar(x + width/2, asr_after, width, label='After Defense', color='skyblue')
    
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Defense Effectiveness Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(defense_names)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.RESULT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()
