import argparse
import torch
import os
from config import Config
from utils.data_loader import get_dataloaders
from models.spiking_resnet19 import SpikingResNet19
from models.spiking_vgg16 import SpikingVGG16
from models.nmnist_net import NMNISTNet
from attacks.triggers import T_p, T_s
from attacks.backdoor_train import backdoor_train
from evaluation.metrics import clean_accuracy, attack_success_rate
from defenses.fine_tuning import fine_tuning_defense
from defenses.clp import clp_defense
from defenses.anp import anp_defense
from defenses.tsbd import tsbd_defense
from defenses.nad import nad_defense
import copy

def get_model(dataset):
    if dataset == 'nmnist':
        return NMNISTNet().to(Config.DEVICE)
    elif Config.MODEL == 'vgg16':
        return SpikingVGG16().to(Config.DEVICE)
    else:
        return SpikingResNet19().to(Config.DEVICE)

def main():
    parser = argparse.ArgumentParser(description="BadSNN Implementation Suite - Neural Executable Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['attack', 'defense', 'both'], help="Execution mode segment")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'gtsrb', 'cifar100', 'nmnist'], help="Target objective dataset source")
    parser.add_argument('--poisoning_ratio', type=float, default=0.01, help="Poison ratio constraint equivalent handling D_t^p sizes")
    parser.add_argument('--defense', type=str, default='fine_tuning', choices=['fine_tuning', 'anp', 'clp', 'tsbd', 'nad'], help="Countermeasure application subset structure")
    parser.add_argument('--trigger', type=str, default='T_p', choices=['T_p', 'T_s'], help="Execution injection wrapper style constraint mapping")

    args = parser.parse_args()
    
    Config.DATASET = args.dataset
    trigger_func = T_s if args.trigger == 'T_s' else T_p
    
    train_loader, test_loader = get_dataloaders()
    model = get_model(args.dataset)
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    
    backdoor_model = None

    if args.mode in ['attack', 'both']:
        print(f"\\n--- Running Attack Phase [{args.dataset} | Trigger: {args.trigger} | Ratio: {args.poisoning_ratio}] ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Dual-spike backdoor structural injection binding
        backdoor_model, _, _ = backdoor_train(model, train_loader, optimizer, trigger_func=trigger_func, poisoning_ratio=args.poisoning_ratio)
        
        ca = clean_accuracy(backdoor_model, test_loader)
        asr = attack_success_rate(backdoor_model, test_loader, trigger_func=trigger_func)
        
        print(f"Attack Evaluation -> Clean Accuracy (CA): {ca:.2f}% | Attack Success Rate (ASR): {asr:.2f}%")
        torch.save(backdoor_model.state_dict(), os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth"))
    
    if args.mode in ['defense', 'both']:
        print(f"\\n--- Running Defense Phase [{args.defense.upper()}] ---")
        if backdoor_model is None:
            backdoor_model = get_model(args.dataset)
            model_path = os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth")
            if os.path.exists(model_path):
                backdoor_model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
                print("Loaded pre-trained memory module mapping backdoored model.")
            else:
                print("Warning: No pre-trained backdoor model found block. Automatically initiating injection state module to ensure verification viability...")
                opt = torch.optim.Adam(backdoor_model.parameters(), lr=Config.LEARNING_RATE)
                backdoor_model, _, _ = backdoor_train(backdoor_model, train_loader, opt, trigger_func=trigger_func, poisoning_ratio=args.poisoning_ratio)
        
        defended_model = None
        if args.defense == 'fine_tuning':
            defended_model = fine_tuning_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'clp':
            defended_model = clp_defense(copy.deepcopy(backdoor_model))
        elif args.defense == 'anp':
            defended_model = anp_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'tsbd':
            defended_model = tsbd_defense(copy.deepcopy(backdoor_model), train_loader)
        elif args.defense == 'nad':
            teacher = get_model(args.dataset)
            opt_t = torch.optim.Adam(teacher.parameters(), lr=Config.LEARNING_RATE)
            teacher, _, _ = backdoor_train(teacher, train_loader, opt_t, trigger_func=lambda x:x, poisoning_ratio=0.0)
            defended_model = nad_defense(copy.deepcopy(backdoor_model), teacher, train_loader)
            
        ca = clean_accuracy(defended_model, test_loader)
        asr = attack_success_rate(defended_model, test_loader, trigger_func=trigger_func)
        
        print(f"Defense Component Verification Runtime Analysis -> Clean Accuracy (CA): {ca:.2f}% | Attack Success Rate (ASR): {asr:.2f}%")

if __name__ == "__main__":
    main()
