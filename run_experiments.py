import torch
import copy
from config import Config
from utils.data_loader import get_dataloaders
from models.spiking_resnet19 import SpikingResNet19
from models.spiking_vgg16 import SpikingVGG16
from models.nmnist_net import NMNISTNet
from attacks.triggers import T_p
from attacks.backdoor_train import backdoor_train
from defenses.fine_tuning import fine_tuning_defense
from defenses.anp import anp_defense
from defenses.clp import clp_defense
from defenses.tsbd import tsbd_defense
from defenses.nad import nad_defense
from evaluation.metrics import clean_accuracy, attack_success_rate
from evaluation.visualize import plot_accuracy_vs_poisoning, plot_defense_comparison

def get_model(dataset):
    if dataset == 'nmnist':
        return NMNISTNet().to(Config.DEVICE)
    elif Config.MODEL == 'vgg16':
        return SpikingVGG16().to(Config.DEVICE)
    else:
        return SpikingResNet19().to(Config.DEVICE)

def run_table1():
    print("\\n=== Replicating Table 1: Attack Effectiveness ===")
    results = {}
    old_dataset = Config.DATASET
    
    for dataset in ['cifar10', 'gtsrb', 'cifar100', 'nmnist']:
        Config.DATASET = dataset
        train_loader, test_loader = get_dataloaders()
        
        dataset_results = {'CA': [], 'ASR': []}
        
        for ratio in Config.POISONING_RATIOS:
            print(f"\\nEvaluating {dataset.upper()} with Poisoning Ratio: {ratio}")
            model = get_model(dataset)
            optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
            
            # Attack Training Simulation
            model, _, _ = backdoor_train(model, train_loader, optimizer, trigger_func=T_p, poisoning_ratio=ratio)
            
            ca = clean_accuracy(model, test_loader)
            asr = attack_success_rate(model, test_loader, trigger_func=T_p)
            
            print(f"[{dataset}] Ratio: {ratio} -> CA: {ca:.2f}% | ASR: {asr:.2f}%")
            dataset_results['CA'].append(ca)
            dataset_results['ASR'].append(asr)
            
        plot_accuracy_vs_poisoning(Config.POISONING_RATIOS, dataset_results['CA'], dataset_results['ASR'], 
                                   save_name=f"{dataset}_acc_vs_poisoning.png")
        results[dataset] = dataset_results
        
    Config.DATASET = old_dataset
    return results

def run_table2():
    print("\\n=== Replicating Table 2: Defense Robustness ===")
    train_loader, test_loader = get_dataloaders()
    
    print("Training robust baseline backdoored model...")
    base_model = get_model(Config.DATASET)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=Config.LEARNING_RATE)
    base_model, _, _ = backdoor_train(base_model, train_loader, optimizer, trigger_func=T_p, poisoning_ratio=0.05)
    
    asr_before = attack_success_rate(base_model, test_loader, trigger_func=T_p)
    print(f"Baseline ASR before defenses: {asr_before:.2f}%")
    
    defenses = ['fine_tuning', 'anp', 'clp', 'tsbd', 'nad']
    asr_after_list = []
    
    for defense_name in defenses:
        print(f"\\nEvaluating Defense: {defense_name.upper()}")
        model_copy = copy.deepcopy(base_model)
        
        if defense_name == 'fine_tuning':
            defended_model = fine_tuning_defense(model_copy, train_loader)
        elif defense_name == 'anp':
            defended_model = anp_defense(model_copy, train_loader)
        elif defense_name == 'clp':
            defended_model = clp_defense(model_copy)
        elif defense_name == 'tsbd':
            defended_model = tsbd_defense(model_copy, train_loader)
        elif defense_name == 'nad':
            teacher = get_model(Config.DATASET)
            opt_t = torch.optim.Adam(teacher.parameters(), lr=Config.LEARNING_RATE)
            teacher, _, _ = backdoor_train(teacher, train_loader, opt_t, trigger_func=lambda x:x, poisoning_ratio=0.0)
            defended_model = nad_defense(model_copy, teacher, train_loader)
            
        asr_after = attack_success_rate(defended_model, test_loader, trigger_func=T_p)
        print(f"ASR after {defense_name.upper()}: {asr_after:.2f}%")
        asr_after_list.append(asr_after)
        
    plot_defense_comparison(defenses, [asr_before]*len(defenses), asr_after_list, save_name="defense_robustness.png")

if __name__ == '__main__':
    run_table1()
    run_table2()
