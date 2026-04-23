import argparse
import torch
import os
import copy
import time

from config import Config
from utils.data_loader import get_dataloaders
from models.spiking_resnet19 import SpikingResNet19
from models.spiking_vgg16 import SpikingVGG16
from models.nmnist_net import NMNISTNet
from attacks.triggers import T_p, T_s
from attacks.backdoor_train import backdoor_train, create_poison_loader
from evaluation.metrics import clean_accuracy, attack_success_rate
from defenses.fine_tuning import fine_tuning_defense
from defenses.clp import clp_defense
from defenses.anp import anp_defense
from defenses.tsbd import tsbd_defense
from defenses.nad import nad_defense
from utils.monitor import TrainingMonitor


def get_model(dataset):
    """Return the correct model architecture for each dataset (Correction 10)."""
    specs = Config.DATASET_SPECS[dataset]
    arch        = specs['model']
    num_classes = specs['num_classes']
    if arch == 'nmnist_net':
        return NMNISTNet(num_classes=num_classes).to(Config.DEVICE)
    elif arch == 'vgg16':
        return SpikingVGG16(num_classes=num_classes).to(Config.DEVICE)
    else:
        return SpikingResNet19(num_classes=num_classes).to(Config.DEVICE)


def get_trigger(dataset, trigger_arg):
    """
    Return the correct inference trigger for each dataset (Correction 11).
      - CIFAR-10, GTSRB, CIFAR-100 : T_p (power transformation)
      - N-MNIST                     : T_s (neuromorphic noise); T_o not used
    If the user explicitly passed --trigger on the CLI, that overrides.
    """
    if trigger_arg is not None:
        return T_s if trigger_arg == 'T_s' else T_p
    # Default per dataset
    default = Config.DATASET_SPECS[dataset]['trigger']
    return T_s if default == 'T_s' else T_p


def main():
    parser = argparse.ArgumentParser(description="BadSNN Implementation")
    parser.add_argument('--mode',            type=str, required=True,
                        choices=['attack', 'defense', 'both'])
    parser.add_argument('--dataset',         type=str, default='cifar10',
                        choices=['cifar10', 'gtsrb', 'cifar100', 'nmnist'])
    parser.add_argument('--poisoning_ratio', type=float, default=None,
                        help="Override dataset-default poisoning ratio")
    parser.add_argument('--defense',         type=str, default='fine_tuning',
                        choices=['fine_tuning', 'anp', 'clp', 'tsbd', 'nad'])
    parser.add_argument('--trigger',         type=str, default=None,
                        choices=['T_p', 'T_s'],
                        help="Override dataset-default trigger (default: T_p for static, T_s for N-MNIST)")
    parser.add_argument('--epochs',          type=int, default=Config.EPOCHS)
    args = parser.parse_args()

    Config.DATASET = args.dataset
    Config.EPOCHS  = args.epochs

    specs           = Config.DATASET_SPECS[args.dataset]
    poisoning_ratio = args.poisoning_ratio if args.poisoning_ratio is not None else specs['poisoning_ratio']
    tau_t           = specs['tau_t']
    trigger_func    = get_trigger(args.dataset, args.trigger)

    train_loader, test_loader = get_dataloaders()

    os.makedirs(Config.SAVE_DIR,   exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    backdoor_model = None

    # ------------------------------------------------------------------ #
    #  ATTACK PHASE                                                        #
    # ------------------------------------------------------------------ #
    if args.mode in ['attack', 'both']:
        print(f"\n{'='*80}")
        print(f"ATTACK PHASE")
        print(f"{'='*80}")
        print(f"  Dataset:         {args.dataset}")
        print(f"  Trigger:         {trigger_func.__name__}")
        print(f"  Poisoning Ratio: {poisoning_ratio}")
        print(f"  Epochs:          {Config.EPOCHS}")
        print(f"  V_thr_n/t/a:     {Config.V_THR_N} / {Config.V_THR_T} / {Config.V_THR_A}")
        print(f"  tau_t:           {tau_t}")
        print(f"  Model:           {specs['model']} ({specs['num_classes']} classes)")
        print(f"{'='*80}\n")

        model = get_model(args.dataset)

        # Compute D_t_p partition ONCE before the training loop (Correction 8)
        poison_loader = create_poison_loader(
            train_loader,
            target_label    = Config.TARGET_LABEL,
            poisoning_ratio = poisoning_ratio,
            seed            = Config.SEED,
        )

        monitor   = TrainingMonitor(enable_plots=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=1e-5
        )

        csv_path = os.path.join(
            Config.RESULT_DIR,
            f"{args.dataset}_{trigger_func.__name__}_training_log.csv"
        )
        with open(csv_path, "w") as f:
            f.write("Epoch,Loss,Loss_Nominal,Loss_Malicious,Base_CA,CA_Attack,ASR,Time_Elapsed\n")

        start_time  = time.time()
        best_ca     = 0
        best_asr    = 0
        best_ckpt   = None

        for epoch in range(Config.EPOCHS):
            backdoor_model, t_loss, _, loss_n, loss_t = backdoor_train(
                model,
                train_loader,
                poison_loader,
                optimizer,
                tau_t=tau_t,
            )
            scheduler.step()

            should_evaluate = (
                epoch % 5 == 0
                or epoch == Config.EPOCHS - 1
                or epoch in [1, 2, 3, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
            )

            if should_evaluate:
                # Base CA: nominal hyperparameters
                base_ca = clean_accuracy(backdoor_model, test_loader, mode='nominal')
                # CA under attack thresholds
                ca_attack = clean_accuracy(backdoor_model, test_loader, mode='attack')
                # ASR: attack hyperparameters + trigger, non-target samples only
                asr = attack_success_rate(backdoor_model, test_loader, trigger_func=trigger_func)

                elapsed = time.time() - start_time
                monitor.print_status(
                    epoch, Config.EPOCHS, t_loss, loss_n, loss_t,
                    base_ca, ca_attack, asr, warmup=False
                )

                with open(csv_path, "a") as f:
                    f.write(
                        f"{epoch},{t_loss:.4f},{loss_n:.4f},{loss_t:.4f},"
                        f"{base_ca:.2f},{ca_attack:.2f},{asr:.2f},{elapsed:.1f}\n"
                    )

                if base_ca > best_ca:
                    best_ca   = base_ca
                    best_ckpt = f"{args.dataset}_backdoor_best_ca.pth"
                    torch.save(
                        backdoor_model.state_dict(),
                        os.path.join(Config.SAVE_DIR, best_ckpt)
                    )

                if asr > best_asr:
                    best_asr = asr

                if epoch % 10 == 0 or epoch == Config.EPOCHS - 1:
                    plot_path = os.path.join(
                        Config.RESULT_DIR,
                        f"{args.dataset}_{trigger_func.__name__}_epoch{epoch}.png"
                    )
                    monitor.plot_metrics(save_path=plot_path)

                if monitor.health_status == "CRITICAL" and epoch > 15:
                    print("\n" + "="*80)
                    print("CRITICAL HEALTH CHECK FAILED — stopping training")
                    print("="*80)
                    print(monitor.get_summary())
                    break

        torch.save(
            backdoor_model.state_dict(),
            os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth")
        )
        print(monitor.get_summary())
        print(f"\nTraining complete. Total time: {(time.time()-start_time)/60:.1f} min")
        print(f"Best Base CA: {best_ca:.2f}%  |  Best ASR: {best_asr:.2f}%")
        if best_ckpt:
            print(f"Best CA checkpoint: {Config.SAVE_DIR}{best_ckpt}")

    # ------------------------------------------------------------------ #
    #  DEFENSE PHASE                                                       #
    # ------------------------------------------------------------------ #
    if args.mode in ['defense', 'both']:
        print(f"\n--- Defense Phase [{args.defense.upper()}] ---")

        if backdoor_model is None:
            backdoor_model = get_model(args.dataset)
            model_path = os.path.join(Config.SAVE_DIR, f"{args.dataset}_backdoor.pth")
            if os.path.exists(model_path):
                backdoor_model.load_state_dict(
                    torch.load(model_path, map_location=Config.DEVICE)
                )
                print("Loaded pre-trained backdoor model.")
            else:
                print("No pre-trained backdoor model found — training one now.")
                opt = torch.optim.Adam(backdoor_model.parameters(), lr=Config.LEARNING_RATE)
                poison_loader_def = create_poison_loader(
                    train_loader, Config.TARGET_LABEL, poisoning_ratio, seed=Config.SEED
                )
                backdoor_model, _, _, _, _ = backdoor_train(
                    backdoor_model, train_loader, poison_loader_def, opt, tau_t=tau_t
                )

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
            opt_t   = torch.optim.Adam(teacher.parameters(), lr=Config.LEARNING_RATE)
            # Train a clean teacher with poisoning_ratio=0 (Pass 2 loop simply won't run)
            poison_loader_empty = create_poison_loader(
                train_loader, Config.TARGET_LABEL, poisoning_ratio=0.0, seed=Config.SEED
            )
            teacher, _, _, _, _ = backdoor_train(
                teacher, train_loader, poison_loader_empty, opt_t, tau_t=tau_t
            )
            defended_model = nad_defense(copy.deepcopy(backdoor_model), teacher, train_loader)

        ca  = clean_accuracy(defended_model, test_loader, mode='nominal')
        asr = attack_success_rate(defended_model, test_loader, trigger_func=trigger_func)
        print(f"Defense [{args.defense}] -> Base CA: {ca:.2f}%  |  ASR: {asr:.2f}%")


if __name__ == "__main__":
    main()
