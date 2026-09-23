"""Self-train the ORIGINAL X-Fi baseline (group A) with per-run named artifacts.

Faithful wrapper around the untouched original training logic: it replicates run.py's exact
setup (XRF55_Datase + collate_fn_padd + X_Fi(model_depth=5, num_classes=55) +
torch.manual_seed(3407) + utils.har_train with AdamW@1e-4) and changes ONLY where artifacts
go: <save-root>/<run-name>/ instead of the shared ./pre-trained_weights/ folder. run.py and
utils.py themselves are NOT modified (JEPAREADME §7) — utils.har_train already writes
'checkpoint_<timestamp>.pth' under its save_dir argument, so pointing save_dir at a per-run
directory gives run grouping with zero changes to the original training code. Like the
original, it saves the LAST model once at the end (no best-by-val selection — that matches
how the official released checkpoint was produced).

Overwrite safety: each run gets its own directory; the timestamped checkpoint filename is
unique per run, so nothing is ever overwritten (original run.py also never overwrites, but
it mixes all runs into ./pre-trained_weights/ with only a timestamp to tell them apart).
"""
import argparse
import os
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from XRF55_Dataset import XRF55_Datase
from utils import collate_fn_padd, har_train
from X_Fi import X_Fi
from jepa_pretrain import stratified_subset_indices, stratified_val_indices


def main():
    parser = argparse.ArgumentParser('Original X-Fi supervised baseline (group A), run-named')
    parser.add_argument('--dataset', type=str, default='../data/XRF55_Dataset_split')
    parser.add_argument('--run-name', type=str, default=None,
                        help='unique run name; artifacts go to <save-root>/<run-name>/. '
                             'Defaults to the current time to the second.')
    parser.add_argument('--save-root', type=str, default='./xfi_baseline_runs')
    parser.add_argument('--epochs', type=int, default=100, help='original protocol: 100')
    parser.add_argument('--batch-size', type=int, default=16, help='original protocol: 16')
    parser.add_argument('--lr', type=float, default=1e-4, help='original protocol: 1e-4')
    parser.add_argument('--label-fraction', type=float, default=1.0,
                        help='few-shot protocol: stratified per-class fraction of train labels. '
                             'Candidates exclude the standard 5%% val split (seed 3407+7) so the '
                             'labeled samples match the JEPA few-shot runs exactly.')
    parser.add_argument('--dry-run', action='store_true',
                        help='build data/model, print planned output paths, exit (no training)')
    args = parser.parse_args()

    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    run_dir = os.path.join(args.save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'args.yaml'), 'w') as f:  # snapshot for telling runs apart
        yaml.safe_dump(vars(args), f, sort_keys=False)
    print(f'run name: {run_name}\nrun dir:   {run_dir}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Available training resources: {device}')

    # identical to run.py: same datasets, same collate, same seed position, same model
    train_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=True)
    test_dataset = XRF55_Datase(root_dir=args.dataset, scene='all', is_train=False)
    print(f'train/test: {len(train_dataset)}/{len(test_dataset)}')

    if args.label_fraction < 1.0:
        # few-shot: same labeled pool as the JEPA few-shot runs (post-val part, same seeds),
        # so original-vs-JEPA comparisons at a fraction see identical labeled samples
        val_idx = set(stratified_val_indices(train_dataset.RFID_name_list, 0.05, 3407 + 7))
        candidates = [i for i in range(len(train_dataset)) if i not in val_idx]
        labeled_idx = stratified_subset_indices(train_dataset.RFID_name_list, candidates,
                                                args.label_fraction, 3407 + 11)
        train_dataset = Subset(train_dataset, labeled_idx)
        print(f'label fraction {args.label_fraction}: {len(labeled_idx)} labeled train samples '
              f'(standard val split excluded from the pool)')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn_padd)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                 collate_fn=collate_fn_padd)

    torch.manual_seed(3407)  # same position as run.py (after loaders, before model init)
    model = X_Fi(model_depth=5, num_classes=55)
    model.to(device)

    if args.dry_run:
        print('dry run OK: model built, loaders ready. Checkpoint will be written to')
        print(f'  {os.path.join(run_dir, "checkpoint_<timestamp>.pth")} (last model, original behavior)')
        return

    criterion = torch.nn.CrossEntropyLoss()
    har_train(model=model, train_loader=train_dataloader, test_loader=test_dataloader,
              num_epochs=args.epochs, learning_rate=args.lr, criterion=criterion,
              device=device, save_dir=run_dir, val_random_seed=3407)


if __name__ == '__main__':
    main()
