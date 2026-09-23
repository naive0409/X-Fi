"""Stage-2 downstream evaluation with method switching (JEPAREADME §4.4, §5.4; human req #5).

All methods are evaluated with the SAME routine: accuracy on each of the 7 fixed modality
subsets (sample-granularity, fixing the original multi_test's batch-granularity averaging,
JEPAREADME §5.2 trap 7) + worst-subset + macro average. Results are printed and saved to JSON.

--method:
  xfi_supervised    original X_Fi + full checkpoint (official release or run.py output) -> group A
  jepa_linear_probe JEPA ckpt: freeze projector+X_Fusion, train a fresh head only      -> group B
  jepa_finetune     JEPA ckpt: unfreeze projector+X_Fusion at low lr + train head      -> group B'
  random_init       random projector+X_Fusion + linear probe (lower bound)             -> group C

Notes:
- Probe/finetune training mirrors the original supervised protocol: one modality_list per
  batch sampled with utils-style probabilities (config-driven), CE loss on labels.
- A frozen backbone stays in eval() during probing (deterministic features, BN running
  stats); finetune puts projector+X_Fusion back to train() — the X_Fi_Backbone.train() guard
  keeps the shared frozen feature extractor in eval() either way.
- torch.load is patched with map_location=device because the original X_Fi.py loads its
  backbone .pt files without map_location (they are CUDA archives; see JEPACHANGES).
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from X_Fi import X_Fi
from X_Fi_backbone import X_Fi_Backbone, classification_Head
from XRF55_Dataset import XRF55_Datase
from jepa_pretrain import stratified_val_indices, stratified_subset_indices

try:  # optional dependency: TB logging is enabled automatically once tensorboard is installed
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

MODALITY_IDX = {'mmWave': 0, 'WiFi': 1, 'RFID': 2}
SUBSETS = [
    ('mmWave', [True, False, False]),
    ('WiFi', [False, True, False]),
    ('RFID', [False, False, True]),
    ('mmWave+WiFi', [True, True, False]),
    ('mmWave+RFID', [True, False, True]),
    ('WiFi+RFID', [False, True, True]),
    ('all', [True, True, True]),
]


def jepa_collate(batch):
    wifi = torch.FloatTensor(np.array([t[0] for t in batch]))
    rfid = torch.FloatTensor(np.array([t[1] for t in batch]))
    mmwave = torch.FloatTensor(np.array([t[2] for t in batch]))
    labels = torch.LongTensor([int(t[3]) for t in batch])
    return mmwave, wifi, rfid, labels


@torch.no_grad()
def evaluate_subsets(forward_fn, loader, device, desc='eval', max_batches=0):
    """forward_fn(mm, wifi, rfid, modality_list) -> logits (B, num_classes).
    Returns dict subset->accuracy, sample-granularity (JEPAREADME §5.2 trap 7).
    max_batches > 0 caps the number of batches (debug only — accuracies then partial)."""
    correct = {name: 0 for name, _ in SUBSETS}
    total = 0
    for i, (mm, wifi, rfid, labels) in enumerate(tqdm(loader, desc=desc)):
        if max_batches and i >= max_batches:
            break
        mm, wifi, rfid = mm.to(device), wifi.to(device), rfid.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        for name, ml in SUBSETS:
            pred = forward_fn(mm, wifi, rfid, ml).argmax(dim=1)
            correct[name] += (pred == labels).sum().item()
    accs = {name: correct[name] / max(1, total) for name, _ in SUBSETS}
    accs['worst_subset'] = min(accs[name] for name, _ in SUBSETS)
    accs['macro_avg'] = sum(accs[name] for name, _ in SUBSETS) / len(SUBSETS)
    return accs


def print_accs(accs):
    for name, _ in SUBSETS:
        print(f'  {name:14s} acc: {accs[name]:.4f}')
    print(f'  {"WORST":14s}     : {accs["worst_subset"]:.4f}   macro avg: {accs["macro_avg"]:.4f}')


def build_backbone_from_ckpt(ckpt_path, device):
    backbone = X_Fi_Backbone(model_depth=5, num_classes=55).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['online_state'])
    print(f"loaded JEPA online encoder from {ckpt_path} (epoch {ckpt.get('epoch')})")
    return backbone


@torch.no_grad()
def val_accuracy(forward_fn, val_loader, device, selection_metric):
    """Validation metric for best-checkpoint selection: cheap all-modality accuracy by
    default, or the full 7-subset suite when selection_metric is macro_avg/worst_subset."""
    if selection_metric == 'acc_all':
        correct = total = 0
        for mm, wifi, rfid, labels in val_loader:
            mm, wifi, rfid = mm.to(device), wifi.to(device), rfid.to(device)
            labels = labels.to(device)
            pred = forward_fn(mm, wifi, rfid, [True, True, True]).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        return {'acc_all': correct / max(1, total)}
    return evaluate_subsets(forward_fn, val_loader, device, desc='val')


def probe_or_finetune(backbone, method, cfg, device, loader, steps_cap=0, tb_writer=None,
                      val_loader=None, run_dir='.'):
    "Train a fresh head (probe) or head + projector/X_Fusion (finetune); returns the head."
    torch.manual_seed(cfg['seed'] + 100)  # head init independent of backbone init
    head = classification_Head(512, 55).to(device)

    for p in backbone.parameters():
        p.requires_grad_(False)
    if method == 'jepa_finetune':
        for n, p in backbone.linear_projector.named_parameters():
            p.requires_grad_(True)
        for n, p in backbone.X_Fusion_block.named_parameters():
            if not n.startswith('classification_head'):
                p.requires_grad_(True)

    stage = cfg['linear_probe'] if method == 'jepa_linear_probe' or method == 'random_init' \
        else cfg['finetune']
    if method == 'jepa_finetune':
        optimizer = torch.optim.AdamW(
            [{'params': head.parameters(), 'lr': stage['head_lr']},
             {'params': [p for p in backbone.linear_projector.parameters() if p.requires_grad],
              'lr': stage['backbone_lr']},
             {'params': [p for p in backbone.X_Fusion_block.parameters() if p.requires_grad],
              'lr': stage['backbone_lr']}],
            weight_decay=stage['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=stage['lr'],
                                      weight_decay=stage['weight_decay'])

    total_steps = len(loader) * stage['epochs']
    warmup_steps = len(loader) * stage['warmup_epochs']

    if stage.get('schedule', 'cosine') == 'none':
        # constant lr — mirrors utils.har_train (no scheduler). NOTE: warmup_epochs: 0 alone
        # would degenerate to pure cosine here, hence the explicit 'none' option.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    subset_rng = np.random.RandomState(cfg['seed'] + 3)
    keep_probs = np.array(cfg['data']['modality_keep_probs'])

    # ---- best-by-validation checkpoint tracking (val split from train, never the test set)
    selection_metric = cfg['eval'].get('selection_metric', 'acc_all')
    eval_every = int(stage.get('eval_every', 1))
    best_score, best_epoch = -float('inf'), -1

    for epoch in range(stage['epochs']):
        backbone.train() if method == 'jepa_finetune' else backbone.eval()
        head.train()
        # a frozen backbone must stay in eval() (deterministic features, BN running stats);
        # finetune trains projector/X_Fusion in train() — extractor stays eval via the guard
        if method == 'jepa_finetune':
            backbone.feature_extractor.eval()

        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for i, (mm, wifi, rfid, labels) in enumerate(tqdm(loader, desc=f'epoch {epoch + 1}')):
            if steps_cap and i >= steps_cap:
                break
            mm, wifi, rfid = mm.to(device), wifi.to(device), rfid.to(device)
            labels = labels.to(device)

            while True:  # one modality_list per batch, mirrors utils.py protocol
                draws = subset_rng.random_sample(3) < keep_probs
                if draws.any():
                    break
            ml = [bool(v) for v in draws]

            logits = head(backbone(mm, wifi, rfid, ml))
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping')
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(1) == labels).float().mean().item()
            n_batches += 1
        print(f'Epoch:{epoch + 1}, head train acc:{epoch_acc / max(1, n_batches):.4f}, '
              f'loss:{epoch_loss / max(1, n_batches):.6f}')
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train', epoch_loss / max(1, n_batches), epoch + 1)
            tb_writer.add_scalar('Accuracy/train', epoch_acc / max(1, n_batches), epoch + 1)

        if val_loader is not None and (epoch + 1) % eval_every == 0:
            backbone.eval()
            head.eval()

            def val_forward(mm, wifi, rfid, ml):
                return head(backbone(mm, wifi, rfid, ml))

            m = val_accuracy(val_forward, val_loader, device, selection_metric)
            print(f'  val {selection_metric}: {m[selection_metric]:.4f}')
            if tb_writer is not None:
                tb_writer.add_scalar(f'Val/{selection_metric}', m[selection_metric], epoch + 1)
            if m[selection_metric] > best_score:
                best_score, best_epoch = m[selection_metric], epoch + 1
                torch.save({'method': method, 'backbone_state': backbone.state_dict(),
                            'head_state': head.state_dict(), 'epoch': epoch + 1,
                            'val_metrics': m}, os.path.join(run_dir, 'best.pth'))
                print(f'  [best] {selection_metric}={best_score:.4f} @ epoch {best_epoch} '
                      f'-> best.pth updated')

    torch.save({'method': method, 'backbone_state': backbone.state_dict(),
                'head_state': head.state_dict(), 'epoch': stage['epochs'], 'val_metrics': None},
               os.path.join(run_dir, 'last.pth'))
    if best_epoch > 0:
        print(f"checkpoints saved to {run_dir}: best.pth (epoch {best_epoch}, "
              f"{selection_metric}={best_score:.4f}), last.pth (epoch {stage['epochs']})")
    else:
        print(f'checkpoints saved to {run_dir}: last.pth only (no validation split configured)')
    return head


def main():
    parser = argparse.ArgumentParser('X-Fi-JEPA stage-2 downstream evaluation')
    parser.add_argument('--method', required=True,
                        choices=['xfi_supervised', 'jepa_linear_probe', 'jepa_finetune',
                                 'random_init'])
    parser.add_argument('--config', type=str, default='configs/eval_methods.yaml')
    parser.add_argument('--dataset', type=str, default=None, help='override data.root')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='JEPA checkpoint (jepa_pretrain.py output), methods jepa_*')
    parser.add_argument('--xfi-weights', type=str, default=None,
                        help='full X_Fi checkpoint for --method xfi_supervised')
    parser.add_argument('--epochs', type=int, default=None, help='override training epochs')
    parser.add_argument('--limit-train-batches', type=int, default=0, help='debug cap')
    parser.add_argument('--limit-eval-batches', type=int, default=0, help='debug cap')
    parser.add_argument('--save-json', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None,
                        help='unique run name; results/config/tensorboard of this run are kept '
                             'under jepa_eval_results/<run-name>/. Defaults to the current time '
                             'to the second.')
    parser.add_argument('--label-fraction', type=float, default=1.0,
                        help='few-shot protocol: fraction of train labels (stratified per class) '
                             'used for probe/finetune training; drawn AFTER the val split so '
                             'val/test stay clean. 1.0 = all labels (default).')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset:
        cfg['data']['root'] = args.dataset
    if args.epochs is not None:
        cfg['linear_probe']['epochs'] = args.epochs
        cfg['finetune']['epochs'] = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Available downstream resources: {device}; method = {args.method}')
    torch.manual_seed(cfg['seed'])  # model init (head init reseeds seed+100 inside probe)

    # ---- per-run directory (named by --run-name, default: local time to the second)
    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    run_dir = os.path.join('jepa_eval_results', run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:  # snapshot for telling runs apart
        yaml.safe_dump(cfg, f, sort_keys=False)
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=run_dir)
        print(f'tensorboard: tensorboard --logdir jepa_eval_results')
    else:
        tb_writer = None
        print('[warn] tensorboard is not installed -> TB logging disabled '
              '(enable with: conda run -n xfi pip install tensorboard)')
    print(f'run name: {run_name}\nrun dir:   {run_dir}')

    # shim: original X_Fi.py / checkpoints assume torch.load default device; released .pt
    # archives hold CUDA tensors, so force everything onto the selected device
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'map_location': device})

    train_ds_full = XRF55_Datase(cfg['data']['root'], 'all', True)
    val_split = float(cfg['data'].get('val_split', 0.0))
    val_loader = None
    if val_split > 0 and args.method != 'xfi_supervised':
        # stratified by label (parsed from filenames); val is ONLY for best-ckpt selection,
        # never the test set (no test leakage into model selection)
        val_idx = stratified_val_indices(train_ds_full.RFID_name_list, val_split, cfg['seed'] + 7)
        train_idx = sorted(set(range(len(train_ds_full))) - set(val_idx))
        train_ds = Subset(train_ds_full, train_idx)
        val_loader = DataLoader(Subset(train_ds_full, val_idx),
                                batch_size=cfg['data']['train_batch_size'], shuffle=False,
                                num_workers=cfg['data']['num_workers'], collate_fn=jepa_collate)
        print(f'train/val split: {len(train_idx)}/{len(val_idx)} (val used only for best.pth selection)')
    else:
        train_ds = train_ds_full
        train_idx = list(range(len(train_ds_full)))

    # ---- few-shot protocol: stratified per-class subsampling of the POST-VAL train part, so
    # ---- every compared method sees the IDENTICAL labeled samples (same fraction + seed)
    label_fraction = float(args.label_fraction)
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    if label_fraction < 1.0 and args.method != 'xfi_supervised':
        labeled_idx = stratified_subset_indices(train_ds_full.RFID_name_list, train_idx,
                                                label_fraction, cfg['seed'] + 11)
        train_ds = Subset(train_ds_full, labeled_idx)
        lab_labels = [int(os.path.basename(os.path.normpath(train_ds_full.RFID_name_list[i])).split('_')[1]) - 1
                      for i in labeled_idx]
        per_class = np.bincount(lab_labels, minlength=55)
        print(f'label fraction {label_fraction}: {len(labeled_idx)} labeled train samples '
              f'(per class min/max: {per_class.min()}/{per_class.max()}); '
              f'val {n_val} (unchanged, drawn before subsampling)')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['data']['train_batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'], collate_fn=jepa_collate)
    test_loader = DataLoader(
        XRF55_Datase(cfg['data']['root'], 'all', False),
        batch_size=cfg['data']['test_batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'], collate_fn=jepa_collate)
    if args.limit_eval_batches:
        test_loader = DataLoader(test_loader.dataset,
                                 batch_size=cfg['data']['test_batch_size'], shuffle=False,
                                 num_workers=cfg['data']['num_workers'],
                                 collate_fn=jepa_collate)

    best_info, accs_last = {}, None
    if args.method == 'xfi_supervised':
        if args.xfi_weights is None:
            raise SystemExit('--xfi-weights is required for xfi_supervised')
        model = X_Fi(model_depth=5, num_classes=55).to(device)
        model.load_state_dict(torch.load(args.xfi_weights, map_location=device))
        model.eval()
        accs = evaluate_subsets(model, test_loader, device, desc='xfi_supervised',
                                max_batches=args.limit_eval_batches)
        accs_last = accs  # eval-only method: no best/last distinction
    else:
        backbone = None
        if args.method == 'random_init':
            backbone = X_Fi_Backbone(model_depth=5, num_classes=55).to(device)
            print('random-init backbone (group C lower bound)')
        else:
            if args.pretrained is None:
                raise SystemExit(f'--pretrained is required for {args.method}')
            backbone = build_backbone_from_ckpt(args.pretrained, device)
        head = probe_or_finetune(backbone, args.method, cfg, device, train_loader,
                                 steps_cap=args.limit_train_batches, tb_writer=tb_writer,
                                 val_loader=val_loader, run_dir=run_dir)
        backbone.eval()
        head.eval()

        def forward_fn(mm, wifi, rfid, ml):
            return head(backbone(mm, wifi, rfid, ml))

        # headline numbers = the best-by-validation checkpoint; the last model is reported
        # alongside so the two conventions can be compared (both on the untouched test set)
        best_path = os.path.join(run_dir, 'best.pth')
        best_info = {}
        if os.path.exists(best_path):
            bk = torch.load(best_path, map_location=device)
            backbone.load_state_dict(bk['backbone_state'])
            head.load_state_dict(bk['head_state'])
            backbone.eval()
            head.eval()
            accs = evaluate_subsets(forward_fn, test_loader, device, desc=f'{args.method}/best',
                                    max_batches=args.limit_eval_batches)
            best_info = {'best_epoch': bk['epoch'], 'best_val_metrics': bk.get('val_metrics')}
            print(f"[best-by-val @ epoch {bk['epoch']}]")
            print_accs(accs)
        accs_last = evaluate_subsets(forward_fn, test_loader, device, desc=f'{args.method}/last',
                                     max_batches=args.limit_eval_batches)
        if best_info:
            print('[last model]')
            print_accs(accs_last)
        else:
            accs = accs_last

    print(f'\n==== results ({args.method}) ====')
    print_accs(accs)

    if tb_writer is not None:
        for name, _ in SUBSETS:
            tb_writer.add_scalar(f'Accuracy/{name}', accs[name], 0)
        tb_writer.add_scalar('Accuracy/worst_subset', accs['worst_subset'], 0)
        tb_writer.add_scalar('Accuracy/macro_avg', accs['macro_avg'], 0)
        tb_writer.close()

    out_path = args.save_json
    if out_path is None and cfg['eval']['save_json'] == 'auto':
        out_path = os.path.join(run_dir, 'results.json')
    if out_path:
        with open(out_path, 'w') as f:
            json.dump({'run_name': run_name, 'method': args.method, 'pretrained': args.pretrained,
                       'xfi_weights': args.xfi_weights, 'dataset': cfg['data']['root'],
                       'limit_eval_batches': args.limit_eval_batches,
                       'label_fraction': label_fraction if args.method != 'xfi_supervised' else None,
                       'n_train_labeled': len(train_ds) if args.method != 'xfi_supervised' else None,
                       'n_val': n_val,
                       'headline': 'best_by_val' if best_info else 'last_model',
                       **best_info, 'accuracy': accs, 'accuracy_last_model': accs_last}, f,
                      indent=2)
        print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
