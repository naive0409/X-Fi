"""Stage-2 downstream evaluation for the csi_jepa unimodal base (PROJECT_LOG §9, Phase A1).

A reduced copy of jepa_downstream.py: a unimodal model has exactly ONE evaluation condition
(WiFi), so the 7-subset suite collapses to a single sample-granularity test accuracy (the
JSON keeps worst_subset/macro_avg keys, set equal, only for schema compatibility with the
multimodal results).

--method:
  csi_probe     frozen JEPA encoder + fresh classification_Head(dim, 55)      -> group B
  csi_finetune  unfreeze the encoder at low lr + head (mirror recipe)         -> group B'
  csi_scratch   RANDOM-INIT encoder + the identical finetune recipe           -> group C
                (G1: pretraining must beat this; SSL signature = gap widening
                as --label-fraction shrinks)

Architecture is taken from the pretrained checkpoint's embedded config when available
(self-describing), else from the `model:` block of the eval config (scratch path).
Head class imported from X_Fi_backbone so the pooled-mean+LN+Linear protocol is bit-identical
to the multimodal pipeline. Few-shot --label-fraction reuses the NESTED stratified sampler
so every compared method sees the identical labeled samples. Run from XRF55_HAR/.
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

from X_Fi_backbone import classification_Head
from csi_jepa.data import WiFiOnlyDataset, wifi_collate
from csi_jepa.model import CSIJEPAEncoder
from jepa_pretrain import stratified_subset_indices, stratified_val_indices

try:  # optional dependency: TB logging is enabled automatically once tensorboard is installed
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


@torch.no_grad()
def evaluate_accuracy(encoder, head, loader, device, desc='eval', max_batches=0):
    """Sample-granularity test accuracy on the untouched test set (single WiFi condition)."""
    encoder.eval()
    head.eval()
    correct = total = 0
    for i, (wifi, labels) in enumerate(tqdm(loader, desc=desc)):
        if max_batches and i >= max_batches:
            break
        wifi, labels = wifi.to(device), labels.to(device)
        pred = head(encoder(wifi)).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def build_encoder(model_cfg, device):
    encoder = CSIJEPAEncoder(
        channels=int(model_cfg.get('channels', 270)),
        time_steps=int(model_cfg.get('time_steps', 1000)),
        patch_c=int(model_cfg['patch_c']),
        patch_t=int(model_cfg['patch_t']),
        dim=int(model_cfg['dim']),
        depth=int(model_cfg['depth']),
        num_heads=int(model_cfg['num_heads']),
        ffn_dim=int(model_cfg['ffn_dim']),
        dropout=float(model_cfg.get('dropout', 0.0)),
        input_norm=model_cfg.get('input_norm', 'none'),
    ).to(device)
    return encoder


def probe_or_finetune(encoder, method, model_cfg, cfg, device, loader, steps_cap=0,
                      tb_writer=None, val_loader=None, run_dir='.'):
    "Train a fresh head (probe) or head + encoder (finetune/scratch); saves best/last."
    dim = int(model_cfg['dim'])
    torch.manual_seed(cfg['seed'] + 100)  # head init independent of encoder init
    head = classification_Head(dim, 55).to(device)

    for p in encoder.parameters():
        p.requires_grad_(False)
    finetuning = method in ('csi_finetune', 'csi_scratch')
    if finetuning:
        for p in encoder.parameters():
            p.requires_grad_(True)

    stage = cfg['finetune'] if finetuning else cfg['linear_probe']
    if finetuning:
        optimizer = torch.optim.AdamW(
            [{'params': head.parameters(), 'lr': stage['head_lr']},
             {'params': [p for p in encoder.parameters() if p.requires_grad],
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

    # ---- best-by-validation checkpoint tracking (val split from train, never the test set)
    selection_metric = cfg['eval'].get('selection_metric', 'acc_wifi')
    eval_every = int(stage.get('eval_every', 1))
    best_score, best_epoch = -float('inf'), -1

    for epoch in range(stage['epochs']):
        encoder.train() if finetuning else encoder.eval()
        head.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for i, (wifi, labels) in enumerate(tqdm(loader, desc=f'epoch {epoch + 1}')):
            if steps_cap and i >= steps_cap:
                break
            wifi, labels = wifi.to(device), labels.to(device)
            logits = head(encoder(wifi))
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
        print(f'Epoch:{epoch + 1}, train acc:{epoch_acc / max(1, n_batches):.4f}, '
              f'loss:{epoch_loss / max(1, n_batches):.6f}')
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train', epoch_loss / max(1, n_batches), epoch + 1)
            tb_writer.add_scalar('Accuracy/train', epoch_acc / max(1, n_batches), epoch + 1)

        if val_loader is not None and (epoch + 1) % eval_every == 0:
            encoder.eval()
            head.eval()
            correct = total = 0
            with torch.no_grad():
                for wifi, labels in val_loader:
                    wifi, labels = wifi.to(device), labels.to(device)
                    pred = head(encoder(wifi)).argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            m = {'acc_wifi': correct / max(1, total)}
            print(f'  val {selection_metric}: {m[selection_metric]:.4f}')
            if tb_writer is not None:
                tb_writer.add_scalar(f'Val/{selection_metric}', m[selection_metric], epoch + 1)
            if m[selection_metric] > best_score:
                best_score, best_epoch = m[selection_metric], epoch + 1
                torch.save({'method': method, 'encoder_state': encoder.state_dict(),
                            'head_state': head.state_dict(), 'epoch': epoch + 1,
                            'val_metrics': m}, os.path.join(run_dir, 'best.pth'))
                print(f'  [best] {selection_metric}={best_score:.4f} @ epoch {best_epoch} '
                      f'-> best.pth updated')

    torch.save({'method': method, 'encoder_state': encoder.state_dict(),
                'head_state': head.state_dict(), 'epoch': stage['epochs'],
                'val_metrics': None}, os.path.join(run_dir, 'last.pth'))
    if best_epoch > 0:
        print(f"checkpoints saved to {run_dir}: best.pth (epoch {best_epoch}, "
              f"{selection_metric}={best_score:.4f}), last.pth (epoch {stage['epochs']})")
    else:
        print(f'checkpoints saved to {run_dir}: last.pth only (no validation split configured)')
    return head


def main():
    parser = argparse.ArgumentParser('csi_jepa stage-2 downstream evaluation (unimodal)')
    parser.add_argument('--method', required=True,
                        choices=['csi_probe', 'csi_finetune', 'csi_scratch'])
    parser.add_argument('--config', type=str, default='configs/csi_eval_mirror.yaml')
    parser.add_argument('--dataset', type=str, default=None, help='override data.root')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='csi_jepa pretrain checkpoint (jepa_csi_pretrain.py output); '
                             'required for csi_probe/csi_finetune')
    parser.add_argument('--epochs', type=int, default=None, help='override training epochs')
    parser.add_argument('--limit-train-batches', type=int, default=0, help='debug cap')
    parser.add_argument('--limit-eval-batches', type=int, default=0, help='debug cap')
    parser.add_argument('--save-json', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None,
                        help='unique run name; artifacts kept under csi_eval_results/<run-name>/. '
                             'Defaults to the current time to the second.')
    parser.add_argument('--label-fraction', type=float, default=1.0,
                        help='few-shot protocol: fraction of train labels (stratified per class) '
                             'used for training; drawn AFTER the val split so val/test stay '
                             'clean; NESTED across fractions (same seed). 1.0 = all labels.')
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
    torch.manual_seed(cfg['seed'])  # encoder init (scratch); head init reseeds seed+100

    # ---- per-run directory (named by --run-name, default: local time to the second)
    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    run_dir = os.path.join('csi_eval_results', run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=run_dir)
        print('tensorboard: tensorboard --logdir csi_eval_results')
    else:
        tb_writer = None
        print('[warn] tensorboard is not installed -> TB logging disabled (CSV n/a; '
              'results.json keeps final metrics)')
    print(f'run name: {run_name}\nrun dir:   {run_dir}')

    # ---- architecture: checkpoint-embedded config wins over the eval config's model block
    model_cfg = dict(cfg['model'])
    ckpt = None
    if args.method in ('csi_probe', 'csi_finetune'):
        if args.pretrained is None:
            raise SystemExit(f'--pretrained is required for {args.method}')
        ckpt = torch.load(args.pretrained, map_location=device)
        emb = ckpt.get('config', {}).get('model')
        if emb:
            model_cfg = emb
            print(f"architecture from checkpoint ({args.pretrained}, "
                  f"epoch {ckpt.get('epoch')})")
    encoder = build_encoder(model_cfg, device)
    if ckpt is not None:
        encoder.load_state_dict(ckpt['online_state'], strict=True)
        print('loaded pretrained encoder weights (strict=True)')
    else:
        print('random-init encoder (group C lower bound)')

    train_ds_full = WiFiOnlyDataset(cfg['data']['root'], 'all', True)
    val_split = float(cfg['data'].get('val_split', 0.0))
    val_loader = None
    if val_split > 0:
        val_idx = stratified_val_indices(train_ds_full.wifi_name_list, val_split, cfg['seed'] + 7)
        train_idx = sorted(set(range(len(train_ds_full))) - set(val_idx))
        train_ds = Subset(train_ds_full, train_idx)
        val_loader = DataLoader(Subset(train_ds_full, val_idx),
                                batch_size=cfg['data']['train_batch_size'], shuffle=False,
                                num_workers=cfg['data']['num_workers'], collate_fn=wifi_collate)
        print(f'train/val split: {len(train_idx)}/{len(val_idx)} '
              f'(val used only for best.pth selection)')
    else:
        train_ds = train_ds_full
        train_idx = list(range(len(train_ds_full)))

    # ---- few-shot protocol: stratified per-class subsampling of the POST-VAL train part, so
    # ---- every compared method sees the IDENTICAL labeled samples (same fraction + seed)
    label_fraction = float(args.label_fraction)
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    if label_fraction < 1.0:
        labeled_idx = stratified_subset_indices(train_ds_full.wifi_name_list, train_idx,
                                                label_fraction, cfg['seed'] + 11)
        train_ds = Subset(train_ds_full, labeled_idx)
        lab_labels = [int(os.path.basename(os.path.normpath(
            train_ds_full.wifi_name_list[i])).split('_')[1]) - 1 for i in labeled_idx]
        per_class = np.bincount(lab_labels, minlength=55)
        print(f'label fraction {label_fraction}: {len(labeled_idx)} labeled train samples '
              f'(per class min/max: {per_class.min()}/{per_class.max()}); '
              f'val {n_val} (unchanged, drawn before subsampling)')

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['train_batch_size'], shuffle=True,
                              num_workers=cfg['data']['num_workers'], collate_fn=wifi_collate)
    test_loader = DataLoader(WiFiOnlyDataset(cfg['data']['root'], 'all', False),
                             batch_size=cfg['data']['test_batch_size'], shuffle=False,
                             num_workers=cfg['data']['num_workers'], collate_fn=wifi_collate)

    head = probe_or_finetune(encoder, args.method, model_cfg, cfg, device, train_loader,
                             steps_cap=args.limit_train_batches, tb_writer=tb_writer,
                             val_loader=val_loader, run_dir=run_dir)

    # headline numbers = best-by-val when a val split exists, else last-model (mirror recipe);
    # the last model is reported alongside either way (both on the untouched test set)
    best_path = os.path.join(run_dir, 'best.pth')
    best_info = {}
    accs = None
    if os.path.exists(best_path):
        bk = torch.load(best_path, map_location=device)
        encoder.load_state_dict(bk['encoder_state'])
        head.load_state_dict(bk['head_state'])
        accs = evaluate_accuracy(encoder, head, test_loader, device,
                                 desc=f'{args.method}/best', max_batches=args.limit_eval_batches)
        best_info = {'best_epoch': bk['epoch'], 'best_val_metrics': bk.get('val_metrics')}
        print(f"[best-by-val @ epoch {bk['epoch']}] WiFi acc: {accs:.4f}")
    last_ck = torch.load(os.path.join(run_dir, 'last.pth'), map_location=device)
    encoder.load_state_dict(last_ck['encoder_state'])
    head.load_state_dict(last_ck['head_state'])
    accs_last = evaluate_accuracy(encoder, head, test_loader, device,
                                  desc=f'{args.method}/last',
                                  max_batches=args.limit_eval_batches)
    if accs is None:
        accs = accs_last
    else:
        print(f'[last model] WiFi acc: {accs_last:.4f}')

    print(f'\n==== results ({args.method}) ====')
    print(f'  WiFi acc: {accs:.4f}')

    if tb_writer is not None:
        tb_writer.add_scalar('Accuracy/WiFi', accs, 0)
        tb_writer.close()

    out_path = args.save_json
    if out_path is None and cfg['eval']['save_json'] == 'auto':
        out_path = os.path.join(run_dir, 'results.json')
    if out_path:
        # worst_subset/macro_avg kept ONLY for schema compatibility with the multimodal
        # results; a unimodal model has a single condition, so all three are equal.
        acc_dict = {'WiFi': accs, 'worst_subset': accs, 'macro_avg': accs}
        acc_last = {'WiFi': accs_last, 'worst_subset': accs_last, 'macro_avg': accs_last}
        with open(out_path, 'w') as f:
            json.dump({'run_name': run_name, 'method': args.method, 'pretrained': args.pretrained,
                       'dataset': cfg['data']['root'],
                       'arch': {'dim': model_cfg['dim'], 'depth': model_cfg['depth'],
                                'patch_c': model_cfg['patch_c'], 'patch_t': model_cfg['patch_t']},
                       'limit_eval_batches': args.limit_eval_batches,
                       'label_fraction': label_fraction,
                       'n_train_labeled': len(train_ds),
                       'n_val': n_val,
                       'headline': 'best_by_val' if best_info else 'last_model',
                       **best_info, 'accuracy': acc_dict,
                       'accuracy_last_model': acc_last}, f, indent=2)
        print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
