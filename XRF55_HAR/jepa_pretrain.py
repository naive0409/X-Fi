"""Stage-1 JEPA pretraining on XRF55 (JEPAREADME §3, §4.3, §5.1). No labels used.

Per-batch steps (JEPAREADME §5.1):
  1. batch (three modalities; labels ignored)
  2. modality-level mask -> ONE context_modality_list per batch (same mechanism as the
     original X-Fi, whose collate samples one list per batch; per-sample token masks below)
  3. token-level mask -> (B, 3, 32) bool, only present modalities, rate from config
  4. z_cm = online backbone(context_modality_list, token_mask)
  5. z_tgt = EMA target branch(full inputs, all modalities, no token mask), no_grad
  6. z_hat = predictor(z_cm)
  7. loss = SmoothL1(z_hat, z_tgt)
  8. optimizer.step() on linear_projector + X_Fusion (classification head excluded) + predictor
     + mask_token
  9. EMA update (momentum 0.996 -> 1.0 cosine over training progress)

Masking is sampled in the train loop with an independent numpy RandomState (seed+1), NOT in
collate (JEPAREADME §5.2 trap 1); all-false context subsets are resampled (trap 6). Monitoring
per epoch: loss, token-level std of z_tgt over the batch dim (collapse watch, §3.5), per-subset
sampling frequencies, actual token-mask rate. Checkpoints store online/ema/predictor/optimizer
state_dicts separately. Run from XRF55_HAR/ (backbone .pt relative default paths).
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:  # optional dependency: TB logging is enabled automatically once tensorboard is installed
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from XRF55_Dataset import XRF55_Datase
from X_Fi_backbone import X_Fi_Backbone
from jepa_modules import Predictor, EMATargetEncoder, AuxModalityHeads

MODALITY_NAMES = ['mmWave', 'WiFi', 'RFID']
MODALITY_IDX = {'mmWave': 0, 'WiFi': 1, 'RFID': 2}
SUBSET_NAMES = ['mmWave', 'WiFi', 'RFID', 'mmWave+WiFi', 'mmWave+RFID', 'WiFi+RFID', 'all']


def jepa_collate(batch):
    "Stack the three modalities; NO modality sampling here (moved to the train loop, trap 1)."
    wifi = torch.FloatTensor(np.array([t[0] for t in batch]))
    rfid = torch.FloatTensor(np.array([t[1] for t in batch]))
    mmwave = torch.FloatTensor(np.array([t[2] for t in batch]))
    labels = torch.LongTensor([float(t[3]) for t in batch])  # kept for interface parity, unused
    return mmwave, wifi, rfid, labels


def stratified_val_indices(rfid_name_list, val_split, seed):
    """Per-label stratified validation indices, parsed from FILENAMES (no data loading).
    Mirrors XRF55_Dataset.py's label rule: basename field 2 minus 1."""
    import os as _os
    labels = [int(_os.path.basename(_os.path.normpath(f)).split('_')[1]) - 1
              for f in rfid_name_list]
    rng = np.random.RandomState(seed)
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    val_idx = []
    for y, idxs in by_label.items():
        idxs = np.asarray(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val].tolist())
    return sorted(val_idx)


def stratified_subset_indices(rfid_name_list, candidate_idx, fraction, seed):
    """Per-label stratified subset of `candidate_idx` with the given label fraction.

    Used for the few-shot protocol (--label-fraction): draw identical labeled samples for
    every method being compared. With a fixed seed the per-label shuffle order is reused
    and a prefix is taken, so subsets of different fractions are NESTED (10% ⊃ 5% ⊃ 1%).
    """
    import os as _os
    label_of = {i: int(_os.path.basename(_os.path.normpath(f)).split('_')[1]) - 1
                for i, f in enumerate(rfid_name_list)}
    rng = np.random.RandomState(seed)
    by_label = {}
    for i in candidate_idx:
        by_label.setdefault(label_of[i], []).append(i)
    out = []
    for y, idxs in by_label.items():
        idxs = np.asarray(idxs)
        rng.shuffle(idxs)
        n = max(1, int(round(len(idxs) * fraction)))
        out.extend(idxs[:n].tolist())
    return sorted(out)


@torch.no_grad()
def ssl_val_evaluate(backbone, ema, predictor, val_loader, device):
    """Deterministic SSL validation on held-out samples: full modalities, no token mask.
    Returns (smooth_l1_loss, explained_variance) where explained_variance uses the same
    1 - 2*loss/var convention as the D19 analysis."""
    backbone.eval()
    tot_loss, tot_var, tot_n = 0.0, 0.0, 0
    for mm, wifi, rfid, _labels in val_loader:
        mm, wifi, rfid = mm.to(device), wifi.to(device), rfid.to(device)
        z_cm = backbone(mm, wifi, rfid, [True, True, True])
        z_tgt = ema(mm, wifi, rfid)
        z_hat = predictor(z_cm)
        b = mm.size(0)
        tot_loss += torch.nn.functional.smooth_l1_loss(z_hat, z_tgt).item() * b
        tot_var += z_tgt.var().item() * b
        tot_n += b
    avg_loss = tot_loss / max(1, tot_n)
    avg_var = tot_var / max(1, tot_n)
    return avg_loss, 1.0 - 2.0 * avg_loss / max(1e-8, avg_var)


def main():
    parser = argparse.ArgumentParser('X-Fi-JEPA stage-1 pretraining')
    parser.add_argument('--config', type=str, default='configs/jepa_baseline.yaml')
    parser.add_argument('--dataset', type=str, default=None,
                        help='override data.root from the config')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--limit-train-batches', type=int, default=0,
                        help='debug: only run the first N batches per epoch (0 = off)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override optim.epochs (debug/smoke tests)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='unique run name; every artifact of this run (checkpoints, CSV, '
                             'tensorboard, config snapshot) is kept under '
                             '<log.save_dir>/<run-name>/ and nothing is overwritten across '
                             'runs. Defaults to the current time to the second.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg['data']['root'] = args.dataset
    if args.epochs is not None:
        cfg['optim']['epochs'] = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Available pretraining resources: {device}')

    # ---- seeds: model init & data order share the torch global RNG (fixed sequence);
    # ---- mask sampling uses an independent numpy RandomState (trap 8).
    torch.manual_seed(cfg['seed'])
    mask_rng = np.random.RandomState(cfg['seed'] + 1)

    # ---- per-run directory: named by --run-name (default: local time to the second), so
    # ---- successive runs never overwrite each other's weights/logs
    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    save_dir = os.path.join(cfg['log']['save_dir'], run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:  # snapshot for telling runs apart
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'run name: {run_name}\nrun dir:   {save_dir}')

    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=save_dir)
        print(f'tensorboard: tensorboard --logdir {cfg["log"]["save_dir"]}')
    else:
        tb_writer = None
        print('[warn] tensorboard is not installed -> TB logging disabled '
              '(enable with: conda run -n xfi pip install tensorboard); CSV keeps all metrics')

    train_dataset_full = XRF55_Datase(root_dir=cfg['data']['root'], scene='all', is_train=True)
    n_workers = int(cfg['data'].get('num_workers', 4))

    # ---- stratified train/val split (labels parsed from filenames, no data loading); the
    # ---- val part is ONLY used to pick the best_val checkpoint (SSL loss on unseen data)
    val_split = float(cfg['data'].get('val_split', 0.0))
    val_loader = None
    if val_split > 0:
        val_idx = stratified_val_indices(train_dataset_full.RFID_name_list, val_split,
                                         cfg['seed'] + 7)
        train_idx = sorted(set(range(len(train_dataset_full))) - set(val_idx))
        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset = Subset(train_dataset_full, val_idx)
        val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False,
                                num_workers=n_workers, collate_fn=jepa_collate,
                                persistent_workers=n_workers > 0)
        print(f'pretrain data: train {len(train_idx)} / val {len(val_idx)} samples '
              f'(val_split={val_split}, used only for best_val checkpoint selection)')
    else:
        train_dataset = train_dataset_full
        print(f'pretrain dataset: {len(train_dataset)} samples (no val split)')

    loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=n_workers,
        collate_fn=jepa_collate,
        # pin_memory locks page-locked host memory for in-flight batches (~850MB/batch at
        # batch 256 with ~8 in flight). On servers with tight locked-memory limits this
        # makes cudaHostAlloc fail ("CUDA error: invalid argument" in the pin thread), so
        # it is config-controlled and OFF by default (H2D copy is tiny vs compute here).
        pin_memory=bool(cfg['data'].get('pin_memory', False)) and device.type == 'cuda',
        prefetch_factor=(int(cfg['data'].get('prefetch_factor', 2)) if n_workers > 0 else None),
        persistent_workers=n_workers > 0,
    )
    steps_per_epoch = len(loader)
    if args.limit_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)

    # ---- models (JEPAREADME §4.3-1)
    backbone = X_Fi_Backbone(
        model_depth=cfg['model']['model_depth'],
        num_classes=cfg['model']['num_classes'],
        backbone_root=cfg['model']['backbone_root'],
    ).to(device)
    ema = EMATargetEncoder(backbone, base_momentum=cfg['ema']['base_momentum']).to(device)
    pcfg = cfg['model']['predictor']
    predictor = Predictor(
        dim=512, num_tokens=32, depth=pcfg['depth'], num_heads=pcfg['num_heads'],
        ffn_dim=pcfg['ffn_dim'], dropout=pcfg['dropout'],
    ).to(device)

    # ---- cross-modal auxiliary prediction (JEPACHANGES §4-14): decode each ABSENT
    # ---- modality's EMA-side projected features out of the aggregate prediction
    aux_cfg = cfg.get('aux_loss', {}) or {}
    aux_weight = float(aux_cfg.get('weight', 0.0))
    aux_ramp_epochs = int(aux_cfg.get('ramp_epochs', 5))
    aux_norm = aux_cfg.get('normalize', 'both')
    aux = None
    if aux_weight > 0:
        aux = AuxModalityHeads().to(device)
        print(f'aux loss ON: weight {aux_weight} (linear ramp over {aux_ramp_epochs} epochs), '
              f'normalize={aux_norm}, heads=3xLinear(512,512)')

    # ---- optimizer: projector + X_Fusion (classification head excluded: unused by the backbone)
    # ---- + predictor + out_norm + mask_token. Norm/bias params (ndim<=1) are excluded from
    # ---- weight decay (standard practice; decaying BN/LN scale parameters is harmful).
    fusion_params = [p for n, p in backbone.X_Fusion_block.named_parameters()
                     if not n.startswith('classification_head')]
    decay_params, no_decay_params = [], []
    _all_trainable = (list(backbone.linear_projector.parameters())
                      + fusion_params
                      + list(predictor.parameters())
                      + list(backbone.out_norm.parameters()))
    if aux is not None:
        _all_trainable += list(aux.parameters())
    for p in _all_trainable:
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)
    no_decay_params.append(backbone.mask_token)
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': cfg['optim']['weight_decay']},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=cfg['optim']['lr'])

    total_steps = steps_per_epoch * cfg['optim']['epochs']
    warmup_steps = steps_per_epoch * cfg['optim']['warmup_epochs']

    def lr_lambda(step):
        if step < warmup_steps:  # linear warmup
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # cosine to 0
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, global_step = 0, 0
    best_val = float('inf')
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt['online_state'])
        ema.load_state_dict(ckpt['ema_state'])
        predictor.load_state_dict(ckpt['predictor_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        if aux is not None:
            if ckpt.get('aux_state'):
                aux.load_state_dict(ckpt['aux_state'])
            else:
                print('[warn] resuming a pre-aux checkpoint: aux heads start fresh')
        print(f"resumed from {args.resume} at epoch {start_epoch} (best_val={best_val:.6f})")

    mask_cfg = cfg['mask']
    keep_probs = np.array(mask_cfg['modality_keep_probs'], dtype=np.float64)
    token_rate = float(mask_cfg['token_mask_rate'])

    # ---- BN warmup: run forward-only batches in train mode so the projector's BatchNorm
    # ---- running stats converge (the frozen extractors' output scales differ a lot per
    # ---- modality), then sync them into the EMA branch (weights still identical at this
    # ---- point). Without this, the eval-mode target branch starts from (0, 1) init stats
    # ---- and drifts for hundreds of steps. Skipped when resuming (stats already in ckpt).
    n_warm = int(cfg.get('bn_warmup_batches', 50))
    if args.resume is None and n_warm > 0:
        backbone.train()
        print(f'BN warmup: {n_warm} forward-only batches (full modalities, no optimizer step)')
        warmed = 0
        for mm, wifi, rfid, _labels in loader:
            with torch.no_grad():
                backbone(mm.to(device), wifi.to(device), rfid.to(device), [True, True, True])
            warmed += 1
            if warmed >= n_warm:
                break
        ema.sync_buffers_from(backbone)
        print(f'BN warmup done ({warmed} batches); BN running buffers synced into EMA branch')

    csv_path = os.path.join(save_dir, 'train_log.csv')
    if not os.path.exists(csv_path) and start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'loss', 'ztgt_std', 'token_mask_rate', 'lr']
                + [f'freq_{s}' for s in SUBSET_NAMES] + ['momentum', 'val_loss', 'val_expl_var',
                                                         'aux_loss'])

    n_params = sum(p.numel() for p in decay_params + no_decay_params)
    print(f'training {n_params / 1e6:.2f}M params (projector + X_Fusion + predictor + out_norm + mask_token)')

    for epoch in range(start_epoch, cfg['optim']['epochs']):
        backbone.train()   # guard keeps the shared frozen extractor in eval()
        predictor.train()  # EMA branch is permanently eval() by construction

        epoch_loss, epoch_steps = 0.0, 0
        ztgt_std_sum, mask_rate_sum = 0.0, 0.0
        aux_sum, aux_n = 0.0, 0
        subset_count = {s: 0 for s in SUBSET_NAMES}
        momentum_used = None

        progress = tqdm(loader, total=steps_per_epoch, desc=f'epoch {epoch + 1}')
        for i, (mmwave, wifi, rfid, _labels) in enumerate(progress):
            if args.limit_train_batches > 0 and i >= args.limit_train_batches:
                break
            mmwave = mmwave.to(device, non_blocking=True)
            wifi = wifi.to(device, non_blocking=True)
            rfid = rfid.to(device, non_blocking=True)
            bsz = mmwave.size(0)

            # step 2: modality-level context mask — ONE list per batch, mirroring the original
            # X-Fi collate mechanism (the backbone signature takes a single list; per-sample
            # granularity is provided by the token mask below). Non-empty guaranteed (trap 6).
            while True:
                draws = mask_rng.random_sample(3) < keep_probs
                if draws.any():
                    break
            modality_list = [bool(v) for v in draws]

            # step 3: token-level mask, per sample, present modalities only
            token_mask_np = mask_rng.random_sample((bsz, 3, 32)) < token_rate
            token_mask_np[:, [j for j in range(3) if not modality_list[j]], :] = False
            token_mask = torch.from_numpy(token_mask_np).to(device)

            # first-epoch mask bookkeeping (JEPAREADME §5.2 trap 1)
            if epoch == 0 and i < 3 and cfg['log']['log_masks_first_epoch']:
                print(f'[mask record] step {i}: context_modality_list={modality_list}, '
                      f'token_mask_rate(actual)='
                      f'{token_mask[:, [j for j in range(3) if modality_list[j]], :].float().mean().item():.4f}')

            # steps 4-6
            z_cm = backbone(mmwave, wifi, rfid, modality_list, token_mask=token_mask)
            with torch.no_grad():
                # full inputs, all modalities, no token mask; parts = per-modality projected
                # features (EMA side) for the auxiliary cross-modal prediction loss
                z_tgt, ema_parts = ema.forward_with_parts(mmwave, wifi, rfid)
            z_hat = predictor(z_cm)

            # step 7: latent prediction loss on the full token-level representation
            loss = F.smooth_l1_loss(z_hat, z_tgt.detach())

            # step 7b: cross-modal auxiliary prediction — for each modality ABSENT from the
            # context, decode its EMA-side projected features out of the aggregate prediction.
            # Targets AND predictions are parameter-free-LayerNorm'd (scale-free: no affine
            # params, no EMA coupling — structurally immune to the D17 scale feedback).
            aux_loss_val = None
            if aux is not None:
                absent = [j for j, present in enumerate(modality_list) if not present]
                if absent:
                    lam = aux_weight * min(1.0, (epoch + 1) / max(1, aux_ramp_epochs))
                    preds = aux(z_hat, absent)
                    tgts = [ema_parts[m].detach() for m in absent]
                    if aux_norm in ('both', 'pred'):
                        preds = [F.layer_norm(p, (512,)) for p in preds]
                    if aux_norm in ('both', 'target'):
                        tgts = [F.layer_norm(t, (512,)) for t in tgts]
                    aux_loss = torch.stack(
                        [F.smooth_l1_loss(p, t) for p, t in zip(preds, tgts)]).mean()
                    loss = loss + lam * aux_loss
                    aux_loss_val = aux_loss.item()
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping batch')
                continue

            # step 8
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # step 9: EMA update with cosine momentum schedule
            progress_frac = global_step / max(1, total_steps)
            momentum_used = ema.update(backbone, progress=progress_frac)
            global_step += 1
            if aux_loss_val is not None:
                aux_sum += aux_loss_val
                aux_n += 1
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_step', loss.item(), global_step)

            epoch_loss += loss.item()
            epoch_steps += 1
            ztgt_std_sum += z_tgt.std(dim=0).mean().item()  # token-level std over batch dim
            mask_rate_sum += token_mask[:, [j for j in range(3) if modality_list[j]], :] \
                .float().mean().item()
            for s in SUBSET_NAMES:
                if s == 'all':
                    hit = all(modality_list)
                else:
                    hit = all(modality_list[MODALITY_IDX[b]] for b in s.split('+'))
                if hit:
                    subset_count[s] += 1
            progress.set_postfix(loss=f'{loss.item():.4f}',
                                 ztgt_std=f'{ztgt_std_sum / epoch_steps:.4f}')

        # ---- epoch bookkeeping / monitoring (JEPAREADME §3.5)
        n = max(1, epoch_steps)
        freqs = {s: subset_count[s] / n for s in SUBSET_NAMES}
        epoch_metrics = dict(
            epoch=epoch + 1,
            loss=epoch_loss / n,
            ztgt_std=ztgt_std_sum / n,
            token_mask_rate=mask_rate_sum / n,
            lr=scheduler.get_last_lr()[0],
            **{f'freq_{s}': freqs[s] for s in SUBSET_NAMES},
            momentum=momentum_used,
        )
        aux_mean = aux_sum / aux_n if aux_n else None
        print('Epoch:{}, JEPA loss:{:.6f}, z_tgt std:{:.4f}, token mask rate:{:.4f}'.format(
            epoch + 1, epoch_metrics['loss'], epoch_metrics['ztgt_std'],
            epoch_metrics['token_mask_rate'])
            + (f', aux loss:{aux_mean:.6f}' if aux_mean is not None else ''))
        print('  subset sampling freq: ' + ', '.join(f'{s}={freqs[s]:.2%}' for s in SUBSET_NAMES))
        if epoch_metrics['ztgt_std'] < 0.1:
            print('  [warn] z_tgt std < 0.1: representation may be collapsing (JEPAREADME §3.5); '
                  'consider the VICReg-style fallback regularizer')
        # ---- SSL validation on held-out samples (deterministic protocol: full modalities,
        # ---- no token mask) -> best_val checkpoint selection
        val_loss, val_ev = None, None
        if val_loader is not None:
            val_loss, val_ev = ssl_val_evaluate(backbone, ema, predictor, val_loader, device)
            print(f'  val: loss {val_loss:.6f}, explained var {val_ev:.4f}')

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch_metrics[k] for k in
                                    ['epoch', 'loss', 'ztgt_std', 'token_mask_rate', 'lr']
                                    + [f'freq_{s}' for s in SUBSET_NAMES] + ['momentum']]
                                   + ([f'{val_loss:.6f}', f'{val_ev:.4f}'] if val_loader else ['', ''])
                                   + ([f'{aux_mean:.6f}'] if aux_mean is not None else ['']))
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train_epoch', epoch_metrics['loss'], epoch + 1)
            tb_writer.add_scalar('Metrics/ztgt_std', epoch_metrics['ztgt_std'], epoch + 1)
            tb_writer.add_scalar('Metrics/token_mask_rate', epoch_metrics['token_mask_rate'], epoch + 1)
            tb_writer.add_scalar('Optim/lr', epoch_metrics['lr'], epoch + 1)
            tb_writer.add_scalar('Optim/ema_momentum', epoch_metrics['momentum'] or 0.0, epoch + 1)
            for s in SUBSET_NAMES:
                tb_writer.add_scalar(f'Sampling/freq_{s}', freqs[s], epoch + 1)
            if val_loss is not None:
                tb_writer.add_scalar('Val/loss', val_loss, epoch + 1)
                tb_writer.add_scalar('Val/explained_var', val_ev, epoch + 1)
            if aux_mean is not None:
                tb_writer.add_scalar('Loss/aux_train_epoch', aux_mean, epoch + 1)

        # ---- checkpoint: three state_dicts kept separate (JEPAREADME §7)
        ckpt = {
            'online_state': backbone.state_dict(),
            'ema_state': ema.state_dict(),
            'predictor_state': predictor.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val,
            'aux_state': aux.state_dict() if aux is not None else None,
            'config': cfg,
        }
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            ckpt['best_val_loss'] = best_val
            ckpt['val_metrics'] = {'val_loss': val_loss, 'val_expl_var': val_ev}
            torch.save(ckpt, os.path.join(save_dir, 'best_val.pth'))
            print(f'  [best] val loss {best_val:.6f} -> best_val.pth updated')
        if (epoch + 1) % cfg['log']['save_every_epochs'] == 0:
            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth'))

    if tb_writer is not None:
        tb_writer.close()
    print('pretraining done.')


if __name__ == '__main__':
    main()
