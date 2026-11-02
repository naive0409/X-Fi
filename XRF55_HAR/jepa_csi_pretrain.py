"""Stage-1 JEPA pretraining for the csi_jepa unimodal base (PROJECT_LOG §9, Phase A1).

A reduced copy of jepa_pretrain.py (the project's 'copy with targeted changes' convention)
for the from-scratch WiFi-JEPA-style encoder:

  REMOVED  modality machinery (modality_list / subset schedule / keep_probs) — the model
           sees WiFi CSI only; cross-modal auxiliary heads; BN warmup + buffer sync
           (the encoder has no BatchNorm — that failure class is structurally gone).
  KEPT     train+test unlabeled corpus (include_test_split, disclosed protocol), stratified
           val from the train part only (D34: SSL val is diagnostic, downstream is the
           arbiter), independent mask RNG (trap 8), per-run dirs + TensorBoard + CSV,
           checkpoint set last/epoch_N/best_val, ztgt_std collapse watch (D19).
  NEW      2D block masking on the (grid_c=9, grid_t=10) token grid — rectangular blocks
           (rect_c x rect_t) to a target rate + per-sample link dropout (one full RX link
           masked, the WiFi-JEPA link-mask analog).

Per-batch steps:
  1. batch of raw WiFi CSI (B, 270, 1000); labels ignored
  2. grid token mask (B, 90) bool sampled in-loop with the independent RandomState
  3. z_ctx = online encoder(x, token_mask)
  4. z_tgt = EMA target encoder(x), full input, no mask, no_grad
  5. z_hat = predictor(z_ctx)
  6. loss = SmoothL1(z_hat, z_tgt); optimizer step; EMA update (0.996 -> 1.0 cosine)

Run from XRF55_HAR/.
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

try:  # optional dependency: TB logging is enabled automatically once tensorboard is installed
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from csi_jepa.data import WiFiOnlyDataset, wifi_collate
from csi_jepa.model import CSIJEPAEncoder, CSIEMATargetEncoder
from jepa_modules import Predictor
from jepa_pretrain import stratified_val_indices


@torch.no_grad()
def ssl_val_evaluate(encoder, ema, predictor, val_loader, device):
    """Deterministic SSL validation: full input, no token mask. Returns
    (smooth_l1_loss, explained_variance) with the D19 1 - 2*loss/var convention."""
    encoder.eval()
    tot_loss, tot_var, tot_n = 0.0, 0.0, 0
    for wifi, _labels in val_loader:
        wifi = wifi.to(device)
        z_ctx = encoder(wifi)
        z_tgt = ema(wifi)
        z_hat = predictor(z_ctx)
        b = wifi.size(0)
        tot_loss += F.smooth_l1_loss(z_hat, z_tgt).item() * b
        tot_var += z_tgt.var().item() * b
        tot_n += b
    avg_loss = tot_loss / max(1, tot_n)
    avg_var = tot_var / max(1, tot_n)
    return avg_loss, 1.0 - 2.0 * avg_loss / max(1e-8, avg_var)


@torch.no_grad()
def compute_channel_stats(dataset, batch_size, num_workers):
    """Per-channel mean/std over the whole corpus (input_norm='channel_std' only).
    One read-only pass; stats are installed into the model, not the dataloader (trap 9)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=wifi_collate)
    tot, tot_sq, n = None, None, 0
    for wifi, _ in loader:
        x = wifi.double()
        tot = x.sum(dim=(0, 2)) if tot is None else tot + x.sum(dim=(0, 2))
        tot_sq = (x ** 2).sum(dim=(0, 2)) if tot_sq is None else tot_sq + (x ** 2).sum(dim=(0, 2))
        n += x.shape[0] * x.shape[2]
    mean = tot / n
    std = torch.sqrt((tot_sq / n - mean ** 2).clamp(min=0))
    return mean.float(), std.float()


def build_grid_mask(mask_rng, bsz, grid_c, grid_t, rect_c, rect_t, rate, link_prob):
    """(B, grid_c*grid_t) bool mask: rectangular blocks to the target rate + link dropout.

    Blocks: n_blocks = round(L*rate/area) rectangles drawn without replacement over the
    start grid (overlaps allowed — same simplicity as the 1D block code). Link dropout:
    per sample, with prob link_prob one full channel group (row of grid_t tokens) is
    additionally masked (the WiFi-JEPA link-mask analog)."""
    L = grid_c * grid_t
    g = np.zeros((bsz, grid_c, grid_t), dtype=bool)
    n_starts = (grid_c - rect_c + 1) * (grid_t - rect_t + 1)
    n_blocks = max(1, int(round(L * rate / (rect_c * rect_t))))
    for s in range(bsz):
        for st in mask_rng.permutation(n_starts)[:n_blocks]:
            r, c = divmod(int(st), grid_t - rect_t + 1)
            g[s, r:r + rect_c, c:c + rect_t] = True
        if mask_rng.random_sample() < link_prob:
            g[s, mask_rng.randint(grid_c), :] = True
    return g.reshape(bsz, L)


def main():
    parser = argparse.ArgumentParser('csi_jepa stage-1 pretraining (unimodal, from scratch)')
    parser.add_argument('--config', type=str, default='configs/csi_jepa_base.yaml')
    parser.add_argument('--dataset', type=str, default=None,
                        help='override data.root from the config')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--limit-train-batches', type=int, default=0,
                        help='debug: only run the first N batches per epoch (0 = off)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override optim.epochs (debug/smoke tests)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='unique run name; every artifact of this run is kept under '
                             '<log.save_dir>/<run-name>/. Defaults to the current time.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg['data']['root'] = args.dataset
    if args.epochs is not None:
        cfg['optim']['epochs'] = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Available pretraining resources: {device}')

    torch.manual_seed(cfg['seed'])
    mask_rng = np.random.RandomState(cfg['seed'] + 1)

    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    save_dir = os.path.join(cfg['log']['save_dir'], run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'run name: {run_name}\nrun dir:   {save_dir}')

    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=save_dir)
        print(f'tensorboard: tensorboard --logdir {cfg["log"]["save_dir"]}')
    else:
        tb_writer = None
        print('[warn] tensorboard is not installed -> TB logging disabled '
              '(CSV keeps all metrics)')

    mcfg = cfg['model']
    train_part = WiFiOnlyDataset(root_dir=cfg['data']['root'], scene='all', is_train=True)
    n_workers = int(cfg['data'].get('num_workers', 4))

    include_test = bool(cfg['data'].get('include_test_split', False))
    if include_test:
        test_part = WiFiOnlyDataset(root_dir=cfg['data']['root'], scene='all', is_train=False)
        corpus = ConcatDataset([train_part, test_part])
        n_test_added = len(test_part)
    else:
        corpus = train_part
        n_test_added = 0

    val_split = float(cfg['data'].get('val_split', 0.0))
    val_loader = None
    val_idx = stratified_val_indices(train_part.wifi_name_list, val_split,
                                     cfg['seed'] + 7) if val_split > 0 else []
    train_idx = sorted(set(range(len(train_part))) - set(val_idx))
    if include_test:
        train_idx = train_idx + list(range(len(train_part), len(train_part) + n_test_added))
    train_dataset = Subset(corpus, train_idx)
    if val_split > 0:
        val_loader = DataLoader(Subset(corpus, val_idx), batch_size=cfg['data']['batch_size'],
                                shuffle=False, num_workers=n_workers, collate_fn=wifi_collate,
                                persistent_workers=n_workers > 0)
    if include_test:
        print(f'pretrain corpus: {len(train_dataset)} train / {len(val_idx)} val samples '
              f'(train {len(train_part) - len(val_idx)} + test {n_test_added} UNLABELED — '
              f'disclosed; val drawn from train part only)')
    else:
        print(f'pretrain data: train {len(train_dataset)} / val {len(val_idx)} samples '
              f'(val_split={val_split}, used only for best_val checkpoint selection)')

    loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=n_workers,
        collate_fn=wifi_collate,
        # pin_memory OFF by default: locked-memory limit crash, D16 (see jepa_pretrain.py)
        pin_memory=bool(cfg['data'].get('pin_memory', False)) and device.type == 'cuda',
        prefetch_factor=(int(cfg['data'].get('prefetch_factor', 2)) if n_workers > 0 else None),
        persistent_workers=n_workers > 0,
    )
    steps_per_epoch = len(loader)
    if args.limit_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)

    encoder = CSIJEPAEncoder(
        channels=int(mcfg.get('channels', 270)),
        time_steps=int(mcfg.get('time_steps', 1000)),
        patch_c=int(mcfg['patch_c']),
        patch_t=int(mcfg['patch_t']),
        dim=int(mcfg['dim']),
        depth=int(mcfg['depth']),
        num_heads=int(mcfg['num_heads']),
        ffn_dim=int(mcfg['ffn_dim']),
        dropout=float(mcfg.get('dropout', 0.0)),
        input_norm=mcfg.get('input_norm', 'none'),
    ).to(device)
    ema = CSIEMATargetEncoder(encoder, base_momentum=cfg['ema']['base_momentum']).to(device)
    pcfg = mcfg['predictor']
    predictor = Predictor(
        dim=int(mcfg['dim']), num_tokens=encoder.num_tokens, depth=pcfg['depth'],
        num_heads=pcfg['num_heads'], ffn_dim=pcfg['ffn_dim'], dropout=pcfg['dropout'],
    ).to(device)
    print(f'token grid: {encoder.tokenizer.grid_c} links x {encoder.tokenizer.grid_t} time steps '
          f'= {encoder.num_tokens} tokens (patch {mcfg["patch_c"]}x{mcfg["patch_t"]})')
    print(f'encoder params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M, '
          f'predictor params: {sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M')

    if encoder.tokenizer.input_norm_mode == 'channel_std':
        mean, std = compute_channel_stats(train_dataset, cfg['data']['batch_size'], n_workers)
        encoder.tokenizer.set_input_stats(mean.to(device), std.to(device))
        ema.encoder.tokenizer.set_input_stats(mean.to(device), std.to(device))
        print('input_norm=channel_std: corpus stats installed into both branches')

    # ---- optimizer over ALL trainable params (encoder is fully trainable, unlike rounds
    # ---- 1-5) + predictor; weight-decay exclusion for ndim<=1 params and the mask token.
    decay_params, no_decay_params = [], []
    for name, p in encoder.named_parameters():
        (no_decay_params if (p.ndim <= 1 or name.endswith('mask_token')) else decay_params)\
            .append(p)
    for p in predictor.parameters():
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)
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
        encoder.load_state_dict(ckpt['online_state'])
        ema.load_state_dict(ckpt['ema_state'])
        predictor.load_state_dict(ckpt['predictor_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"resumed from {args.resume} at epoch {start_epoch} (best_val={best_val:.6f})")

    mask_cfg = cfg['mask']
    rect_c = max(1, min(encoder.tokenizer.grid_c, int(mask_cfg.get('rect_c', 2))))
    rect_t = max(1, min(encoder.tokenizer.grid_t, int(mask_cfg.get('rect_t', 2))))
    token_rate = float(mask_cfg.get('token_mask_rate', 0.5))
    link_prob = float(mask_cfg.get('link_dropout_prob', 0.3))
    print(f'grid mask: rect {rect_c}x{rect_t} to rate {token_rate} '
          f'+ link dropout p={link_prob}')

    csv_path = os.path.join(save_dir, 'train_log.csv')
    if not os.path.exists(csv_path) and start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'ztgt_std', 'token_mask_rate', 'lr',
                                    'momentum', 'val_loss', 'val_expl_var'])

    n_params = sum(p.numel() for g in (decay_params, no_decay_params) for p in g)
    print(f'training {n_params / 1e6:.2f}M params (encoder + predictor)')

    for epoch in range(start_epoch, cfg['optim']['epochs']):
        encoder.train()    # EMA branch is permanently eval() by construction
        predictor.train()

        epoch_loss, epoch_steps = 0.0, 0
        ztgt_std_sum, mask_rate_sum = 0.0, 0.0
        momentum_used = None

        progress = tqdm(loader, total=steps_per_epoch, desc=f'epoch {epoch + 1}')
        for i, (wifi, _labels) in enumerate(progress):
            if args.limit_train_batches > 0 and i >= args.limit_train_batches:
                break
            wifi = wifi.to(device, non_blocking=True)
            bsz = wifi.size(0)

            # grid token mask, per sample (independent RandomState — trap 8)
            token_mask = torch.from_numpy(build_grid_mask(
                mask_rng, bsz, encoder.tokenizer.grid_c, encoder.tokenizer.grid_t,
                rect_c, rect_t, token_rate, link_prob)).to(device)

            if epoch == 0 and i < 3 and cfg['log']['log_masks_first_epoch']:
                print(f'[mask record] step {i}: token_mask_rate(actual)='
                      f'{token_mask.float().mean().item():.4f}')

            z_ctx = encoder(wifi, token_mask=token_mask)
            with torch.no_grad():
                z_tgt = ema(wifi)  # full input, no mask
            z_hat = predictor(z_ctx)

            loss = F.smooth_l1_loss(z_hat, z_tgt.detach())
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping batch')
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_frac = global_step / max(1, total_steps)
            momentum_used = ema.update(encoder, progress=progress_frac)
            global_step += 1
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_step', loss.item(), global_step)

            epoch_loss += loss.item()
            epoch_steps += 1
            ztgt_std_sum += z_tgt.std(dim=0).mean().item()  # token-level std over batch dim
            mask_rate_sum += token_mask.float().mean().item()
            progress.set_postfix(loss=f'{loss.item():.4f}',
                                 ztgt_std=f'{ztgt_std_sum / epoch_steps:.4f}')

        # ---- epoch bookkeeping / monitoring
        n = max(1, epoch_steps)
        epoch_metrics = dict(
            epoch=epoch + 1, loss=epoch_loss / n, ztgt_std=ztgt_std_sum / n,
            token_mask_rate=mask_rate_sum / n, lr=scheduler.get_last_lr()[0],
            momentum=momentum_used,
        )
        print('Epoch:{}, JEPA loss:{:.6f}, z_tgt std:{:.4f}, token mask rate:{:.4f}'.format(
            epoch + 1, epoch_metrics['loss'], epoch_metrics['ztgt_std'],
            epoch_metrics['token_mask_rate']))
        if epoch_metrics['ztgt_std'] < 0.1:
            print('  [warn] z_tgt std < 0.1: representation may be collapsing; '
                  'downstream remains the arbiter (D34)')
        val_loss, val_ev = None, None
        if val_loader is not None:
            val_loss, val_ev = ssl_val_evaluate(encoder, ema, predictor, val_loader, device)
            print(f'  val: loss {val_loss:.6f}, explained var {val_ev:.4f}')

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch_metrics[k] for k in
                 ['epoch', 'loss', 'ztgt_std', 'token_mask_rate', 'lr', 'momentum']]
                + ([f'{val_loss:.6f}', f'{val_ev:.4f}'] if val_loader else ['', '']))
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train_epoch', epoch_metrics['loss'], epoch + 1)
            tb_writer.add_scalar('Metrics/ztgt_std', epoch_metrics['ztgt_std'], epoch + 1)
            tb_writer.add_scalar('Metrics/token_mask_rate', epoch_metrics['token_mask_rate'],
                                 epoch + 1)
            tb_writer.add_scalar('Optim/lr', epoch_metrics['lr'], epoch + 1)
            tb_writer.add_scalar('Optim/ema_momentum', epoch_metrics['momentum'] or 0.0, epoch + 1)
            if val_loss is not None:
                tb_writer.add_scalar('Val/loss', val_loss, epoch + 1)
                tb_writer.add_scalar('Val/explained_var', val_ev, epoch + 1)

        # ---- checkpoint set (same keys as jepa_pretrain minus aux_state)
        ckpt = {
            'online_state': encoder.state_dict(),
            'ema_state': ema.state_dict(),
            'predictor_state': predictor.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val,
            'config': cfg,
        }
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            ckpt['best_val_loss'] = best_val
            ckpt['val_metrics'] = {'val_loss': val_loss, 'val_expl_var': val_ev}
            torch.save(ckpt, os.path.join(save_dir, 'best_val.pth'))
            print(f'  [best] val loss {best_val:.6f} -> best_val.pth updated '
                  f'(diagnostic only — D34)')
        if (epoch + 1) % cfg['log']['save_every_epochs'] == 0:
            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth'))

    if tb_writer is not None:
        tb_writer.close()
    print('csi_jepa pretraining done.')


if __name__ == '__main__':
    main()
