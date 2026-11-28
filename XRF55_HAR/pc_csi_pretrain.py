"""pc_csi_pretrain.py — 点云条件化 CSI-JEPA 预训练（Phase PC，PROJECT_LOG §12）。

dloc_csi_pretrain_fp.py 的复制减法 + 场景模态替换（2D raster -> 3D 点云 patch）：
  - CSI = (T, 2, 256) re/im 快照（rooms10/sionna 同构），tokenizer 网格 (1,8)=8 token
  - 场景 tokens = 环境点云 patch（FPS+kNN 预分组，逐场景缓存；PointPatchTokenizer）
  - scene_mode: pc（主）/ shuffled（错配查表检测）/ onehot（id 对照）/ none（CSI-only）
  - EMA 目标 = CSI-only 视图（target_mode: csi_only 默认）或 [pc; csi] 全视图
    （target_mode: dual，Sonata 式对照——场景恒定输出风险见 D47）
  - 物理辅助：P1（pc tokens -> 逐 (TX,频带) mean|CSI| 8 维，无参数 LN 两侧，λ0.3
    ramp5）+ P2（pc tokens -> 路径统计 3 维，标准化域回归，config 门控）+
    center_aux（Point-JMAE 式被 scene_token_dropout 掩掉的 patch 中心预测）
  - geo_dropout 在 PointPatchTokenizer 内（training 态自动生效）

Run from XRF55_HAR/.
"""
import argparse
import copy
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from csi_jepa.pc_data import PCSceneCorpus, PCDataset, pc_collate, stratified_subset_by_scene
from csi_jepa.pointcloud import PCSIJEPA, P1Head, P2Head, CenterAuxHead, PatchProvider
from csi_jepa.model import CSIEMATargetEncoder
from jepa_csi_pretrain import build_grid_mask
from jepa_modules import Predictor


@torch.no_grad()
def compute_channel_stats(loader, device):
    "per-(行) 均值/方差（input_norm='channel_std'；buffer 进模型，dataloader 恒原始值）。"
    tot, tot_sq, n = None, None, 0
    for batch in loader:
        x = batch['csi'].to(device).double()
        tot = x.sum(dim=(0, 2)) if tot is None else tot + x.sum(dim=(0, 2))
        tot_sq = (x ** 2).sum(dim=(0, 2)) if tot_sq is None else tot + x.sum(dim=(0, 2))
        n += x.shape[0] * x.shape[2]
    mean = (tot / n).float()
    std = torch.sqrt(((tot_sq / n) - mean.double() ** 2).clamp(min=0)).float()
    return mean, std.clamp(min=1e-6)


def scene_grouped_inputs(model, batch, provider, device, rng, scene_dropout):
    """批内唯一场景分组取 pc tokens 输入；shuffled 模式置换场景归属（错配检测）。

    返回 (patches (U,K,S,3), center_norm (U,K,3), scen_eff (B,), aux)；aux =
    (drop_mask (B,K), inv (B,))，none 模式 aux=None。"""
    scen = batch['scenario'].to(device)
    perm = None
    if model.scene_mode == 'shuffled':
        perm = torch.randperm(scen.shape[0], device=device)
    scen_eff = scen[perm] if perm is not None else scen
    if model.scene_mode == 'none':
        return None, None, scen_eff, None
    uniq, inv = torch.unique(scen_eff, return_inverse=True)
    # 唯一行的 key 按 corpus 索引取（scenario id = corpus.scenes 下标）
    keys = [_CORPUS.scenes[int(u)]['key'] for u in uniq.tolist()]
    pts = [provider.get(k, corpus_env(k)) for k in keys]
    patches = torch.as_tensor(np.stack([p['patches'] for p in pts]), device=device)
    center_norm = torch.as_tensor(np.stack([p['center_norm'] for p in pts]),
                                  device=device).float()
    drop = None
    if scene_dropout > 0:
        drop = torch.from_numpy(rng.random_sample((scen.shape[0], model.n_patches))
                                < scene_dropout).to(device)
    return patches, center_norm, scen_eff, (drop, inv)


_ENV_CACHE = {}


def corpus_env(key):
    "场景 env_pc 点（fps 优先）；模块级缓存（每场景一张静态图）。"
    if key in _ENV_CACHE:
        return _ENV_CACHE[key]
    pts = _CORPUS.scene_env_points(_CORPUS.scene_id[key])
    _ENV_CACHE[key] = np.asarray(pts, dtype=np.float32)
    return _ENV_CACHE[key]


_CORPUS = None


def forward_context(model, csi, token_mask, sg, drop_mask=None):
    "[scene; csi] context 前向（pc/onehot/none 分支统一入口）。返回 (B, num_tokens, dim)。"
    patches, center_norm, scen_eff, aux = sg
    if model.scene_mode == 'none':
        return model(csi, token_mask=token_mask)
    drop, inv = aux
    if model.scene_mode == 'pc':
        s = model.patch_net(patches, center_norm)          # (U,K,dim)
        scene_tokens = s[inv]                              # gather 回逐样本
    else:                                                  # onehot
        uniq_ids = torch.unique(scen_eff, return_inverse=True)[0]   # (U,) 场景 id 本身
        s = model.scene(center_norm, uniq_ids)                       # (U,K,dim)
        scene_tokens = s[inv]                                        # gather 回逐样本
    if drop_mask is not None:
        scene_tokens = torch.where(drop_mask.unsqueeze(-1), model.mask_token,
                                   scene_tokens)
    return _forward_with_scene_tokens(model, csi, token_mask, scene_tokens)


def _forward_with_scene_tokens(model, csi, token_mask, scene_tokens):
    "与 dloc forward_scene_grouped 同款：场景 tokens 直通入口（避免重复编码）。"
    return model(csi, token_mask=token_mask, scene_tokens=scene_tokens)


def p1_loss_fn(p1_head, scene_tokens, p1_table, scen):
    "P1：无参数 LN 两侧 + SmoothL1（目标 = 逐场景 (TX,频带) 统计表 (S,1,8)）。"
    pred = p1_head(scene_tokens)                                    # (B, 8)
    tgt = p1_table[scen].reshape(scen.shape[0], -1)                 # (B, 8)
    pred_n = F.layer_norm(pred, (pred.shape[-1],))
    tgt_n = F.layer_norm(tgt, (tgt.shape[-1],))
    return F.smooth_l1_loss(pred_n, tgt_n)


def main():
    global _CORPUS
    parser = argparse.ArgumentParser('point-cloud conditioned CSI-JEPA pretraining')
    parser.add_argument('--config', type=str, default='configs/pc_jepa_base.yaml')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--sionna-root', type=str, default=None)
    parser.add_argument('--rooms10-root', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--limit-train-batches', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.split:
        cfg['data']['split'] = args.split
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
    tb_writer = SummaryWriter(log_dir=save_dir) if SummaryWriter is not None else None

    # ---- 数据
    dc = cfg['data']
    _CORPUS = PCSceneCorpus(rooms10_root=args.rooms10_root or dc.get('rooms10_root'),
                            sionna_root=args.sionna_root or dc.get('sionna_root'),
                            env_cache=dc.get('env_cache'),
                            snr_tier=dc.get('snr_tier', 20))
    train_idx, test_idx, split_note = _CORPUS.split(
        dc['split'], test_scenes=dc.get('test_scenes'))
    # SSL 诊断 val：逐场景 train 块尾部 5%（连续块，不打乱轨迹顺序）
    by_sc = {}
    for si, t in train_idx:
        by_sc.setdefault(si, []).append(t)
    val = []
    for si, ts in by_sc.items():
        ts = sorted(ts)
        k = max(1, int(round(len(ts) * dc.get('val_split', 0.05))))
        val += [(si, t) for t in ts[-k:]]
    train_idx = sorted(set(train_idx) - set(val))
    print(f"split={dc['split']} ({split_note}): SSL corpus {len(train_idx)} "
          f"(val {len(val)} from train blocks; held-out {len(test_idx)}); "
          f"scenes: {[s['key'] for s in _CORPUS.scenes]}")

    n_workers = int(dc.get('num_workers', 4))
    loader = DataLoader(PCDataset(_CORPUS, train_idx, cache_max_gb=dc.get('cache_max_gb', 6)),
                        batch_size=dc['batch_size'], shuffle=True,
                        num_workers=n_workers, collate_fn=pc_collate,
                        pin_memory=bool(dc.get('pin_memory', False)))
    val_loader = DataLoader(PCDataset(_CORPUS, val, cache_max_gb=dc.get('cache_max_gb', 6)),
                            batch_size=dc['batch_size'], shuffle=False,
                            num_workers=n_workers, collate_fn=pc_collate)
    steps_per_epoch = len(loader)
    if args.limit_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)

    # ---- 模型（组合而非改动）
    mcg = cfg['model']
    scene_mode = mcg['scene_mode']
    model = PCSIJEPA(dim=mcg['dim'], depth=mcg['depth'], num_heads=mcg['num_heads'],
                     ffn_dim=mcg['ffn_dim'], dropout=mcg.get('dropout', 0.0),
                     scene_mode='pc' if scene_mode == 'shuffled' else scene_mode,
                     n_scenarios=len(_CORPUS.scenes), n_patches=mcg['n_patches'],
                     geo_dropout=mcg.get('geo_dropout', 0.0)).to(device)
    target_mode = mcg.get('target_mode', 'csi_only')
    ema_view = model.csi_only_view() if target_mode == 'csi_only' else model.full_view()
    ema = CSIEMATargetEncoder(ema_view, base_momentum=cfg['ema']['base_momentum']).to(device)
    pcfg = mcg['predictor']
    predictor = Predictor(dim=mcg['dim'], num_tokens=model.num_tokens, depth=pcfg['depth'],
                          num_heads=pcfg['num_heads'], ffn_dim=pcfg['ffn_dim']).to(device)
    p1_head = P1Head(mcg['dim']).to(device) if scene_mode != 'none' else None
    p2_head = (P2Head(mcg['dim']).to(device)
               if mcg.get('p2', {}).get('weight', 0) > 0 and scene_mode != 'none' else None)
    center_head = (CenterAuxHead(mcg['dim']).to(device)
                   if mcg.get('center_aux', False) and scene_mode == 'pc' else None)
    print(f"scene_mode={scene_mode} target={target_mode}: "
          f"{model.n_scene_tokens} pc + {model.num_tokens - model.n_scene_tokens} csi "
          f"= {model.num_tokens} tokens; model "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, predictor "
          f"{sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M")

    # ---- P1/P2 目标表（逐场景统计；train 场景的前 90% 快照——零泄漏）
    p1_table = None
    if p1_head is not None:
        p1_table = torch.as_tensor(
            np.stack([_CORPUS.scene_p1(si) for si in range(len(_CORPUS.scenes))]),
            device=device)                                     # (S, 1, 8)
    p2_table = None
    if p2_head is not None:
        rows = []
        for s in _CORPUS.scenes:
            with np.load(os.path.join(s['dir'], 'sidecar_pc.npz')) as f:
                rows.append(f['p2_stats'])
            if not os.path.exists(os.path.join(s['dir'], 'sidecar_pc.npz')):
                rows[-1] = _p2_from_cir(s['cir_path'])
        p2_raw = np.stack(rows).astype(np.float32)             # (S, 3)
        p2_mu, p2_sd = p2_raw.mean(axis=0), p2_raw.std(axis=0) + 1e-6
        p2_table = torch.as_tensor((p2_raw - p2_mu) / p2_sd, device=device)

    # ---- pc patch provider（逐场景预分组缓存）
    provider = PatchProvider(n_patches=mcg['n_patches'],
                             patch_size=mcg.get('patch_size', 32))

    # ---- channel_std 统计量（train 语料；装进两分支 tokenizer buffer）
    if model.tokenizer.input_norm_mode == 'channel_std':
        mean, std = compute_channel_stats(
            DataLoader(PCDataset(_CORPUS, train_idx, cache_max_gb=6),
                       batch_size=dc['batch_size'], shuffle=False,
                       num_workers=n_workers, collate_fn=pc_collate), device)
        model.tokenizer.set_input_stats(mean, std)
        ema.encoder.tokenizer.set_input_stats(mean, std)
        print('input_norm=channel_std: train-语料统计已装入两分支')

    # ---- 优化器（ndim<=1 与 mask_token 不衰减——D17 纪律）
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        (no_decay_params if (p.ndim <= 1 or name.endswith('mask_token'))
         else decay_params).append(p)
    aux_modules = [m for m in (predictor, p1_head, p2_head, center_head) if m is not None]
    for m in aux_modules:
        for p in m.parameters():
            (no_decay_params if p.ndim <= 1 else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': cfg['optim']['weight_decay']},
         {'params': no_decay_params, 'weight_decay': 0.0}], lr=cfg['optim']['lr'])
    total_steps = steps_per_epoch * cfg['optim']['epochs']
    warmup_steps = steps_per_epoch * cfg['optim']['warmup_epochs']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, global_step, best_val = 0, 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['online_state'])
        ema.load_state_dict(ckpt['ema_state'])
        predictor.load_state_dict(ckpt['predictor_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        for key, mod in (('p1_state', p1_head), ('p2_state', p2_head),
                         ('center_state', center_head)):
            if mod is not None and ckpt.get(key):
                mod.load_state_dict(ckpt[key])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f'resumed from {args.resume} at epoch {start_epoch}')

    mcfg = cfg['mask']
    token_rate = float(mcfg.get('token_mask_rate', 0.5))
    rect_t = int(mcfg.get('rect_t', 2))                    # (1 AP x 2 频带) 块
    link_prob = float(mcfg.get('link_dropout_prob', 0.3))  # 整行（=整 TX）掩
    scene_dropout = float(mcfg.get('scene_token_dropout', 0.0))
    p1_w = float(cfg.get('p1', {}).get('weight', 0.3))
    p1_ramp = int(cfg.get('p1', {}).get('ramp_epochs', 5))
    p2_w = float(mcg.get('p2', {}).get('weight', 0.0))
    center_w = float(mcg.get('center_aux_weight', 0.0))
    rng = np.random.RandomState(cfg['seed'] + 2)

    csv_path = os.path.join(save_dir, 'train_log.csv')
    if start_epoch == 0 and not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'p1', 'p2', 'center', 'ztgt_std',
                                    'scene_std', 'mask_rate', 'lr', 'momentum',
                                    'val_loss', 'val_expl_var'])

    for epoch in range(start_epoch, cfg['optim']['epochs']):
        model.train()
        predictor.train()
        epoch_loss, epoch_steps = 0.0, 0
        acc = {'p1': 0.0, 'p2': 0.0, 'center': 0.0, 'ztgt': 0.0, 'scene': 0.0,
               'mask': 0.0}
        p1_n = p2_n = 0
        momentum_used = None
        progress = tqdm(loader, total=steps_per_epoch, desc=f'epoch {epoch + 1}')
        for i, batch in enumerate(progress):
            if args.limit_train_batches > 0 and i >= steps_per_epoch:
                break
            csi = batch['csi'].to(device, non_blocking=True)
            bsz = csi.size(0)

            # CSI 掩码：网格 (1,8) 上 (1x2) 块 x round(8*rate/2) + 整行 drop
            grid_mask = torch.from_numpy(build_grid_mask(
                mask_rng, bsz, 1, 8, 1, rect_t, token_rate, link_prob)).to(device)

            sg = scene_grouped_inputs(model, batch, provider, device, rng,
                                      scene_dropout)
            _, _, scen_eff, aux = sg
            drop_mask = aux[0] if aux is not None else None
            z_ctx = forward_context(model, csi, grid_mask, sg, drop_mask)
            with torch.no_grad():
                z_tgt = ema(csi) if target_mode == 'csi_only' else \
                    ema.encoder(csi, _patches_full(batch, provider, device),
                                _centers_full(batch, provider, device),
                                _uniq_ids(batch, device))
            z_hat = predictor(z_ctx)

            n_scene = model.n_scene_tokens
            if target_mode == 'csi_only':
                loss = F.smooth_l1_loss(z_hat[:, n_scene:], z_tgt)
            else:                                              # dual：全 token
                loss = F.smooth_l1_loss(z_hat, z_tgt)

            p1_v = p2_v = c_v = 0.0
            if p1_head is not None and n_scene > 0:
                lam = p1_w * min(1.0, (epoch + 1) / max(1, p1_ramp))
                p1_loss = p1_loss_fn(p1_head, z_hat[:, :n_scene], p1_table, scen_eff)
                loss = loss + lam * p1_loss
                p1_v = float(p1_loss.item())
            if p2_head is not None and n_scene > 0:
                pred2 = p2_head(z_hat[:, :n_scene])
                p2_loss = F.smooth_l1_loss(pred2, p2_table[scen_eff])
                loss = loss + p2_w * p2_loss
                p2_v = float(p2_loss.item())
                p2_n += 1
            if center_head is not None and drop_mask is not None and drop_mask.any():
                pred_c = center_head(z_hat[:, :n_scene][drop_mask])
                tgt_c = _centers_for_scene(batch, provider, device)[drop_mask]
                c_loss = F.mse_loss(pred_c, tgt_c)
                loss = loss + center_w * c_loss
                c_v = float(c_loss.item())
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping')
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_frac = global_step / max(1, total_steps)
            momentum_used = ema.update(ema_view, progress=progress_frac)
            global_step += 1
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_step', loss.item(), global_step)

            epoch_loss += loss.item()
            epoch_steps += 1
            acc['p1'] += p1_v
            acc['p2'] += p2_v
            acc['center'] += c_v
            acc['ztgt'] += z_tgt.std(dim=0).mean().item()
            if n_scene > 0:
                acc['scene'] += z_ctx[:, :n_scene].std(dim=0).mean().item()
            acc['mask'] += grid_mask.float().mean().item()
            progress.set_postfix(loss=f'{loss.item():.4f}',
                                 ztgt_std=f"{acc['ztgt'] / epoch_steps:.4f}")

        n = max(1, epoch_steps)
        val_loss, val_ev = _ssl_val(model, ema, predictor, val_loader, device,
                                    provider, target_mode) if val_loader else (None, None)
        print(f"Epoch:{epoch + 1}, loss:{epoch_loss / n:.6f}, "
              f"p1:{acc['p1'] / n:.4f}, p2:{acc['p2'] / max(1, p2_n):.4f}, "
              f"center:{acc['center'] / n:.4f}, ztgt_std:{acc['ztgt'] / n:.4f}, "
              f"scene_std:{acc['scene'] / n:.4f}, mask:{acc['mask'] / n:.4f}"
              + (f', val:{val_loss:.6f} ev:{val_ev:.4f}' if val_loss is not None else ''))
        if tb_writer is not None:
            for k, v in (('Loss/train_epoch', epoch_loss / n), ('Loss/p1', acc['p1'] / n),
                         ('Loss/p2', acc['p2'] / max(1, p2_n)),
                         ('Metrics/ztgt_std', acc['ztgt'] / n),
                         ('Metrics/scene_std', acc['scene'] / n),
                         ('Metrics/token_mask_rate', acc['mask'] / n),
                         ('Optim/lr', scheduler.get_last_lr()[0]),
                         ('Optim/ema_momentum', momentum_used or 0.0)):
                tb_writer.add_scalar(k, v, epoch + 1)
            if val_loss is not None:
                tb_writer.add_scalar('Val/loss', val_loss, epoch + 1)
                tb_writer.add_scalar('Val/explained_var', val_ev, epoch + 1)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch + 1, f'{epoch_loss / n:.6f}', f"{acc['p1'] / n:.6f}",
                 f"{acc['p2'] / max(1, p2_n):.6f}", f"{acc['center'] / n:.6f}",
                 f"{acc['ztgt'] / n:.4f}", f"{acc['scene'] / n:.4f}",
                 f"{acc['mask'] / n:.4f}", scheduler.get_last_lr()[0], momentum_used]
                + ([f'{val_loss:.6f}', f'{val_ev:.4f}'] if val_loss is not None
                   else ['', '']))

        ckpt = {'online_state': model.state_dict(), 'ema_state': ema.state_dict(),
                'predictor_state': predictor.state_dict(),
                'p1_state': p1_head.state_dict() if p1_head is not None else None,
                'p2_state': p2_head.state_dict() if p2_head is not None else None,
                'center_state': center_head.state_dict() if center_head is not None else None,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(), 'epoch': epoch,
                'global_step': global_step, 'best_val_loss': best_val, 'config': cfg}
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            ckpt['best_val_loss'] = best_val
            torch.save(ckpt, os.path.join(save_dir, 'best_val.pth'))
            print(f'  [best] val {best_val:.6f} -> best_val.pth（诊断 only — D34）')
        if (epoch + 1) % cfg['log']['save_every_epochs'] == 0:
            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth'))

    if tb_writer is not None:
        tb_writer.close()
    print('pc csi pretraining done.')


def _patches_full(batch, provider, device):
    "dual 目标用：批内每样本的 pc patches（无分组——目标侧一次性成本可接受）。"
    keys = batch['scene_key']
    ps = [provider.get(k, corpus_env(k))['patches'] for k in keys]
    return torch.as_tensor(np.stack(ps), device=device)


def _centers_full(batch, provider, device):
    keys = batch['scene_key']
    cs = [provider.get(k, corpus_env(k))['center_norm'] for k in keys]
    return torch.as_tensor(np.stack(cs), device=device).float()


def _uniq_ids(batch, device):
    return torch.unique(batch['scenario'].to(device), return_inverse=True)[1]


def _centers_for_scene(batch, provider, device):
    keys = batch['scene_key']
    cs = [provider.get(k, corpus_env(k))['center_norm'] for k in keys]
    return torch.as_tensor(np.stack(cs), device=device).float()


def _p2_from_cir(cir_path):
    "rooms10 无 sidecar 时现算 P2 统计（前 90% 快照）。"
    with np.load(cir_path) as f:
        valid = f['valid'][:int(round(f['valid'].shape[0] * 0.9))]
        tau = f['absolute_delay_s'][:valid.shape[0]]
        coef = f['coefficients'][:valid.shape[0]]
    n_paths = valid.sum(axis=1).astype(np.float32)
    first = np.array([tau[i][valid[i]].min() * 1e9 if valid[i].any() else 0.0
                      for i in range(len(valid))], dtype=np.float32)
    gain = np.array([10.0 * np.log10(max(float((np.abs(coef[i][valid[i]]) ** 2).sum()),
                                         1e-30)) if valid[i].any() else -300.0
                     for i in range(len(valid))], dtype=np.float32)
    return np.array([n_paths.mean(), float(np.median(first)), gain.mean()],
                    dtype=np.float32)


@torch.no_grad()
def _ssl_val(model, ema, predictor, val_loader, device, provider, target_mode):
    "确定性 SSL 诊断：完整输入、无掩码。返回 (loss, expl_var)。"
    model.eval()
    tot_loss, tot_var, tot_n = 0.0, 0.0, 0
    n_scene = model.n_scene_tokens
    for batch in val_loader:
        csi = batch['csi'].to(device)
        sg = scene_grouped_inputs(model, batch, provider, device,
                                  np.random.RandomState(0), 0.0)
        z_ctx = forward_context(model, csi, None, sg)
        if target_mode == 'csi_only':
            z_tgt = ema(csi)
            loss = F.smooth_l1_loss(predictor(z_ctx)[:, n_scene:], z_tgt,
                                    reduction='none').mean()
        else:
            z_tgt = ema.encoder(csi, _patches_full(batch, provider, device),
                                _centers_full(batch, provider, device),
                                _uniq_ids(batch, device))
            loss = F.smooth_l1_loss(predictor(z_ctx), z_tgt, reduction='none').mean()
        tot_loss += loss.item() * csi.size(0)
        tot_var += z_tgt.var().item() * csi.size(0)
        tot_n += csi.size(0)
    avg = tot_loss / max(1, tot_n)
    var = tot_var / max(1, tot_n)
    return avg, 1.0 - 2.0 * avg / max(1e-8, var)


if __name__ == '__main__':
    main()
