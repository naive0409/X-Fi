"""DLoc floorplan 条件化 JEPA 预训练（Phase B-v2 试点，PROJECT_LOG §11）。

jepa_csi_pretrain.py 的复制减法版（项目惯例）。与 A1 的结构差异：
  - CSI = DLoc (32, 234) re/im 快照（3-AP 补零），tokenizer 网格 (4 AP, 9 频带) = 36 tokens
  - 场景 tokens 拼接进上下文（[scene; csi]，场景 tokens 保留到输出），场景分支只经
    context 注意力获得梯度（EMA 目标 = CSI-only）
  - 批内 ≤8 个唯一场景：场景编码按唯一场景分组前向再 gather（512 份重复栅格不进显存）
  - 补零 AP 行不进掩码、不进损失
  - P1 物理辅助：池化场景 tokens 预测逐 (场景, AP, 频带) 平均 |CSI|（train 块统计，无参数
    LN 两侧，λ=0.3 ramp 5；LOSO 下 test 场景不在表内 -> P1-val 即迁移诊断）
  - scene_mode: raster（主）/ shuffled（batch 内 floorplan 错配——查表检测）/ onehot（B2
    对照，id 替换栅格）/ none（CSI-only，DLoc 版 A1）；onehot/none 由 model 分支处理，
    shuffled 在本脚本的数据编排层实现（把样本的场景归属按 batch 置换后走 raster 通路）

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from csi_jepa.dloc_data import DLocCorpus, DLocDataset, N_BANDS, N_AP_SLOTS, dloc_collate
from csi_jepa.floorplan import DLocRasterBuilder, FPCSIJEPA, P1Head
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
        tot_sq = (x ** 2).sum(dim=(0, 2)) if tot_sq is None else tot_sq + (x ** 2).sum(dim=(0, 2))
        n += x.shape[0] * x.shape[2]
    mean = (tot / n).float()
    std = torch.sqrt(((tot_sq / n) - mean.double() ** 2).clamp(min=0)).float()
    return mean, std.clamp(min=1e-6)


@torch.no_grad()
def ssl_val_evaluate(model, ema, predictor, val_loader, device, p1_head, p1_table,
                     raster_gpu, cents_gpu, valid_gpu):
    "确定性 SSL 诊断：完整输入、无掩码、场景 tokens 直通；返回 (loss, expl_var, p1_loss)。"
    model.eval()
    tot_loss, tot_var, tot_p1, tot_n = 0.0, 0.0, 0.0, 0
    n_scene = model.n_scene_tokens
    for batch in val_loader:
        csi = batch['csi'].to(device)
        scen = batch['scenario'].to(device)
        z_ctx, _ = forward_scene_grouped(model, csi, None, scen, raster_gpu, cents_gpu,
                                         valid_gpu)
        z_tgt = ema(csi)
        z_hat = predictor(z_ctx)
        n_csi = model.num_tokens - n_scene
        ctx_csi = z_ctx[:, n_scene:]
        valid_tok = token_validity(batch['valid_ap'].to(device), model)
        elem_mask = valid_tok.unsqueeze(-1).float()
        b = csi.size(0)
        loss = F.smooth_l1_loss(ctx_csi, z_tgt, reduction='none')
        tot_loss += (loss * elem_mask).sum().item() \
            / (elem_mask.sum().clamp(min=1).item() * ctx_csi.shape[-1]) * b
        tot_var += z_tgt.var().item() * b
        if p1_head is not None and n_scene > 0:
            lam_loss = p1_loss_fn(p1_head, z_ctx[:, :n_scene], scen, p1_table)
            tot_p1 += lam_loss.item() * b
        tot_n += b
    avg = tot_loss / max(1, tot_n)
    var = tot_var / max(1, tot_n)
    return avg, 1.0 - 2.0 * avg / max(1e-8, var), tot_p1 / max(1, tot_n)


def forward_scene_grouped(model, csi, token_mask, scen, raster_gpu, cents_gpu, valid_gpu,
                          perm=None, drop_mask=None):
    """批内唯一场景分组前向：场景编码只算 unique 个，再 gather 回逐样本。
    perm 非 None 时先把样本的场景归属按 perm 置换（shuffled 对照：floorplan 错配）；
    drop_mask 非 None 时把对应样本的场景 tokens 换成 mask_token（scene_token_dropout）。"""
    scen_eff = scen[perm] if perm is not None else scen
    uniq, inv = torch.unique(scen_eff, return_inverse=True)
    if model.scene is not None:
        sel_r = raster_gpu[uniq]
        sel_c = cents_gpu[uniq]
        sel_v = valid_gpu[uniq]
        if model.scene_mode == 'onehot':
            s = model.scene(sel_r, sel_c, sel_v, uniq)
        else:
            s = model.scene(sel_r, sel_c, sel_v)
        scene_tokens = s[inv]
        if drop_mask is not None:
            scene_tokens = torch.where(drop_mask.view(-1, 1, 1), model.mask_token,
                                       scene_tokens)
    else:
        scene_tokens = None
    z_ctx = model(csi, token_mask=token_mask, scene_tokens=scene_tokens)
    return z_ctx, scen_eff


def token_validity(valid_ap, model):
    "(B,4) 有效 AP -> (B, n_csi_tokens) token 有效旗标（补零 AP 行不进损失/掩码）。"
    b = valid_ap.size(0)
    grid_c = model.tokenizer.grid_c
    tok = valid_ap[:, :grid_c]                          # 前 grid_c 个 AP 槽位
    return tok.unsqueeze(-1).expand(b, grid_c, model.tokenizer.grid_t).reshape(
        b, grid_c * model.tokenizer.grid_t)


def p1_loss_fn(p1_head, scene_tokens, scen, p1_table):
    "P1：无参数 LN 两侧 + valid 掩码的 SmoothL1。目标 = 逐场景 (AP,频带) 统计表 (8,4,9)。"
    pred = p1_head(scene_tokens)                                    # (B, 36)
    tgt = p1_table[scen]                                            # (B, 4, 9)
    b = scen.size(0)
    pred_n = F.layer_norm(pred, (pred.shape[-1],))
    tgt_n = F.layer_norm(tgt.reshape(b, -1), (tgt.shape[-1] * tgt.shape[-2],))
    loss = F.smooth_l1_loss(pred_n, tgt_n, reduction='none').reshape(b, N_AP_SLOTS, N_BANDS)
    return loss.sum() / max(1, b * N_AP_SLOTS * N_BANDS)


def main():
    parser = argparse.ArgumentParser('dloc floorplan-conditioned JEPA pretraining')
    parser.add_argument('--config', type=str, default='configs/dloc_jepa_fp.yaml')
    parser.add_argument('--split', type=str, default=None,
                        help='override data.split: interp|rand|loso|building')
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

    mcg = cfg['model']
    split_mode = cfg['data']['split']
    corpus = DLocCorpus(cfg['data']['cache_root'])
    train_idx, test_idx, split_note = corpus.split(split_mode)
    train_idx, val_idx = corpus.val_indices(train_idx, cfg['data'].get('val_split', 0.05))
    train_scenarios = corpus.train_scenarios(split_mode)
    print(f'split={split_mode} ({split_note}): SSL corpus {len(train_idx)} '
          f'(val {len(val_idx)} from train part; held-out test {len(test_idx)} — '
          f'test-split CSI 本样本序不进预训练：LOSO/building 场景级隔离，interp/rand 块隔离)')
    print(f'train scenarios: {train_scenarios}')

    builder = DLocRasterBuilder(
        sorted({corpus.global_index[g][0] for g in train_idx + test_idx}),
        canvas=mcg['canvas'])
    grid_hw = builder.grid_hw()
    print(f'raster: canvas={mcg["canvas"]} cell 0.2m patch {mcg.get("raster_patch", 20)} '
          f'-> grid {grid_hw[0]}x{grid_hw[1]}')

    train_ds = DLocDataset(corpus, train_idx)
    n_workers = int(cfg['data'].get('num_workers', 4))
    loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True,
                        num_workers=n_workers, collate_fn=dloc_collate,
                        pin_memory=bool(cfg['data'].get('pin_memory', False)),
                        prefetch_factor=(int(cfg['data'].get('prefetch_factor', 2))
                                         if n_workers > 0 else None),
                        persistent_workers=n_workers > 0)
    val_loader = DataLoader(DLocDataset(corpus, val_idx), batch_size=cfg['data']['batch_size'],
                            shuffle=False, num_workers=n_workers, collate_fn=dloc_collate,
                            persistent_workers=n_workers > 0)
    steps_per_epoch = len(loader)
    if args.limit_train_batches > 0:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)

    # ---- 模型（组合而非改动：A1 文件零改动）
    scene_mode = mcg['scene_mode']
    model = FPCSIJEPA(
        dim=mcg['dim'], depth=mcg['depth'], num_heads=mcg['num_heads'],
        ffn_dim=mcg['ffn_dim'], dropout=mcg.get('dropout', 0.0),
        scene_mode='raster' if scene_mode == 'shuffled' else scene_mode,
        grid_hw=grid_hw, raster_patch=mcg.get('raster_patch', 20), n_scenarios=8,
        csi_repr=mcg.get('csi_repr', 'reim'),
    ).to(device)
    ema = CSIEMATargetEncoder(model.csi_only_view(),
                              base_momentum=cfg['ema']['base_momentum']).to(device)
    pcfg = mcg['predictor']
    predictor = Predictor(dim=mcg['dim'], num_tokens=model.num_tokens, depth=pcfg['depth'],
                          num_heads=pcfg['num_heads'], ffn_dim=pcfg['ffn_dim'],
                          dropout=pcfg.get('dropout', 0.0)).to(device)
    p1_head = P1Head(mcg['dim']).to(device) if mcg['scene_mode'] != 'none' else None
    print(f'scene_mode={scene_mode}: {model.n_scene_tokens} scene + '
          f'{model.num_tokens - model.n_scene_tokens} csi = {model.num_tokens} tokens; '
          f'model {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, '
          f'predictor {sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M')

    # ---- P1 目标表（train 场景块统计；LOSO 下 test 场景不在表内 -> P1-val=迁移诊断）
    p1_table = None
    if p1_head is not None:
        p1_table = torch.from_numpy(corpus.p1_target_table(train_scenarios)).to(device)

    # ---- 栅格常驻 GPU（8 张 × ~0.8MB）；场景编码按唯一场景分组
    scen_all = sorted({corpus.global_index[g][0] for g in train_idx + test_idx})
    raster_np = np.stack([builder.build(sc,
                                        corpus_side(builder, corpus, sc, 'ap_centroids'),
                                        corpus_side(builder, corpus, sc, 'valid_ap'))
                          for sc in scen_all])
    raster_gpu = torch.from_numpy(raster_np).to(device)
    cents_np = np.stack([corpus_side(builder, corpus, sc, 'ap_centroids') for sc in scen_all])
    valid_np = np.stack([corpus_side(builder, corpus, sc, 'valid_ap') for sc in scen_all])
    cents_gpu = torch.from_numpy(cents_np).float().to(device)
    valid_gpu = torch.from_numpy(valid_np).to(device)
    scen_to_row = {s: i for i, s in enumerate(scen_all)}

    if model.tokenizer.input_norm_mode == 'channel_std':
        mean, std = compute_channel_stats(
            DataLoader(DLocDataset(corpus, train_idx), batch_size=cfg['data']['batch_size'],
                       shuffle=False, num_workers=n_workers, collate_fn=dloc_collate), device)
        model.tokenizer.set_input_stats(mean, std)
        ema.encoder.tokenizer.set_input_stats(mean, std)
        print('input_norm=channel_std: train-语料统计已装入两分支')

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        (no_decay_params if (p.ndim <= 1 or name.endswith('mask_token')) else decay_params)\
            .append(p)
    for p in list(predictor.parameters()) + list(p1_head.parameters() if p1_head else []):
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': cfg['optim']['weight_decay']},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=cfg['optim']['lr'])
    total_steps = steps_per_epoch * cfg['optim']['epochs']
    warmup_steps = steps_per_epoch * cfg['optim']['warmup_epochs']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, global_step, best_val = 0, 0, float('inf')
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['online_state'])
        ema.load_state_dict(ckpt['ema_state'])
        predictor.load_state_dict(ckpt['predictor_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if p1_head is not None and ckpt.get('p1_state'):
            p1_head.load_state_dict(ckpt['p1_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    mcfg = cfg['mask']
    rect_c = max(1, min(model.tokenizer.grid_c, int(mcfg.get('rect_c', 1))))
    rect_t = max(1, min(model.tokenizer.grid_t, int(mcfg.get('rect_t', 2))))
    token_rate = float(mcfg.get('token_mask_rate', 0.5))
    link_prob = float(mcfg.get('link_dropout_prob', 0.3))
    scene_drop = float(mcfg.get('scene_token_dropout', 0.0))
    p1_w = float(cfg.get('p1', {}).get('weight', 0.3))
    p1_ramp = int(cfg.get('p1', {}).get('ramp_epochs', 5))

    csv_path = os.path.join(save_dir, 'train_log.csv')
    if not os.path.exists(csv_path) and start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'p1_loss', 'ztgt_std', 'scene_std',
                                    'token_mask_rate', 'lr', 'momentum', 'val_loss',
                                    'val_expl_var', 'val_p1'])

    scene_rng = np.random.RandomState(cfg['seed'] + 2)
    for epoch in range(start_epoch, cfg['optim']['epochs']):
        model.train()
        predictor.train()
        epoch_loss, epoch_steps = 0.0, 0
        ztgt_std_sum, scene_std_sum, mask_rate_sum, p1_sum = 0.0, 0.0, 0.0, 0.0
        p1_n = 0
        momentum_used = None
        progress = tqdm(loader, total=steps_per_epoch, desc=f'epoch {epoch + 1}')
        for i, batch in enumerate(progress):
            if args.limit_train_batches > 0 and i >= args.limit_train_batches:
                break
            csi = batch['csi'].to(device, non_blocking=True)
            scen = batch['scenario'].to(device, non_blocking=True)
            valid_ap = batch['valid_ap'].to(device, non_blocking=True)
            bsz = csi.size(0)

            valid_tok = token_validity(valid_ap, model)             # (B, n_csi)
            grid_mask = torch.from_numpy(build_grid_mask(
                mask_rng, bsz, model.tokenizer.grid_c, model.tokenizer.grid_t,
                rect_c, rect_t, token_rate, link_prob)).to(device)
            token_mask = grid_mask & valid_tok                       # 补零行不掩

            perm = None
            if scene_mode == 'shuffled':
                perm = torch.from_numpy(scene_rng.permutation(bsz)).to(device)
            drop = None
            if scene_drop > 0:
                drop = torch.from_numpy(
                    (scene_rng.random_sample(bsz) < scene_drop)).to(device)

            z_ctx, scen_eff = forward_scene_grouped(
                model, csi, token_mask, scen, raster_gpu, cents_gpu, valid_gpu, perm=perm,
                drop_mask=drop)
            with torch.no_grad():
                z_tgt = ema(csi)                                     # CSI-only 目标
            z_hat = predictor(z_ctx)

            n_scene = model.n_scene_tokens
            ctx_csi = z_hat[:, n_scene:]
            elem_mask = valid_tok.unsqueeze(-1).float()
            loss = (F.smooth_l1_loss(ctx_csi, z_tgt, reduction='none') * elem_mask).sum() \
                / (elem_mask.sum().clamp(min=1) * ctx_csi.shape[-1])   # 元素数 = token数 × dim

            p1_loss_val = None
            if p1_head is not None and n_scene > 0:
                lam = p1_w * min(1.0, (epoch + 1) / max(1, p1_ramp))
                p1_loss = p1_loss_fn(p1_head, z_ctx[:, :n_scene], scen_eff, p1_table)
                loss = loss + lam * p1_loss
                p1_loss_val = p1_loss.item()
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping')
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_frac = global_step / max(1, total_steps)
            momentum_used = ema.update(model.csi_only_view(), progress=progress_frac)
            global_step += 1
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/train_step', loss.item(), global_step)

            epoch_loss += loss.item()
            epoch_steps += 1
            ztgt_std_sum += z_tgt.std(dim=0).mean().item()
            if n_scene > 0:
                scene_std_sum += z_ctx[:, :n_scene].std(dim=0).mean().item()  # 场景塌缩监视
            mask_rate_sum += token_mask.float().mean().item()
            if p1_loss_val is not None:
                p1_sum += p1_loss_val
                p1_n += 1
            progress.set_postfix(loss=f'{loss.item():.4f}',
                                 ztgt_std=f'{ztgt_std_sum / epoch_steps:.4f}')

        n = max(1, epoch_steps)
        val_loss, val_ev, val_p1 = ssl_val_evaluate(
            model, ema, predictor, val_loader, device, p1_head, p1_table, raster_gpu,
            cents_gpu, valid_gpu) if val_loader is not None else (None, None, None)
        print(f'Epoch:{epoch + 1}, loss:{epoch_loss / n:.6f}, '
              f'p1:{p1_sum / max(1, p1_n):.4f}, ztgt_std:{ztgt_std_sum / n:.4f}, '
              f'scene_std:{scene_std_sum / n:.4f}, mask:{mask_rate_sum / n:.4f}'
              + (f', val:{val_loss:.6f} ev:{val_ev:.4f} val_p1:{val_p1:.4f}'
                 if val_loss is not None else ''))
        if tb_writer is not None:
            for k, v in (('Loss/train_epoch', epoch_loss / n), ('Loss/p1_train', p1_sum / max(1, p1_n)),
                         ('Metrics/ztgt_std', ztgt_std_sum / n),
                         ('Metrics/scene_std', scene_std_sum / n),
                         ('Metrics/token_mask_rate', mask_rate_sum / n),
                         ('Optim/lr', scheduler.get_last_lr()[0]),
                         ('Optim/ema_momentum', momentum_used or 0.0)):
                tb_writer.add_scalar(k, v, epoch + 1)
            if val_loss is not None:
                tb_writer.add_scalar('Val/loss', val_loss, epoch + 1)
                tb_writer.add_scalar('Val/explained_var', val_ev, epoch + 1)
                tb_writer.add_scalar('Val/p1', val_p1, epoch + 1)

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{epoch_loss / n:.6f}',
                                    f'{p1_sum / max(1, p1_n):.6f}',
                                    f'{ztgt_std_sum / n:.4f}', f'{scene_std_sum / n:.4f}',
                                    f'{mask_rate_sum / n:.4f}', scheduler.get_last_lr()[0],
                                    momentum_used]
                                   + ([f'{val_loss:.6f}', f'{val_ev:.4f}', f'{val_p1:.6f}']
                                      if val_loss is not None else ['', '', '']))

        ckpt = {'online_state': model.state_dict(), 'ema_state': ema.state_dict(),
                'predictor_state': predictor.state_dict(),
                'p1_state': p1_head.state_dict() if p1_head is not None else None,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(), 'epoch': epoch,
                'global_step': global_step, 'best_val_loss': best_val, 'config': cfg}
        torch.save(ckpt, os.path.join(save_dir, 'last.pth'))
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            ckpt['best_val_loss'] = best_val
            ckpt['val_metrics'] = {'val_loss': val_loss, 'val_expl_var': val_ev,
                                   'val_p1': val_p1}
            torch.save(ckpt, os.path.join(save_dir, 'best_val.pth'))
            print(f'  [best] val {best_val:.6f} -> best_val.pth (诊断 only — D34)')
        if (epoch + 1) % cfg['log']['save_every_epochs'] == 0:
            torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch + 1:03d}.pth'))

    if tb_writer is not None:
        tb_writer.close()
    print('dloc fp pretraining done.')


def corpus_side(builder, corpus, scenario, key):
    "取场景 sidecar 字段（ap_centroids/valid_ap）。"
    import numpy as np
    with np.load(os.path.join(corpus.cache_root, f'sidecar_{scenario}.npz')) as f:
        return {k: f[k] for k in f.files}[key]


if __name__ == '__main__':
    main()
