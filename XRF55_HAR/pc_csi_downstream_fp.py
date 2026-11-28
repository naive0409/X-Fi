"""pc_csi_downstream_fp.py — 点云条件化基座的下游定位评估（Phase PC）。

pc_csi_downstream_fp.py 母本 = dloc_csi_downstream_fp.py（复制减法 + 场景模态替换）：
  - 任务：UE 轨迹 XY 定位回归（头 = 池化 mean + LN + Linear(dim,2)，SmoothL1 β=1m）
  - 方法：fp_probe（冻结 encoder + 新头）/ fp_finetune（mirror 配方）/ fp_scratch
  - 头结构三挡 --head-arch pool|split|film（D51 诊断遗留；pool 为主口径）
  - --scene-mode shuffled（评估期 floorplan/点云错配 = 严格查表检测）
  - 划分 interp（逐场景轨迹连续块 90/10，主口径）/ loso（--test-scenes 整场景留出）
  - 指标：mean/median/p75/p90 欧氏误差 + %<1m + 逐场景表

Run from XRF55_HAR/.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from csi_jepa.pc_data import PCSceneCorpus, PCDataset, pc_collate, stratified_subset_by_scene
from csi_jepa.pointcloud import PCSIJEPA, PatchProvider

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class LocalHead(torch.nn.Module):
    "镜像 classification_Head 协议（池化 mean -> LN -> Linear），输出 (x, y) 米。"

    def __init__(self, dim=256, num_classes=2):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(dim, num_classes)

    def forward(self, tokens, n_scene=0):
        return self.fc(self.norm(torch.mean(tokens, dim=1)))


class SplitPoolHead(torch.nn.Module):
    "诊断头（D51）：pc/csi tokens 分池化各 LN 后拼接 -> Linear；n_scene=0 退化。"

    def __init__(self, dim=256, num_classes=2):
        super().__init__()
        self.norm_c = torch.nn.LayerNorm(dim)
        self.norm_s = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(2 * dim, num_classes)

    def forward(self, tokens, n_scene=0):
        c = self.norm_c(torch.mean(tokens[:, n_scene:], dim=1))
        if n_scene > 0:
            s = self.norm_s(torch.mean(tokens[:, :n_scene], dim=1))
            x = torch.cat([c, s], dim=-1)
        else:
            x = torch.cat([c, torch.zeros_like(c)], dim=-1)
        return self.fc(x)


class FiLMHead(torch.nn.Module):
    "诊断头（D51）：pc 池化向量 FiLM 调制 csi 池化特征 -> Linear；n_scene=0 退化。"

    def __init__(self, dim=256, num_classes=2):
        super().__init__()
        self.norm_c = torch.nn.LayerNorm(dim)
        self.norm_s = torch.nn.LayerNorm(dim)
        self.film = torch.nn.Linear(dim, 2 * dim)
        self.fc = torch.nn.Linear(dim, num_classes)

    def forward(self, tokens, n_scene=0):
        c = self.norm_c(torch.mean(tokens[:, n_scene:], dim=1))
        if n_scene > 0:
            gamma, beta = self.film(
                self.norm_s(torch.mean(tokens[:, :n_scene], dim=1))).chunk(2, dim=-1)
            c = c * (1.0 + gamma) + beta
        return self.fc(c)


HEADS = {'pool': LocalHead, 'split': SplitPoolHead, 'film': FiLMHead}


def euc_metrics(pred_xy, label_xy):
    "pred/label (N,2) 米 -> 欧氏误差指标族。"
    err = np.linalg.norm(pred_xy - label_xy, axis=1)
    return {'mean_err_m': float(err.mean()), 'median_err_m': float(np.median(err)),
            'p75_err_m': float(np.percentile(err, 75)),
            'p90_err_m': float(np.percentile(err, 90)),
            'pct_within_1m': float((err < 1.0).mean())}


_ENV_CACHE = {}


def corpus_env(corpus, key):
    if key in _ENV_CACHE:
        return _ENV_CACHE[key]
    pts = corpus.scene_env_points(corpus.scene_id[key])
    _ENV_CACHE[key] = np.asarray(pts, dtype=np.float32)
    return _ENV_CACHE[key]


def scene_tokens_for(encoder, provider, corpus, batch, device, perm=None):
    "唯一场景分组的 pc tokens（评估期 perm 非 None = 错配 floorplan/点云）。"
    scen = batch['scenario'].to(device)
    scen_eff = scen[perm] if perm is not None else scen
    if encoder.scene_mode == 'none':
        return None
    uniq, inv = torch.unique(scen_eff, return_inverse=True)
    # 唯一行的 key 直接按 corpus 索引取（scenario id = corpus.scenes 下标；
    # 不可用 batch['scene_key'][u]——那是按批内位置索引，错位）
    keys = [corpus.scenes[int(u)]['key'] for u in uniq.tolist()]
    pts = [provider.get(k, corpus_env(corpus, k)) for k in keys]
    patches = torch.as_tensor(np.stack([p['patches'] for p in pts]), device=device)
    center_norm = torch.as_tensor(np.stack([p['center_norm'] for p in pts]),
                                  device=device).float()
    if encoder.scene_mode == 'onehot':
        s = encoder.scene(center_norm, uniq)
    else:
        s = encoder.patch_net(patches, center_norm)
    return s[inv]


@torch.no_grad()
def evaluate(encoder, head, loader, device, provider, corpus, eval_perm=False):
    "返回 (pred_xy (N,2), label_xy (N,2), scene_keys)。"
    encoder.eval()
    head.eval()
    preds, labels, keys = [], [], []
    for batch in tqdm(loader, desc='eval'):
        csi = batch['csi'].to(device)
        perm = None
        if eval_perm:
            perm = torch.randperm(csi.size(0), device=device)
        s_tokens = scene_tokens_for(encoder, provider, corpus, batch, device, perm)
        z = encoder(csi, scene_tokens=s_tokens)
        preds.append(head(z, n_scene=encoder.n_scene_tokens).cpu().numpy())
        labels.append(batch['label_xy'].numpy())
        keys += batch['scene_key']
    return np.concatenate(preds), np.concatenate(labels), keys


def main():
    parser = argparse.ArgumentParser('point-cloud conditioned downstream localization')
    parser.add_argument('--method', required=True,
                        choices=['fp_probe', 'fp_finetune', 'fp_scratch'])
    parser.add_argument('--config', type=str, default='configs/pc_eval_fp_mirror.yaml')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--test-scenes', type=str, nargs='*', default=None,
                        help='loso 留出场景 key（如 room3 / scene_5）')
    parser.add_argument('--rooms10-root', type=str, default=None)
    parser.add_argument('--sionna-root', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--scene-mode', type=str, default=None,
                        choices=['pc', 'onehot', 'shuffled', 'none'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--label-fraction', type=float, default=1.0)
    parser.add_argument('--limit-train-batches', type=int, default=0)
    parser.add_argument('--head-arch', type=str, default='pool', choices=list(HEADS))
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--save-json', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.split:
        cfg['data']['split'] = args.split
    if args.epochs is not None:
        cfg['linear_probe']['epochs'] = args.epochs
        cfg['finetune']['epochs'] = args.epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Available downstream resources: {device}; method = {args.method}')
    torch.manual_seed(cfg['seed'])

    run_name = (args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')).replace(os.sep, '_')
    run_dir = os.path.join(cfg['log']['save_dir'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    tb_writer = SummaryWriter(log_dir=run_dir) if SummaryWriter is not None else None
    print(f'run name: {run_name}\nrun dir:   {run_dir}')

    # ---- 架构：checkpoint 内嵌 config 优先（自描述）；scratch 用 eval config
    model_cfg = dict(cfg['model'])
    ckpt = None
    if args.method in ('fp_probe', 'fp_finetune'):
        if args.pretrained is None:
            raise SystemExit(f'--pretrained is required for {args.method}')
        ckpt = torch.load(args.pretrained, map_location=device)
        emb = ckpt.get('config', {}).get('model', {})
        if emb:
            model_cfg.update({k: emb[k] for k in
                              ('dim', 'depth', 'num_heads', 'ffn_dim', 'dropout',
                               'scene_mode', 'n_patches', 'geo_dropout') if k in emb})
            print(f"architecture from checkpoint ({args.pretrained}, epoch {ckpt.get('epoch')})")
    if args.scene_mode is not None and args.method in ('fp_probe', 'fp_finetune'):
        model_cfg['scene_mode'] = 'pc' if args.scene_mode == 'shuffled' else args.scene_mode
        print(f'scene_mode overridden to {args.scene_mode}')
    eval_scene_perm = (args.scene_mode == 'shuffled')

    dc = cfg['data']
    corpus = PCSceneCorpus(rooms10_root=args.rooms10_root or dc.get('rooms10_root'),
                           sionna_root=args.sionna_root or dc.get('sionna_root'),
                           env_cache=dc.get('env_cache'), snr_tier=dc.get('snr_tier', 20))
    train_idx, test_idx, split_note = corpus.split(
        dc['split'], test_scenes=args.test_scenes or dc.get('test_scenes'))
    print(f"split={dc['split']} ({split_note}): train {len(train_idx)} / test {len(test_idx)}")

    grid_hw = None
    encoder = PCSIJEPA(dim=int(model_cfg['dim']), depth=int(model_cfg['depth']),
                       num_heads=int(model_cfg['num_heads']),
                       ffn_dim=int(model_cfg['ffn_dim']),
                       dropout=float(model_cfg.get('dropout', 0.0)),
                       scene_mode=model_cfg.get('scene_mode', 'pc'),
                       n_scenarios=len(corpus.scenes),
                       n_patches=int(model_cfg.get('n_patches', 64)),
                       geo_dropout=float(model_cfg.get('geo_dropout', 0.0))).to(device)
    if ckpt is not None:
        encoder.load_state_dict(ckpt['online_state'], strict=True)
        print('loaded pretrained encoder weights (strict=True)')
    else:
        print('random-init encoder (scratch control)')
    provider = PatchProvider(n_patches=int(model_cfg.get('n_patches', 64)),
                             patch_size=int(model_cfg.get('patch_size', 32)))

    if args.label_fraction < 1.0:
        train_idx = stratified_subset_by_scene(train_idx, args.label_fraction,
                                               seed=cfg['seed'] + 11)
        print(f'label fraction {args.label_fraction}: {len(train_idx)} labeled samples')

    n_workers = int(dc.get('num_workers', 4))
    train_loader = DataLoader(PCDataset(corpus, train_idx,
                                        cache_max_gb=dc.get('cache_max_gb', 6)),
                              batch_size=dc['train_batch_size'], shuffle=True,
                              num_workers=n_workers, collate_fn=pc_collate)
    test_loader = DataLoader(PCDataset(corpus, test_idx,
                                       cache_max_gb=dc.get('cache_max_gb', 6)),
                             batch_size=dc['test_batch_size'], shuffle=False,
                             num_workers=n_workers, collate_fn=pc_collate)

    head = HEADS[args.head_arch](int(model_cfg['dim'])).to(device)
    print(f'head arch: {args.head_arch}')
    torch.manual_seed(cfg['seed'] + 100)
    finetuning = args.method in ('fp_finetune', 'fp_scratch')
    stage = cfg['finetune'] if finetuning else cfg['linear_probe']
    for p in encoder.parameters():
        p.requires_grad_(False)
    if finetuning:
        for p in encoder.parameters():
            p.requires_grad_(True)
    if finetuning:
        optimizer = torch.optim.AdamW(
            [{'params': head.parameters(), 'lr': stage['head_lr']},
             {'params': [p for p in encoder.parameters() if p.requires_grad],
              'lr': stage['backbone_lr']}],
            weight_decay=stage['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=stage['lr'],
                                      weight_decay=stage['weight_decay'])
    if stage.get('schedule', 'cosine') == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    else:
        total_steps = len(train_loader) * stage['epochs']
        warmup_steps = len(train_loader) * stage['warmup_epochs']

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(stage['epochs']):
        encoder.train() if finetuning else encoder.eval()
        head.train()
        epoch_loss, n_batches = 0.0, 0
        for i, batch in enumerate(tqdm(train_loader, desc=f'epoch {epoch + 1}')):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            csi = batch['csi'].to(device)
            s_tokens = scene_tokens_for(encoder, provider, corpus, batch, device)
            z = encoder(csi, scene_tokens=s_tokens)
            loss = F.smooth_l1_loss(head(z, n_scene=encoder.n_scene_tokens),
                                    batch['label_xy'].to(device), beta=1.0)
            if not torch.isfinite(loss):
                print(f'[warn] non-finite loss at epoch {epoch + 1} step {i}; skipping')
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f'Epoch:{epoch + 1}, train loss:{epoch_loss / max(1, n_batches):.4f}')
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/train', epoch_loss / max(1, n_batches), epoch + 1)

    torch.save({'method': args.method, 'encoder_state': encoder.state_dict(),
                'head_state': head.state_dict(), 'epoch': stage['epochs']},
               os.path.join(run_dir, 'last.pth'))

    # ---- 测试集评估：headline（mirror 无 val -> last）+ 逐场景表
    saved = torch.load(os.path.join(run_dir, 'last.pth'), map_location=device)
    encoder.load_state_dict(saved['encoder_state'])
    head.load_state_dict(saved['head_state'])
    preds, labels, keys = evaluate(encoder, head, test_loader, device, provider,
                                   corpus, eval_perm=eval_scene_perm)
    metrics = euc_metrics(preds, labels)
    per_scene = {}
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        per_scene[key] = euc_metrics(preds[idx], labels[idx])
        print(f"  {key:12s} mean {per_scene[key]['mean_err_m']:.2f}m  "
              f"median {per_scene[key]['median_err_m']:.2f}m  "
              f"%<1m {per_scene[key]['pct_within_1m']:.1%}")
    print(f'\n==== results ({args.method}, scene_mode={args.scene_mode or model_cfg.get("scene_mode")}) ====')
    print(f'  mean {metrics["mean_err_m"]:.2f}m | median {metrics["median_err_m"]:.2f}m | '
          f'p75 {metrics["p75_err_m"]:.2f}m | p90 {metrics["p90_err_m"]:.2f}m | '
          f'%<1m {metrics["pct_within_1m"]:.1%}')
    if tb_writer is not None:
        tb_writer.add_scalar('Metrics/mean_err_m', metrics['mean_err_m'], 0)
        tb_writer.add_scalar('Metrics/median_err_m', metrics['median_err_m'], 0)
        tb_writer.add_scalar('Metrics/pct_within_1m', metrics['pct_within_1m'], 0)
        tb_writer.close()

    out_path = args.save_json or os.path.join(run_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({'run_name': run_name, 'method': args.method,
                   'scene_mode': args.scene_mode or model_cfg.get('scene_mode'),
                   'eval_scene_perm': eval_scene_perm, 'head_arch': args.head_arch,
                   'pretrained': args.pretrained, 'split': dc['split'],
                   'test_scenes': args.test_scenes or dc.get('test_scenes'),
                   'label_fraction': args.label_fraction, 'n_train': len(train_idx),
                   'n_test': len(test_idx), 'metrics': metrics,
                   'per_scene': per_scene}, f, indent=2)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
