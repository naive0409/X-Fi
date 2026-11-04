"""DLoc floorplan 条件化基座的下游定位评估（Phase B-v2，PROJECT_LOG §11）。

jepa_csi_downstream.py 的复制减法版 + 任务替换（HAR 分类 -> XY 定位回归）：
  - 头：池化 mean（全部 168 tokens，含场景 tokens——保留到输出是我们的设计差异）+ LN +
    Linear(dim, 2)；SmoothL1(beta=1m) 原始米制
  - 指标：mean/median 欧氏误差 + p75/p90 + %<1m（逐场景表）
  - 方法：fp_probe（冻结 encoder + 新头）/ fp_finetune（mirror 配方）/ fp_scratch（随机
    初始化同配方 = 下界对照）
  - scene_mode（--scene-mode 或 ckpt 内嵌 config）：raster 主线；onehot（B2 对照——id 替换
    栅格，须配对应预训练 run）；none（CSI-only）；shuffled 评估期 = 用错误 floorplan（严格
    查表检测：预训 raster 主线 + 评估期错配，或错配预训 + 正确评估）
  - 划分与预训练一致（--split）；标签效率 --label-fraction 复用嵌套分层抽样（场景键）

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from csi_jepa.dloc_data import (SCENARIOS, DLocCorpus, DLocDataset, dloc_collate)
from csi_jepa.floorplan import DLocRasterBuilder, FPCSIJEPA, P1Head  # noqa: F401 (P1 评测诊断)
from jepa_modules import PreLNBlock  # noqa: F401

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class LocalHead(torch.nn.Module):
    "镜像 classification_Head 协议（池化 mean -> LN -> Linear），输出 (x, y) 米。"

    def __init__(self, dim=256, num_classes=2):
        super(LocalHead, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(dim, num_classes)

    def forward(self, tokens):
        x = torch.mean(tokens, dim=1)
        return self.fc(self.norm(x))


def build_encoder(model_cfg, device, grid_hw):
    return FPCSIJEPA(
        dim=int(model_cfg['dim']), depth=int(model_cfg['depth']),
        num_heads=int(model_cfg['num_heads']), ffn_dim=int(model_cfg['ffn_dim']),
        dropout=float(model_cfg.get('dropout', 0.0)),
        scene_mode=model_cfg.get('scene_mode', 'raster'), grid_hw=grid_hw,
        raster_patch=int(model_cfg.get('raster_patch', 20)), n_scenarios=8,
        csi_repr=model_cfg.get('csi_repr', 'reim'),
    ).to(device)


def euc_metrics(pred_xy, label_xy):
    "pred/label (N,2) 米 -> 误差指标（米）。"
    err = np.linalg.norm(pred_xy - label_xy, axis=1)
    return {'mean_err_m': float(err.mean()), 'median_err_m': float(np.median(err)),
            'p75_err_m': float(np.percentile(err, 75)),
            'p90_err_m': float(np.percentile(err, 90)),
            'pct_within_1m': float((err < 1.0).mean())}


def scene_tokens_for(encoder, raster_gpu, cents_gpu, valid_gpu, scen_eff):
    "唯一场景分组的场景 tokens；onehot 模式需传 scenario_id（与预训练 forward_scene_grouped 同分支）。"
    uniq, inv = torch.unique(scen_eff, return_inverse=True)
    if encoder.scene is None:
        return None
    if encoder.scene_mode == 'onehot':
        return encoder.scene(raster_gpu[uniq], cents_gpu[uniq], valid_gpu[uniq], uniq)[inv]
    return encoder.scene(raster_gpu[uniq], cents_gpu[uniq], valid_gpu[uniq])[inv]


@torch.no_grad()
def evaluate(model, head, loader, device, raster_gpu, cents_gpu, valid_gpu,
             scene_eval_perm=False):
    "返回 (pred_xy (N,2), label_xy (N,2))；scene_eval_perm=True 时按 batch 错配 floorplan。"
    model.eval()
    head.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc='eval'):
        csi = batch['csi'].to(device)
        scen = batch['scenario'].to(device)
        perm = None
        if scene_eval_perm:
            perm = torch.randperm(csi.size(0), device=device)
        scen_eff = scen[perm] if perm is not None else scen
        z_ctx = model(csi, scene_tokens=scene_tokens_for(model, raster_gpu, cents_gpu,
                                                         valid_gpu, scen_eff))
        preds.append(head(z_ctx).cpu().numpy())
        labels.append(batch['label_xy'].numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser('dloc fp downstream localization evaluation')
    parser.add_argument('--method', required=True,
                        choices=['fp_probe', 'fp_finetune', 'fp_scratch'])
    parser.add_argument('--config', type=str, default='configs/dloc_eval_fp_mirror.yaml')
    parser.add_argument('--split', type=str, default=None, help='override data.split')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='fp 预训练 checkpoint；fp_probe/fp_finetune 必填')
    parser.add_argument('--scene-mode', type=str, default=None,
                        choices=['raster', 'onehot', 'shuffled', 'none'],
                        help='覆盖 ckpt 内嵌 scene_mode（评估期错配检测用）')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--limit-train-batches', type=int, default=0)
    parser.add_argument('--label-fraction', type=float, default=1.0,
                        help='少量标签：逐场景嵌套分层抽样（同 seed 各方法同样本）')
    parser.add_argument('--adapt-fraction', type=float, default=0.0,
                        help='LOSO/楼间适配：把 held-out 场景文件序前 X%% 样本并入训练（连续空间'
                             '条带，与评估区不重叠）；0=零样本 probe（默认）')
    parser.add_argument('--save-json', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
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
    run_dir = os.path.join('dloc_eval_results', run_name)
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
                               'scene_mode', 'canvas', 'raster_patch', 'csi_repr')
                              if k in emb})
            print(f"architecture from checkpoint ({args.pretrained}, epoch {ckpt.get('epoch')})")
    if args.scene_mode is not None and args.method in ('fp_probe', 'fp_finetune'):
        model_cfg['scene_mode'] = 'raster' if args.scene_mode == 'shuffled' else args.scene_mode
        print(f'scene_mode overridden to {args.scene_mode}')
    scene_mode = model_cfg.get('scene_mode', 'raster')
    # shuffled 评估期错配：主线 raster 模型 + 评估时 batch 内错配 floorplan
    eval_scene_perm = (args.scene_mode == 'shuffled')

    split_mode = cfg['data']['split']
    corpus = DLocCorpus(cfg['data']['cache_root'])
    train_idx, test_idx, split_note = corpus.split(split_mode)

    # ---- 目标场景适配（--adapt-fraction）：held-out 场景文件序前 X% 并入训练
    # （连续空间条带；与评估区不重叠——游走顺序保证空间不相交）
    adapt = float(args.adapt_fraction)
    n_adapted = 0
    if adapt > 0:
        test_scens = sorted({corpus.global_index[g][0] for g in test_idx})
        newly = []
        for sc in test_scens:
            offs = sorted(g for g in test_idx if corpus.global_index[g][0] == sc)
            k = max(1, int(round(len(offs) * adapt)))
            newly += offs[:k]
        train_idx = sorted(train_idx + newly)
        test_idx = sorted(set(test_idx) - set(newly))
        n_adapted = len(newly)
        print(f'adapt {adapt:.0%}: {n_adapted} held-out 场景样本并入训练 '
              f'（文件序前缀条带）')
    print(f'split={split_mode} ({split_note}): train {len(train_idx)} / test {len(test_idx)}')

    builder = DLocRasterBuilder(sorted({corpus.global_index[g][0] for g in train_idx + test_idx}),
                                canvas=model_cfg.get('canvas', 'global'))
    grid_hw = builder.grid_hw()
    encoder = build_encoder(model_cfg, device, grid_hw)
    if ckpt is not None:
        encoder.load_state_dict(ckpt['online_state'], strict=True)
        print('loaded pretrained encoder weights (strict=True)')
    else:
        print('random-init encoder (scratch control)')

    scen_all = sorted({corpus.global_index[g][0] for g in train_idx + test_idx})
    side = {sc: dict(np.load(os.path.join(cfg['data']['cache_root'],
                                          f'sidecar_{sc}.npz'))) for sc in scen_all}
    raster_gpu = torch.from_numpy(np.stack([
        builder.build(sc, side[sc]['ap_centroids'], side[sc]['valid_ap'])
        for sc in scen_all])).float().to(device)
    cents_gpu = torch.from_numpy(np.stack([side[sc]['ap_centroids']
                                           for sc in scen_all])).float().to(device)
    valid_gpu = torch.from_numpy(np.stack([side[sc]['valid_ap']
                                           for sc in scen_all])).to(device)
    scen_row = {s: i for i, s in enumerate(scen_all)}

    # ---- 少量标签：逐场景嵌套分层（场景内随机，seed 固定，10%⊃5%⊃1%）
    label_fraction = float(args.label_fraction)
    if label_fraction < 1.0:
        rng = np.random.RandomState(cfg['seed'] + 11)
        by_sc = {}
        for g in train_idx:
            by_sc.setdefault(corpus.global_index[g][0], []).append(g)
        labeled = []
        for sc, idxs in by_sc.items():
            idxs = np.asarray(idxs)
            rng.shuffle(idxs)
            k = max(1, int(round(len(idxs) * label_fraction)))
            labeled += idxs[:k].tolist()
        train_idx = sorted(labeled)
        print(f'label fraction {label_fraction}: {len(train_idx)} labeled samples')

    n_workers = int(cfg['data'].get('num_workers', 4))
    train_loader = DataLoader(DLocDataset(corpus, train_idx),
                              batch_size=cfg['data']['train_batch_size'], shuffle=True,
                              num_workers=n_workers, collate_fn=dloc_collate,
                              persistent_workers=n_workers > 0)
    test_loader = DataLoader(DLocDataset(corpus, test_idx),
                             batch_size=cfg['data']['test_batch_size'], shuffle=False,
                             num_workers=n_workers, collate_fn=dloc_collate,
                             persistent_workers=n_workers > 0)

    # ---- 训练头（probe）或头+encoder（finetune/scratch）
    head = LocalHead(int(model_cfg['dim'])).to(device)
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
    total_steps = len(train_loader) * stage['epochs']
    warmup_steps = len(train_loader) * stage['warmup_epochs']
    if stage.get('schedule', 'cosine') == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
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
            scen = batch['scenario'].to(device)
            labels_xy = batch['label_xy'].to(device)
            z_ctx = encoder(csi, scene_tokens=scene_tokens_for(
                encoder, raster_gpu, cents_gpu, valid_gpu, scen))
            loss = F.smooth_l1_loss(head(z_ctx), labels_xy, beta=1.0)
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

    # ---- 测试集评估：headline（mirror 无 val -> last）；逐场景表 + 总表
    encoder.load_state_dict(torch.load(os.path.join(run_dir, 'last.pth'),
                                       map_location=device)['encoder_state'])
    head.load_state_dict(torch.load(os.path.join(run_dir, 'last.pth'),
                                    map_location=device)['head_state'])
    preds, labels = evaluate(encoder, head, test_loader, device, raster_gpu, cents_gpu,
                             valid_gpu, scene_eval_perm=eval_scene_perm)
    metrics = euc_metrics(preds, labels)
    per_scene = {}
    for sc in SCENARIOS:
        idx = [i for i, g in enumerate(test_idx)
               if corpus.global_index[g][0] == sc]
        if not idx:
            continue
        per_scene[sc] = euc_metrics(preds[idx], labels[idx])
        print(f"  {sc:22s} mean {per_scene[sc]['mean_err_m']:.2f}m  "
              f"median {per_scene[sc]['median_err_m']:.2f}m  "
              f"%<1m {per_scene[sc]['pct_within_1m']:.1%}")
    print(f'\n==== results ({args.method}, scene_mode={args.scene_mode or scene_mode}) ====')
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
                   'scene_mode': args.scene_mode or scene_mode,
                   'eval_scene_perm': eval_scene_perm,
                   'pretrained': args.pretrained, 'split': split_mode,
                   'adapt_fraction': adapt, 'n_adapted': n_adapted,
                   'label_fraction': label_fraction, 'n_train': len(train_idx),
                   'n_test': len(test_idx), 'metrics': metrics,
                   'per_scene': per_scene}, f, indent=2)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
