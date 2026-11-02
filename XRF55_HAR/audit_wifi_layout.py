"""只读审计：确认 XRF55 WiFi CSI 的通道分解与数值特性（Phase A 前置，PROJECT_LOG §9）。

背景：csi_jepa tokenizer 需要在 (270, 1000) 上定 patch 尺寸。工作假设（论文 §3.1 +
初步相关性检查）：270 = 9 链路 × 30 子载波（3 接收笔记本 × 3 天线），1000 = 1 秒
200 包重采样；存储为纯幅度（无相位）。本脚本跨场景抽样验证，不写任何文件、不动数据。

检查项：
  1. 形状/类型（期望 (270, 1000) float）
  2. 幅度-only 判定（全样本 min >= 0）
  3. 通道块相关性：30 通道块内 vs 跨 30 块（同 90 超块） vs 跨 90 超块
     （期望 块内 >> 块间、超块内 > 超块间 → 支持 9×30 嵌套于 3×90）
  4. 时间轴 lag-1 / lag-100 自相关（支持 100 采样时间片切分）
  5. 90 个 (30×100) patch 的均值/标准差分布（token 信息量 sanity check）
"""
import glob
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'XRF55_Dataset_split')  # <repo>/data/XRF55_Dataset_split

N_PER_SCENE = 2  # train 每场景抽几个文件；test 每场景 1 个


def list_files(split):
    files = []
    for scene in sorted(os.listdir(os.path.join(ROOT, split, 'WiFi'))):
        pattern = os.path.join(ROOT, split, 'WiFi', scene, scene, '*.npy')
        sub = sorted(glob.glob(pattern))
        files += sub[:N_PER_SCENE if split == 'train_data' else 1]
    return files


def block_stats(corr, block, superblock=None):
    """返回 (块内平均|corr|, 跨块但同超块, 跨超块)（superblock 为 None 时只返回块内/跨块）。"""
    n = corr.shape[0]
    idx = np.arange(n)
    same = np.abs(corr[(idx[:, None] // block) == (idx[None, :] // block)])
    cross = np.abs(corr[(idx[:, None] // block) != (idx[None, :] // block)])
    if superblock is None:
        return same[same < 1.0].mean(), cross.mean(), None
    in_super = (idx[:, None] // block != idx[None, :] // block) & \
               ((idx[:, None] // superblock) == (idx[None, :] // superblock))
    out_super = (idx[:, None] // superblock) != (idx[None, :] // superblock)
    return same[same < 1.0].mean(), np.abs(corr[in_super]).mean(), np.abs(corr[out_super]).mean()


def audit_file(path):
    x = np.load(path)
    report = {'shape': x.shape, 'dtype': str(x.dtype), 'min': float(x.min()), 'max': float(x.max())}
    c, t = x.shape

    # 3) 通道相关性（样本 = 时间步）
    cc = np.corrcoef(x)  # (c, c)
    if c % 90 == 0 and c % 30 == 0:
        b30_same, b30_cross, b90_cross = block_stats(cc, 30, 90)
        report['corr_within30'] = float(b30_same)
        report['corr_across30_same90'] = float(b30_cross)
        report['corr_across90'] = float(b90_cross)
    elif c % 30 == 0:
        b30_same, b30_cross, _ = block_stats(cc, 30)
        report['corr_within30'] = float(b30_same)
        report['corr_across30'] = float(b30_cross)

    # 4) 时间自相关（样本 = 通道）
    tx = np.corrcoef(x) if False else None
    xt = x - x.mean(axis=1, keepdims=True)
    denom = (xt ** 2).sum(axis=1)
    lag1 = (xt[:, 1:] * xt[:, :-1]).sum(axis=1) / denom
    lag100 = (xt[:, 100:] * xt[:, :-100]).sum(axis=1) / denom
    report['time_lag1'] = float(lag1.mean())
    report['time_lag100'] = float(lag100.mean())

    # 5) patch 统计：(9, 30, 1000) → 9×10 个 (30, 100) patch
    if c % 30 == 0 and t % 100 == 0:
        g = x.reshape(c // 30, 30, t // 100, 100)  # (links, subcarriers, tsteps, samples)
        patches = g.transpose(0, 2, 1, 3).reshape(-1, 30 * 100)
        report['patch_mean_range'] = (float(patches.mean(axis=1).min()), float(patches.mean(axis=1).max()))
        report['patch_std_range'] = (float(patches.std(axis=1).min()), float(patches.std(axis=1).max()))
        report['patch_std_of_means'] = float(patches.mean(axis=1).std())
    return report


def main():
    for split in ('train_data', 'test_data'):
        files = list_files(split)
        print(f'\n=== {split}: {len(files)} 个抽样文件 ===')
        agg = {}
        for f in files:
            r = audit_file(f)
            tag = os.path.relpath(f, ROOT)
            if r['shape'] != (270, 1000):
                print(f'[形状异常] {tag}: {r["shape"]}')
                continue
            for k, v in r.items():
                if k in ('shape', 'dtype'):
                    continue
                agg.setdefault(k, []).append(v)
            print(f'{tag}: min={r["min"]:.3f} max={r["max"]:.3f} '
                  f'within30={r.get("corr_within30", float("nan")):.3f} '
                  f'cross30@90={r.get("corr_across30_same90", float("nan")):.3f} '
                  f'cross90={r.get("corr_across90", float("nan")):.3f} '
                  f'tlag1={r["time_lag1"]:.3f} tlag100={r["time_lag100"]:.3f} '
                  f'patchμ∈[{r["patch_mean_range"][0]:.2f},{r["patch_mean_range"][1]:.2f}] '
                  f'patchσ∈[{r["patch_std_range"][0]:.2f},{r["patch_std_range"][1]:.2f}]')
        print('--- 汇总（跨文件均值） ---')
        for k, v in sorted(agg.items()):
            flat = [x for item in v for x in (item if isinstance(item, tuple) else (item,))]
            print(f'  {k}: {np.mean(flat):.4f}')


if __name__ == '__main__':
    main()
