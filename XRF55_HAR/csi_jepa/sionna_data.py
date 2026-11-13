"""Sionna 合成数据集的数据层（xfi 训练侧消费；缓存由 sionna_gen 管线产出）。

与 DLocCorpus/DLocDataset 的关系：缓存格式**逐位同构**（CSI_<sc>.npy (N,32,234)
float32 + sidecar_<sc>.npz），故 Dataset/collate 直接复用 dloc_data 的实现；差异只在
语料组织——场景数是动态的（数百~2600 个合成场景），划分按场景而不是按场景名写死：
  interp          逐场景 UE 随机 90/10（合成 UE 位置 iid，无 DLoc 游走泄漏问题；
                  seed 固定与 rand 同口径）
  holdout_scenes  前 90% 场景全量训练、后 10% 场景全量测试（跨场景零样本轴的
                  主口径——sionna 上的 "building holdout" 类比，测试含数百场景）
floorplan 张量：sionna_gen/make_tensors.py 已从 PNG 派生 occ/unk/sdf npz；本模块的
SionnaRasterProvider 惰性加载并现算 ch4 AP 高斯（DLocRasterBuilder.build 的同款数学），
对训练脚本呈现与 DLocRasterBuilder 相同的接口（grid_hw/build/ap_window_origin）。
"""
import os
import re

import numpy as np

from .dloc_data import DLocDataset, dloc_collate, N_AP_SLOTS  # noqa: F401 (复用导出)

CELL_M = 0.2
PATCH_M = 4.0
GAUSS_SIGMA_M = 0.5


class SionnaCorpus:
    """动态场景数语料（sionna_gen 产物）。接口与 DLocCorpus 对齐：
    split/train_scenarios/val_indices/p1_target_table/global_index/index_scenario。"""

    def __init__(self, cache_root):
        self.cache_root = cache_root
        scen = sorted(f[4:-4] for f in os.listdir(cache_root)
                      if re.fullmatch(r'CSI_scene_\d+\.npy', f))
        if not scen:
            raise FileNotFoundError(f'{cache_root} 下无 CSI_scene_*.npy（先跑 '
                                    f'sionna_gen.build_cache）')
        self.scenarios = scen
        self.scen_samples, self.scen_offset, total = [], [], 0
        for sc in scen:
            with np.load(os.path.join(cache_root, f'sidecar_{sc}.npz')) as f:
                n = f['labels'].shape[0]
            self.scen_samples.append(n)
            self.scen_offset.append(total)
            total += n
        self.total = total
        self.global_index = [(sc, i) for sc, n in zip(scen, self.scen_samples)
                             for i in range(n)]
        self.index_scenario = [sc for sc, n in zip(scen, self.scen_samples)
                               for i in range(n)]

    def split(self, mode):
        "返回 (train_idx, test_idx, split_note)。"
        rng = np.random.RandomState(3407 + 13)
        tr, te = [], []
        if mode == 'interp':
            for sc, a, b in self._scene_bounds():
                perm = rng.permutation(len(a) + len(b))
                off = self.scen_offset[self.scenarios.index(sc)]
                tr += [off + i for i in perm[:len(a)]]
                te += [off + i for i in perm[len(a):]]
            note = 'per-scene random 90/10 (iid UE positions)'
        elif mode == 'holdout_scenes':
            k = int(round(len(self.scenarios) * 0.9))
            for j, sc in enumerate(self.scenarios):
                off = self.scen_offset[j]
                n = self.scen_samples[j]
                if j < k:
                    tr += [off + i for i in range(n)]
                else:
                    te += [off + i for i in range(n)]
            note = (f'train=first {k} scenes (all UE), '
                    f'test=last {len(self.scenarios) - k} scenes (zero-shot new scenes)')
        else:
            raise ValueError(f'SionnaCorpus 不支持划分 {mode}（可选 interp/holdout_scenes）')
        return sorted(tr), sorted(te), note

    def _scene_bounds(self, train_frac=0.9):
        out = []
        for j, sc in enumerate(self.scenarios):
            n = self.scen_samples[j]
            n_train = int(round(n * train_frac))
            out.append((sc, list(range(n_train)), list(range(n_train, n))))
        return out

    def train_scenarios(self, mode):
        if mode == 'holdout_scenes':
            k = int(round(len(self.scenarios) * 0.9))
            return self.scenarios[:k]
        return list(self.scenarios)

    def val_indices(self, train_idx, val_frac=0.05):
        "SSL 诊断 val：逐场景 train 尾部 5%（iid 位置，任意子集皆无泄漏）。"
        by_sc = {}
        for g in train_idx:
            by_sc.setdefault(self.global_index[g][0], []).append(g)
        val = []
        for idxs in by_sc.values():
            k = max(1, int(round(len(idxs) * val_frac)))
            val += idxs[-k:]
        return sorted(set(train_idx) - set(val)), sorted(val)

    def p1_target_table(self, train_scenarios):
        "(n_scenes, 4, 9)：train 场景行 = sidecar p1_stats；非 train 行 = train 聚合。"
        per_sc = np.zeros((len(self.scenarios), N_AP_SLOTS, 9), dtype=np.float32)
        num = np.zeros((N_AP_SLOTS, 9), dtype=np.float64)
        den = np.zeros(N_AP_SLOTS, dtype=np.float64)
        train_set = set(train_scenarios)
        for j, sc in enumerate(self.scenarios):
            with np.load(os.path.join(self.cache_root, f'sidecar_{sc}.npz')) as f:
                p1, valid = f['p1_stats'], f['valid_ap']
            if sc in train_set:
                per_sc[j] = p1
                w = self.scen_samples[j]
                num += p1 * valid[:, None] * w
                den += valid * w
        for j, sc in enumerate(self.scenarios):
            if sc not in train_set:
                per_sc[j] = (num / np.maximum(den, 1)[:, None]).astype(np.float32)
        return per_sc


class SionnaRasterProvider:
    """PNG 派生张量的惰性加载器 + AP 高斯现算（接口对齐 DLocRasterBuilder）。
    2600 场景全量常驻约 1.6GB，故 LRU 限量缓存；命中失败的逐场景解压加载。"""

    def __init__(self, tensors_root, scenarios, cache_scenes=256):
        self.root = tensors_root
        self.cache_scenes = cache_scenes
        self._tables = {}
        self._preload_shapes(scenarios)

    def _preload_shapes(self, scenarios):
        self.H, self.W = 160, 320
        one = self._table(scenarios[0])
        assert one['occ'].shape == (self.H, self.W), one['occ'].shape

    def _table(self, sc):
        if sc not in self._tables:
            if len(self._tables) >= self.cache_scenes:   # 简单 FIFO 淘汰
                self._tables.pop(next(iter(self._tables)))
            with np.load(os.path.join(self.root, f'{sc}.npz')) as f:
                self._tables[sc] = {k: f[k] for k in
                                    ('occ', 'unk', 'sdf', 'ap_centroids', 'valid_ap',
                                     'world_origin')}
        return self._tables[sc]

    def grid_hw(self):
        return self.H // int(PATCH_M / CELL_M), self.W // int(PATCH_M / CELL_M)  # (8,16)

    def build(self, scenario, ap_centroids=None, valid_ap=None):
        "(4,160,320) float32：occ/unk/sdf + ch4 AP 高斯（σ=0.5m，DLocRasterBuilder 同款）。"
        t = self._table(scenario)
        H, W = self.H, self.W
        out = np.stack([t['occ'], t['unk'], t['sdf']])       # (3,H,W)
        ox, oy = t['world_origin']
        y_max = oy + self.H * CELL_M                          # 画布顶 = origin_y + 29m
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.zeros((H, W), dtype=np.float32)
        cents = t['ap_centroids'] if ap_centroids is None else ap_centroids
        valid = t['valid_ap'] if valid_ap is None else valid_ap
        for a in range(len(valid)):
            if not valid[a]:
                continue
            col = (cents[a][0] - ox) / CELL_M
            row = (y_max - cents[a][1]) / CELL_M
            gg = np.exp(-((xx - col) ** 2 + (yy - row) ** 2)
                        / (2 * (GAUSS_SIGMA_M / CELL_M) ** 2))
            g = np.clip(g + gg, 0, 1)
        return np.concatenate([out, g[None]], axis=0)

    def ap_window_origin(self, scenario):
        return tuple(self._table(scenario)['world_origin'])


def corpus_factory(cfg):
    "训练脚本按 data.corpus 选语料（dloc|sionna）——接入 diff 见 SIONNA_PIPELINE.md §9。"
    if cfg.get('corpus', 'dloc') == 'sionna':
        return SionnaCorpus(cfg['cache_root'])
    from .dloc_data import DLocCorpus
    return DLocCorpus(cfg['cache_root'])
