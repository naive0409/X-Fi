"""DLoc (WILD) 数据层：memmap Dataset / collate / 划分（Phase B-v2，PROJECT_LOG §11）。

数据事实（2026-11-02 审计）：8 场景 108,008 样本；每样本 (32, 234) float32
（4 AP 组 × 8 re/im 交错行 × 234 子载波；3-AP 场景第 4 组补零）；labels (N,2) XY 米；
文件顺序 = 1cm 步进空间游走、时间戳乱序 → **划分必须用文件序块**（随机划分会把 1cm
相邻样本跨放进 split，虚高）。缓存由 build_dloc_cache.py 生成。
"""
import os

import numpy as np
import torch

SCENARIOS = ['jacobs_July28', 'jacobs_July28_2', 'jacobs_Aug16_1', 'jacobs_Aug16_3',
             'jacobs_Aug16_4_ref', 'July16', 'July18', 'July22_2_ref']
SCENARIO_ID = {s: i for i, s in enumerate(SCENARIOS)}
BUILDING = {s: ('jacobs' if s.startswith('jacobs') else 'atkinson') for s in SCENARIOS}
TEST_SCENARIO_LOSO = 'jacobs_Aug16_3'   # S-LOSO 的 held-out 场景
N_AP_SLOTS, N_SUBCARRIERS, BAND_W = 4, 234, 26
N_BANDS = N_SUBCARRIERS // BAND_W


class DLocCorpus:
    """8 场景缓存的全局索引与划分。划分约定（PROJECT_LOG §11）：
      interp   逐场景文件序连续块 90/10（主口径——随机划分会把 1cm 相邻样本跨 split，虚高）
      rand     逐场景随机 90/10（文献可比，同 seed 与 interp 构成对照诊断）
      loso     训 7 场景，测试 jacobs_Aug16_3 全部样本（跨条件：零样本 probe + 少标签适配）
      building 训一栋楼测另一栋（jacobs↔atkinson，含 4AP→3AP 切换）
    """

    def __init__(self, cache_root):
        self.cache_root = cache_root
        self.scen_samples, self.scen_offset = [], []
        total = 0
        for sc in SCENARIOS:
            with np.load(os.path.join(cache_root, f'sidecar_{sc}.npz')) as f:
                n = f['labels'].shape[0]
            self.scen_samples.append(n)
            self.scen_offset.append(total)
            total += n
        self.total = total
        # 全局样本号 -> (场景名, 场景内样本号)；场景 id 串（场景键分层/诊断用）
        self.global_index = [(sc, i) for sc, n in zip(SCENARIOS, self.scen_samples)
                             for i in range(n)]
        self.index_scenario = [sc for sc, n in zip(SCENARIOS, self.scen_samples)
                               for i in range(n)]

    def _bounds(self, scenario, train_frac=0.9):
        n = self.scen_samples[SCENARIO_ID[scenario]]
        n_train = int(round(n * train_frac))
        return list(range(n_train)), list(range(n_train, n))

    def split(self, mode):
        "返回 (train_idx, test_idx, split_note)——全局样本号列表。"
        rng = np.random.RandomState(3407 + 13)
        tr, te = [], []
        if mode in ('interp', 'rand'):
            for sc in SCENARIOS:
                a, b = self._bounds(sc)
                if mode == 'rand':
                    perm = rng.permutation(len(a) + len(b))
                    a, b = list(perm[:len(a)]), list(perm[len(a):])
                off = self.scen_offset[SCENARIO_ID[sc]]
                tr += [off + i for i in a]
                te += [off + i for i in b]
            note = {'interp': 'per-scenario contiguous block 90/10',
                    'rand': 'per-scenario random 90/10'}[mode]
        elif mode == 'loso':
            for sc in SCENARIOS:
                off = self.scen_offset[SCENARIO_ID[sc]]
                if sc == TEST_SCENARIO_LOSO:
                    te += [off + i for i in range(self.scen_samples[SCENARIO_ID[sc]])]
                else:
                    a, _ = self._bounds(sc)
                    tr += [off + i for i in a]
            note = f'leave-one-scenario-out: test={TEST_SCENARIO_LOSO} (all samples)'
        elif mode == 'building':
            for sc in SCENARIOS:
                a, _ = self._bounds(sc)
                off = self.scen_offset[SCENARIO_ID[sc]]
                (tr if BUILDING[sc] == 'jacobs' else te).extend(off + i for i in a)
            note = 'train=jacobs(5 scenarios, 90% blocks) test=atkinson(3 scenarios)'
        else:
            raise ValueError(mode)
        return sorted(tr), sorted(te), note

    def train_scenarios(self, mode):
        "该划分下参与训练的场景（P1 统计聚合与 SSL 语料范围用）。"
        if mode == 'loso':
            return [s for s in SCENARIOS if s != TEST_SCENARIO_LOSO]
        if mode == 'building':
            return [s for s in SCENARIOS if BUILDING[s] == 'jacobs']
        return list(SCENARIOS)

    def val_indices(self, train_idx, val_frac=0.05):
        "SSL 诊断 val：逐场景取 train 块尾部 5%（连续块，不打乱游走顺序）。"
        by_sc = {}
        for g in train_idx:
            by_sc.setdefault(self.global_index[g][0], []).append(g)
        val = []
        for idxs in by_sc.values():
            k = max(1, int(round(len(idxs) * val_frac)))
            val += idxs[-k:]                                    # 块尾
        val = sorted(val)
        return sorted(set(train_idx) - set(val)), val

    def p1_target_table(self, train_scenarios):
        """P1 目标表 (8, 4, 9)：train 场景行 = 各自块统计；非 train 行（如 LOSO 的 test
        场景）= train 场景聚合（训练中不会被索引到；下游用其真值算 P1 迁移诊断——
        几何模型应接近真值、查表模型只能给出聚合先验）。"""
        per_sc = np.zeros((len(SCENARIOS), N_AP_SLOTS, N_BANDS), dtype=np.float32)
        num = np.zeros((N_AP_SLOTS, N_BANDS), dtype=np.float64)
        den = np.zeros(N_AP_SLOTS, dtype=np.float64)
        for sc in SCENARIOS:
            with np.load(os.path.join(self.cache_root, f'sidecar_{sc}.npz')) as f:
                p1, valid = f['p1_stats'], f['valid_ap']
            if sc in train_scenarios:
                per_sc[SCENARIO_ID[sc]] = p1
                w = self.scen_samples[SCENARIO_ID[sc]]
                num += p1 * valid[:, None] * w
                den += valid * w
        per_sc[~np.isin(np.arange(len(SCENARIOS)),
                        [SCENARIO_ID[s] for s in train_scenarios])] = \
            (num / np.maximum(den, 1)[:, None]).astype(np.float32)
        return per_sc


class DLocDataset(torch.utils.data.Dataset):
    """memmap CSI + 标签 + 场景/AP 元数据。floorplan 张量由 DLocRasterBuilder 在脚本侧
    按 (scenario, ap) 生成并常驻内存（29 张 × ~0.8MB，见 csi_jepa/floorplan.py）。"""

    def __init__(self, corpus, index):
        self.corpus = corpus
        self.index = np.asarray(index, dtype=np.int64)      # 全局样本号
        self._tables = {}                                    # scenario -> (csi mmap, sidecar)

    def __len__(self):
        return len(self.index)

    def _table(self, scenario):
        if scenario not in self._tables:
            csi = np.load(os.path.join(self.corpus.cache_root, f'CSI_{scenario}.npy'),
                          mmap_mode='r')
            with np.load(os.path.join(self.corpus.cache_root, f'sidecar_{scenario}.npz')) as f:
                side = {k: f[k] for k in f.files}
            self._tables[scenario] = (csi, side)
        return self._tables[scenario]

    def __getitem__(self, i):
        g = int(self.index[i])
        sc, local = self.corpus.global_index[g]
        csi, side = self._table(sc)
        return {
            'csi': np.array(csi[local]),                     # (32, 234) float32
            'label_xy': side['labels'][local],               # (2,) 米
            'scenario': SCENARIO_ID[sc],
            'ap_centroids': side['ap_centroids'],            # (4,2) 米
            'valid_ap': side['valid_ap'],                    # (4,) bool
            'global_idx': g,
        }


def dloc_collate(batch):
    "原始值堆叠（陷阱 9：dataloader 不做归一化）；csi (B,32,234) float32。"
    return {
        'csi': torch.FloatTensor(np.stack([b['csi'] for b in batch])),
        'label_xy': torch.FloatTensor(np.stack([b['label_xy'] for b in batch])),
        'scenario': torch.LongTensor([b['scenario'] for b in batch]),
        'ap_centroids': torch.FloatTensor(np.stack([b['ap_centroids'] for b in batch])),
        'valid_ap': torch.BoolTensor(np.stack([b['valid_ap'] for b in batch])),
        'global_idx': torch.LongTensor([b['global_idx'] for b in batch]),
    }
