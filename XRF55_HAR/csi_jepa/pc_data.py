"""pc_data：点云条件化数据层（Phase PC）——dataset_10rooms 与 sionna 生成库统一接口。

两库目录同构（Phase 1 对齐的成果）：
  rooms10  <root>/room<k>/{radio/{csi.npz,link_summary.csv,cir.npz*}, env_pc@cache}
  sionna   <root>/scene_<id>/{radio/{csi.npz,link_summary.csv,run.json}, env_pc.npz}
差异只有 env_pc 的存放位置（rooms10 在 data/rooms10_cache，由 build_rooms10_cache
预提取；sionna 就在场景目录内），由 PCSceneCorpus 归一。
消费层只依赖 csi.npz + link_summary.csv（**cir 不读**——生成库不落盘 cir，用户决策
2026-11-27；rooms10 自带的 cir 仅在 P2 现算回退里用，见 _p2_from_cir）。

CSI 表示统一为 (T, 2, 256) float32 re/im 交错行（estimated_csi 指定 SNR 档，默认
20dB；陷阱 9：dataloader 恒原始值，归一化在模型内 channel_std buffer）。
标签 = 逐快照 UE XY（cir.npz ue_xyz_m）。P1/P2 目标逐场景统计（train 块前 90%，
零泄漏纪律同 Phase B）。

划分（PCDataset.split 模式，与 dloc_data 同哲学）：
  interp  逐场景轨迹连续块 90/10（主口径——轨迹时间相邻，随机划分虚高）
  rand    逐场景随机 90/10（文献可比对照）
  loso    留出整个场景（跨场景零样本/适配，价值主场；test_scenes 指定）
"""
import glob
import json
import os

import numpy as np
import torch

ROOMS10_ENV_CACHE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '..', 'data', 'rooms10_cache')


def _labels_from_link_summary(link_csv):
    """link_summary.csv -> (labels (N,2) [x,y] 米, N)。两库同列名（ue_x_m/ue_y_m）。"""
    import csv as _csv
    xs, ys = [], []
    with open(link_csv) as f:
        for row in _csv.DictReader(f):
            xs.append(float(row['ue_x_m']))
            ys.append(float(row['ue_y_m']))
    return np.stack([xs, ys], axis=1).astype(np.float32), len(xs)


def _csi_rows(csi_npz, snr_tier=20):
    """csi.npz -> (T, 2, 256) float32（estimated_csi 第 snr_tier 档的 re/im 行）。"""
    with np.load(csi_npz) as f:
        tiers = list(f['target_snr_db'])
        ti = tiers.index(snr_tier) if snr_tier in tiers else 1
        est = f['estimated_csi'][ti]                     # (T, 256) complex64
        offsets = f['subcarrier_offsets_hz']
    out = np.empty((est.shape[0], 2, est.shape[1]), dtype=np.float32)
    out[:, 0, :] = est.real
    out[:, 1, :] = est.imag
    return out, offsets


def _scene_p1p2(csi_rows, first_frac=0.9):
    """逐场景 P1 (1,8)（逐 (TX,频带) mean|CSI|，前 first_frac 快照）+ P2 (3,)。
    频带 = 256//32 = 8（与 CSI tokenizer patch_t=32 对齐）。"""
    n_train = int(round(len(csi_rows) * first_frac))
    mag = np.sqrt(csi_rows[:n_train, 0] ** 2 + csi_rows[:n_train, 1] ** 2)
    n_bands = csi_rows.shape[2] // 32
    p1 = mag.reshape(n_train, n_bands, 32).mean(axis=(0, 2))[None, :]
    p1 = p1.astype(np.float32)
    return p1


class PCSceneCorpus:
    """跨源场景表 + 全局样本索引 + 划分。"""

    def __init__(self, rooms10_root=None, sionna_root=None,
                 env_cache=ROOMS10_ENV_CACHE, snr_tier=20):
        self.scenes = []            # dict(source, key, dir, csi_path, cir_path, env_path)
        rooms10_root = rooms10_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '..', '3D_dataset', 'dataset_10rooms')
        self.env_cache = env_cache
        self.snr_tier = snr_tier
        if rooms10_root and os.path.isdir(rooms10_root):
            for k in range(1, 11):
                rd = os.path.join(rooms10_root, f'room{k}')
                if not os.path.isdir(os.path.join(rd, 'radio')):
                    continue
                env = os.path.join(env_cache, f'env_pc_room{k}.npz')
                self.scenes.append({
                    'source': 'rooms10', 'key': f'room{k}', 'dir': rd,
                    'csi_path': os.path.join(rd, 'radio', 'csi.npz'),
                    'cir_path': os.path.join(rd, 'radio', 'cir.npz'),
                    'env_path': env})
        if sionna_root and os.path.isdir(sionna_root):
            for sd in sorted(glob.glob(os.path.join(sionna_root, 'scene_*'))):
                if not os.path.isdir(os.path.join(sd, 'radio')):
                    continue
                key = os.path.basename(sd)
                self.scenes.append({
                    'source': 'sionna', 'key': key, 'dir': sd,
                    'csi_path': os.path.join(sd, 'radio', 'csi.npz'),
                    'cir_path': os.path.join(sd, 'radio', 'cir.npz'),
                    'env_path': os.path.join(sd, 'env_pc.npz')})
        if not self.scenes:
            raise RuntimeError('未发现任何场景（检查 rooms10_root/sionna_root）')
        self.scene_id = {s['key']: i for i, s in enumerate(self.scenes)}
        # 全局样本表：(scene_idx, t_idx)——惰性长度（读 cir 的 T；rooms10/sionna 均 480）
        self._lengths = None

    def lengths(self):
        if self._lengths is None:
            lens = []
            for s in self.scenes:
                _, n = _labels_from_link_summary(
                    os.path.join(s['dir'], 'radio', 'link_summary.csv'))
                lens.append(n)
            self._lengths = lens
        return self._lengths

    def global_index(self):
        """[(scene_idx, t_idx)]——与 dloc_data.global_index 同风格的扁平样本表。"""
        out = []
        for si, n in enumerate(self.lengths()):
            out += [(si, t) for t in range(n)]
        return out

    def scene_env_points(self, scene_idx):
        "场景 env_pc：优先 xyz_fps（模型规模），缺 cache 时退化 xyz 的 FPS 由消费层处理。"
        p = self.scenes[scene_idx]['env_path']
        with np.load(p) as f:
            return f['xyz_fps'] if 'xyz_fps' in f.files else f['xyz']

    def scene_p1(self, scene_idx):
        "P1 目标 (1,8)——sionna 库读 sidecar_pc.npz；rooms10 现算（零泄漏：前 90%）。"
        s = self.scenes[scene_idx]
        sc = os.path.join(s['dir'], 'sidecar_pc.npz')
        if os.path.exists(sc):
            with np.load(sc) as f:
                return f['p1_stats'].astype(np.float32)
        rows, _ = _csi_rows(s['csi_path'], self.snr_tier)
        return _scene_p1p2(rows)

    def split(self, mode, test_scenes=None, train_frac=0.9, seed=3407):
        """返回 (train_samples, test_samples, note)；样本 = (scene_idx, t_idx)。

        interp/rand：逐场景连续/随机块 90/10；loso：test_scenes 整场景入测。"""
        rng = np.random.RandomState(seed + 13)
        tr, te = [], []
        lengths = self.lengths()
        if mode in ('interp', 'rand'):
            for si, n in enumerate(lengths):
                idx = np.arange(n)
                n_train = int(round(n * train_frac))
                if mode == 'rand':
                    perm = rng.permutation(n)
                    tr_idx, te_idx = perm[:n_train], perm[n_train:]
                else:
                    tr_idx, te_idx = idx[:n_train], idx[n_train:]
                tr += [(si, int(t)) for t in tr_idx]
                te += [(si, int(t)) for t in te_idx]
            note = {'interp': 'per-scene trajectory-prefix block 90/10',
                    'rand': 'per-scene random 90/10'}[mode]
        elif mode == 'loso':
            keys = [s['key'] for s in self.scenes]
            test_set = set(test_scenes or [])
            assert test_set <= set(keys), f'test_scenes {test_scenes} 不在场景表 {keys}'
            for si, n in enumerate(lengths):
                if self.scenes[si]['key'] in test_set:
                    te += [(si, t) for t in range(n)]
                else:
                    tr += [(si, t) for t in range(n)]
            note = f'leave-scenes-out: test={sorted(test_set)} (all snapshots)'
        else:
            raise ValueError(mode)
        return sorted(tr), sorted(te), note


class PCDataset(torch.utils.data.Dataset):
    """逐快照样本：csi (2,256) 原始值 + label_xy + scenario idx。场景级惰性缓存。"""

    def __init__(self, corpus: PCSceneCorpus, samples, cache_max_gb=6.0):
        self.corpus = corpus
        self.samples = np.asarray(samples, dtype=object)
        self.cache_max_bytes = int(cache_max_gb * 1e9)
        self._cache = {}                     # scene_idx -> (csi_rows, labels)
        self._cache_bytes = 0

    def __len__(self):
        return len(self.samples)

    def _scene_data(self, si):
        if si in self._cache:
            return self._cache[si]
        s = self.corpus.scenes[si]
        rows, _ = _csi_rows(s['csi_path'], self.corpus.snr_tier)
        labels, _ = _labels_from_link_summary(
            os.path.join(s['dir'], 'radio', 'link_summary.csv'))
        nbytes = rows.nbytes + labels.nbytes
        if self._cache_bytes + nbytes > self.cache_max_bytes and self._cache:
            # FIFO 淘汰（简单够用；场景内访问呈簇状，FIFO 足以避免反复 IO）
            old = next(iter(self._cache))
            self._cache_bytes -= self._cache[old][0].nbytes + self._cache[old][1].nbytes
            del self._cache[old]
        self._cache[si] = (rows, labels)
        self._cache_bytes += nbytes
        return self._cache[si]

    def __getitem__(self, i):
        si, t = self.samples[i]
        rows, labels = self._scene_data(int(si))
        return {'csi': rows[int(t)],
                'label_xy': labels[int(t)],
                'scenario': int(si),
                't_idx': int(t),
                'scene_key': self.corpus.scenes[int(si)]['key']}


def pc_collate(batch):
    "原始值堆叠（陷阱 9：不做归一化）。csi (B,2,256) float32。"
    return {
        'csi': torch.FloatTensor(np.stack([b['csi'] for b in batch])),
        'label_xy': torch.FloatTensor(np.stack([b['label_xy'] for b in batch])),
        'scenario': torch.LongTensor([b['scenario'] for b in batch]),
        't_idx': torch.LongTensor([b['t_idx'] for b in batch]),
        'scene_key': [b['scene_key'] for b in batch],
    }


def stratified_subset_by_scene(samples, frac, seed=3407 + 11):
    """少量标签：逐场景嵌套分层抽样（同 seed 各方法同样本；10%⊃5%⊃1%）。"""
    rng = np.random.RandomState(seed)
    by_sc = {}
    for si, t in samples:
        by_sc.setdefault(si, []).append(t)
    out = []
    for si, ts in by_sc.items():
        ts = np.sort(np.asarray(ts))
        rng.shuffle(ts)
        k = max(1, int(round(len(ts) * frac)))
        out += [(si, int(t)) for t in ts[:k]]
    return sorted(out)
