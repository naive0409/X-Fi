"""build_dloc_cache.py — DLoc (WILD) CSI 一次性预提取（Phase B-v2，PROJECT_LOG §11）。

channels_release/*.mat（MATLAB v7.3/HDF5, compound real+imag f8）→ 每场景一个 memmap 友好的
.npy：CSI_<scenario>.npy (N, 32, 234) float32。

行布局（FPCSIJEPA tokenizer 约定）：4 个 AP 组 × 8 行 = [ant0_re, ant0_im, ant1_re, ant1_im,
ant2_re, ant2_im, ant3_re, ant3_im]（re/im 交错——AoA 跨天线相位梯度与跨子载波相位梯度即
几何信号，且避免相位卷绕；`csi_repr: mag` 消融在训练侧从本缓存即时取模，无需第二种缓存）。
3-AP 场景（atkinson）第 4 组补零，valid_ap 旗标标记有效 AP。

sidecar sidecar_<scenario>.npz：labels (N,2) XY 米、n_ap、valid_ap (4,) bool、
ap_centroids (4,2) 米（ap_metadata.json 天线质心，补零同 padding）、
p1_stats (4, 9) float32 —— 逐 (AP, 频带) 平均 |CSI|，**只从每场景文件序前 90%（S-INTERP
train 块）统计**（P1 目标零 test 泄漏；LOSO 时由训练侧聚合其 train 场景的本表）。

磁盘预算 ~3.3GB（盘点时卷余 34GB）。用法：
  conda run -n xfi python build_dloc_cache.py            # 全部 8 场景
  conda run -n xfi python build_dloc_cache.py July16     # 单场景
"""
import json
import os
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'floorplan_dataset', 'DLoc_pt_code', 'channels_release')
OUT = os.path.join(REPO, 'data', 'dloc_cache')
N_AP_SLOTS = 4
N_SUBCARRIERS = 234
BAND_W = 26                      # 234 = 26 x 9（tokenizer patch_t）
N_BANDS = N_SUBCARRIERS // BAND_W
TRAIN_FRAC = 0.9                 # S-INTERP：文件序前 90% 为 train 块

SCENARIOS = ['jacobs_July28', 'jacobs_July28_2', 'jacobs_Aug16_1', 'jacobs_Aug16_3',
             'jacobs_Aug16_4_ref', 'July16', 'July18', 'July22_2_ref']


def ap_centroids(meta, scenario):
    "ap_metadata.json 的 aps=[AP1: [[x*4],[y*4]], AP2: [...], ...] → 每 AP 质心 (4,2)，3-AP 补零。"
    per_ap = meta[scenario]['aps']
    out = np.zeros((N_AP_SLOTS, 2), dtype=np.float32)
    for i, pair in enumerate(per_ap):
        out[i] = (float(np.mean(pair[0])), float(np.mean(pair[1])))
    return out, len(per_ap)


def build_scenario(scenario):
    out_path = os.path.join(OUT, f'CSI_{scenario}.npy')
    side_path = os.path.join(OUT, f'sidecar_{scenario}.npz')
    if os.path.exists(out_path) and os.path.exists(side_path):
        print(f'[skip] {scenario} 已有缓存')
        return
    with h5py.File(os.path.join(SRC, f'channels_{scenario}.mat'), 'r') as f:
        ch = f['channels'][()]          # (n_ap, 4, 234, N) compound {real, imag} f8
        labels = f['labels'][()].T      # h5py (2, N) → (N, 2) [x, y] 米
    n_ap, n_ant, n_sub, n = ch.shape
    assert (n_ap, n_ant, n_sub) in ((3, 4, 234), (4, 4, 234)), (scenario, ch.shape)
    re = ch['real'].astype(np.float32)  # (n_ap, 4, 234, N)
    im = ch['imag'].astype(np.float32)
    del ch
    # (N, n_ap, 4ant, 2c, 234sub) —— (ant, c) 展平即 ant-major/re-im-minor 行序
    interleaved = np.stack([re, im], axis=3).transpose(3, 0, 1, 2, 4)
    out = np.zeros((n, N_AP_SLOTS * 2 * n_ant, n_sub), dtype=np.float32)
    out[:, :n_ap * 2 * n_ant, :] = interleaved.reshape(n, n_ap * 2 * n_ant, n_sub)
    del re, im, interleaved
    np.save(out_path, out)
    print(f'{scenario}: CSI {out.shape} float32 -> {os.path.getsize(out_path) / 1e9:.2f} GB')

    with open(os.path.join(os.path.dirname(SRC), 'channels_release', 'ap_metadata.json')) as f:
        meta = json.load(f)
    cents, _ = ap_centroids(meta, scenario)
    valid = np.zeros(N_AP_SLOTS, dtype=bool)
    valid[:n_ap] = True
    n_train = int(round(n * TRAIN_FRAC))
    mag = np.abs(out[:n_train, :n_ap * 8, :])            # 仅 train 块（P1 零泄漏）
    p1 = np.zeros((N_AP_SLOTS, N_BANDS), dtype=np.float32)
    for ap in range(n_ap):
        rows = slice(ap * 8, ap * 8 + 8)
        bands = mag[:, rows, :].reshape(n_train, 8, N_BANDS, BAND_W)
        p1[ap] = bands.mean(axis=(0, 1, 3))              # (9,) 每频带平均幅度
    np.savez(side_path, labels=labels.astype(np.float32), n_ap=n_ap, valid_ap=valid,
             ap_centroids=cents, p1_stats=p1)
    print(f'  sidecar: labels {labels.shape}, ap_centroids {cents.tolist()}, '
          f'p1_stats {p1.shape}（train 前 {n_train}/{n}）')


def main():
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:] or SCENARIOS
    for s in wanted:
        if s not in SCENARIOS:
            print(f'[error] 未知场景 {s}（可选: {SCENARIOS}）')
            return
    for s in wanted:
        build_scenario(s)


if __name__ == '__main__':
    main()
