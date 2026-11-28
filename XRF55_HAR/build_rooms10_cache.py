"""build_rooms10_cache.py — dataset_10rooms 的环境点云一次性预提取（Phase PC）。

3D_dataset/dataset_10rooms/room<k>/lidar/*.npz（80 帧，xyz_m 已是世界坐标）→
逐体素(0.1m)中位数融合静态点云：动态目标（行人）在任一体素内只停留少数帧，
按"体素被 ≥50% 帧观测"过滤天然剔除；静态墙体/地面/障碍物每帧可见而保留。
融合点的中位数坐标比体素中心更贴近真实表面（亚格元精度）。

产物（data/rooms10_cache/）：
  env_pc_room<k>.npz   xyz (N_static,3) float32 + xyz_fps (2048,3) + bbox (2,3)
                       + n_frames_kept 统计
  manifest.json        房间清单（lidar 帧数 / radio 快照数 / env_pc 路径 / 尺寸）
  radio/csi.npz 不复制——消费层直接读原路径（npz 原生），manifest 记录路径。

用法（XRF55_HAR/ 下）：
  conda run -n xfi python build_rooms10_cache.py                 # 全 10 房间
  conda run -n xfi python build_rooms10_cache.py --rooms 1 3     # 指定房间
  conda run -n xfi python build_rooms10_cache.py --cov-frac 0.3  # 覆盖率阈值消融
"""
import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, '..', '3D_dataset', 'dataset_10rooms')
OUT = os.path.join(REPO, '..', 'data', 'rooms10_cache')
VOXEL_M = 0.1                     # 融合体素（与 floorplan native 分辨率同款）
COV_FRAC = 0.5                    # 体素保留阈值：被 ≥50% 帧观测才算静态
FPS_POINTS = 2048
FPS_SEED = 3407


def fuse_room_lidar(room_dir, cov_frac=COV_FRAC):
    """80 帧 xyz 融合静态点云。返回 (static_pts (N,3) f32, n_frames, stats dict)。"""
    files = sorted(glob.glob(os.path.join(room_dir, 'lidar', 'lidar_*.npz')))
    assert files, f'{room_dir}/lidar 无帧'
    vox_ids, pts_all, frame_ids = [], [], []
    n_frames = len(files)
    for fi, fp in enumerate(files):
        with np.load(fp) as f:
            xyz = f['xyz_m'].astype(np.float32)     # 世界坐标（agent 已核实）
        vid = np.floor(xyz / VOXEL_M).astype(np.int64)
        key = vid[:, 0] * 7_000_009 + vid[:, 1] * 2003 + vid[:, 2]   # 3D->1D 哈希
        vox_ids.append(key)
        pts_all.append(xyz)
        frame_ids.append(np.full(len(xyz), fi, dtype=np.int16))
    key = np.concatenate(vox_ids)
    pts = np.concatenate(pts_all)
    fr = np.concatenate(frame_ids)
    order = np.argsort(key, kind='stable')
    key, pts, fr = key[order], pts[order], fr[order]
    uniq, start = np.unique(key, return_index=True)
    counts = np.diff(np.append(start, len(key)))
    # 每体素独立观测帧数（同帧同体素多点只计 1 帧——逐组 unique）
    frame_cov = np.zeros(len(uniq), dtype=np.int32)
    for gi, (s, c) in enumerate(zip(start, counts)):
        frame_cov[gi] = len(np.unique(fr[s:s + c]))
    keep = frame_cov >= max(1, int(round(cov_frac * n_frames)))
    # 每保留体素取中位数坐标（逐组，体素量 ~万级可承受）
    static = np.zeros((int(keep.sum()), 3), dtype=np.float32)
    oi = 0
    for gi in np.where(keep)[0]:
        s, c = start[gi], counts[gi]
        static[oi] = np.median(pts[s:s + c], axis=0)
        oi += 1
    stats = {'n_frames': n_frames, 'n_voxels': int(len(uniq)),
             'n_static_voxels': int(keep.sum()),
             'cov_frac': cov_frac, 'voxel_m': VOXEL_M}
    return static, n_frames, stats


def farthest_point_sampling(pts, k, rng):
    "与 pointcloud_exporter.farthest_point_sampling 同实现（确定性 FPS）。"
    n = len(pts)
    sel = np.empty(min(k, n), dtype=np.int64)
    first = int(rng.randint(n))
    sel[0] = first
    dist = np.linalg.norm(pts - pts[first], axis=1)
    for i in range(1, len(sel)):
        j = int(np.argmax(dist))
        sel[i] = j
        dist = np.minimum(dist, np.linalg.norm(pts - pts[j], axis=1))
    return pts[sel]


def main():
    ap = argparse.ArgumentParser('dataset_10rooms 环境点云缓存')
    ap.add_argument('--rooms', type=int, nargs='*', default=list(range(1, 11)))
    ap.add_argument('--cov-frac', type=float, default=COV_FRAC)
    ap.add_argument('--fps-points', type=int, default=FPS_POINTS)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    mf_path = os.path.join(OUT, 'manifest.json')
    if os.path.exists(mf_path):
        with open(mf_path) as f:
            manifest = json.load(f)
    for k in args.rooms:
        room_dir = os.path.join(SRC, f'room{k}')
        out_path = os.path.join(OUT, f'env_pc_room{k}.npz')
        if os.path.exists(out_path) and not args.overwrite:
            print(f'[skip] room{k} 已有缓存')
            continue
        static, n_frames, stats = fuse_room_lidar(room_dir, args.cov_frac)
        rng = np.random.RandomState(FPS_SEED + k)
        xyz_fps = farthest_point_sampling(static.astype(np.float64),
                                          args.fps_points, rng).astype(np.float32)
        bbox = np.stack([static.min(axis=0), static.max(axis=0)])
        np.savez_compressed(out_path, xyz=static, xyz_fps=xyz_fps, bbox=bbox,
                            stats=json.dumps(stats))
        radio_dir = os.path.join(room_dir, 'radio')
        with np.load(os.path.join(radio_dir, 'csi.npz')) as f:
            n_snap = int(f['ideal_csi'].shape[0])
        manifest[f'room{k}'] = {
            'room_dir': os.path.abspath(room_dir),
            'env_pc': os.path.abspath(out_path),
            'radio_dir': os.path.abspath(radio_dir),
            'n_snapshots': n_snap, **stats}
        print(f'room{k}: {n_frames} 帧 -> {stats["n_static_voxels"]} 静态体素 '
              f'({len(static)} pts, cov>={args.cov_frac:.0%}); radio {n_snap} 快照')
    with open(mf_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'manifest -> {mf_path}')


if __name__ == '__main__':
    main()
