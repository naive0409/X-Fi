"""make_tensors：floorplan PNG -> 模型输入 4 通道张量（预处理产物，非数据集资产）。

数据集里 floorplan 只存 PNG（用户决策 2026-11-13）；本脚本从 PNG **精确逆采样**重建
native 0.1m 语义栅格（与 floorplan_renderer 的 (j*8)//5 采样互为 bijection，零失配），
再按 FLOORPLAN_SPEC §5b / csi_jepa DLocRasterBuilder 的同款数学产出张量：
  ch1 占据   0.2m 格 max-pool
  ch2 SDF    native EDT、符号距离 tanh(sdf/1m) 后 mean-pool（亚格元墙距穿过降采样）
  ch3 unknown 画布补白（covered=False）+ "阻挡且距最近自由区 >1m" 启发式
  ch4 AP 高斯图在训练侧加载时按 ap_centroids/valid_ap 现算（本脚本只存坐标——
      与 DLocRasterBuilder.build 的职责划分一致）
输出 tensors/scene_<id>.npz：occ/unk/sdf (160,320) float32 + ap_centroids (4,2) +
valid_ap (4,) + n_ap + world_origin (2,)（画布世界左下角，恒 (-1,-1)）。
"""
import json
import os

import numpy as np
from PIL import Image

from .floorplan_renderer import COLOR_OCC, COLOR_UNK

CELL_M = 0.2
PATCH_M = 4.0
UNKNOWN_FREE_DIST_M = 1.0


def _pool2(x, mode):
    "0.1m -> 0.2m 的 2x 池化（H,W 偶数化）。"
    h, w = x.shape
    x = x[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2)
    return x.max(axis=(1, 3)) if mode == 'max' else x.mean(axis=(1, 3))


def png_to_native(png_path, cfg):
    """PNG -> native 0.1m (free, occ, covered) 布尔栅格 (290,640)，行 0 = y_max。
    逆采样索引 = ceil(j·1.6) = (j·8+4)//5——与 renderer 正向 floor(i·0.625)=(i·5)//8
    精确互逆（png[k] 必落回 native[k·0.625 的唯一宿主格]）。逆采样点若落在网格线/
    AP 点上按 free 归类：二者只画在自由区（renderer 保证），语义不失真。"""
    img = np.asarray(Image.open(png_path).convert('RGB'))
    H = int(round(cfg.canvas_m[1] / cfg.native_res_m))       # 290
    W = int(round(cfg.canvas_m[0] / cfg.native_res_m))       # 640
    ridx = (np.arange(H) * 8 + 4) // 5                       # ceil(j*8/5)
    cidx = (np.arange(W) * 8 + 4) // 5
    samp = img[np.ix_(ridx, cidx)]                           # (H, W, 3)
    occ = np.all(samp == np.array(COLOR_OCC), axis=-1)
    unk = np.all(samp == np.array(COLOR_UNK), axis=-1)
    free = ~(occ | unk)
    covered = ~unk
    return free, occ, covered


def scene_tensor(png_dir, cfg):
    """PNG 目录（单场景逐 AP 图；取 ap0 的图作几何源——各 AP 图几何相同）->
    (3, 160, 320) float32 通道栈 [occ, unk, sdf]（AP 高斯在训练侧现算）。"""
    pngs = sorted(p for p in os.listdir(png_dir) if p.endswith('.png'))
    assert pngs, f'无 floorplan PNG: {png_dir}'
    free, occ, covered = png_to_native(os.path.join(png_dir, pngs[0]), cfg)

    from scipy import ndimage
    res = cfg.native_res_m
    d_free = ndimage.distance_transform_edt(~free) * res     # 非自由格到最近自由区(米)
    d_occ = ndimage.distance_transform_edt(free) * res       # 自由格到最近阻挡(米)
    sdf = np.where(free, d_occ, -d_free)
    unknown = (~covered) | (occ & (d_free > cfg.unknown_free_dist_m))

    ch_occ = _pool2(occ.astype(np.float32), 'max')
    ch_unk = _pool2(unknown.astype(np.float32), 'max')
    ch_sdf = _pool2(np.tanh(sdf / 1.0), 'mean')

    gh = int(round(cfg.canvas_m[1] / CELL_M))
    gw = int(round(cfg.canvas_m[0] / CELL_M))
    H = int(np.ceil(gh / PATCH_M)) * int(PATCH_M / CELL_M)   # 145 -> 160（patch 对齐 pad）
    W = gw                                                    # 320
    out = np.zeros((3, H, W), dtype=np.float32)
    oh, ow = ch_occ.shape
    out[0, :oh, :ow] = ch_occ
    out[1, :oh, :ow] = ch_unk
    out[1, oh:, :] = 1.0                                      # patch pad 区 = unknown
    out[2, :oh, :ow] = ch_sdf
    return out


def process_scene(scene_dir, dirs_map, cfg, overwrite=False):
    "单场景：PNG -> tensors/scene_<id>.npz（ap 坐标取 metadata）。"
    meta_path = os.path.join(scene_dir, 'metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)
    room = f"scene_{meta['scene_id']}"
    out_path = os.path.join(dirs_map['tensors'], f'{room}.npz')
    if os.path.exists(out_path) and not overwrite:
        print(f'[skip] {out_path}')
        return out_path

    n_ap = len(meta['aps'])
    cents = np.zeros((4, 2), dtype=np.float32)
    for i, ap in enumerate(meta['aps']):
        cents[i] = (ap['x'], ap['y'])
    valid = np.zeros(4, dtype=bool)
    valid[:n_ap] = True

    chans = scene_tensor(os.path.join(dirs_map['floorplan_ds'], room), cfg)
    origin = (-cfg.margin_m, -cfg.margin_m)                  # 与 renderer 锚定一致
    np.savez_compressed(out_path, occ=chans[0], unk=chans[1], sdf=chans[2],
                        ap_centroids=cents, valid_ap=valid, n_ap=n_ap,
                        world_origin=np.asarray(origin, dtype=np.float32))
    print(f'{room}: tensor {chans.shape} -> {out_path}')
    return out_path


if __name__ == '__main__':
    from .config import load_config, out_dirs
    import argparse
    ap = argparse.ArgumentParser('floorplan PNG -> tensors')
    ap.add_argument('--config', default=None)
    ap.add_argument('--scenes', type=int, nargs='*', default=None, help='指定 scene id；默认全部')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    cfg = load_config(args.config)
    dirs_map = out_dirs(cfg)
    root = dirs_map['scenes']
    ids = args.scenes or sorted(int(d.split('_')[1]) for d in os.listdir(root)
                                if d.startswith('scene_'))
    for sid in ids:
        process_scene(os.path.join(root, f'scene_{sid}'), dirs_map, cfg, args.overwrite)
