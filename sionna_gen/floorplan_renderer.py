"""floorplan_renderer：场景几何 -> FLOORPLAN_SPEC 标准 floorplan PNG（数据集资产）。

规范要点（floorplan_dataset/FLOORPLAN_SPEC.md，全局常量永不 resize）：
  - SCALE 16 px/m、画布 1024x464（64x29m）、内容包围盒左下角锚定 MARGIN=1m
  - 语义调色板：free #ffffff / occupied #1a1a1a / unknown #c9c9cf；
    网格 #d9d9de 1px 每 1m 只画在自由区；AP 标记 #dc2626 实心圆 0.5m 白描边
  - 一图一 AP（AP 身份由文件名/manifest 承载）；不画文字/坐标轴/多 AP/箭头
  - 行 0 = y_max、列 0 = x_min（与 DLoc 掩码约定一致，张量路径可复用）
实现：几何先光栅化到 native 0.1m 主格（画布 [-1,63]x[-1,28] 世界域），PNG 以
最近邻采样 1.6 px/格 生成——make_tensors 用同一映射做**精确**逆采样重建 native，
保证 PNG<->张量零失配（spec §5b"同源同约定、永不失配"）。
门楣（z>=2.1m）不画：floorplan 是 1m 高接收视角的障碍图（门洞应呈开放）。
"""
import csv
import os

import numpy as np
from PIL import Image

COLOR_FREE = (255, 255, 255)        # #ffffff
COLOR_OCC = (26, 26, 26)            # #1a1a1a
COLOR_UNK = (201, 201, 207)         # #c9c9cf
COLOR_GRID = (217, 217, 222)        # #d9d9de
COLOR_AP = (220, 38, 38)            # #dc2626
# 张量路径的采样点只会落在这三色/网格/AP 上；网格与 AP 只画在自由区 -> 归 free
FREE_CLASSES = {'free', 'grid', 'ap'}

DRAW_Z_MIN = 1.2        # 只画 z_range[0] < 该值的障碍（1m 高 UE 视角；排除门楣）


def build_native(meta, cfg):
    """几何 -> native 0.1m 语义栅格。返回 (occ, free, covered, origin)：
    occ/free/covered (290, 640) bool，行 0 = y_max；origin = 画布世界左下角 (x,y)。
    covered=False 的格子即 unknown（画布空白，spec §3 的"未知"）。"""
    cm = cfg.canvas_m
    ox, oy = -cfg.margin_m, -cfg.margin_m            # 内容左下角锚定 MARGIN -> 画布原点
    res = cfg.native_res_m
    H = int(round(cm[1] / res))                      # 290
    W = int(round(cm[0] / res))                      # 640
    w, d = meta['room']['w'], meta['room']['d']

    # 单元中心世界坐标（row0=y_max）：y = oy + cm[1] - (i+0.5)*res
    xs = ox + (np.arange(W) + 0.5) * res
    ys = oy + cm[1] - (np.arange(H) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)                     # (H, W)

    covered = (XX >= 0) & (XX <= w) & (YY >= 0) & (YY <= d)
    occ = np.zeros((H, W), dtype=bool)

    from matplotlib.path import Path
    for it in meta['items']:
        if it['kind'] in ('floor', 'ceiling'):
            continue
        if it['z_range'][0] >= DRAW_Z_MIN:           # 门楣不画
            continue
        pts = np.asarray(it['footprint'], dtype=np.float64)
        poly = Path(np.column_stack([pts[:, 0], pts[:, 1]]))
        occ |= poly.contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(H, W)

    free = covered & ~occ
    return occ, free, covered, (ox, oy)


def render_scene_floorplans(meta, cfg, dirs_map):
    "逐 AP 渲染 PNG（spec §6 文件组织）并返回 manifest 行列表。"
    scale = cfg.scale_px_per_m
    cw, chh = cfg.canvas_px
    occ, free, covered, (ox, oy) = build_native(meta, cfg)

    # native -> PNG 最近邻采样映射（精确整数运算）：1 native 格 = scale*res = 8/5 px。
    # 正向 png[i] = native[(i*5)//8]；精确逆映射 native[j] = png[ceil(j*8/5)] =
    # png[(j*8+4)//5]（make_tensors.png_to_native 用，二者互逆、零失配）。
    # 末行/列 clip 仅防御。
    row_map = ((np.arange(chh) * 5) // 8).clip(0, occ.shape[0] - 1)
    col_map = ((np.arange(cw) * 5) // 8).clip(0, occ.shape[1] - 1)
    occ_png = occ[np.ix_(row_map, col_map)]
    free_png = free[np.ix_(row_map, col_map)]

    room = f"scene_{meta['scene_id']}"
    out_dir = os.path.join(dirs_map['floorplan_ds'], room)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for k, ap in enumerate(meta['aps']):
        img = np.empty((chh, cw, 3), dtype=np.uint8)
        img[...] = COLOR_UNK                          # 底色 = unknown（画布空白）
        img[free_png] = COLOR_FREE
        img[occ_png] = COLOR_OCC
        # 1m 网格（1px，只画在自由区）
        grid = np.zeros((chh, cw), dtype=bool)
        grid[::scale, :] = True
        grid[:, ::scale] = True
        img[grid & free_png] = COLOR_GRID
        # AP 标记：实心圆 0.5m（8px）+ 白描边；AP 必在自由区
        px = int(round((ap['x'] - ox) * scale))
        py = int(round((oy + cfg.canvas_m[1] - ap['y']) * scale))
        yy, xx = np.mgrid[0:chh, 0:cw]
        r2 = (xx - px) ** 2 + (yy - py) ** 2
        img[r2 <= 16] = COLOR_AP                      # 半径 4px 实心
        img[(r2 > 16) & (r2 <= 25)] = COLOR_FREE      # 1px 白描边（描边处为自由区语义）
        # PNG 内嵌 metadata（spec §6）
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text('px_per_m', str(scale))
        info.add_text('canvas_w_px', str(cw))
        info.add_text('canvas_h_px', str(chh))
        info.add_text('dataset', 'SIONNA')
        fname = f'{room}_ap{k}.png'
        Image.fromarray(img).save(os.path.join(out_dir, fname), pnginfo=info)
        rows.append({'file': os.path.join('SIONNA', room, fname), 'dataset': 'SIONNA',
                     'room': room, 'ap': k, 'device': 'sionna_iso4',
                     'ap_x': f"{ap['x']:.3f}", 'ap_y': f"{ap['y']:.3f}",
                     'canvas_w_px': cw, 'canvas_h_px': chh, 'px_per_m': scale,
                     'world_origin_x': ox, 'world_origin_y': oy})
    return rows


def write_manifest(dirs_map, all_rows):
    "manifest.csv（FLOORPLAN_SPEC §6 字段；存在则重写为全量）。"
    path = os.path.join(dirs_map['floorplan_ds'], 'manifest.csv')
    fields = ['file', 'dataset', 'room', 'ap', 'device', 'ap_x', 'ap_y',
              'canvas_w_px', 'canvas_h_px', 'px_per_m', 'world_origin_x', 'world_origin_y']
    with open(path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(all_rows)
    return path
