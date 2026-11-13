"""sanity_check：单场景端到端体检（pilot 前必过；conda run -n sionna 跑）。

检查项：
  1) 管线产物齐全且形状正确（XML/PLY/PNG/npz/CSI (50,32,234)）
  2) 自由空间衰减斜率：LOS 链路 log|h|² vs log(d) 拟合斜率 ≈ -2
  3) 穿墙路径存在（非 LOS 链路路径数 > 0，refraction 生效）
  4) CSI 幅度量级表（逐 AP×频带 mean |CSI|；--compare-dloc 与 DLoc 真实数据并排）
  5) --radio-map：每 AP 接收功率面图（仅可视化诊断，不入数据集）
  6) preview/ 下诊断图：floorplan overlay + AP/UE + |CSI|-距离散点
用法：
  conda run -n sionna python -m sionna_gen.sanity_check            # scene_0 全套
  conda run -n sionna python -m sionna_gen.sanity_check --scene-id 7 --radio-map
"""
import argparse
import json
import os
import time

import numpy as np

from .config import load_config, out_dirs, scene_seed, BAND_W, N_BANDS
from .scene_generator import generate_scene
from .floorplan_renderer import render_scene_floorplans, build_native
from .make_tensors import process_scene
from .rt_runner import trace_scene, subcarrier_frequencies

C_M_S = 299_792_458.0


def fs_slope_check(meta, h_re, h_im, cfg):
    "LOS 判据：min-tau ≈ d/c（2ns 容差）；对 LOS 链路做 log-log 功率-距离拟合。"
    n_ue, n_ap, n_ant, n_sub = h_re.shape
    aps = np.array([[a['x'], a['y']] for a in meta['aps']])
    ues = np.array([[u['x'], u['y']] for u in meta['ues']])
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    scene = load_scene(os.path.join(cfg.scene_dir, 'scene.xml'))
    scene.frequency = cfg.fc_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='V')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='V')
    for i, ap in enumerate(meta['aps']):
        scene.add(Transmitter(name=f'ap{i}',
                              position=mi.Point3f(float(ap['x']), float(ap['y']),
                                                  float(ap['z']))))
    for i, ue in enumerate(meta['ues']):
        scene.add(Receiver(name=f'ue{i}',
                           position=mi.Point3f(float(ue['x']), float(ue['y']),
                                               cfg.ue_height_m)))
    solver = PathSolver(deterministic=True)
    paths = solver(scene, max_depth=cfg.max_depth, samples_per_src=cfg.samples_per_src,
                   los=cfg.los, specular_reflection=cfg.specular_reflection,
                   refraction=cfg.refraction, diffraction=cfg.diffraction,
                   synthetic_array=True, seed=scene_seed(cfg, meta['scene_id']))
    _, tau = paths.cir(normalize_delays=False, out_type='numpy')
    tau = np.asarray(tau)                                    # [rx, tx, num_paths]

    slopes = []
    n_los = 0
    for a in range(n_ap):
        ds, pw = [], []
        for u in range(n_ue):
            d = np.hypot(*(aps[a] - ues[u]))
            tmin = tau[u, a][tau[u, a] > 0].min() if (tau[u, a] > 0).any() else np.inf
            if abs(tmin - d / C_M_S) < 2e-9:                 # LOS（直达与几何距离一致）
                h = np.sqrt(h_re[u, a] ** 2 + h_im[u, a] ** 2)   # (4 ant, 234)
                ds.append(d)
                pw.append(float((h ** 2).mean()))
                n_los += 1
        if len(ds) >= 5:
            s, b = np.polyfit(np.log(ds), np.log(np.asarray(pw)), 1)
            slopes.append(s)
            print(f'  AP{a}: LOS 链路 {len(ds)} 条, log-log 斜率 {s:.2f} (期望 ≈ -2)')
    print(f'  全场景 LOS 链路 {n_los}/{n_ue * n_ap}；'
          f'平均斜率 {np.mean(slopes):.2f}' if slopes else '  无足够 LOS 链路做拟合')
    nlos = (np.asarray(tau) > 0).sum(axis=-1)
    print(f'  每 (UE,AP) 路径数: min {nlos.min()}, 中位 {np.median(nlos):.0f}, '
          f'max {nlos.max()}（非零即有反射/穿墙分量）')


def p1_table(mag_csi, n_ap, n_train):
    "与 build_dloc_cache 同款 (4,9) mean|CSI| 表。"
    from .config import N_AP_SLOTS
    p1 = np.zeros((N_AP_SLOTS, N_BANDS), dtype=np.float32)
    for a in range(n_ap):
        p1[a] = mag_csi[:n_train, a * 8:(a + 1) * 8, :] \
            .reshape(n_train, 8, N_BANDS, BAND_W).mean(axis=(0, 1, 3))
    return p1


def radio_map_png(meta, cfg, dirs):
    "每 AP 接收功率面图（仅可视化，不入数据集）。"
    import mitsuba as mi
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

    scene = load_scene(os.path.join(cfg.scene_dir, 'scene.xml'))
    scene.frequency = cfg.fc_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='V')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern='iso', polarization='V')
    for i, ap in enumerate(meta['aps']):
        scene.add(Transmitter(name=f'ap{i}',
                              position=mi.Point3f(float(ap['x']), float(ap['y']),
                                                  float(ap['z']))))
    solver = RadioMapSolver()
    rm = solver(scene, cell_size=(0.25, 0.25), samples_per_tx=200_000,
                max_depth=cfg.max_depth, los=cfg.los,
                specular_reflection=cfg.specular_reflection, refraction=cfg.refraction,
                diffraction=cfg.diffraction, seed=scene_seed(cfg, meta['scene_id']))
    try:
        gain = np.asarray(rm.path_gain)                      # 期望 [num_tx, cy, cx] 线性
    except Exception as e:
        print(f'  [warn] path_gain 读取失败（{e!r}）；跳过 radio map 图'
              f'——用 rm.show() 交互查看')
        plt.close('all')
        return
    fig, axes = plt.subplots(1, len(meta['aps']), figsize=(6 * len(meta['aps']), 5))
    axes = np.atleast_1d(axes)
    for a, ax in enumerate(axes):
        ax.imshow(10 * np.log10(gain[a] + 1e-30), origin='lower', cmap='viridis')
        ax.set_title(f'AP{a} path gain (dB)')
    fig.savefig(os.path.join(dirs['preview'],
                             f"radio_map_scene_{meta['scene_id']}.png"), dpi=130)
    plt.close(fig)
    print(f'  radio map -> preview/radio_map_scene_{meta["scene_id"]}.png')


def overlay_png(meta, cfg, dirs, h_re, h_im):
    "floorplan overlay + AP/UE + |CSI|-距离散点。"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    occ, free, covered, origin = build_native(meta, cfg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax.imshow((~free).T, origin='lower',
              extent=[origin[0], origin[0] + cfg.canvas_m[0],
                      origin[1], origin[1] + cfg.canvas_m[1]], cmap='gray')
    for i, ap in enumerate(meta['aps']):
        ax.plot(ap['x'], ap['y'], 'r^', ms=10)
        ax.annotate(f'AP{i}', (ap['x'], ap['y']), color='r')
    ues = np.array([[u['x'], u['y']] for u in meta['ues']])
    ax.plot(ues[:, 0], ues[:, 1], 'b.', ms=4)
    ax.set_xlim(origin[0], origin[0] + meta['room']['w'] + 2)
    ax.set_ylim(origin[1], origin[1] + meta['room']['d'] + 2)
    ax.set_title('geometry (black=obstacle), AP(red)/UE(blue)')

    ax = axes[1]
    aps = np.array([[a['x'], a['y']] for a in meta['aps']])
    mag = np.sqrt(h_re ** 2 + h_im ** 2).mean(axis=(2, 3))   # (n_ue, n_ap)
    for a in range(len(meta['aps'])):
        ds = np.hypot(aps[a, 0] - ues[:, 0], aps[a, 1] - ues[:, 1])
        ax.scatter(ds, mag[:, a], s=8, alpha=0.7, label=f'AP{a}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('mean |CSI|')
    ax.legend()
    ax.set_title('|CSI| vs distance')
    fig.tight_layout()
    out = os.path.join(dirs['preview'], f'sanity_scene_{meta["scene_id"]}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  overlay -> {os.path.relpath(out)}')


def main():
    ap = argparse.ArgumentParser('单场景端到端体检')
    ap.add_argument('--scene-id', type=int, default=0)
    ap.add_argument('--config', default=None)
    ap.add_argument('--radio-map', action='store_true')
    ap.add_argument('--compare-dloc', default=None,
                    help='data/dloc_cache 路径：并排打印 DLoc 真实数据 p1_stats')
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = out_dirs(cfg)
    for k in ('scenes', 'floorplan_ds', 'tensors', 'cache', 'preview'):
        os.makedirs(dirs[k], exist_ok=True)
    scene_dir = os.path.join(dirs['scenes'], f'scene_{args.scene_id}')
    cfg.scene_dir = scene_dir                                # 检查函数复用

    print(f'== sanity: scene_{args.scene_id} ==')
    if not os.path.exists(os.path.join(scene_dir, 'metadata.json')):
        generate_scene(args.scene_id, cfg, dirs)
    meta = json.load(open(os.path.join(scene_dir, 'metadata.json')))

    rows = render_scene_floorplans(meta, cfg, dirs)
    print(f'1) floorplan PNG x{len(rows)}（manifest 行见 build_cache 全量写入）')
    process_scene(scene_dir, dirs, cfg)

    import mitsuba as mi
    print(f'   mitsuba variant: {mi.variant()}（cuda_ad_mono_polarized = GPU 生效）')
    t0 = time.perf_counter()
    csi_path, timing = trace_scene(scene_dir, dirs, cfg)
    if timing:
        print(f'2) solve {timing["solve_s"]}s + cfr {timing["cfr_s"]}s '
              f'（校准估算表：pilot 100 场景 ≈ {100 * (timing["solve_s"] + timing["cfr_s"] + 2) / 60:.0f} min）')
    else:
        print(f'2) 已有缓存（--overwrite 重跑）；管线耗时 {time.perf_counter() - t0:.1f}s')

    csi = np.load(csi_path)
    with np.load(os.path.join(dirs['cache'], f'sidecar_scene_{args.scene_id}.npz')) as f:
        side = {k: f[k] for k in f.files}
    n_ap = int(side['n_ap'])
    # 每 AP 8 行 = [ant0_re, ant0_im, ..., ant3_im]；步 2 切出 re/im 行（AP-major）
    h_re = csi[:, :n_ap * 8:2, :].reshape(csi.shape[0], n_ap, 4, -1)
    h_im = csi[:, 1:n_ap * 8:2, :].reshape(csi.shape[0], n_ap, 4, -1)
    assert csi.shape == (cfg.n_ue, 32, cfg.n_subcarriers), csi.shape
    print(f'3) CSI {csi.shape} float32；valid_ap {side["valid_ap"].tolist()}; '
          f'ap_centroids {side["ap_centroids"].tolist()}')

    print('4) 自由空间衰减检查：')
    fs_slope_check(meta, h_re, h_im, cfg)

    n_train = int(round(cfg.n_ue * 0.9))
    p1 = p1_table(np.abs(csi), n_ap, n_train)
    print('5) 合成 CSI mean|CSI|（AP x 9 频带）:')
    for a in range(n_ap):
        print('   ' + ' '.join(f'{v:.4f}' for v in p1[a]))
    if args.compare_dloc and os.path.isdir(args.compare_dloc):
        print('   DLoc 真实数据（各场景 p1_stats，量级对比用）:')
        for fn in sorted(os.listdir(args.compare_dloc)):
            if fn.startswith('sidecar_') and fn.endswith('.npz'):
                with np.load(os.path.join(args.compare_dloc, fn)) as f:
                    d1 = f['p1_stats']
                sc = fn[8:-4]
                print(f'   {sc:20s} ' + ' '.join(f'{v:.4f}' for v in d1[0]) + '  (AP0 行)')

    overlay_png(meta, cfg, dirs, h_re, h_im)
    if args.radio_map:
        radio_map_png(meta, cfg, dirs)
    print('== sanity 完成（目检 preview/ 图；斜率≈-2、量级与 DLoc 同数量级即通过） ==')


if __name__ == '__main__':
    main()
