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


def sanity_rooms10(args, cfg, dirs):
    """rooms10 布局体检：产物齐全 / cir<->csi 恒等 / NMSE≈-SNR / FS 斜率 / LOS 与
    人体遮挡统计 / env_pc / 轨迹。全部离线（不重新 solve，消费 rt 阶段产物）。"""
    from .pointcloud_exporter import export_env_pc
    from .trajectory import blocker_boxes, point_free
    from .config import BANDWIDTH_PRESETS

    scene_dir = os.path.join(dirs['scenes'], f'scene_{args.scene_id}')
    scene_out = os.path.join(dirs['dataset'], f'scene_{args.scene_id}')
    fails = []

    def check(name, ok, detail=''):
        print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
        if not ok:
            fails.append(name)

    print(f'== sanity (rooms10): scene_{args.scene_id} ==')
    if not os.path.exists(os.path.join(scene_dir, 'metadata.json')):
        generate_scene(args.scene_id, cfg, dirs)
    with open(os.path.join(scene_dir, 'metadata.json')) as f:
        meta = json.load(f)
    traj = meta['trajectory']

    # 1) 环境点云
    pc_path = os.path.join(scene_out, 'env_pc.npz')
    if not os.path.exists(pc_path):
        n_full, _ = export_env_pc(scene_dir, pc_path, cfg, args.scene_id)
        print(f'  （env_pc 缺失，已补导出：{n_full} pts）')
    with np.load(pc_path) as f:
        xyz, xyz_fps = f['xyz'], f['xyz_fps']
        bbox = f['bbox']
    w, d = meta['room']['w'], meta['room']['d']
    check('env_pc 形状', xyz.ndim == 2 and xyz.shape[1] == 3
          and xyz.shape[0] == cfg.pc_full_points, f'{xyz.shape}')
    check('env_pc bbox 在房间内', bbox[0, 0] >= -1.0 and bbox[1, 0] <= w + 1.0
          and bbox[1, 2] <= meta['room']['h'] + 1.0, f'{bbox.round(2).tolist()}')
    cell = 0.2
    occ_cells = len({(round(x / cell), round(y / cell)) for x, y in xyz[:, :2]})
    print(f'  env_pc 密度 ~{occ_cells} 个 0.2m 柱, FPS {len(xyz_fps)} pts')

    # 2) radio 产物齐全（cir 不落盘——用户决策；信道元数据在 link_summary.csv）
    radio = os.path.join(scene_out, 'radio')
    need = ['csi.npz', 'link_summary.csv', 'run.json']
    missing = [k for k in need if not os.path.exists(os.path.join(radio, k))]
    if not os.path.exists(os.path.join(scene_out, 'sidecar_pc.npz')):
        missing.append('sidecar_pc.npz (scene_out 根)')
    check('radio 产物齐全', not missing, f'缺 {missing}' if missing else '')

    with np.load(os.path.join(radio, 'csi.npz')) as f:
        csi = {k: f[k] for k in f.files}
    import csv as _csv
    with open(os.path.join(radio, 'link_summary.csv')) as f:
        link = list(_csv.DictReader(f))
    T = cfg.n_snapshots
    check('csi 形状/时长', csi['ideal_csi'].shape == (T, cfg.n_subcarriers)
          and len(link) == T,
          f"ideal {csi['ideal_csi'].shape}, link rows {len(link)}")

    # 3) NMSE ≈ -SNR
    nmse = csi['nmse_db']
    target = -csi['target_snr_db'].astype(float)
    check('NMSE ≈ -SNR', bool(np.all(np.abs(nmse - target) < 1.0)),
          f'nmse {nmse.round(2).tolist()} vs target {target.tolist()}')

    # 4) 距离衰减/量级合理性（几何 LOS 快照）：
    #    a) sum|coef|² 是室内多径和功率，衰减慢于自由空间（实测典型 -0.8 ~ -1.8，
    #       自由空间 -2 只对孤立直射径成立）——判"存在距离衰减"（-3.0 < slope < -0.5）
    #    b) 绝对量级：median(gain + FSPL_loss) ∈ [-6, +15]dB（多径超出有界，
    #       排除单位/dB-线性类灾难错误）
    dd = np.array([float(r['distance_over_c_ns']) * 0.299792458 for r in link])
    gain_db = np.array([float(r['sum_path_gain_db']) for r in link])
    is_los = np.array([r['los'] == '1' for r in link])
    dd_los = dd[is_los]
    if is_los.sum() >= 30 and dd_los.max() / max(dd_los.min(), 0.5) >= 2.0:
        pw = 10.0 ** (gain_db[is_los] / 10.0)
        s, _ = np.polyfit(np.log(dd_los), np.log(pw), 1)
        lam = C_M_S / cfg.fc_hz
        excess = float(np.median(gain_db[is_los]
                                 + 20.0 * np.log10(4 * np.pi * dd_los / lam)))
        check('距离衰减/量级合理', (-3.0 < s < -0.5) and (-6.0 <= excess <= 15.0),
              f'slope {s:.2f}（多径和功率，慢于 -2 属正常）, '
              f'FSPL 超出中位 {excess:+.1f}dB @ {int(is_los.sum())} LOS 快照, '
              f'd ∈ [{dd_los.min():.1f}, {dd_los.max():.1f}]m')
    else:
        print(f'  [info] 跳过衰减拟合：LOS 快照 {int(is_los.sum())} 个 / 距离跨度不足'
              f'（轨迹模式常见，无统计意义）')

    # 5) LOS / 人体遮挡统计
    n_los = sum(r['los'] == '1' for r in link)
    n_blk = sum(r['los_blocked_by_human'] == '1' for r in link)
    frac_los = n_los / T
    frac_blk = n_blk / T
    print(f'  LOS 快照 {n_los}/{T}（{frac_los:.0%}）；'
          f'人体遮挡 LOS {n_blk}（{frac_blk:.0%}）')
    if cfg.human_occluder and frac_blk == 0.0:
        print(f'  [info] 该场景人体未遮挡任何视线（LOS {frac_los:.0%}）——躯干穿越判据'
              f'要求 TX 落在人体侧 ~50° 锥角内，单场景 0 合法；机制有效性由数据集级'
              f'统计把关（有遮挡场景占比，参考 v4 实测 30%）')

    # 7) 轨迹自由空间复检（抽 50 点）
    boxes = blocker_boxes(meta, cfg.traj_clearance_m)
    pos = np.asarray(traj['positions'])
    ok_pts = [point_free(p[0], p[1], boxes, w, d) for p in pos[::max(1, T // 50)]]
    check('轨迹点在自由空间', all(ok_pts), f'{sum(ok_pts)}/{len(ok_pts)}')

    # 8) run.json 快照数一致（cir 不落盘；元数据在 link_summary/run.json）
    check('link_summary 行数 = 快照数', len(link) == T, f'rows {len(link)}')

    # 9) sidecar / run.json
    with np.load(os.path.join(scene_out, 'sidecar_pc.npz')) as f:
        sc = {k: f[k] for k in f.files}
    check('sidecar p1/p2 形状', sc['p1_stats'].shape == (1, 8)
          and sc['p2_stats'].shape == (3,), f"p1 {sc['p1_stats'].shape} p2 {sc['p2_stats']}")
    with open(os.path.join(radio, 'run.json')) as f:
        run = json.load(f)
    for k in ('sionna_rt_version', 'tx_xyz_m', 'frequency_hz', 'snapshots',
              'los_snapshots', 'human_blockage'):
        if k not in run:
            fails.append(f'run.json 缺键 {k}')
    print(f'  run.json: snapshots {run.get("snapshots")}, los {run.get("los_snapshots")}, '
          f'bandwidth {run.get("bandwidth", {}).get("preset")}, '
          f'human {run.get("human_blockage") is not None}')

    print(f'\n== sanity 完成：{"全部通过" if not fails else f"{len(fails)} 项 FAIL: {fails}"} ==')
    return fails


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
    if cfg.layout == 'rooms10':
        os.makedirs(dirs['dataset'], exist_ok=True)
        sanity_rooms10(args, cfg, dirs)
        return

    for k in ('scenes', 'floorplan_ds', 'tensors', 'cache', 'preview'):
        os.makedirs(dirs[k], exist_ok=True)
    scene_dir = os.path.join(dirs['scenes'], f'scene_{args.scene_id}')
    cfg.scene_dir = scene_dir                                # 检查函数复用（deprecated dloc 路径）

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
