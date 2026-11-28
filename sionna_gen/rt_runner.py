"""rt_runner：单场景 GPU 射追 -> dloc_cache 同构 CSI 缓存。

流程（Sionna RT 2.2 已核实 API）：
  load_scene(scene.xml) -> scene.frequency -> PlanarArray(AP 2x2 iso / UE 单天线 iso)
  -> 每 AP 一个 Transmitter、每 UE 一个 Receiver -> PathSolver(deterministic=True)
  -> paths.cfr(frequencies, normalize_delays=False, out_type="numpy")（每子载波频响
     H(f)=Σ a_i^b e^{-j2πfτ_i}，shape [rx, rx_ant, tx, tx_ant, t, f]）
  -> 缓存布局与 build_dloc_cache 逐位同构：CSI_<sc>.npy (N,32,234) float32，
     每 AP 8 行 = [ant0_re, ant0_im, ..., ant3_im]（ant-major、re/im-minor），
     不足 4 AP 补零；sidecar_<sc>.npz = labels/ap_centroids/valid_ap/n_ap/p1_stats
     （p1 只用前 90% 样本，与 DLoc 的 train 块纪律一致）。
子载波频点：f_k = fc + (k - (n-1)/2)·Δf（居中于 fc，Nexmon 口径）。
确定性：solver 级 deterministic=True + 每场景独立 seed（PathSolver 默认非确定，勿改回）。
"""
import csv
import json
import os
import time

import numpy as np

from .config import N_AP_SLOTS, BAND_W, N_BANDS, scene_seed

SUBCARRIERS_FILE = 'CSI_{}.npy'
SIDECAR_FILE = 'sidecar_{}.npz'


def subcarrier_frequencies(cfg):
    "f_k = fc + (k-(n-1)/2)·Δf，(n,) float64。"
    k = np.arange(cfg.n_subcarriers)
    return cfg.fc_hz + (k - (cfg.n_subcarriers - 1) / 2.0) * cfg.delta_f_hz


def trace_scene(scene_dir, dirs_map, cfg, overwrite=False):
    """单场景射追 + 缓存。返回 (csi_path, timing dict)；已有缓存则 skip。"""
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    with open(os.path.join(scene_dir, 'metadata.json')) as f:
        meta = json.load(f)
    scenario = f"scene_{meta['scene_id']}"
    csi_path = os.path.join(dirs_map['cache'], SUBCARRIERS_FILE.format(scenario))
    side_path = os.path.join(dirs_map['cache'], SIDECAR_FILE.format(scenario))
    if os.path.exists(csi_path) and os.path.exists(side_path) and not overwrite:
        print(f'[skip] {scenario} 已有缓存')
        return csi_path, None

    scene = load_scene(os.path.join(scene_dir, 'scene.xml'), merge_shapes=True)
    scene.frequency = cfg.fc_hz
    scene.tx_array = PlanarArray(num_rows=cfg.tx_rows, num_cols=cfg.tx_cols,
                                 vertical_spacing=cfg.tx_spacing_wl,
                                 horizontal_spacing=cfg.tx_spacing_wl,
                                 pattern=cfg.tx_pattern, polarization=cfg.tx_polarization)
    scene.rx_array = PlanarArray(num_rows=cfg.rx_rows, num_cols=cfg.rx_cols,
                                 vertical_spacing=cfg.rx_spacing_wl,
                                 horizontal_spacing=cfg.rx_spacing_wl,
                                 pattern=cfg.rx_pattern, polarization=cfg.rx_polarization)
    for i, ap in enumerate(meta['aps']):
        scene.add(Transmitter(name=f'ap{i}',
                              position=mi.Point3f(float(ap['x']), float(ap['y']),
                                                  float(ap['z'])),
                              orientation=mi.Point3f(0.0, 0.0, 0.0)))
    for i, ue in enumerate(meta['ues']):
        scene.add(Receiver(name=f'ue{i}',
                           position=mi.Point3f(float(ue['x']), float(ue['y']),
                                               cfg.ue_height_m),
                           orientation=mi.Point3f(0.0, 0.0, 0.0)))

    seed = scene_seed(cfg, meta['scene_id'])
    solver = PathSolver(deterministic=True)
    t0 = time.perf_counter()
    paths = solver(scene,
                   max_depth=cfg.max_depth,
                   samples_per_src=cfg.samples_per_src,
                   synthetic_array=cfg.synthetic_array,
                   los=cfg.los,
                   specular_reflection=cfg.specular_reflection,
                   diffuse_reflection=cfg.diffuse_reflection,
                   refraction=cfg.refraction,
                   diffraction=cfg.diffraction,
                   seed=seed)
    t1 = time.perf_counter()
    freqs = subcarrier_frequencies(cfg)
    h_re, h_im = paths.cfr(frequencies=freqs,
                           normalize_delays=cfg.normalize_delays,
                           out_type='numpy')
    t2 = time.perf_counter()
    # [rx, rx_ant, tx, tx_ant, t(=1), f] -> 每 AP (n_ue, 4 ant, 234) 复数
    h_re = np.asarray(h_re)[:, 0, :, :, 0, :]
    h_im = np.asarray(h_im)[:, 0, :, :, 0, :]
    n_ue, n_ap, n_ant, n_sub = h_re.shape
    assert n_sub == cfg.n_subcarriers and n_ap == len(meta['aps']), \
        (h_re.shape, len(meta['aps']))

    labels = np.asarray([[u['x'], u['y']] for u in meta['ues']], dtype=np.float32)
    out = np.zeros((n_ue, N_AP_SLOTS * 2 * n_ant, n_sub), dtype=np.float32)
    for a in range(n_ap):
        inter = np.stack([h_re[:, a], h_im[:, a]], axis=2)   # (n_ue, 4, 2, 234)
        out[:, a * 8:(a + 1) * 8, :] = inter.reshape(n_ue, 8, n_sub)

    cents = np.zeros((N_AP_SLOTS, 2), dtype=np.float32)
    for i, ap in enumerate(meta['aps']):
        cents[i] = (ap['x'], ap['y'])
    valid = np.zeros(N_AP_SLOTS, dtype=bool)
    valid[:n_ap] = True

    # p1_stats (4,9)：逐 (AP,频带) 平均 |CSI|，只用前 90%（与 DLoc train 块纪律一致）
    n_train = int(round(n_ue * 0.9))
    mag = np.abs(out[:n_train])
    p1 = np.zeros((N_AP_SLOTS, N_BANDS), dtype=np.float32)
    for a in range(n_ap):
        rows = slice(a * 8, a * 8 + 8)
        p1[a] = mag[:, rows, :].reshape(n_train, 8, N_BANDS, BAND_W).mean(axis=(0, 1, 3))

    np.save(csi_path, out)
    np.savez(side_path, labels=labels, n_ap=n_ap, valid_ap=valid, ap_centroids=cents,
             p1_stats=p1)
    timing = {'scenario': scenario, 'solve_s': round(t1 - t0, 2),
              'cfr_s': round(t2 - t1, 2), 'n_ue': n_ue, 'n_ap': n_ap}
    print(f'{scenario}: solve {timing["solve_s"]}s, cfr {timing["cfr_s"]}s '
          f'-> CSI {out.shape}')
    return csi_path, timing


def append_timing(dirs_map, timing):
    "实测耗时追加 sionna_dataset/rt_timing.csv（估算表校准用）。"
    path = os.path.join(dirs_map['root'], 'rt_timing.csv')
    new = not os.path.exists(path)
    fields = ['scenario', 'mode', 'solve_s', 'cfr_s', 'prune_s', 'n_snapshots',
              'n_tx', 'human']
    new_row = {'scenario': timing.get('scenario'), 'mode': timing.get('mode', 'dloc'),
               'solve_s': timing.get('solve_s'), 'cfr_s': timing.get('cfr_s'),
               'prune_s': timing.get('prune_s'), 'n_snapshots': timing.get('n_ue'),
               'n_tx': timing.get('n_ap'), 'human': timing.get('human')}
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(new_row)


# =================================================================
# rooms10 对齐模式（Phase PC 主线）：单 TX-SISO × 轨迹快照
# =================================================================
C_M_S = 299_792_458.0


def _band_freqs(n_sub, delta_f, fc):
    "f_k = fc + (k-(n-1)/2)*delta_f（与 subcarrier_frequencies 同式，可变 n）。"
    k = np.arange(n_sub)
    return fc + (k - (n_sub - 1) / 2.0) * delta_f


def trace_scene_rooms10(scene_dir, dataset_dir, cfg, overwrite=False,
                        bandwidth_override=None):
    """单场景 rooms10 对齐射追 + dataset_10rooms 同构产物。

    solve A batched（T 个快照位置全作为 Receiver）；人体开时追加 T 次逐快照
    solve（人体 mesh 经 XML transform 注入场景——sionna-rt 2.2 不暴露路径折点
    坐标，场景内遮挡是唯一物理精确实现）。CSI 从系数/时延频域合成（cir<->csi
    严格互推由构造保证；**cir 不落盘**——用户决策：数据集不含 cir，系数仅在
    内存中用于 CSI 合成与统计，信道元数据在 link_summary.csv）。

    产物（dataset_dir/scene_<id>/radio/）：
      csi.npz          ideal_csi (T,256) c64 / estimated_csi (4,T,256) c64 /
                       valid_snapshot (T,) / subcarrier_offsets_hz / subcarrier_hz /
                       pilot_symbols / noise_variance (4,) / target_snr_db / nmse_db /
                       metadata_json（含 bandwidth_preset）
      csi_<other>.npz  双口径另一 preset 同 schema（also_write_other_preset 时）
      link_summary.csv 镜像 10rooms 列（los/num_paths/first_path_delay_ns/
                       sum_path_gain_db/human 位姿/UE 坐标/时间戳）
      run.json         镜像 10rooms 键
      （上级目录）sidecar_pc.npz：p1_stats (1,8)（窄带口径，前 90% 快照）+
                       p2_stats (3,)（n_paths/first_delay_ns/mean_gain_db）

    返回 timing dict（append_timing 用）。"""
    import json as _json
    import mitsuba as mi
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    from .config import BANDWIDTH_PRESETS, N_PATH_SLOTS, scene_seed

    with open(os.path.join(scene_dir, 'metadata.json')) as f:
        meta = _json.load(f)
    scenario = f"scene_{meta['scene_id']}"
    radio_dir = os.path.join(dataset_dir, scenario, 'radio')
    os.makedirs(radio_dir, exist_ok=True)
    done_flag = os.path.join(radio_dir, 'run.json')
    if os.path.exists(done_flag) and not overwrite:
        print(f'[skip] {scenario} 已有 radio 产物')
        return None

    preset_main = (bandwidth_override or getattr(cfg, 'bandwidth_preset', 'narrow'))
    preset_other = 'wide' if preset_main == 'narrow' else 'narrow'
    df_main = BANDWIDTH_PRESETS[preset_main]
    df_other = BANDWIDTH_PRESETS[preset_other]
    freqs = _band_freqs(cfg.n_subcarriers, df_main, cfg.fc_hz)

    scene = load_scene(os.path.join(scene_dir, 'scene.xml'), merge_shapes=True)
    scene.frequency = cfg.fc_hz
    scene.tx_array = PlanarArray(num_rows=cfg.tx_rows, num_cols=cfg.tx_cols,
                                 vertical_spacing=cfg.tx_spacing_wl,
                                 horizontal_spacing=cfg.tx_spacing_wl,
                                 pattern=cfg.tx_pattern, polarization=cfg.tx_polarization)
    scene.rx_array = PlanarArray(num_rows=cfg.rx_rows, num_cols=cfg.rx_cols,
                                 vertical_spacing=cfg.rx_spacing_wl,
                                 horizontal_spacing=cfg.rx_spacing_wl,
                                 pattern=cfg.rx_pattern, polarization=cfg.rx_polarization)
    ap = meta['aps'][0]                                   # rooms10：恒单 TX
    scene.add(Transmitter(name='tx', position=mi.Point3f(float(ap['x']), float(ap['y']),
                                                         float(ap['z'])),
                          orientation=mi.Point3f(0.0, 0.0, 0.0)))
    traj = meta['trajectory']
    ue_xyz = np.asarray(traj['positions'], dtype=np.float64)          # (T, 3)
    capture_t = np.asarray(traj['capture_time_s'], dtype=np.float64)
    yaw_t = np.asarray(traj['heading_yaw_rad'], dtype=np.float64)
    tx_xyz = np.array([ap['x'], ap['y'], ap['z']])
    T = len(ue_xyz)
    for i in range(T):
        scene.add(Receiver(name=f'ue{i}',
                           position=mi.Point3f(float(ue_xyz[i, 0]), float(ue_xyz[i, 1]),
                                               float(ue_xyz[i, 2])),
                           orientation=mi.Point3f(0.0, 0.0, 0.0)))

    seed = scene_seed(cfg, meta['scene_id'])
    solver = PathSolver(deterministic=True)
    t0 = time.perf_counter()

    def _solve(batched_rx):
        paths = solver(scene, max_depth=cfg.max_depth,
                       samples_per_src=cfg.samples_per_src,
                       synthetic_array=cfg.synthetic_array, los=cfg.los,
                       specular_reflection=cfg.specular_reflection,
                       diffuse_reflection=cfg.diffuse_reflection,
                       refraction=cfg.refraction, diffraction=cfg.diffraction,
                       seed=seed)
        a, tau = paths.cir(normalize_delays=False, out_type='numpy')
        a = np.asarray(a)
        if a.ndim == 6:                                        # [rx,ant,tx,ant,P,1]
            a = a[..., 0]
        return a.reshape(len(batched_rx), -1), np.asarray(tau).reshape(len(batched_rx), -1)

    def _truncate(coef, tau):
        "能量序截断/补零到 N_PATH_SLOTS（|coef| 降序，无效 tau=0 排尾）。"
        v = tau > 0.0
        key = np.where(v, np.abs(coef), -np.inf)
        k = min(coef.shape[1], N_PATH_SLOTS)
        order = np.argsort(-key, axis=1, kind='stable')[:, :k]
        coef = np.take_along_axis(coef, order, axis=1)
        tau = np.take_along_axis(tau, order, axis=1)
        v = np.take_along_axis(v, order, axis=1)
        if k < N_PATH_SLOTS:                       # 路径数不足 26 -> 右侧补零槽
            pad = N_PATH_SLOTS - k
            coef = np.concatenate([coef, np.zeros((len(coef), pad), dtype=coef.dtype)],
                                  axis=1)
            tau = np.concatenate([tau, np.zeros((len(tau), pad), dtype=tau.dtype)],
                                 axis=1)
            v = np.concatenate([v, np.zeros((len(v), pad), dtype=v.dtype)], axis=1)
        return coef, tau, v

    # ---- solve A：无人体，T 个快照位置全作为 Receiver（一次 batched solve）
    coef, tau, valid = _truncate(*_solve(ue_xyz))
    P = N_PATH_SLOTS
    d_3d = np.linalg.norm(ue_xyz - tx_xyz[None, :], axis=1)
    los_without_human = _los_flags(tau, valid, d_3d)
    t_A = time.perf_counter()

    # ---- 人体遮挡（用户决策保留 + config 开关；mode = 逐快照 in-scene solve，
    #      人体 mesh 经 XML transform 注入——sionna-rt 2.2 不暴露路径折点坐标，
    #      场景内遮挡是唯一物理精确的实现）
    human_rows = np.zeros((T, 3), dtype=np.float64)
    blocked = np.zeros(T, dtype=bool)
    los = los_without_human.copy()
    human_xml = None
    if cfg.human_occluder:
        from .human_occluder import (human_center, human_mesh, write_human_ply,
                                     human_to_world_matrix)
        hc0 = human_center(ue_xyz[0, :2], float(yaw_t[0]), cfg.human_offset_m)
        v, f = human_mesh(0.0, 0.0, 0.0, cfg.human_profile)    # 局部系（脚底原点，yaw=0）
        human_ply = os.path.join(scene_dir, 'human.ply')
        write_human_ply(human_ply, v, f)
        base_xml = open(os.path.join(scene_dir, 'scene.xml')).read()
        human_block = (
            '\t<shape type="ply" id="human_occluder">\n'
            '\t\t<string name="filename" value="{ply}"/>\n'
            '\t\t<boolean name="face_normals" value="true"/>\n'
            '\t\t<transform name="to_world"><matrix value="{mat}"/></transform>\n'
            '\t\t<bsdf type="itu-radio-material" id="mat-itu_{mat_name}">\n'
            '\t\t\t<string name="type" value="{mat_name}"/>\n'
            '\t\t\t<float name="thickness" value="0.1"/>\n'
            '\t\t</bsdf>\n'
            '\t</shape>\n')
        los = np.zeros(T, dtype=bool)
        coef_h = np.zeros_like(coef)
        tau_h = np.zeros_like(tau)
        valid_h = np.zeros_like(valid)
        for i in range(T):
            hc = human_center(ue_xyz[i, :2], float(yaw_t[i]), cfg.human_offset_m)
            human_rows[i] = [hc[0], hc[1], cfg.human_height_m / 2.0]
            mat = human_to_world_matrix(hc[0], hc[1], float(yaw_t[i]))
            mat_str = ' '.join(f'{v:.6f}' for v in mat.reshape(-1))
            xml_i = base_xml.replace(
                '</scene>',
                human_block.format(ply=human_ply, mat=mat_str,
                                   mat_name=cfg.human_material) + '</scene>')
            xml_path = os.path.join(scene_dir, f'_scene_snap{i}.xml')
            with open(xml_path, 'w') as f:
                f.write(xml_i)
            sc_i = load_scene(xml_path, merge_shapes=True)
            sc_i.frequency = cfg.fc_hz
            sc_i.tx_array = PlanarArray(num_rows=cfg.tx_rows, num_cols=cfg.tx_cols,
                                        vertical_spacing=cfg.tx_spacing_wl,
                                        horizontal_spacing=cfg.tx_spacing_wl,
                                        pattern=cfg.tx_pattern,
                                        polarization=cfg.tx_polarization)
            sc_i.rx_array = PlanarArray(num_rows=cfg.rx_rows, num_cols=cfg.rx_cols,
                                        vertical_spacing=cfg.rx_spacing_wl,
                                        horizontal_spacing=cfg.rx_spacing_wl,
                                        pattern=cfg.rx_pattern,
                                        polarization=cfg.rx_polarization)
            sc_i.add(Transmitter(name='tx',
                                 position=mi.Point3f(float(ap['x']), float(ap['y']),
                                                     float(ap['z'])),
                                 orientation=mi.Point3f(0.0, 0.0, 0.0)))
            sc_i.add(Receiver(name='ue0',
                              position=mi.Point3f(float(ue_xyz[i, 0]),
                                                  float(ue_xyz[i, 1]),
                                                  float(ue_xyz[i, 2])),
                              orientation=mi.Point3f(0.0, 0.0, 0.0)))
            c_i, t_i = _solve(ue_xyz[i:i + 1])
            coef_h[i:i + 1], tau_h[i:i + 1], valid_h[i:i + 1] = _truncate(c_i, t_i)
            os.remove(xml_path)
        coef, tau, valid = coef_h, tau_h, valid_h
        los = _los_flags(tau, valid, d_3d)
        blocked = los_without_human & ~los
        human_xml = 'per-snapshot in-scene solve'
    prune_s = time.perf_counter() - t_A

    # ---- CSI 频域合成（剪枝后系数——cir<->csi 严格互推）
    def synth_csi(freqs_):
        ph = np.exp(-2j * np.pi * freqs_[None, None, :] * tau[:, :, None])
        return (coef[:, :, None] * ph).sum(axis=1)          # (T, F) complex

    noise_rng = np.random.RandomState(seed + 5)

    def csi_product(ideal, preset_name, freqs_, df):
        offsets = freqs_ - cfg.fc_hz
        mean_power = float((np.abs(ideal) ** 2).mean())
        tiers = list(cfg.snr_tiers_db)
        nv = np.array([mean_power * 10.0 ** (-s / 10.0) for s in tiers])
        est = np.stack([ideal + np.sqrt(nv[t] / 2.0) * (
            noise_rng.standard_normal(ideal.shape) + 1j * noise_rng.standard_normal(
                ideal.shape)) for t in range(len(tiers))]).astype(np.complex64)
        nmse = 10.0 * np.log10(np.maximum(nv, 1e-30) / max(mean_power, 1e-30))
        pilots = np.ones(cfg.n_subcarriers, dtype=np.complex64)
        md = _json.dumps({
            'bandwidth_preset': preset_name, 'fc_hz': cfg.fc_hz,
            'n_subcarriers': cfg.n_subcarriers, 'delta_f_hz': float(df),
            'csi_definition': 'ideal = sum_p coef*exp(-2j pi f tau) over valid paths; '
                              'estimated = ideal + AWGN(tier); pilot-LS at unit pilots',
            'scene_id': int(meta['scene_id'])})
        return {
            'ideal_csi': ideal, 'estimated_csi': est,
            'valid_snapshot': (valid.any(axis=1)),
            'subcarrier_offsets_hz': offsets.astype(np.float64),
            'subcarrier_hz': freqs_.astype(np.float64),
            'pilot_symbols': pilots,
            'noise_variance': nv.astype(np.float32),
            'target_snr_db': np.asarray(tiers, dtype=np.int64),
            'frame_snr_db': (-nmse).astype(np.float32),
            'nmse_db': nmse.astype(np.float32),
            'metadata_json': md,
        }, nv

    ideal_main = synth_csi(freqs).astype(np.complex64)
    csi_main, _ = csi_product(ideal_main, preset_main, freqs, df_main)
    np.savez_compressed(os.path.join(radio_dir, 'csi.npz'), **csi_main)
    if getattr(cfg, 'also_write_other_preset', True):
        other_freqs = _band_freqs(cfg.n_subcarriers, df_other, cfg.fc_hz)
        ideal_other = synth_csi(other_freqs).astype(np.complex64)
        csi_other, _ = csi_product(ideal_other, preset_other, other_freqs, df_other)
        np.savez_compressed(os.path.join(radio_dir, f'csi_{preset_other}.npz'),
                            **csi_other)

    # ---- 多普勒：sionna-rt 2.2 不暴露路径折点坐标（interactions 只给物体索引），
    #      到达向不可得；cir.npz 不落盘（用户决策 2026-11-27：数据集不含 cir——
    #      系数/时延仅在内存中用于 CSI 频域合成与统计，逐快照信道元数据在
    #      link_summary.csv：los/num_paths/first_path_delay_ns/sum_path_gain_db/
    #      human 位姿/UE 坐标）。run.json 如实注明。

    # ---- link_summary.csv（10rooms 同列）
    with open(os.path.join(radio_dir, 'link_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imu_sample_index', 'sensor_time_s', 'capture_time_s', 'ue_x_m',
                    'ue_y_m', 'ue_z_m', 'los', 'num_paths', 'distance_over_c_ns',
                    'first_path_delay_ns', 'sum_path_gain_db', 'human_x_m', 'human_y_m',
                    'human_z_m', 'human_yaw_rad', 'los_without_human',
                    'los_blocked_by_human'])
        for i in range(T):
            n_paths = int(valid[i].sum())
            first = float(tau[i][valid[i]].min()) if n_paths else 0.0
            gain = 10.0 * np.log10(max(float((np.abs(coef[i][valid[i]]) ** 2).sum()),
                                       1e-30)) if n_paths else -300.0
            w.writerow([i, capture_t[i] + cfg.sensor_time_offset_s, capture_t[i],
                        f'{ue_xyz[i, 0]:.6f}', f'{ue_xyz[i, 1]:.6f}', f'{ue_xyz[i, 2]:.6f}',
                        int(los[i]), n_paths, f'{d_3d[i] / C_M_S * 1e9:.6f}',
                        f'{first * 1e9:.6f}', f'{gain:.6f}',
                        f'{human_rows[i, 0]:.6f}', f'{human_rows[i, 1]:.6f}',
                        f'{human_rows[i, 2]:.6f}', f'{yaw_t[i]:.6f}',
                        int(los_without_human[i]), int(blocked[i])])

    # ---- run.json（镜像 10rooms 键 + 本库新增）
    with open(os.path.join(radio_dir, 'run.json'), 'w') as f:
        _json.dump({
            'median_snapshot_rate_hz': cfg.snapshot_rate_hz,
            'max_snapshot_gap_s': float(np.diff(capture_t).max()) if T > 1 else 0.0,
            'sionna_rt_version': _sionna_version(),
            'scene_xml': os.path.abspath(os.path.join(scene_dir, 'scene.xml')),
            'trajectory': {'source': 'analytic', 'type': traj['type'],
                           'speed_nominal_m_s': cfg.traj_speed_m_s},
            'tx_xyz_m': [float(v) for v in tx_xyz],
            'frequency_hz': cfg.fc_hz,
            'samples_per_src': cfg.samples_per_src,
            'seed': int(seed),
            'max_reflections': cfg.max_depth,
            'path_gain_db_definition':
                '10 log10(sum of squared path coefficient magnitudes)',
            'timing': 'Independent quasi-static RT snapshots at trajectory sample times',
            'los_snapshots': int(los.sum()),
            'snapshots_with_paths': int(valid.any(axis=1).sum()),
            'snapshots': int(T),
            'bandwidth': {'preset': preset_main, 'n_subcarriers': cfg.n_subcarriers,
                          'delta_f_hz': float(df_main),
                          'also_written': preset_other if
                          getattr(cfg, 'also_write_other_preset', True) else None},
            'human_blockage': None if not cfg.human_occluder else {
                'model': 'Rigid upright pedestrian profile (8-segment, 10rooms-same), '
                         'in-scene absorber mesh, per-snapshot solve, yaw follows '
                         'trajectory tangent',
                'height_m': cfg.human_height_m,
                'offset_m': cfg.human_offset_m,
                'material_approx': cfg.human_material,
                'implementation': human_xml,
                'blocked_snapshots': int(blocked.sum())},
            'path_doppler_note': 'not computed (sionna-rt 2.2 does not expose per-path '
                                 'interaction coordinates; arrival-direction Doppler '
                                 'not computable); cir not exported (dataset decision '
                                 '2026-11-27: per-snapshot channel metadata lives in '
                                 'link_summary.csv)',
        }, f, indent=2)

    # ---- sidecar_pc.npz（P1/P2 物理辅助目标；只取前 90% 快照——train 块纪律）
    n_train = int(round(T * 0.9))
    n_bands = cfg.n_subcarriers // 32                     # 256/32 = 8 频带（=CSI patch_t）
    p1 = np.abs(ideal_main[:n_train]).reshape(n_train, n_bands, 32).mean(axis=(0, 2))
    p1 = p1[None, :].astype(np.float32)                   # (1, 8) 单 TX 槽位
    n_paths_tr = valid[:n_train].sum(axis=1).astype(np.float32)
    first_tr = np.array([tau[i][valid[i]].min() * 1e9 if valid[i].any() else 0.0
                         for i in range(n_train)], dtype=np.float32)
    gain_tr = np.array([10.0 * np.log10(max(float((np.abs(coef[i][valid[i]]) ** 2).sum()),
                                            1e-30)) if valid[i].any() else -300.0
                        for i in range(n_train)], dtype=np.float32)
    p2 = np.array([n_paths_tr.mean(), float(np.median(first_tr)), gain_tr.mean()],
                  dtype=np.float32)                       # (3,)
    np.savez_compressed(os.path.join(os.path.dirname(radio_dir), 'sidecar_pc.npz'),
                        p1_stats=p1, p2_stats=p2, n_snapshots_train=n_train,
                        bandwidth_preset='narrow')

    t_end = time.perf_counter()
    solve_s = t_A - t0
    post_s = t_end - t_A
    timing = {'scenario': scenario, 'mode': 'rooms10',
              'solve_s': round(solve_s, 2), 'cfr_s': round(post_s, 2),
              'prune_s': round(prune_s, 2), 'n_ue': T, 'n_ap': 1,
              'human': bool(cfg.human_occluder)}
    print(f'{scenario}: solveA {solve_s:.1f}s + human/post {post_s:.1f}s '
          f'-> csi (T={T}, P={P}, cir 不落盘); LOS {int(los.sum())}/{T}, '
          f'human-blocked {int(blocked.sum())}')
    return timing


def _los_flags(tau, valid, d_3d):
    "LOS 判定：存在 tau ≈ d/c（2ns 容差）的有效路径。"
    los = np.zeros(tau.shape[0], dtype=bool)
    for i in range(tau.shape[0]):
        cand = valid[i] & (np.abs(tau[i] - d_3d[i] / C_M_S) < 2e-9)
        los[i] = bool(cand.any())
    return los


def _sionna_version():
    try:
        import sionna.rt
        return str(getattr(sionna.rt, '__version__', 'unknown'))
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    import argparse
    import mitsuba as mi
    from .config import load_config, out_dirs
    ap = argparse.ArgumentParser('单场景射追（sanity 用；layout 由 config 决定）')
    ap.add_argument('--scene-dir', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    import sionna.rt                                     # 先触发后端选择（否则 variant=None）
    print(f'mitsuba variant: {mi.variant()}')            # 期望 cuda_ad_mono_polarized（生成机）
    cfg = load_config(args.config)
    dirs_map = out_dirs(cfg)
    if cfg.layout == 'rooms10':
        os.makedirs(dirs_map['dataset'], exist_ok=True)
        t = trace_scene_rooms10(args.scene_dir, dirs_map['dataset'], cfg, args.overwrite)
    else:
        os.makedirs(dirs_map['cache'], exist_ok=True)
        _, t = trace_scene(args.scene_dir, dirs_map, cfg, args.overwrite)
    if t:
        append_timing(dirs_map, t)
