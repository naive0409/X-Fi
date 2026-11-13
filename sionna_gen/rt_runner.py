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
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'solve_s', 'cfr_s', 'n_ue', 'n_ap'])
        if new:
            w.writeheader()
        w.writerow(timing)


if __name__ == '__main__':
    import argparse
    import mitsuba as mi
    from .config import load_config, out_dirs
    ap = argparse.ArgumentParser('单场景射追（sanity 用）')
    ap.add_argument('--scene-dir', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    print(f'mitsuba variant: {mi.variant()}')            # 期望 cuda_ad_mono_polarized
    cfg = load_config(args.config)
    dirs_map = out_dirs(cfg)
    os.makedirs(dirs_map['cache'], exist_ok=True)
    _, t = trace_scene(args.scene_dir, dirs_map, cfg, args.overwrite)
    if t:
        append_timing(dirs_map, t)
