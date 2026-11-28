"""build_cache：管线编排器（生成 -> [QC floorplan] -> 环境点云 -> 射追产物）。

阶段可独立执行/断点续跑（各阶段产物已存在即 skip，--overwrite 强制重做）。
layout='rooms10'（Phase PC 主线，对齐 dataset_10rooms）：
  scenes     scene_generator：scene_<id>/{scene.xml, meshes/, metadata.json(含轨迹)}
  floorplans floorplan_renderer：QC 可选（PNG 不入数据集消费路径）
  env_pc     pointcloud_exporter：dataset/scene_<id>/env_pc.npz（mesh 表面采样）
  rt         rt_runner.rooms10：dataset/scene_<id>/radio/{cir,csi,csi_<other>,link_summary,run.json}
             + sidecar_pc.npz（单 TX-SISO × 轨迹快照；人体=求解后几何剪枝）
layout='dloc'（deprecated）：原四阶段不变（scenes/floorplans/tensors/rt）。

用法（conda run -n sionna，GPU 任务跑前先 nvidia-smi）：
  python -m sionna_gen.build_cache --num-scenes 3                       # 冒烟
  python -m sionna_gen.build_cache --start 0 --num-scenes 1000 \
      --bandwidth wide --human-frac 0.2                                 # 扩量（分片）
  python -m sionna_gen.build_cache --num-scenes 1 --stages scenes       # 只出几何
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np

from .config import load_config, out_dirs, scene_seed
from .scene_generator import generate_scene
from .pointcloud_exporter import export_env_pc

STAGES_DLOC = ('scenes', 'floorplans', 'tensors', 'rt')
STAGES_ROOMS10 = ('scenes', 'floorplans', 'env_pc', 'rt')


def main():
    ap = argparse.ArgumentParser('Sionna 合成数据集编排器')
    ap.add_argument('--config', default=None, help='yaml 覆盖 GenConfig')
    ap.add_argument('--start', type=int, default=0, help='起始 scene id')
    ap.add_argument('--num-scenes', type=int, default=1)
    ap.add_argument('--stages', nargs='*', default=None,
                    help=f"缺省 = 布局对应全阶段（rooms10: {STAGES_ROOMS10}）")
    ap.add_argument('--bandwidth', choices=['narrow', 'wide'], default=None,
                    help='主口径覆盖（另一口径同 solve 顺带写出 csi_<other>.npz）')
    ap.add_argument('--human-frac', type=float, default=None,
                    help='开人体遮挡的场景占比（0-1；缺省 = cfg.human_occluder 全开/全关）')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.bandwidth:
        cfg.bandwidth_preset = args.bandwidth    # delta_f 由 preset 查表（不改 cfg 值）
    if args.human_frac is not None:
        cfg.human_frac = float(args.human_frac)
    dirs = out_dirs(cfg)
    if cfg.layout == 'rooms10':
        stages = tuple(args.stages) if args.stages else STAGES_ROOMS10
        for k in ('scenes', 'dataset', 'preview'):
            os.makedirs(dirs[k], exist_ok=True)
    else:
        stages = tuple(args.stages) if args.stages else STAGES_DLOC
        for k in ('scenes', 'floorplan_ds', 'tensors', 'cache'):
            os.makedirs(dirs[k], exist_ok=True)
    unknown = set(stages) - (set(STAGES_ROOMS10) if cfg.layout == 'rooms10'
                             else set(STAGES_DLOC))
    assert not unknown, f'未知阶段 {unknown}（layout={cfg.layout}）'

    # 已有 manifest 则载入（floorplans QC 阶段增量追加、全量重写）
    manifest_rows = []
    mf = os.path.join(dirs['floorplan_ds'], 'manifest.csv')
    if 'floorplans' in stages and os.path.exists(mf) and not args.overwrite:
        with open(mf) as f:
            manifest_rows = list(csv.DictReader(f))
    manifest_rooms = {r['room'] for r in manifest_rows}

    t_all = time.perf_counter()
    for sid in range(args.start, args.start + args.num_scenes):
        scene_dir = os.path.join(dirs['scenes'], f'scene_{sid}')
        cfg_scene = cfg
        if cfg.layout == 'rooms10' and getattr(cfg, 'human_frac', None) is not None:
            cfg_scene = copy.copy(cfg)
            cfg_scene.human_occluder = bool(
                (int(scene_seed(cfg, sid)) % 1000) < cfg.human_frac * 1000)

        # ---- scenes
        if 'scenes' in stages:
            if os.path.exists(os.path.join(scene_dir, 'metadata.json')) and not args.overwrite:
                print(f'[skip] scene_{sid} 已生成')
            else:
                generate_scene(sid, cfg_scene, dirs)

        if not os.path.exists(os.path.join(scene_dir, 'metadata.json')):
            print(f'[error] scene_{sid} 缺 metadata（需先跑 scenes 阶段）')
            continue

        if cfg.layout == 'rooms10':
            dataset_dir = dirs['dataset']
            scene_out = os.path.join(dataset_dir, f'scene_{sid}')

            # ---- floorplans（QC 可选）
            if 'floorplans' in stages:
                room = f'scene_{sid}'
                if room in manifest_rooms and not args.overwrite:
                    print(f'[skip] {room} floorplan 已渲染')
                else:
                    from .floorplan_renderer import render_scene_floorplans, write_manifest
                    manifest_rows = [r for r in manifest_rows if r['room'] != room]
                    with open(os.path.join(scene_dir, 'metadata.json')) as f:
                        meta = json.load(f)
                    manifest_rows += render_scene_floorplans(meta, cfg, dirs)
                    write_manifest(dirs, manifest_rows)
                    manifest_rooms.add(room)

            # ---- env_pc（mesh 表面采样 -> dataset/scene_<id>/env_pc.npz）
            if 'env_pc' in stages:
                pc_path = os.path.join(scene_out, 'env_pc.npz')
                if os.path.exists(pc_path) and not args.overwrite:
                    print(f'[skip] scene_{sid} env_pc 已导出')
                else:
                    n_full, n_faces = export_env_pc(scene_dir, pc_path, cfg, sid)
                    print(f'scene_{sid}: env_pc {n_full} pts ({n_faces} faces)')

            # ---- rt（单 TX-SISO × 轨迹快照 + 人体剪枝）
            if 'rt' in stages:
                from .rt_runner import trace_scene_rooms10, append_timing
                t = trace_scene_rooms10(scene_dir, dataset_dir, cfg_scene,
                                        args.overwrite,
                                        bandwidth_override=args.bandwidth)
                if t:
                    append_timing(dirs, t)
        else:
            # deprecated dloc 路径（原逻辑不动）
            from .floorplan_renderer import render_scene_floorplans, write_manifest
            from .make_tensors import process_scene
            from .rt_runner import trace_scene, append_timing

            if 'floorplans' in stages:
                room = f'scene_{sid}'
                if room in manifest_rooms and not args.overwrite:
                    print(f'[skip] {room} floorplan 已渲染')
                else:
                    manifest_rows = [r for r in manifest_rows if r['room'] != room]
                    with open(os.path.join(scene_dir, 'metadata.json')) as f:
                        meta = json.load(f)
                    manifest_rows += render_scene_floorplans(meta, cfg, dirs)
                    write_manifest(dirs, manifest_rows)
                    manifest_rooms.add(room)
            if 'tensors' in stages:
                process_scene(scene_dir, dirs, cfg, args.overwrite)
            if 'rt' in stages:
                _, t = trace_scene(scene_dir, dirs, cfg, args.overwrite)
                if t:
                    append_timing(dirs, t)

    print(f'\n全部完成：{args.num_scenes} 场景，总耗时 {time.perf_counter() - t_all:.0f}s')
    timing_csv = os.path.join(dirs['root'], 'rt_timing.csv')
    if os.path.exists(timing_csv):
        with open(timing_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            mean_solve = float(np.mean([float(r['solve_s']) for r in rows]))
            print(f'实测单场景 solve 均值 {mean_solve:.2f}s（{len(rows)} 条记录）'
                  f'——用此值校准 SIONNA_PIPELINE.md 估算表')


if __name__ == '__main__':
    main()
