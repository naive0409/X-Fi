"""build_cache：管线编排器（生成 -> 渲染 floorplan -> 张量 -> 射追缓存）。

阶段可独立执行/断点续跑（各阶段产物已存在即 skip，--overwrite 强制重做）：
  scenes     scene_generator：scene_<id>/{scene.xml, meshes/, metadata.json}
  floorplans floorplan_renderer：逐 AP PNG（数据集 floorplan 资产）+ manifest.csv
  tensors    make_tensors：PNG -> (3,160,320) npz（预处理，非数据集资产）
  rt         rt_runner：CSI_*.npy + sidecar_*.npz（dloc_cache 同构，训练直接消费）

用法（conda run -n sionna，GPU 任务跑前先 nvidia-smi）：
  python -m sionna_gen.build_cache --num-scenes 100                # pilot 全流程
  python -m sionna_gen.build_cache --start 100 --num-scenes 2500 \
      --stages scenes floorplans tensors rt                        # 扩量
  python -m sionna_gen.build_cache --num-scenes 1 --stages scenes floorplans
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .config import load_config, out_dirs
from .scene_generator import generate_scene
from .floorplan_renderer import render_scene_floorplans, write_manifest
from .make_tensors import process_scene
from .rt_runner import trace_scene, append_timing

STAGES = ('scenes', 'floorplans', 'tensors', 'rt')


def main():
    ap = argparse.ArgumentParser('Sionna 合成数据集编排器')
    ap.add_argument('--config', default=None, help='yaml 覆盖 GenConfig')
    ap.add_argument('--start', type=int, default=0, help='起始 scene id')
    ap.add_argument('--num-scenes', type=int, default=1)
    ap.add_argument('--stages', nargs='*', default=list(STAGES), choices=STAGES)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = out_dirs(cfg)
    for k in ('scenes', 'floorplan_ds', 'tensors', 'cache'):
        os.makedirs(dirs[k], exist_ok=True)

    # 已有 manifest 则载入（floorplans 阶段增量追加、全量重写）
    manifest_rows = []
    mf = os.path.join(dirs['floorplan_ds'], 'manifest.csv')
    if 'floorplans' in args.stages and os.path.exists(mf) and not args.overwrite:
        with open(mf) as f:
            manifest_rows = list(csv.DictReader(f))
    manifest_rooms = {r['room'] for r in manifest_rows}

    t_all = time.perf_counter()
    for sid in range(args.start, args.start + args.num_scenes):
        scene_dir = os.path.join(dirs['scenes'], f'scene_{sid}')

        # ---- scenes
        if 'scenes' in args.stages:
            if os.path.exists(os.path.join(scene_dir, 'metadata.json')) and not args.overwrite:
                print(f'[skip] scene_{sid} 已生成')
            else:
                generate_scene(sid, cfg, dirs)

        if not os.path.exists(os.path.join(scene_dir, 'metadata.json')):
            print(f'[error] scene_{sid} 缺 metadata（需先跑 scenes 阶段）')
            continue

        # ---- floorplans（数据集资产：逐 AP PNG + manifest）
        if 'floorplans' in args.stages:
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

        # ---- tensors（PNG 派生预处理）
        if 'tensors' in args.stages:
            process_scene(scene_dir, dirs, cfg, args.overwrite)

        # ---- rt（GPU 射追 + dloc 同构缓存）
        if 'rt' in args.stages:
            _, t = trace_scene(scene_dir, dirs, cfg, args.overwrite)
            if t:
                append_timing(dirs, t)

    print(f'\n全部完成：{args.num_scenes} 场景 x {cfg.n_ue} UE，'
          f'总耗时 {time.perf_counter() - t_all:.0f}s')
    timing_csv = os.path.join(dirs['root'], 'rt_timing.csv')
    if os.path.exists(timing_csv):
        with open(timing_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            mean_solve = float(np.mean([float(r['solve_s']) for r in rows]))
            print(f'实测单场景 solve 均值 {mean_solve:.2f}s（{len(rows)} 场景）'
                  f'——用此值校准 SIONNA_PIPELINE.md 估算表')


if __name__ == '__main__':
    main()
