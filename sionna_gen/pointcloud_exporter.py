"""pointcloud_exporter：场景 PLY mesh -> 静态环境点云 env_pc.npz（Isaac lidar 替代）。

环境是静态量——每场景一张图即可（dataset_10rooms 的 80 帧 lidar 中 95% 是冗余；
本库直接从 mesh 表面采样，天然无行人、无遮挡死角、无强度量化）。产物
env_pc.npz 与消费层约定：
  xyz      (pc_full_points, 3) float32   全分辨率表面均匀采样（米，世界系）
  xyz_fps  (pc_fps_points, 3)  float32   FPS 下采样（模型输入规模，训练侧零成本）
  bbox     (2, 3) float64                场景点云包围盒 [min; max]
  meta     json str                      场景 id/种子/点数
FPS 用确定性种子（scene_seed + 11），同一场景重复导出逐位一致。

采样口径（2026-11-28 用户决策后）：
  - **全局面积加权**：40960 点按全部三角面面积占比一次性分配（旧版逐 mesh 等额，
    门楣/家具等小面片密度虚高十几倍——用户目检发现的失真，已修复）
  - **默认不含房顶**（env_pc_include_ceiling=False，对齐真实 SLAM 采集；RT 场景
    天花板保留物理反射，只从点云排除）
"""
import glob
import json
import os

import numpy as np

from .config import scene_seed

PC_SEED_OFFSET = 11


def load_ply_ascii(path):
    "scene_generator.write_ply 的 ASCII PLY（顶点 + 三角面）。返回 (verts, faces)。"
    verts, faces = [], []
    with open(path) as f:
        header = True
        n_v = n_f = 0
        for line in f:
            if header:
                if line.startswith('element vertex'):
                    n_v = int(line.split()[-1])
                elif line.startswith('element face'):
                    n_f = int(line.split()[-1])
                elif line.startswith('end_header'):
                    header = False
                continue
            parts = line.split()
            if len(verts) < n_v:
                verts.append([float(v) for v in parts[:3]])
            else:
                faces.append([int(v) for v in parts[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def farthest_point_sampling(pts, k, rng):
    """确定性 FPS（首点取 rng 种子点；O(kN)，numpy 向量化）。"""
    n = len(pts)
    assert k <= n
    first = int(rng.randint(n))
    sel = np.empty(k, dtype=np.int64)
    sel[0] = first
    dist = np.linalg.norm(pts - pts[first], axis=1)
    for i in range(1, k):
        j = int(np.argmax(dist))
        sel[i] = j
        d = np.linalg.norm(pts - pts[j], axis=1)
        dist = np.minimum(dist, d)
    return pts[sel]


def export_env_pc(scene_dir, out_path, cfg, scene_id):
    """mesh 目录 -> env_pc.npz。返回 (n_full, n_faces_total) 供日志。

    全场景 mesh 合并成一个三角面池后按面积占比 multinomial 分配点数（全局均匀
    密度，逐点噪声由大数定律抹平），再重心坐标均匀采样；房顶 mesh 按
    env_pc_include_ceiling=False 排除。"""
    mesh_paths = sorted(glob.glob(os.path.join(scene_dir, 'meshes', '*.ply')))
    assert mesh_paths, f'{scene_dir} 无 mesh'
    skip = set()
    if not getattr(cfg, 'env_pc_include_ceiling', False):
        meta_path = os.path.join(scene_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            skip = {it['mesh'] for it in meta['items'] if it['kind'] == 'ceiling'}

    rng = np.random.RandomState(scene_seed(cfg, scene_id) + PC_SEED_OFFSET)
    tris, n_faces, n_used = [], 0, 0
    for p in mesh_paths:
        verts, faces = load_ply_ascii(p)
        n_faces += len(faces)
        if os.path.basename(p) in skip:
            continue
        tris.append(verts[faces])
        n_used += 1
    assert tris, f'{scene_dir} 过滤后无可采样 mesh（skip={sorted(skip)}）'
    tri = np.concatenate(tris, axis=0)                    # (F, 3, 3)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                           axis=1)                        # 2*面积
    total = area2.sum()
    assert total > 0, 'mesh 退化（总面积为 0）'
    counts = rng.multinomial(cfg.pc_full_points, area2 / total)

    chunks = []
    for f_idx, cnt in enumerate(counts):
        if cnt == 0:
            continue
        a, b, c = tri[f_idx]
        r1, r2 = np.sqrt(rng.random(cnt)), rng.random(cnt)
        chunks.append(((1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b
                       + (r1 * r2)[:, None] * c))
    xyz = np.concatenate(chunks, axis=0)
    perm = rng.permutation(len(xyz))
    xyz = xyz[perm[:cfg.pc_full_points]].astype(np.float32)
    xyz_fps = farthest_point_sampling(
        xyz.astype(np.float64), min(cfg.pc_fps_points, len(xyz)), rng).astype(np.float32)
    bbox = np.stack([xyz.min(axis=0), xyz.max(axis=0)]).astype(np.float64)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, xyz=xyz, xyz_fps=xyz_fps, bbox=bbox,
                        meta=json.dumps({'scene_id': int(scene_id),
                                         'n_full': int(len(xyz)),
                                         'n_fps': int(len(xyz_fps)),
                                         'n_faces': int(n_faces),
                                         'n_meshes': len(mesh_paths),
                                         'n_meshes_sampled': int(n_used),
                                         'ceiling_included': bool(
                                             getattr(cfg, 'env_pc_include_ceiling',
                                                     False))}))
    return len(xyz), n_faces
