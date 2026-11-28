"""scene_generator：程序化生成室内场景（Mitsuba XML + PLY，无需 Blender）。

配方（复现 MUSE-FM §IV-A-1 + 用户确认的适度扩展）：
  - 外壳 room_w x room_d x room_h（默认 10x10x3m）：地板 concrete、外围墙 brick、天花板 concrete
  - 1-2 面轴对齐分房隔墙（marble），墙上开门洞（墙拆段 + 门楣）
  - 0-3 面随机朝向追加隔墙（marble，MUSE-FM 配方）
  - 0-2 根贯通大理石圆柱（marble，MUSE-FM 配方）
  - 0-4 件家具级低障碍（wood/metal，0.5-0.9m 高）
  - AP 1-4 个（近墙候选点采样，高 2.5m）、UE 50 个（开放空间拒绝采样，高 1m）
材质走 `mat-itu_*` 命名约定（scene_utils.process_xml 自动转 ITU 材质）；墙/板厚 =
wall_thickness_m（thin-box 模型：几何盒厚 = RadioMaterial thickness）。
输出：scene.xml + meshes/*.ply + metadata.json（全几何/设备坐标/种子——渲染器与
调试的单一几何来源）。

注意：追加物放置采用"先验 footprint、后提交"两段式，被拒候选不留孤儿 mesh。
"""
import json
import os

import numpy as np

from .config import scene_seed

MAT_FLOOR = 'concrete'
MAT_CEIL = 'concrete'
MAT_OUTER = 'brick'
MAT_INNER = 'marble'
MATS_FURNITURE = ('wood', 'metal')


# ---------------------------------------------------------------- 几何基元
def box_corners_2d(cx, cy, sx, sy, yaw):
    "yaw 旋转矩形的 4 个 2D 角点（footprint 用）。"
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = sx / 2.0, sy / 2.0
    pts = [(dx, dy), (-dx, dy), (-dx, -dy), (dx, -dy)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def box_ply(cx, cy, cz, sx, sy, sz, yaw):
    "(12 三角面) 盒：局部 8 角 -> yaw 旋转 -> 顶点/面列表。"
    c, s = np.cos(yaw), np.sin(yaw)
    verts = []
    for lx in (-sx / 2, sx / 2):
        for ly in (-sy / 2, sy / 2):
            for lz in (-sz / 2, sz / 2):
                verts.append((cx + lx * c - ly * s, cy + lx * s + ly * c, cz + lz))
    # 顶点序: 0..3 底面, 4..7 顶面（z 主序）
    quads = [(0, 1, 3, 2), (4, 5, 7, 6),          # 底 / 顶
             (0, 1, 5, 4), (2, 3, 7, 6),          # 前后
             (0, 2, 6, 4), (1, 3, 7, 5)]          # 左右
    faces = []
    for a, b, cc, dd in quads:
        faces += [(a, b, cc), (a, cc, dd)]
    return verts, faces


def cylinder_ply(cx, cy, cz, r, sz, n=24):
    "n 边棱柱逼近圆柱（贯通 z 向，含底/顶中心点的扇面）。"
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]
    verts = [(x, y, cz - sz / 2) for x, y in ring] + [(x, y, cz + sz / 2) for x, y in ring]
    verts += [(cx, cy, cz - sz / 2), (cx, cy, cz + sz / 2)]     # idx 2n 底心, 2n+1 顶心
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]              # 侧面
        faces += [(i, j, 2 * n)]                                  # 底
        faces += [(n + i, n + j, 2 * n + 1)]                      # 顶
    return verts, faces


def write_ply(path, verts, faces):
    "ASCII PLY（面片朝向由 XML face_normals=true 归一，不依赖绕序）。"
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(verts))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('element face %d\nproperty list uchar int vertex_indices\nend_header\n' % len(faces))
        for v in verts:
            f.write('%.4f %.4f %.4f\n' % v)
        for a, b, c in faces:
            f.write('3 %d %d %d\n' % (a, b, c))


# ---------------------------------------------------------------- 放置辅助
def _aabb(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _aabb_gap(a, b):
    "两 AABB 的间隔（负数=重叠）。"
    dx = max(b[0] - a[2], a[0] - b[2])
    dy = max(b[1] - a[3], a[1] - b[3])
    return max(dx, dy)


def _split_wall_with_doors(rng, length, door_w, margin=0.5):
    "沿墙长轴切出 1-2 个不重叠门洞 -> (墙段 [(a,b)], 门洞 [(a,b)])。"
    doors = []
    n_doors = 1 if rng.rand() < 0.6 else 2
    lo = margin
    for _ in range(n_doors):
        hi = length - margin - door_w
        if hi <= lo:
            break
        p = float(rng.uniform(lo, hi))
        doors.append((p, p + door_w))
        lo = p + door_w + 0.2                               # 下一门不与前一门重叠
    segs, cur = [], 0.0
    for a, b in doors:
        if a - cur > 0.05:
            segs.append((cur, a))
        cur = b
    if length - cur > 0.05:
        segs.append((cur, length))
    return segs, doors


# ---------------------------------------------------------------- 主入口
def generate_scene(scene_id, cfg, out_dirs_map):
    """生成单场景：写 scene.xml + meshes/*.ply + metadata.json。返回 metadata dict。"""
    rng = np.random.RandomState(scene_seed(cfg, scene_id))
    h = cfg.room_h_m
    if getattr(cfg, 'room_size_range_m', None) is not None:
        lo, hi = cfg.room_size_range_m          # 用户决策 2026-11-28：w/d 逐场景 U(7,15)m
        w = float(rng.uniform(lo, hi))
        d = float(rng.uniform(lo, hi))
    else:
        w, d = cfg.room_w_m, cfg.room_d_m
    t = cfg.wall_thickness_m
    scene_dir = os.path.join(out_dirs_map['scenes'], f'scene_{scene_id}')
    mesh_dir = os.path.join(scene_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)

    items = []          # metadata 条目（渲染器/调试的几何来源）
    shapes = []         # (mesh 文件名, 材质名)

    def add_box(kind, material, cx, cy, cz, sx, sy, sz, yaw=0.0):
        name = f'{kind}_{len(shapes)}'
        verts, faces = box_ply(cx, cy, cz, sx, sy, sz, yaw)
        write_ply(os.path.join(mesh_dir, name + '.ply'), verts, faces)
        shapes.append((name, material))
        items.append({'kind': kind, 'material': material, 'mesh': name + '.ply',
                      'footprint': box_corners_2d(cx, cy, sx, sy, yaw),
                      'z_range': [cz - sz / 2, cz + sz / 2]})
        return items[-1]

    def add_cylinder(x, y, r):
        name = f'cylinder_{len(shapes)}'
        verts, faces = cylinder_ply(x, y, h / 2, r, h)
        write_ply(os.path.join(mesh_dir, name + '.ply'), verts, faces)
        shapes.append((name, MAT_INNER))
        fp = [(x + r * np.cos(a), y + r * np.sin(a))
              for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        items.append({'kind': 'cylinder', 'material': MAT_INNER, 'mesh': name + '.ply',
                      'footprint': fp, 'z_range': [0.0, h],
                      'center': [float(x), float(y)], 'radius': r})
        return items[-1]

    # ---- 外壳
    add_box('floor', MAT_FLOOR, w / 2, d / 2, -t / 2, w + 2 * t, d + 2 * t, t)
    add_box('ceiling', MAT_CEIL, w / 2, d / 2, h + t / 2, w + 2 * t, d + 2 * t, t)
    add_box('outer_wall', MAT_OUTER, w / 2, t / 2, h / 2, w + 2 * t, t, h)                    # y=0
    add_box('outer_wall', MAT_OUTER, w / 2, d - t / 2, h / 2, w + 2 * t, t, h)                # y=d
    add_box('outer_wall', MAT_OUTER, t / 2, d / 2, h / 2, t, d, h)                            # x=0
    add_box('outer_wall', MAT_OUTER, w - t / 2, d / 2, h / 2, t, d, h)                        # x=w

    # ---- 分房隔墙（轴对齐 + 门洞；1-2 面 -> 2-4 个房间）
    n_div = int(rng.randint(cfg.n_divider_min, cfg.n_divider_max + 1))
    div_positions = []                             # [(along_x, pos)] 同轴去重
    for _ in range(n_div):
        along_x = bool(rng.rand() < 0.5)      # True: 墙沿 x 走向（y=pos 横切）
        for _try in range(20):
            lim = 0.3 * d if along_x else 0.3 * w
            pos = float(rng.uniform(lim, (d if along_x else w) - lim))
            if all(along_x != ax or abs(pos - q) > 1.5 for ax, q in div_positions):
                break
        div_positions.append((along_x, pos))
        length = w if along_x else d
        segs, doors = _split_wall_with_doors(rng, length, cfg.door_width_m)
        for a, b in segs:
            mid = (a + b) / 2
            if along_x:
                add_box('divider', MAT_INNER, mid, pos, h / 2, b - a, t, h)
            else:
                add_box('divider', MAT_INNER, pos, mid, h / 2, t, b - a, h)
        for a, b in doors:                     # 门楣（门洞上方到墙顶）
            mid = (a + b) / 2
            if along_x:
                add_box('divider', MAT_INNER, mid, pos,
                        (cfg.door_height_m + h) / 2, b - a, t, h - cfg.door_height_m)
            else:
                add_box('divider', MAT_INNER, pos, mid,
                        (cfg.door_height_m + h) / 2, t, b - a, h - cfg.door_height_m)

    # 追加物避让表：full-height 障碍（门楣在 2.1m 以上，不进表——不挡 UE/家具）
    blockers = [_aabb(it['footprint']) for it in items
                if it['kind'] in ('divider', 'partition', 'cylinder', 'furniture')]

    def try_place(footprint_fn, commit_fn, margin):
        "先验 footprint（无副作用），过界/过近即弃；通过才提交（写 mesh + 记 items）。"
        for _ in range(40):
            x = float(rng.uniform(0.4, w - 0.4))
            y = float(rng.uniform(0.4, d - 0.4))
            box = _aabb(footprint_fn(x, y))
            if box[0] < 0.3 or box[1] < 0.3 or box[2] > w - 0.3 or box[3] > d - 0.3:
                continue
            if all(_aabb_gap(box, b) >= margin for b in blockers):
                blockers.append(box)
                commit_fn(x, y)
                return True
        return False                             # 放不下就少一件（多样性优先于硬数量）

    # ---- 追加随机朝向隔墙（0-3，marble，贯通全高——MUSE-FM 配方）
    for _ in range(int(rng.randint(0, cfg.n_partition_max + 1))):
        L = float(rng.uniform(1.5, 4.0))
        yaw = float(rng.uniform(0, np.pi))
        try_place(lambda x, y: box_corners_2d(x, y, L, t, yaw),
                  lambda x, y: add_box('partition', MAT_INNER, x, y, h / 2, L, t, h, yaw),
                  0.5)

    # ---- 圆柱（0-2，marble，贯通全高）
    for _ in range(int(rng.randint(0, cfg.n_cylinder_max + 1))):
        r = float(rng.uniform(0.15, 0.35))
        try_place(lambda x, y: box_corners_2d(x, y, 2 * r, 2 * r, 0.0),
                  lambda x, y: add_cylinder(x, y, r), 0.5)

    # ---- 家具（0-4，低矮 0.5-0.9m；UE 高 1m，视线可越过——散射/遮挡障碍）
    for _ in range(int(rng.randint(0, cfg.n_furniture_max + 1))):
        sx = float(rng.uniform(0.4, 1.2))
        sy = float(rng.uniform(0.4, 2.0))
        fz = float(rng.uniform(0.5, 0.9))
        yaw = float(rng.choice([0.0, np.pi / 2]))
        mat = MATS_FURNITURE[int(rng.rand() > 0.8)]     # 偏 wood（80% wood / 20% metal）
        try_place(lambda x, y: box_corners_2d(x, y, sx, sy, yaw),
                  lambda x, y: add_box('furniture', mat, x, y, fz / 2, sx, sy, fz, yaw),
                  0.3)

    # ---- AP（近墙候选点：4 角内缩 + 4 墙中点；两两距离 >= min(3m, 短边/3)）
    #      rooms10 布局：恒 1 个 TX（单 TX-SISO 对齐 dataset_10rooms）
    weights = np.asarray(cfg.n_ap_weights, dtype=np.float64)
    n_ap = int(rng.choice(np.arange(cfg.n_ap_min, cfg.n_ap_max + 1), p=weights / weights.sum()))
    inset = 0.5
    cands = [(inset, inset), (w - inset, inset), (inset, d - inset), (w - inset, d - inset),
             (w / 2, inset), (w / 2, d - inset), (inset, d / 2), (w - inset, d / 2)]
    aps = []
    min_ap_gap = min(3.0, min(w, d) / 3.0)
    for i in rng.permutation(len(cands)):
        p = cands[i]
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_ap_gap for q in aps):
            aps.append(p)
        if len(aps) == n_ap:
            break
    if not aps:                                       # 防御（理论上不可达）
        aps, n_ap = [(w / 2, d / 2)], 1

    # ---- UE（rooms10：不采静态点，改为解析轨迹——Isaac 回放轨迹的替代品；
    #      dloc：静态点拒绝采样，保持原样）
    wall_boxes = [_aabb(it['footprint']) for it in items
                  if it['kind'] in ('outer_wall', 'divider', 'partition',
                                    'cylinder', 'furniture')]
    wall_boxes = [(b[0] - 0.15, b[1] - 0.15, b[2] + 0.15, b[3] + 0.15) for b in wall_boxes]
    trajectory = None
    if cfg.layout == 'rooms10':
        from .trajectory import generate_trajectory
        geo = {'room': {'w': w, 'd': d, 'h': h}, 'items': items, 'scene_id': scene_id}
        trajectory = generate_trajectory(geo, cfg)
        ues = []
    else:
        ues = []
        guard = 0
        while len(ues) < cfg.n_ue and guard < 50 * cfg.n_ue:
            guard += 1
            x = float(rng.uniform(0.2, w - 0.2))
            y = float(rng.uniform(0.2, d - 0.2))
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in wall_boxes):
                continue
            ues.append((x, y))
        assert len(ues) == cfg.n_ue, f'UE 采样不足: {len(ues)}'

    # ---- metadata
    meta = {
        'scene_id': int(scene_id),
        'seed': scene_seed(cfg, scene_id),
        'layout': cfg.layout,
        'room': {'w': w, 'd': d, 'h': h, 'wall_thickness': t},
        'items': items,
        'aps': [{'x': float(p[0]), 'y': float(p[1]), 'z': cfg.ap_height_m} for p in aps],
        'ues': [{'x': float(p[0]), 'y': float(p[1])} for p in ues],
    }
    if trajectory is not None:
        meta['trajectory'] = trajectory
        # QC/存档：首快照人体 mesh（RT 求解时人体不在场景内——理想吸收体由
        # rt_runner 求解后按几何剪枝实现，见 human_occluder 模块 docstring）
        from .human_occluder import human_center, human_mesh, write_human_ply
        pos = np.asarray(trajectory['positions'][0])
        yaw0 = float(trajectory['heading_yaw_rad'][0])
        hc = human_center(pos[:2], yaw0, cfg.human_offset_m)
        v, f = human_mesh(hc[0], hc[1], yaw0, cfg.human_profile)
        write_human_ply(os.path.join(scene_dir, 'human_preview.ply'), v, f)
    with open(os.path.join(scene_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    # ---- scene XML（mat-itu_* 命名约定 -> process_xml 自动转 ITU 材质）
    bsdf_names = sorted({m for _, m in shapes})
    lines = ['<scene version="2.1.0">', '', '<!-- Materials -->', '']
    for m in bsdf_names:
        lines += [f'\t<bsdf type="itu-radio-material" id="mat-itu_{m}">',
                  f'\t\t<string name="type" value="{m}"/>',
                  f'\t\t<float name="thickness" value="{t}"/>', '\t</bsdf>', '']
    lines += ['<!-- Shapes -->', '']
    for name, m in shapes:
        lines += [f'\t<shape type="ply" id="{name}">',
                  f'\t\t<string name="filename" value="meshes/{name}.ply"/>',
                  '\t\t<boolean name="face_normals" value="true"/>',
                  f'\t\t<ref id="mat-itu_{m}" name="bsdf"/>', '\t</shape>', '']
    lines.append('</scene>')
    with open(os.path.join(scene_dir, 'scene.xml'), 'w') as f:
        f.write('\n'.join(lines))

    if cfg.layout == 'rooms10':
        print(f'scene_{scene_id}: {len(shapes)} shapes, TX 1, trajectory '
              f'{len(trajectory["positions"])} snapshots ({trajectory["type"]}) '
              f'-> {scene_dir}')
    else:
        print(f'scene_{scene_id}: {len(shapes)} shapes, {len(aps)} AP, {len(ues)} UE '
              f'-> {scene_dir}')
    return meta


if __name__ == '__main__':
    from .config import load_config, out_dirs
    import sys
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cfg = load_config()
    generate_scene(sid, cfg, out_dirs(cfg))
