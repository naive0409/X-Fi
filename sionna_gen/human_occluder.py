"""human_occluder：行走人体遮挡（对齐 dataset_10rooms 的 human_blockage 模型）。

两个职责：
  1) mesh 构建——按 10rooms run.json 的 8 段人体轮廓（[z, 横向半径, 前后半径]）
     放样成竖直环带椭柱网格（QC 可视化 / 写入场景文件夹存档）；人体朝向 yaw 跟随
     轨迹切向，中心 = UE 位置旁 human_offset_m（10rooms 恒 0.140m）。
  2) "理想几何吸收体"实现——dataset_10rooms 声明的人体模型。RT 求解阶段不加人体
     mesh（避免 480 次逐快照 solve），而是在求解后对每条路径的每段折线做
     "线段-竖直圆柱"相交测试：任何接触人体的路径段系数置零（能量被人体吸收）。
     这与"理想吸收体"语义一致：人体只吸收、不反射、不透射。主效应（LOS 遮挡）
     与在场景内放置吸收 mesh 等价，且天然获得逐快照移动的人体。

剪枝几何：每段轮廓带近似为一个竖直圆柱（半径取该带椭圆长半轴 = max(横向, 前后)，
保守略大）；线段-圆柱测试在 2D（x,y）做线段-圆相交求穿透区间，再检查该区间的
z 是否与圆柱 z 范围重叠。
"""
import numpy as np

MESH_RING_VERTS = 16      # 每环顶点数（QC mesh 用）


def human_center(ue_xy, yaw, offset_m):
    "人体中心 = UE 旁 offset（人体右手侧，相对轨迹切向）。"
    perp = np.array([np.sin(yaw), -np.cos(yaw)])    # 切向 (cos,sin) 的右侧法向
    return np.asarray(ue_xy, dtype=np.float64) + offset_m * perp


def human_to_world_matrix(cx, cy, yaw):
    """人体局部系（脚底原点、面向 +x）-> 世界系的 4x4 变换（绕 z 转 yaw + 平移）。

    供 Mitsuba XML <transform name="to_world"><matrix value="..."/></transform>
    使用（行主序 16 值，mitsuba 约定）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0, cx],
        [s,  c, 0.0, cy],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def ring_points(cx, cy, z, lat_r, dep_r, yaw, m=MESH_RING_VERTS):
    "单环顶点：椭圆（横/前后半径）随 yaw 旋转 + 平移。"
    a = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    lx, ly = lat_r * np.cos(a), dep_r * np.sin(a)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([cx + lx * c - ly * s, cy + lx * s + ly * c,
                     np.full(m, z)], axis=1)


def human_mesh(cx, cy, yaw, profile, height_m=None):
    """放样人体 mesh：轮廓各高度一环 + 顶/底帽。返回 (verts (V,3), faces (F,3))。"""
    rings = [ring_points(cx, cy, z, lat, dep, yaw) for z, lat, dep in profile]
    verts = np.concatenate(rings, axis=0)
    faces = []
    m = MESH_RING_VERTS
    n_rings = len(rings)
    for k in range(n_rings - 1):
        for i in range(m):
            j = (i + 1) % m
            a, b = k * m + i, k * m + j
            c_, d_ = (k + 1) * m + i, (k + 1) * m + j
            faces += [(a, b, c_), (b, d_, c_)]
    top_c, bot_c = len(verts), len(verts) + 1
    verts = np.concatenate([verts, [rings[-1].mean(axis=0), rings[0].mean(axis=0)]])
    for i in range(m):
        j = (i + 1) % m
        faces.append(((n_rings - 1) * m + i, (n_rings - 1) * m + j, top_c))
        faces.append((i, bot_c, j))
    return verts, np.asarray(faces, dtype=np.int64)


def write_human_ply(path, verts, faces):
    "QC/存档用 ASCII PLY（与 scene_generator.write_ply 同格式）。"
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(verts))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('element face %d\nproperty list uchar int vertex_indices\nend_header\n'
                % len(faces))
        for v in verts:
            f.write('%.4f %.4f %.4f\n' % tuple(v))
        for a, b, c in faces:
            f.write('3 %d %d %d\n' % (a, b, c))


# ---------------------------------------------------------------- 剪枝几何
def human_cylinders(cx, cy, yaw, profile):
    """轮廓 -> 保守竖直圆柱列表 [(x, y, z_lo, z_hi, r)]（r = 带内椭圆长半轴）。

    yaw 只影响椭圆朝向；保守化取长半轴后与 yaw 无关（略大于真实人体，吸收体
    语义下宁可多吸收）。"""
    out = []
    for k in range(len(profile) - 1):
        z0, lat0, dep0 = profile[k]
        z1, lat1, dep1 = profile[k + 1]
        r = max(lat0, dep0, lat1, dep1)
        out.append((cx, cy, z0, z1, r))
    return out


def _seg_circle_2d(p0, p1, c, r):
    """2D 线段 [p0,p1] 与圆 (c,r) 的穿透参数区间 (t0, t1)；无交返回 None。

    解 |p0 + t*(p1-p0) - c|² = r² 的 t 根，取落在 [0,1] 的区间。"""
    d = p1 - p0
    f = p0 - c
    a = float(d @ d)
    if a < 1e-12:                                   # 零长段：当点处理
        if float(f @ f) <= r * r:
            return 0.0, 1.0
        return None
    b = 2.0 * float(d @ f)
    cc = float(f @ f) - r * r
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return None
    sd = np.sqrt(disc)
    t0, t1 = (-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)
    t0, t1 = max(0.0, t0), min(1.0, t1)
    if t0 > t1:
        return None
    return t0, t1


def seg_hits_cylinder(p0, p1, cyl):
    """3D 线段是否接触竖直圆柱 (cx, cy, z_lo, z_hi, r)（接触即算，吸收体保守）。"""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    hit = _seg_circle_2d(p0[:2], p1[:2], np.asarray(cyl[:2]), cyl[4])
    if hit is None:
        return False
    z0, z1 = p0[2], p1[2]
    lo, hi = min(z0, z1), max(z0, z1)
    z_lo, z_hi = cyl[2], cyl[3]
    return (hi >= z_lo - 1e-9) and (lo <= z_hi + 1e-9)


def seg_hits_human(p0, p1, cyls):
    "路径段是否接触人体任一圆柱。"
    return any(seg_hits_cylinder(p0, p1, c) for c in cyls)


def path_hits_human(points, cyls):
    """一条路径（折点序列 (K,3)，含起点与终点；中间为交互点）是否接触人体。"""
    if points is None or len(points) < 2:
        return False
    pts = np.asarray(points, dtype=np.float64)
    # 顶点占位（未命中路径以 nan/0 填充）——0 占位会把路径拉到原点，先过滤退化段
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        if np.linalg.norm(seg[:2]) < 1e-9 and abs(seg[2]) < 1e-9:
            continue
        if seg_hits_human(pts[i], pts[i + 1], cyls):
            return True
    return False
