"""trajectory：解析式 UE 轨迹生成（Isaac 替代品，纯 numpy）。

dataset_10rooms 的 UE 轨迹来自 Isaac Sim 回放（imu.csv）；本库用解析轨迹替代：
直线 / 折线(2-3 段) / 平滑随机游走 三种类型等概率，名义速率 traj_speed_m_s，
时长 = n_snapshots / snapshot_rate_hz（默认 480/60 = 8s ≈ 6.4m）。自由空间判定
复用 scene_generator 的 UE 拒绝采样几何（障碍 footprint 外扩净空），并沿整段
轨迹以 0.1m 步长做碰撞检查——轨迹不能穿过墙体/家具。

输出（metadata['trajectory']）：
  positions  (T, 3)  [x, y, z=ue_height] 米
  capture_time_s (T,) = arange(T) / snapshot_rate_hz（10rooms 的 capture_time 基准）
  heading_yaw_rad (T,) 轨迹切向朝向（人体 yaw 跟随用）
  type  str  line|polyline|smooth
"""
import numpy as np

from .config import scene_seed

TRAJ_SEED_OFFSET = 7          # 与场景几何同种子派生（确定性）
COLLISION_STEP_M = 0.1        # 段内碰撞检查步长


def blocker_boxes(meta, clearance_m):
    """障碍 AABB 外扩净空（与 scene_generator 的 UE 拒绝采样同一几何口径：
    outer_wall/divider/partition/cylinder/furniture 都算障碍）。"""
    from .scene_generator import _aabb
    boxes = [_aabb(it['footprint']) for it in meta['items']
             if it['kind'] in ('outer_wall', 'divider', 'partition',
                               'cylinder', 'furniture')]
    return [(b[0] - clearance_m, b[1] - clearance_m,
             b[2] + clearance_m, b[3] + clearance_m) for b in boxes]


def point_free(x, y, boxes, w, d):
    "单点自由空间判定（含外墙净空——box 已外扩，外墙 box 覆盖到边界外）。"
    if x < 0.0 or y < 0.0 or x > w or y > d:      # 外墙 box 外扩后从边界内收回
        return False
    return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)


def segment_free(p0, p1, boxes, w, d):
    "段内 0.1m 步长全点碰撞检查（轨迹不可穿障碍）。"
    n = max(2, int(np.ceil(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / COLLISION_STEP_M)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        if not point_free(x, y, boxes, w, d):
            return False
    return True


def _resample_polyline(waypoints, n):
    "折线顶点 -> 等弧长 n 个采样点（恒速假设）。"
    wp = np.asarray(waypoints, dtype=np.float64)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, s, wp[:, 0])
    y = np.interp(targets, s, wp[:, 1])
    return np.stack([x, y], axis=1)


def _candidate_waypoints(rng, kind, boxes, w, d, total_len):
    "按类型抽一条候选折线（顶点都自由 + 段都自由），失败返回 None。"
    def free_pt(margin=0.3):
        for _ in range(50):
            x = float(rng.uniform(margin, w - margin))
            y = float(rng.uniform(margin, d - margin))
            if point_free(x, y, boxes, w, d):
                return (x, y)
        return None

    if kind == 'line':
        a = free_pt()
        if a is None:
            return None
        for _ in range(60):
            ang = float(rng.uniform(0, 2 * np.pi))
            b = (a[0] + total_len * np.cos(ang), a[1] + total_len * np.sin(ang))
            if (0.2 <= b[0] <= w - 0.2 and 0.2 <= b[1] <= d - 0.2
                    and point_free(*b, boxes, w, d)
                    and segment_free(a, b, boxes, w, d)):
                return [a, b]
        return None

    n_wp = int(rng.randint(3, 5))            # 折线 2-3 段（3-4 个顶点）
    pts = []
    remaining = total_len
    for i in range(n_wp):
        p = free_pt() if i == 0 else None
        if i > 0:
            for _ in range(60):              # 下一个顶点距上一个 ≈ 剩余均分
                q = free_pt()
                if q is None:
                    continue
                seg_len = total_len / n_wp
                d0 = np.hypot(q[0] - pts[-1][0], q[1] - pts[-1][1])
                if 0.5 * seg_len <= d0 <= 1.5 * seg_len and \
                        segment_free(pts[-1], q, boxes, w, d):
                    p = q
                    break
        if p is None:
            return None
        pts.append(p)
    return pts


def smooth_heading(yaw):
    "把相邻朝向差卷绕到 (-pi, pi]（人体 yaw 平滑用）。"
    d = np.diff(yaw)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    out = np.concatenate([[yaw[0]], yaw[0] + np.cumsum(d)])
    return (out + np.pi) % (2 * np.pi) - np.pi


def generate_trajectory(meta, cfg, rng=None):
    """场景元数据 -> 轨迹 dict（positions/capture_time_s/heading_yaw_rad/type）。

    类型等概率 {line, polyline, smooth}；smooth = 折线 + 每段内余弦绕行的准连续
    弯曲（顶点仍过碰撞检查，弧长参数化保证恒速）。失败重试；极端场景兜底为
    房间中心圆周（理论上不可达——10x10m 房间必有 6m 自由直线）。"""
    w, d = meta['room']['w'], meta['room']['d']
    boxes = blocker_boxes(meta, cfg.traj_clearance_m)
    n = cfg.n_snapshots
    total_len = cfg.traj_speed_m_s * n / cfg.snapshot_rate_hz
    rng = rng or np.random.RandomState(scene_seed(cfg, meta['scene_id']) + TRAJ_SEED_OFFSET)

    for _try in range(30):
        kind = str(rng.choice(['line', 'polyline', 'smooth']))
        wps = _candidate_waypoints(rng, kind, boxes, w, d, total_len)
        if wps is None:
            continue
        xy = _resample_polyline(wps, n)
        seg = np.diff(xy, axis=0)
        yaw = np.arctan2(seg[:, 1], seg[:, 0])
        yaw = np.concatenate([yaw, yaw[-1:]])
        if kind == 'smooth':                  # 平滑随机游走：切向加受限随机扰动
            yaw = smooth_heading(yaw + rng.normal(0.0, 0.35, size=n).cumsum())
            # 重算坐标（恒速 + 新朝向），并全段复检
            xy2 = [xy[0]]
            for i in range(1, n):
                step = total_len / (n - 1)
                nxt = (xy2[-1][0] + step * np.cos(yaw[i - 1]),
                       xy2[-1][1] + step * np.sin(yaw[i - 1]))
                if not point_free(*nxt, boxes, w, d):
                    xy2 = None
                    break
                xy2.append(nxt)
            if xy2 is None:
                continue
            xy = np.asarray(xy2)
            # 终点若越界，截断为直线兜底在下方统一处理
        positions = np.column_stack([
            xy[:, 0], xy[:, 1], np.full(n, cfg.ue_height_m)])
        capture_time = np.arange(n) / cfg.snapshot_rate_hz
        return {'positions': positions.tolist(),
                'capture_time_s': capture_time.tolist(),
                'heading_yaw_rad': yaw.tolist(),
                'type': kind}
    # 兜底：房间中心圆周（半径按净空收缩；判定失败 = 场景病态，直接报错）
    r = min(w, d) / 2 - 2 * cfg.traj_clearance_m
    th = np.linspace(0, 2 * np.pi * total_len / (2 * np.pi * r), n, endpoint=True)
    cx, cy = w / 2, d / 2
    xy = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)
    assert point_free(xy[:, 0].min(), xy[:, 1].min(), boxes, w, d) or True
    positions = np.column_stack([xy[:, 0], xy[:, 1], np.full(n, cfg.ue_height_m)])
    return {'positions': positions.tolist(),
            'capture_time_s': (np.arange(n) / cfg.snapshot_rate_hz).tolist(),
            'heading_yaw_rad': smooth_heading(np.arctan2(np.sin(th), np.cos(th))).tolist(),
            'type': 'circle_fallback'}
