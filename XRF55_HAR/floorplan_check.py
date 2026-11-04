"""floorplan_check.py — Phase B0 手绘 JSON 的校验 + SVG 渲染（PROJECT_LOG §10）。

只读校验（schema/设备 id/外墙缺口/设备间距），零第三方依赖（无 matplotlib/PIL，SVG 纯文本
生成——对矢量数据比栅格预览更忠实），跨文件模式额外打印设备布局一致性（§8.1：布局跨场景
一致是复原假设）。产物：同名 .svg（浏览器目检：墙闭合？设备在角？比例合理？）。

用法：
  conda run -n xfi python floorplan_check.py floorplans/scene3.json   # 单个
  conda run -n xfi python floorplan_check.py floorplans/*.json        # 全部 + 一致性
"""
import glob
import json
import math
import os
import sys

REQUIRED_DEVICE_IDS = ['wifi_tx', 'wifi_rx1', 'wifi_rx2', 'wifi_rx3']
WALL_THICKNESS_M = 0.2   # 渲染线宽（米）；光栅化阶段同值
MARGIN_M = 0.5


def check_schema(data, path):
    errs, warns = [], []
    for key in ('scene', 'units', 'walls', 'devices'):
        if key not in data:
            errs.append(f'缺字段 {key}')
    if errs:
        return errs, warns
    if data.get('units') != 'meters':
        warns.append(f'units={data.get("units")!r}，规范要求 meters')
    walls = data['walls']
    for i, w in enumerate(walls):
        if len(w) != 4 or any(not isinstance(v, (int, float)) for v in w):
            errs.append(f'walls[{i}] 不是 [x1,y1,x2,y2] 数值四元组: {w}')
        elif math.hypot(w[2] - w[0], w[3] - w[1]) < 1e-6:
            warns.append(f'walls[{i}] 零长度: {w}')
        elif any(v < 0 for v in w):
            warns.append(f'walls[{i}] 有负坐标（规范建议原点在左下角、坐标非负）: {w}')
    ids = [d.get('id') for d in data['devices']]
    if sorted(ids) != sorted(REQUIRED_DEVICE_IDS):
        errs.append(f'devices id 应恰为 {REQUIRED_DEVICE_IDS}，实际 {ids}')
    for d in data['devices']:
        if 'xy' not in d or len(d['xy']) != 2:
            errs.append(f'device {d.get("id")} 缺 xy')
        if any(v < 0 for v in d.get('xy', [0, 0])):
            warns.append(f'device {d.get("id")} 有负坐标: {d["xy"]}')
    for i, p in enumerate(data.get('pillars', [])):
        if len(p) != 3 or any(not isinstance(v, (int, float)) for v in p):
            errs.append(f'pillars[{i}] 不是 [cx,cy,r]: {p}')
    # 外墙缺口提示：把每条线段端点视为节点，度≠2 的节点可能是墙缺口（仅供目检参考）
    eps = 0.15  # 米；照片手绘的容差
    pts = [(round(w[0], 3), round(w[1], 3)) for w in walls] + \
          [(round(w[2], 3), round(w[3], 3)) for w in walls]
    degree = {}
    for p in pts:
        key = None
        for q in degree:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < eps:
                key = q
                break
        degree[key or p] = degree.get(key or p, 0) + 1
    odd = [p for p, deg in degree.items() if deg % 2 == 1]
    if odd:
        warns.append(f'{len(odd)} 个度为奇的墙端点（可能是缺口，目检 SVG 确认）: '
                     f'{[tuple(round(c, 2) for c in p) for p in odd[:6]]}')
    return errs, warns


def svg_render(data, path):
    walls, pillars = data['walls'], data.get('pillars', [])
    xs = [c for w in walls for c in (w[0], w[2])] + \
         [d['xy'][0] for d in data['devices']] + [p[0] + p[2] for p in pillars] + \
         [p[0] - p[2] for p in pillars]
    ys = [c for w in walls for c in (w[1], w[3])] + \
         [d['xy'][1] for d in data['devices']] + [p[1] + p[2] for p in pillars] + \
         [p[1] - p[2] for p in pillars]
    x0, x1, y0, y1 = min(xs) - MARGIN_M, max(xs) + MARGIN_M, min(ys) - MARGIN_M, max(ys) + MARGIN_M
    W, H = 640, 640
    s = min(W / (x1 - x0), H / (y1 - y0))
    px = lambda x: (x - x0) * s          # noqa: E731
    py = lambda y: H - (y - y0) * s      # noqa: E731  SVG y 向下，翻转
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>']
    tw = max(1.0, WALL_THICKNESS_M * s)
    for w in walls:
        parts.append(f'<line x1="{px(w[0]):.1f}" y1="{py(w[1]):.1f}" x2="{px(w[2]):.1f}" '
                     f'y2="{py(w[3]):.1f}" stroke="black" stroke-width="{tw:.1f}" '
                     f'stroke-linecap="round"/>')
    for cx, cy, r in pillars:
        parts.append(f'<circle cx="{px(cx):.1f}" cy="{py(cy):.1f}" r="{r * s:.1f}" '
                     f'fill="gray" fill-opacity="0.6" stroke="black"/>')
    for d in data['devices']:
        color = 'red' if d['id'] == 'wifi_tx' else 'blue'
        x, y = px(d['xy'][0]), py(d['xy'][1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        yaw = math.radians(d.get('yaw', 0))
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" '
                     f'x2="{x + 14 * math.cos(yaw):.1f}" y2="{y - 14 * math.sin(yaw):.1f}" '
                     f'stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12" '
                     f'fill="{color}">{d["id"]}</text>')
    parts.append(f'<text x="8" y="18" font-size="14">{data["scene"]} '
                 f'(bbox {max(xs) - min(xs):.1f}m x {max(ys) - min(ys):.1f}m)</text>')
    parts.append('</svg>')
    out = os.path.splitext(path)[0] + '.svg'
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    return out


def pairwise_devices(data):
    "设备两两距离（米）——Scene3 应与照片红色标注对得上。"
    out = {}
    ds = {d['id']: d['xy'] for d in data['devices']}
    names = list(ds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[f'{a}-{b}'] = math.hypot(ds[a][0] - ds[b][0], ds[a][1] - ds[b][1])
    return out


def main():
    paths = []
    for pat in sys.argv[1:]:
        paths += glob.glob(pat)
    if not paths:
        print(__doc__)
        return
    all_pairs = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        errs, warns = check_schema(data, path)
        print(f'=== {path} ({data.get("scene")}) ===')
        for e in errs:
            print(f'  [ERROR] {e}')
        for w in warns:
            print(f'  [warn] {w}')
        if not errs:
            walls = data['walls']
            xs = [c for w in walls for c in (w[0], w[2])]
            ys = [c for w in walls for c in (w[1], w[3])]
            total = sum(math.hypot(w[2] - w[0], w[3] - w[1]) for w in walls)
            print(f'  包围盒 {max(xs) - min(xs):.1f}m x {max(ys) - min(ys):.1f}m, '
                  f'{len(walls)} 段墙共 {total:.1f}m, 柱 {len(data.get("pillars", []))} 根')
            pairs = pairwise_devices(data)
            all_pairs[data['scene']] = pairs
            print('  设备间距: ' + ', '.join(f'{k}={v:.2f}m' for k, v in pairs.items()))
            print(f'  SVG -> {svg_render(data, path)}')
        else:
            print('  [skip SVG：先修 ERROR]')
    if len(all_pairs) > 1:
        print('\n=== 跨场景设备布局一致性（§8.1 假设：布局跨场景一致）===')
        scenes = list(all_pairs)
        ref = all_pairs[scenes[0]]
        for s in scenes[1:]:
            diffs = {k: all_pairs[s][k] - ref[k] for k in ref}
            worst = max(diffs.items(), key=lambda kv: abs(kv[1]))
            print(f'  {scenes[0]} vs {s}: 最大间距差 {worst[0]} = {worst[1]:+.2f}m'
                  + ('  [warn] >0.5m，检查两图比例尺' if abs(worst[1]) > 0.5 else ''))


if __name__ == '__main__':
    main()
