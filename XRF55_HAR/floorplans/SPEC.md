# XRF55 floorplan 手绘规范（Phase B0，PROJECT_LOG §10）

> 来源：X-Fi 论文 Fig.2（p21:6）4 张场景照片。D37 范围：2D 平面图，只含**墙 + 柱**，
> 不含家具；设备只含 **WiFi Intel 5300（1 TX + 3 RX，位于感知矩形四角）**，只记平面位置。
> 每场景一个 JSON，命名 `scene1.json` ... `scene4.json`（放本目录）。人工工作量 ~0.5-1 天。

## 手绘流程（建议）

1. **先画 Scene3**：照片上有红色相对距离标注 → 用它标定比例尺（米）。
2. 以房间**左下角为原点**，x 向右、y 向上（所有场景统一）。估一个"1 米 ≈ 多少像素"，
   之后所有坐标按此换算（角度/比例对比绝对尺度更重要，误差 ±0.5m 可接受——§8.1 结论）。
3. 按照片描**墙体线段**（`walls`）：外墙一圈 + 内隔墙；门窗**不画**（v1 范围外，后续可加通道）。
4. **立柱**画进 `pillars`（Scene4 有柱）：`[cx, cy, r]`。
5. **设备**：4 台 Intel 5300 在感知矩形四角，`devices` 记 `id/xy/yaw`（yaw=天线板朝向，度，
   右手系从 x 轴逆时针；拿不准就填照片可见朝向的近似值——朝向 v1 只记录不进模型）。
6. 画完跑 `conda run -n xfi python floorplan_check.py floorplans/sceneN.json` ——自动校验 +
   生成 SVG（浏览器打开 `floorplans/sceneN.svg` 目检：墙闭合？设备在角上？比例合理？）。

## JSON Schema（v1）

```json
{
  "scene": "Scene1",
  "units": "meters",
  "origin_note": "原点=房间左下角，x右 y上（4 场景必须一致）",
  "walls": [[0,0,6,0], [6,0,6,5], [6,5,0,5], [0,5,0,0], [6,0,6,2.5]],
  "pillars": [[3.1, 2.4, 0.3]],
  "devices": [
    {"id": "wifi_tx",  "xy": [0.5, 0.5], "yaw": 45},
    {"id": "wifi_rx1", "xy": [5.5, 0.5], "yaw": 225},
    {"id": "wifi_rx2", "xy": [5.5, 4.5], "yaw": 225},
    {"id": "wifi_rx3", "xy": [0.5, 4.5], "yaw": 225}
  ],
  "notes": "来源：Fig.2 Scene1 照片；比例尺依据 Scene3 的 xx 米标注折算"
}
```

字段说明：
- `walls`: 线段列表 `[x1, y1, x2, y2]`（米）。**闭合性不强制**（校验工具会提示缺口），
  但外墙尽量闭合；墙厚不记（渲染/光栅化时统一加 ~0.2m 厚度）。
- `pillars`: 圆柱 `[cx, cy, r]`，可空列表。
- `devices`: 恰 4 条，id 固定为 `wifi_tx / wifi_rx1 / wifi_rx2 / wifi_rx3`（校验强制）。
- 数值一律米制 float；不要负坐标（原点取房间左下角外推一点也可以，保持非负即可）。

## 校验工具

```bash
cd ~/dev/X-Fi/XRF55_HAR
conda run -n xfi python floorplan_check.py floorplans/scene3.json   # 单个
conda run -n xfi python floorplan_check.py floorplans/*.json        # 全部 + 跨场景一致性
```

输出：schema 校验、外墙缺口/自交提示、设备落位提示、度量摘要（包围盒/墙总长/设备间距——
Scene3 的设备间距应≈照片红色标注值），并写 `sceneN.svg`（浏览器目检）。
