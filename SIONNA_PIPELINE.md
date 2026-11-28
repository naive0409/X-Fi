# SIONNA_PIPELINE.md — Sionna 合成 CSI 数据集：工作流与管线说明

> 目的：用 Sionna RT 射线追踪大规模合成多场景室内 WiFi CSI 数据集，解决 DLoc 试点暴露的
> "2 栋楼 8 场景不够多样" 问题（PROJECT_LOG D53/D54：跨楼零样本 −34% 反转）。方法上复现
> **MUSE-FM**（arXiv 2509.01967，PDF 在 `paper/muse-fm.pdf`）的数据生成流程，并做 WiFi 化
> 适配。本文既是**学习文档**（Sionna 怎么用）也是**管线手册**（我们怎么生成）。

---

## ⚡ Phase PC（2026-11-27）——管线已切至 rooms10 对齐布局（新主线）

> **在另一台机器上执行生成？把 [SIONNA_HANDOFF.md](SIONNA_HANDOFF.md)（自包含移交文档：
> 环境自检/执行序列/QC/回传协议/陷阱清单）交给该机的 coding agent，本文档作背景阅读。**
>
> **与旧文档（DLoc 对齐）的关键差异**。旧布局（2.4GHz / 1-4 AP 2×2 UPA / 静态 UE 点 /
> dloc_cache 同构缓存）deprecated 未删（`layout: 'dloc'` + `legacy_dloc.yaml` 可复现）。

**目标对齐 `3D_dataset/dataset_10rooms`**（用户决策）：3.5GHz、**单 TX-SISO**（TX 近墙角高
2.12m，UE 单天线高 1.25m）、**UE 轨迹 480 快照 @60Hz**（Isaac 回放轨迹 → 解析轨迹替代，
`trajectory.py`：直线/折线/平滑随机游走，自由空间避障）、**可选人体遮挡**
（`human_occluder`：8 段轮廓 mesh 在场景内、逐快照 solve、yaw 跟随轨迹切向；轮廓参数与
10rooms run.json 逐值同源）、输出与 10rooms **逐键同构**（cir.npz/csi.npz/link_summary.csv/
run.json）+ 新增 `env_pc.npz`（mesh 表面采样静态环境点云，替代 Isaac lidar）。

| 项 | 值（Phase PC） |
|---|---|
| RF | **3.5 GHz、256 子载波 ×30kHz = 7.68MHz**（`--bandwidth narrow`，10rooms 同款）；`wide` 预设 = 256×390.625kHz ≈100MHz（张量形状不变，时延分辨率 130ns→10ns）；**双口径同 solve 顺带写出**（cir 共享，近零成本） |
| 产物布局 | `sionna_dataset/scenes/scene_<id>/{scene.xml, meshes/, metadata.json(含轨迹), human.ply}`（几何源）+ **`sionna_dataset/dataset/scene_<id>/{env_pc.npz, radio/{csi,csi_<other>,link_summary.csv,run.json}, sidecar_pc.npz}`**（消费层入口，与 10rooms `room<k>/{lidar→env_pc, radio/}` 同构；**cir 不落盘**——用户决策 2026-11-27，逐快照信道元数据在 link_summary.csv） |
| 双带宽文件 | 主口径 `csi.npz`；另一口径 `csi_wide.npz` / `csi_narrow.npz`（schema 相同，metadata_json 内注 preset） |
| 人体 | `--human-frac 0.2` 按场景种子确定开关；**每开人体场景 = 1 次 batched solve + T 次逐快照 solve**（GPU 上估 ~5-15 min/场景，见下方成本表） |
| P1/P2 目标 | `sidecar_pc.npz`：p1_stats (1,8)（逐 TX×8 频带 mean|CSI|，前 90% 快照）+ p2_stats (3)（n_paths/first_delay_ns/mean_gain）——物理辅助监督 |
| 已知限制 | path_doppler 置零（sionna-rt 2.2 不暴露路径折点坐标，`paths.interactions` 只给物体索引）；人体材质为 ITU 近似（默认 wood），"理想吸收体"以场景内遮挡近似 |

**环境自检（换机器必做；本机两个坑已实测，2026-11-27）**：

```bash
conda create -n sionna python=3.11 -y && conda run -n sionna pip install sionna-rt==2.2.0 matplotlib scipy
# 坑①：drjit 需要 LLVM>=17 的共享库（15 编不了 fmaximum 指令；系统 libLLVM-12 被拒绝）：
conda install -n sionna -c conda-forge -y "libllvm17>=17.0.6,<18"
# 坑②：sionna-rt 2.2 的 CUDA 后端要求 GPU compute capability >= 7.5（本机 V100=7.0 被静默跳过，
#       且与 LLVM 缺失叠加曾表现为 import 空转）。生成机的 GPU 需 Turing 及以上（RTX20xx/T4/A100/V100→不行）。
# 自检（期望：打印 variant OK: cuda_ad_mono_polarized）：
conda run -n sionna python -c "
import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized')
print('variant OK:', mi.variant())"
# 本机（V100，无兼容 GPU）的退路 = CPU llvm 后端：慢 ~10-50×，只够冒烟、不够量产——
CUDA_VISIBLE_DEVICES="" DRJIT_LIBLLVM_PATH=<env>/lib/libLLVM-17.so \
  python -m sionna_gen.build_cache --config sionna_gen/smoke_rooms10.yaml --num-scenes 1
# （实测：CPU 6 快照×50 样本全链路 5s/场景；量产 480 快照×1e5 样本必须在兼容 GPU 的生成机）
```

**Phase PC 运行手册（在生成机上执行；先 `nvidia-smi` 挑卡，`CUDA_VISIBLE_DEVICES=<N>`）**：

```bash
cd X-Fi   # 仓库根
# 冒烟（~1 min；极小参数只验证代码路径，产物弃置）
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --config sionna_gen/smoke_rooms10.yaml --num-scenes 1
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.sanity_check --config sionna_gen/smoke_rooms10.yaml --scene-id 0
# pilot 100 场景（正式参数；估 GPU 1-3 h，rt_timing.csv 校准）
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --num-scenes 100
# 扩量 1000-5000（多卡分片：--start/--num-scenes；人体按 20% 场景开）
CUDA_VISIBLE_DEVICES=<卡A> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start 100 --num-scenes 1400 --human-frac 0.2
CUDA_VISIBLE_DEVICES=<卡B> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start 1500 --num-scenes 1400 --human-frac 0.2
# 窄带兼容子集（200 场景即可；双口径默认已同 solve 写出，无需单独跑）
```

成本表（ pilot 实测回填；本机 CPU 冒烟单 solve 0.2s@50 样本/6 快照仅供参考）：
每场景 = scenes(CPU 秒级) + env_pc(~10s CPU) + solveA(1 次) + [人体场景追加 T 次 solve]。
磁盘 ≈ 6-10 MB/场景（env_pc 2MB + radio ~4MB + 几何 ~1MB）→ 1000 场景 ≈ 6-10GB，5000 ≈ 30-50GB。
**QC**：每个 pilot 场景跑 `sanity_check`（9 项：env_pc/cir 形状/cir↔csi 恒等 1e-5/NMSE≈−SNR/FS
斜率≈−2（LOS≥20）/LOS 统计/轨迹自由空间/多普勒/sidecar）。

---

## 0. 速览（deprecated DLoc 布局速览，Phase PC 见上节）

| 项 | 值 |
|---|---|
| 生成环境 | 独立 conda env `sionna`（python 3.11 + `sionna-rt==2.2.0`），**xfi 环境不动** |
| 训练环境 | `xfi`（读缓存即可；缓存是纯 numpy 文件） |
| 场景配方 | MUSE-FM：10×10×3m、concrete 地板、brick 外墙、0-3 随机隔墙 + 0-2 圆柱（marble）；扩展：1-2 面分房隔墙（带门洞）+ 0-4 家具 |
| RF 配置 | WiFi 化：2.412 GHz、234 子载波 ×156.25 kHz ≈36.6 MHz（Nexmon/DLoc tone 结构，config 可换） |
| 天线 | AP = TX，2×2 UPA（4 天线，iso）；UE = RX 单天线 iso，高 1m；AP 高 2.5m，每场景 1-4 个 |
| 规模 | 每场景 50 个 UE（= MUSE-FM 每场景样本数）；pilot 100 场景，目标 2600 场景 |
| floorplan | **数据集里只有 PNG**（FLOORPLAN_SPEC 标准）；4 通道张量是从 PNG 派生的预处理 |
| 缓存格式 | 与 `data/dloc_cache` 逐位同构 → 现有 csi_jepa 预训练/下游代码直接消费 |
| GPU | 全程 GPU（Mitsuba/Dr.Jit CUDA）；估算见 §7，pilot 实测校准 |

---

## 1. 背景与 MUSE-FM 对照

MUSE-FM 用 Blender 人工建模 + Sionna 射追造了 2600 个室内场景 × 50 样本（13 万条），按
**场景**划分 2000/300/300。它的场景建模规范（论文 §IV-A-1）：

- 统一尺寸 **10m × 10m × 3m**；地板 concrete、墙 brick
- 房间内随机 **0-3 面内部 marble 隔墙**（长度与 3D 方向随机）+ **0-2 根 marble 圆柱**
- **BS 随机置于 (±4.75, ±4.75, 2.5m)**；UE 随机分布、高 1m；每场景 K=4 用户
- 模型输入 "scene graph"：俯视图离散化 **100×100 矩阵**（0=开放空间，2=BS 位置）
- 系统参数（Table II）：28 GHz、48 子载波 @1.8 kHz、BS 16×4 UPA、UE 单天线

**我们的替换**（均已获用户确认）：

| 维度 | MUSE-FM | 我们 |
|---|---|---|
| 频段 | 28 GHz mmWave | **2.4 GHz ISM**（WiFi 感知口径） |
| 子载波 | 48 @1.8kHz | **234 @156.25kHz**（对齐 DLoc/Nexmon，结构可换） |
| 天线 | BS 16×4 UPA | AP **2×2 UPA**（每 AP 8 行 CSI，与 DLoc 缓存同构） |
| 场景 | Blender 单间 | **程序化**（无需 Blender）单间配方 + 1-2 隔墙分房 + 门洞 + 家具 |
| 场景表示 | 100×100 scene graph | FLOORPLAN_SPEC §5b 4 通道张量（0.2m cell，更细） |
| 划分 | 按场景 2000/300/300 | 同纪律：场景级划分（`interp` / `holdout_scenes`） |

---

## 2. 环境安装（用户执行）

```bash
conda create -n sionna python=3.11 -y
conda run -n sionna pip install sionna-rt==2.2.0 matplotlib scipy
conda run -n sionna python -c "import sionna.rt; print(sionna.rt.__version__)"   # → 2.2.0
```

- `sionna-rt` 依赖 `mitsuba==3.9.1`、`drjit==1.5.0`、`numpy>=1.26`、`scipy>=1.14.1`，
  **自身不依赖 torch**（主包 `sionna` 才依赖 torch≥2.9.1——不要装主包，以免动 xfi）。
- import 时自动选 GPU variant `cuda_ad_mono_polarized`；无 GPU 时退回 LLVM CPU（慢）。
- 先 `pip install --dry-run sionna-rt==2.2.0` 看依赖解析再实装。
- 独立 env `sionna` 是用户 2026-11-13 新授权（此前约定只建 xfi；生成/训练环境分离）。

---

## 3. Sionna RT 六步工作流（学习核心）

Sionna = NVIDIA 的通信系统仿真库；`sionna-rt` 是其射追分包，建在 Mitsuba 3（渲染器）+
Dr.Jit（向量化自动微分）上，**路径求解、场计算、频响全部编译成 CUDA kernel 在 GPU 跑**。

### 3.1 场景 = Mitsuba XML + PLY

- 场景文件是 Mitsuba 3 XML，shape 用 `<shape type="ply">` 引用网格；材质是 `<bsdf>`。
- Sionna 约定：材质 id 写成 `mat-itu_concrete` / `itu_brick` 等，加载时 `process_xml`
  自动转成 ITU-R P.2040-4 射频材质（εr/σ 随频率自动更新）；
  或显式 `<bsdf type="itu-radio-material">`。18 种 ITU 材质：concrete/brick/marble/
  wood/metal/glass/plasterboard/...（默认板厚 0.1m）。
- 单位：**米**；Z 轴向上。薄墙假设：墙建模为单面几何 + 材质 `thickness` 承担穿透
  （refraction）；我们的墙盒厚度 = 材质 thickness = 0.1m，口径一致。
- 参考实例：`sionna-rt/src/sionna/rt/scenes/floor_wall/floor_wall.xml`。

### 3.2 六步（API 均已对 sionna-rt 2.2.0 源码核实）

```python
import mitsuba as mi
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

# (1) 加载场景
scene = load_scene('scene.xml')

# (2) 频率（ITU 材质 εr/σ 自动随动）
scene.frequency = 2.412e9

# (3) 天线阵（间距单位 = 波长倍数；pattern: iso|dipole|tr38901；pol: V|H|VH|cross）
scene.tx_array = PlanarArray(num_rows=2, num_cols=2, vertical_spacing=0.5,
                             horizontal_spacing=0.5, pattern='iso', polarization='V')
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                             horizontal_spacing=0.5, pattern='iso', polarization='V')

# (4) 收发机（一次可加多个；求解器算所有 TX-RX 对；位置米、Z 向上）
for i, ap in enumerate(aps):
    scene.add(Transmitter(name=f'ap{i}', position=mi.Point3f(ap.x, ap.y, ap.z)))
for i, ue in enumerate(ues):
    scene.add(Receiver(name=f'ue{i}', position=mi.Point3f(ue.x, ue.y, 1.0)))

# (5) 路径求解（LOS/镜面反射/折射/绕射可独立开关；批量生成必须 deterministic=True）
solver = PathSolver(deterministic=True)
paths = solver(scene, max_depth=3, samples_per_src=1_000_000,
               los=True, specular_reflection=True, refraction=True,
               diffraction=False, synthetic_array=True, seed=42)

# (6) CSI：每子载波频响 H(f)=Σ a_i^b e^{-j2πfτ_i}
#    shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_frequencies]
freqs = 2.412e9 + (np.arange(234) - 116.5) * 156.25e3
h_re, h_im = paths.cfr(frequencies=freqs, normalize_delays=False, out_type='numpy')
```

补充 API：
- `paths.cir(normalize_delays=..., out_type=...)` → 基带 CIR `(a, τ)`（τ 秒）；
  `paths.a / paths.tau / paths.doppler / paths.interactions`（逐路径交互类型，调试用）
- `RadioMapSolver()(scene, cell_size=(0.25,0.25), samples_per_tx=...,
  max_depth=...)` → `rm.path_gain`（每 TX 一张功率面图）。**仅可视化诊断，
  不进数据集、不做模型输入**（用户约定）。

### 3.3 官方教程（`sionna-rt/doc/source/tutorials.rst`）

Introduction（入门必读）/ Scene-Edit（改场景）/ Diffraction / Mobility / Radio-Maps /
Scattering / RCS。跑教程需要 jupyter：`conda run -n sionna pip install jupyter`。

---

## 4. 我们的管线

### 4.1 目录与数据流

```
sionna_gen/                     生成管线（sionna env 运行）
  config.py                     全部参数（GenConfig；yaml 逐键覆盖）
  scene_generator.py            场景 -> scene_<id>/{scene.xml, meshes/*.ply, metadata.json}
  floorplan_renderer.py         几何 -> FLOORPLAN_SPEC 标准 PNG（数据集资产）+ manifest.csv
  make_tensors.py               PNG -> tensors/scene_<id>.npz（预处理，非数据集资产）
  rt_runner.py                  GPU 射追 -> cache/CSI_*.npy + sidecar_*.npz（dloc 同构）
  build_cache.py                编排器（四阶段，断点续跑）
  sanity_check.py               单场景体检（pilot 前必过）

sionna_dataset/                 生成产物（仓库根；建议 .gitignore）
  scenes/scene_<id>/            场景源文件
  floorplan/SIONNA/scene_<id>/scene_<id>_ap<k>.png   # 数据集 floorplan（唯一形式）
  floorplan/SIONNA/manifest.csv                       # FLOORPLAN_SPEC §6 字段
  tensors/scene_<id>.npz        occ/unk/sdf (160,320) + ap 坐标（PNG 派生）
  cache/CSI_scene_<id>.npy      (50, 32, 234) float32 —— 与 dloc_cache 逐位同构
  cache/sidecar_scene_<id>.npz  labels/ap_centroids/valid_ap/n_ap/p1_stats
  preview/                      诊断图（radio map 等，不入数据集）
  rt_timing.csv                 实测耗时（校准 §7）

XRF55_HAR/csi_jepa/sionna_data.py   训练侧消费（xfi env；SionnaCorpus/SionnaRasterProvider）
XRF55_HAR/configs/sionna_jepa_fp.yaml        预训练主配置（raster）
XRF55_HAR/configs/sionna_jepa_fp_none.yaml   CSI-only 对照
```

数据流：**几何（scene_generator）→ PNG（renderer，数据集资产）→ 张量（make_tensors，
PNG 派生）+ CSI（rt_runner，几何→GPU 射追）→ 训练（csi_jepa 读 cache+tensors）**。
注意张量与 CSI 的几何来源不同（PNG vs scene.xml）但同源同坐标约定；renderer↔tensors
之间有**精确可逆**的采样映射（8/5 整数运算），零失配（sanity_check 可验证）。

### 4.2 命令

```bash
cd /home/ubuntu/users/wr/dev/X-Fi          # 仓库根（产物生成在 ./sionna_dataset）
nvidia-smi                                  # GPU 任务前检查

# 第 0 步：单场景体检（生成 + 渲染 + 射追 + 检查全跑一遍，~1 min）
conda run -n sionna python -m sionna_gen.sanity_check --radio-map \
    --compare-dloc data/dloc_cache

# 第 1 步：pilot 100 场景（四阶段全跑，估 5-20 min，见 §7）
conda run -n sionna python -m sionna_gen.build_cache --num-scenes 100

# 第 2 步：扩量到 2600 场景（pilot 通过后）
conda run -n sionna python -m sionna_gen.build_cache --start 100 --num-scenes 2500
```

断点续跑：各阶段产物已存在即 skip；`--overwrite` 强制重做；`--stages scenes floorplans`
可只跑 CPU 阶段。多 GPU：按 `--start/--num-scenes` 分片起多个进程（每卡一个）。

### 4.3 场景配方的随机化明细

| 元素 | 数量 | 随机量 |
|---|---|---|
| 外壳 | 1 | 尺寸默认 10×10×3m（config 可改）；floor concrete / 外墙 brick / 顶 concrete |
| 分房隔墙 | 1-2 | 轴对齐、位置 ∈[0.3,0.7]×边长；每墙 1-2 个门洞（宽 0.9m、高 2.1m，墙拆段+门楣） |
| 追加隔墙 | 0-3 | marble、长 1.5-4m、朝向任意角、贯通全高（MUSE-FM 配方） |
| 圆柱 | 0-2 | marble、半径 0.15-0.35m、贯通全高（MUSE-FM 配方） |
| 家具 | 0-4 | wood(80%)/metal、footprint 0.4-1.2×0.4-2.0m、高 0.5-0.9m |
| AP | 1-4（偏 4） | 近墙候选点（4 角+4 墙中点内缩 0.5m），两两距离 ≥ min(3m, 短边/3)，高 2.5m |
| UE | 50 | 开放空间拒绝采样（避障碍外扩 0.15m、外墙 0.2m），高 1m |
| 种子 | — | `seed = seed_base*1000003 + scene_id`（几何/布置/RT 同源确定性） |

放置纪律：追加物"先验 footprint、后提交"（被拒候选不留孤儿 mesh）；40 次放不下则少一件。

### 4.4 floorplan（数据集资产）规范

逐条执行 `floorplan_dataset/FLOORPLAN_SPEC.md`：SCALE 16px/m、画布 1024×464（64×29m 全局，
与 DLoc 同框）、内容左下角锚定 MARGIN=1m、语义调色板（free #ffffff / occ #1a1a1a /
unknown #c9c9cf、网格 #d9d9de 只画自由区、AP 红点 #dc2626 Ø0.5m 白描边）、**一图一 AP**
、PNG 内嵌 metadata + `manifest.csv`（§6 字段全）。门楣（z≥2.1m）不画——floorplan 是
1m 高接收视角的障碍图。`world_origin` 恒 (-1,-1)。

张量（预处理，非资产）：PNG → 精确逆采样 native 0.1m → ch1 occ（0.2m max-pool）/
ch2 SDF（native EDT、tanh(sdf/1m) mean-pool）/ ch3 unknown（画布补白 + >1m 启发式）/
ch4 AP 高斯（σ=0.5m，训练侧加载时按坐标现算）。数学与 `csi_jepa/floorplan.py` 的
DLocRasterBuilder 同款，张量形状 (160,320) 与 DLoc 完全一致 → SceneEncoder 可直接换语料。

### 4.5 缓存格式（与 dloc_cache 逐位同构）

| 项 | 值 |
|---|---|
| `CSI_scene_<id>.npy` | (50, 32, 234) float32；每 AP 8 行 = [ant0_re, ant0_im, ..., ant3_im]（2×2 UPA）；AP 不足 4 补零 |
| `sidecar_scene_<id>.npz` | labels (50,2) 米；n_ap；valid_ap (4,)；ap_centroids (4,2) 米；p1_stats (4,9)（前 90% 样本统计，零泄漏纪律同 DLoc） |
| 子载波频点 | f_k = 2.412GHz + (k − 116.5)×156.25kHz |
| 相位口径 | `normalize_delays=False`：保留绝对到达相位（TOF 信息，对齐真实 WiFi CSI）；config 可 A/B |

---

## 5. GPU 加速

- **默认就是 GPU**：Mitsuba/Dr.Jit 把路径生成、场计算、cfr 编译成 CUDA kernel；
  CPU 只承担场景文件生成、PNG 光栅化和落盘。
- 验证：脚本打印 `mitsuba variant`，`cuda_ad_mono_polarized` = GPU 生效。
- 并行：单场景内 50 UE×4 AP 已批处理；场景间按 GPU 分片（`--start/--num-scenes`）近线性加速。
- 路径数上不去/太慢的旋钮：`samples_per_src`（默认 1e6）、`max_depth`（默认 3）、
  `diffraction`（默认关——开一阶绕射耗时 ×3-10，留作对照实验）。

## 6. 存储预算

每样本 32×234×4B ≈ 75KB；50 样本/场景 ≈ 3.75MB；**2600 场景 ≈ 10GB CSI**，
张量 ~0.6MB/场景（compressed ≈1.6GB 总），场景 XML+PLY ~1MB/场景。pilot 100 场景 <0.5GB。
磁盘紧张时把 `sionna_dataset` 放 /mnt/DataDrive164 并软链回仓库根（不 move 原始数据的
约定不变，这是新产物）。

## 7. 时间估算（单卡 RTX 4090 级；以 rt_timing.csv 实测为准）

| 阶段 | 设备 | 单场景 | pilot 100 | full 2600 |
|---|---|---|---|---|
| 场景生成（XML+PLY） | CPU | <1s | ~1min | ~30min |
| floorplan PNG + 张量 | CPU | <1s | ~2min | ~45min |
| RT solve（无绕射，depth 3） | GPU | ~1-3s | ~3-5min | ~1-2.5h |
| RT solve（+一阶绕射） | GPU | ~3-10s | ~8-17min | ~3-8h |
| cfr + 写缓存 | GPU+盘 | <0.5s | ~1min | ~20min |

sanity_check 会打印单场景实测并外推 pilot 耗时——把实测值回填到本表。

---

## 8. 常见坑（核对清单）

1. **确定性**：PathSolver 默认非确定（固定 seed 也不保）；批量生成必须 `deterministic=True`。
2. **单位与轴向**：米；Z 向上；PNG/张量 **行 0 = y_max**（DLoc 掩码同约定）。
3. **薄墙模型**：墙=单面几何+材质 thickness（穿透/折射按 slab 算）；墙盒厚=thickness=0.1m。
4. **normalize_delays**：False 才保留绝对 TOF 相位（对齐真实 CSI）；True 只留相对时延。
5. **天线间距单位是波长**（0.5 = half-wavelength），不是米。
6. **子载波频点自己构造**（RT 只给 CIR/CFR 接口）：居中口径 f_k = fc + (k−(n−1)/2)·Δf。
7. **材质 thickness 与几何厚度一致**，否则折射损耗口径错。
8. **PNG 是数据集资产、张量是派生**——不要把张量当数据集 floorplan（用户约定）。
9. **Radio Map 仅可视化**（用户约定），不要写进数据集或喂模型。
10. **划分纪律**：同一场景的样本永不跨 train/test（合成数据重演 D53 的"相邻样本跨 split"
    教训没有借口——UE 虽 iid，仍按场景或按场景内固定 seed 切）。

---

## 9. 训练侧接入（xfi env）

`csi_jepa/sionna_data.py` 已提供与 DLoc 接口对齐的 `SionnaCorpus`（动态场景数；
`interp` / `holdout_scenes` 划分）与 `SionnaRasterProvider`（PNG 派生张量惰性加载 +
AP 高斯现算，接口同 `DLocRasterBuilder`）；Dataset/collate 直接复用 dloc_data（缓存同构）。

待合并后对 `dloc_csi_pretrain_fp.py` / `dloc_csi_downstream_fp.py` 施加的接入 diff
（示意，语料与栅格构造处各一处）：

```python
from csi_jepa.sionna_data import corpus_factory, SionnaRasterProvider

# 语料（原: corpus = DLocCorpus(cfg['data']['cache_root'])）
corpus = corpus_factory(cfg['data'])

# 栅格（原: builder = DLocRasterBuilder(scen_all, canvas=...); raster_gpu = np.stack(
#          [builder.build(sc, side[sc][...], ...)] ...)）
if cfg['data'].get('corpus') == 'sionna':
    builder = SionnaRasterProvider(cfg['data']['tensors_root'], corpus.scenarios)
    raster_gpu = torch.from_numpy(np.stack([builder.build(sc)
                                            for sc in scen_all])).float().to(device)
else:
    builder = DLocRasterBuilder(scen_all, canvas=model_cfg.get('canvas', 'global'))
    raster_gpu = torch.from_numpy(np.stack([
        builder.build(sc, side[sc]['ap_centroids'], side[sc]['valid_ap'])
        for sc in scen_all])).float().to(device)
```

预训练命令（pilot 通过后；先 nvidia-smi）：

```bash
cd XRF55_HAR
CUDA_VISIBLE_DEVICES=0 conda run -n xfi python dloc_csi_pretrain_fp.py \
    --config configs/sionna_jepa_fp.yaml --run-name SI_raster_interp
CUDA_VISIBLE_DEVICES=0 conda run -n xfi python dloc_csi_pretrain_fp.py \
    --config configs/sionna_jepa_fp_none.yaml --run-name SI_none_interp
```

Pilot 训练 gate（预期信号）：合成数据 interp 划分上 `fp_scratch` probe 的 mean err 随
label-fraction 10%→100% 单调下降（定位有信号）；`holdout_scenes` 划分上
raster vs none 的零样本差距 = "几何先验在多场景下是否可迁移"的直接读数——这正是
本数据集存在的科学问题。

---

## 10. 决策点与待办

- [ ] 用户：建 env `sionna` + 实装 `sionna-rt==2.2.0`（先 `--dry-run` 看依赖解析）
- [ ] sanity_check 通过（斜率≈−2、量级与 DLoc 同数量级、PNG 目检）
- [ ] pilot 100 场景 → xfi 下游冒烟 → 扩量决策（2600 场景，多 GPU 分片可选）
- [ ] 训练脚本接入 diff 应用（合并本分支后在 floorplan 工作副本执行）
- [ ] 实测耗时回填 §7；PROJECT_LOG §12 誊录（本文件 §11 为草稿）

## 11. 运行记录（待誊入 PROJECT_LOG.md §12）

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-11-13 | plan 批准：Sionna 合成数据集（MUSE-FM 方法复现 + WiFi 化）；用户决策：独立 env `sionna`、floorplan 只能是 PNG、Radio Map 仅可视化、维护本文档 | sionna_gen/ 管线 + 训练侧 sionna_data.py + 本文档落地 |
| （待填） | sanity_check scene_0 | （斜率/量级/耗时） |
| （待填） | pilot 100 场景 | （实测均值、下游 fp_scratch 冒烟） |
