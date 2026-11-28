# SIONNA_PIPELINE.md — Sionna 合成 CSI 数据集：管线手册 + 运行手册（唯一维护文档）

> 目的：用 Sionna RT 射线追踪大规模合成多场景室内 WiFi CSI 数据集，解决 DLoc 试点暴露的
> "2 栋楼 8 场景不够多样" 问题（PROJECT_LOG D53/D54：跨楼零样本 −34% 反转）。方法上复现
> **MUSE-FM**（arXiv 2509.01967，PDF 在 `paper/muse-fm.pdf`）的数据生成流程，并做 WiFi 化
> 适配。本文既是**学习文档**（Sionna 怎么用）也是**管线手册 + 运行手册**（我们怎么生成）。
>
> **本文档自包含**：原《SIONNA_HANDOFF.md》（生成机移交文档）已于 2026-11-28 并入下方
> Phase PC 节并删除，此后**只维护本文档**。给 coding agent 的话：生成机上的唯一任务是
> 搭环境 → 冒烟 → 体检 → pilot → 批量 1000–5000 场景 + 质检回传；**不要在生成机上训练
> 任何模型**（训练在训练机的 `xfi` 环境做）；所有命令都在仓库根目录（`X-Fi/`）执行；
> 开始前通读 §Phase PC 全节。管线代码行为/参数语义读 `sionna_gen/config.py`（GenConfig
> 注释）；消费层契约读 `XRF55_HAR/csi_jepa/pc_data.py` 模块 docstring；项目背景与判门读
> PROJECT_LOG §12。

---

## ⚡ Phase PC：rooms10 对齐生成管线（主线）——运行手册（2026-11-28 实测回填）

**目标对齐 `3D_dataset/dataset_10rooms`**（用户决策）：3.5GHz、**单 TX-SISO**（TX 近墙角高
2.12m，UE 单天线高 1.25m）、**UE 轨迹 480 快照 @60Hz（恒 8s 时窗）**（解析轨迹替代 Isaac
回放轨迹，`trajectory.py`：直线/折线/平滑随机游走，自由空间避障）、**可选人体遮挡**
（`human_occluder`：8 段轮廓 mesh 在场景内、逐快照 solve、yaw 跟随轨迹切向；轮廓参数与
10rooms run.json 逐值同源）、输出与 10rooms **逐键同构**（csi.npz/link_summary.csv/
run.json）+ 新增 `env_pc.npz`（mesh 表面采样静态环境点云，替代 Isaac lidar）。

### PC.0 状态一览（2026-11-28）

- **pilot 100 场景已完成并通过全套 QC**（RTX 4090×2，17 分钟；判据与实测见 §PC.3/§PC.4）
- 下一步：批量 1000–5000 场景（§PC.3 分片命令；实测 1000 场景 ≈3h、5000 ≈15h 双卡；
  规模与参数待用户确认）
- **当前生成口径**（本日多轮修复与决策后的最终形态）：房间 w,d 逐场景 U(7,15)m、h=3m；
  家具 80% wood；LOS/人体遮挡为几何判定标签；solveA 按 `rt_rx_chunk=64` 分块。
  **此前任何旧口径产物不可比，需重生成。**
- Phase PC 代码主体在 `3D` 分支（commit ff793c3）；2026-11-28 的 bugfix/口径变更
  （rt_runner/config/scene_generator/trajectory/sanity_check）落地后待提交固化。

### PC.1 环境搭建与自检（每步必须过，不过不要继续）

```bash
conda create -n sionna python=3.11 -y
conda run -n sionna pip install sionna-rt==2.2.0 matplotlib scipy
conda install -n sionna -c conda-forge -y "libllvm17>=17.0.6,<18"
```

**自检 ① GPU 兼容（关键）**——sionna-rt 2.2 的 CUDA 后端要求 **GPU compute capability
≥ 7.5**（Turing 及以上：RTX 20xx/T4/A100/30xx/40xx 等）。**V100 = 7.0 不支持**（会被静默
跳过）：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
conda run -n sionna python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print('variant OK:', mi.variant())"
# 期望输出：variant OK: cuda_ad_mono_polarized
```
若报 "no compatible CUDA devices" → 这台机器不能做 GPU 生成，换机器，不要用 CPU 退路量产。

**自检 ② 求解速度**——§PC.3 冒烟跑完后看日志末行的 solve 均值：GPU 正常时单场景（极小
冒烟参数）应亚秒级；若慢 10 倍以上，`mi.variant()` 打印的八成不是 cuda_*，回自检 ①。

**遗留坑（不要重蹈）**：① 系统 `libLLVM-12` 会被 drjit 拒绝、且初始化失败曾表现为
import 无限空转（100% CPU 不报错）——装 `libllvm17` 后消失；② 训练机（V100）的冒烟命令
带 `CUDA_VISIBLE_DEVICES=""` 和 `DRJIT_LIBLLVM_PATH=...`，那是强制 CPU 的退路（慢
~10-50×，只够冒烟），**不要带到生成机**；选卡只用 `CUDA_VISIBLE_DEVICES=<N>`。

### PC.2 管线五分钟理解

```
python -m sionna_gen.build_cache  的四阶段（各阶段产物已存在即 skip，可断点续跑）：
  scenes   scene_generator：程序化房间（外壳 w,d 逐场景 U(7,15)m、h=3m；1-2 面贯通带
           门洞隔墙 + 0-3 隔墙 + 0-2 圆柱 + 0-4 家具，ITU 材质）→
           scenes/scene_<id>/{scene.xml, meshes/*.ply, metadata.json（含 480 快照解析
           轨迹）, human.ply}
  floorplans QC 可选（floorplan PNG，不进消费路径）
  env_pc   pointcloud_exporter：mesh 均匀表面采样 → dataset/scene_<id>/env_pc.npz
           （xyz 40960 点 + FPS 2048 + bbox）
  rt       rt_runner.rooms10：单 TX-SISO × 轨迹 480 快照 → dataset/scene_<id>/radio/
           {csi.npz, csi_<另一口径>.npz, link_summary.csv, run.json} + sidecar_pc.npz
```

| 项 | 值（默认，GenConfig 单点真相在 `sionna_gen/config.py`，yaml 可覆盖、未知键报错） |
|---|---|
| RF | 3.5GHz、256 子载波；**双口径**：narrow = 30kHz SCS（7.68MHz，10rooms 同款，主口径）+ wide ≈390.625kHz（100MHz）；**一次 solve 双写两个 csi.npz**（cir 共享） |
| 天线 | 单 TX（近墙角候选点采样、高 2.12m、iso 单天线 V 极化）→ 单 UE RX（高 1.25m） |
| 轨迹 | 480 快照 @60Hz（恒 8s 时窗），解析直线/折线/平滑随机游走（自由空间避障，名义 0.8 m/s、实际 0.29–0.84） |
| 房间尺寸 | **w,d 逐场景独立 U(7,15)m、h=3m 固定**（用户决策 2026-11-28；`room_size_range_m` 置 null 回固定 10×10 模式） |
| 材质 | 类别固定映射：地板/天花 concrete、外墙 brick、隔墙/隔断/圆柱 marble、家具 wood 80%/metal 20%、人体 wood 近似 |
| 人体 | **默认开**（`human_occluder: true`）：8 段轮廓 mesh 注入场景、逐快照 solve、yaw 跟随轨迹切向；`--human-frac 0.2` = 按场景种子 20% 开（跨分片确定一致，见陷阱 17） |
| RT | PathSolver(deterministic=True)、max_depth=3、samples_per_src=1e6、LOS+镜面+折射开、漫射/绕射关、`rt_rx_chunk=64` 分块（陷阱 13）、seed = 3407*1000003 + scene_id |
| LOS 标签 | **几何视线判定**（2026-11-28 重定义，陷阱 5）：`los`/`los_without_human` = 视线相交测试；`los_blocked_by_human` = 人体躯干轴心穿越 |
| 点云 | 40960 表面采样点 + FPS 2048（模型输入规模） |
| **不含 cir** | **cir 不落盘（用户决策）**：系数/时延仅在内存用于 CSI 合成与统计；逐快照信道元数据全在 `link_summary.csv` |

**产物清单**（每个场景）：
```
sionna_dataset/
├── scenes/scene_<id>/                    # 几何源（训练侧"干净几何消融"要用，勿删）
│   ├── scene.xml  meshes/*.ply  metadata.json  human.ply
├── dataset/scene_<id>/                   # ★ 消费层入口（最终交付物）
│   ├── env_pc.npz                        # xyz(40960,3) f32 / xyz_fps(2048,3) / bbox(2,3) / meta json
│   ├── sidecar_pc.npz                    # p1_stats(1,8)（逐 TX×8 频带 mean|CSI|，前 90% 快照）
│   │                                     # + p2_stats(3)（n_paths/first_delay_ns/mean_gain_db）
│   └── radio/
│       ├── csi.npz                       # 主口径（缺省 narrow）：ideal_csi(T,256)c64、
│       │                                 # estimated_csi(4,T,256)c64（SNR 0/10/20/30dB 档）、
│       │                                 # subcarrier_hz、noise_variance、nmse_db、metadata_json
│       ├── csi_wide.npz                  # 另一口径（主口径为 narrow 时；反之 csi_narrow.npz）
│       ├── link_summary.csv              # 逐快照：时间戳/ue_xyz/los/num_paths/first_path_delay_ns/
│       │                                 # sum_path_gain_db/human 位姿/los_without_human/
│       │                                 # los_blocked_by_human
│       └── run.json                      # 仿真元数据（10rooms 逐键同构 + bandwidth/human/trajectory）
├── rt_timing.csv                         # 每场景 solve/后处理耗时（成本校准用）
└── preview/                              # 仅可视化（floorplan PNG 等，不入数据集）
```

### PC.3 执行序列（按顺序，每步过了再下一步）

**① 冒烟（~1 min，产物弃置）**——极小参数（6 快照 × 50 samples、人体关、写 /tmp），
只验证代码路径；期望日志末行 `实测单场景 solve 均值 ...s`、exit=0：
```bash
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --config sionna_gen/smoke_rooms10.yaml --num-scenes 1
```
**② 体检（对冒烟产物）**——期望 `== sanity 完成：全部通过 ==`（自由空间斜率项在冒烟
规模下会 info 跳过，正常）；再跑人体开变体验证人体路径：
```bash
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.sanity_check --config sionna_gen/smoke_rooms10.yaml --scene-id 0
CUDA_VISIBLE_DEVICES=<N> conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --config sionna_gen/smoke_rooms10_human.yaml --num-scenes 1
# 期望日志含 "human-blocked N"（N≥0 合法，几何判据下人体不一定挡到视线）
```
**③ pilot 100 场景（正式参数）**——✅ 2026-11-28 已完成（双卡分片 17 min，QC 全绿）：
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start   0 --num-scenes 50 --stages scenes env_pc rt
CUDA_VISIBLE_DEVICES=1 conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start 50 --num-scenes 50 --stages scenes env_pc rt
```
说明：不带 `--human-frac` 时 `human_occluder` 默认全开（最重路径的成本校准场景）；
跳过 `floorplans` 阶段以避免多进程并发写 manifest.csv（它本就是 QC 可选、不在消费路径，
需要时单进程补跑 `--stages floorplans`）。

**pilot 通过判据**：sanity 全 PASS；rt_timing.csv 的 solve 均值稳定（max/min <2×）；
抽查 csi.npz 的 nmse_db ≈ [0,−10,−20,−30]；抽场景确认 `link_summary.csv` 行数=480 且
LOS 标签与 metadata 几何一致（los 恒 0 合法——TX 与轨迹被隔墙分隔的锁孔场景；
几何判定法见陷阱 5）。

**④ 批量 1000–5000 场景（多卡分片；断点安全，中断直接重跑同命令续传）**：
```bash
# 例：2 卡分片 4000 场景（各卡 range 不相交；human-frac 各分片保持一致——场景级种子
# 决定，同一场景 id 在任何分片/重跑下人体开关都相同）
CUDA_VISIBLE_DEVICES=0 conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start    0 --num-scenes 2000 --human-frac 0.2 \
  --stages scenes env_pc rt
CUDA_VISIBLE_DEVICES=1 conda run -n sionna --no-capture-output \
  python -u -m sionna_gen.build_cache --start 2000 --num-scenes 2000 --human-frac 0.2 \
  --stages scenes env_pc rt
# 窄带兼容子集无需单独跑（双口径默认同 solve 双写）
```
**⑤ 批量 QC（生成完做一次全量体检扫描）**：
```bash
conda run -n sionna python - <<'EOF'
# 全量轻量 QC：逐场景检查产物齐全 + 行数 + NaN + nmse 口径（不重新 solve）
import glob, json, os
import numpy as np
bad = []
dirs = sorted(glob.glob('sionna_dataset/dataset/scene_*'))
for d in dirs:
    r = os.path.join(d, 'radio')
    try:
        need = ['csi.npz', 'link_summary.csv', 'run.json']
        if any(not os.path.exists(os.path.join(r, f)) for f in need) \
           or not os.path.exists(os.path.join(d, 'env_pc.npz')) \
           or not os.path.exists(os.path.join(d, 'sidecar_pc.npz')):
            bad.append((d, 'missing')); continue
        with np.load(os.path.join(r, 'csi.npz')) as f:
            assert np.isfinite(f['estimated_csi']).all()
            assert f['estimated_csi'].shape[1:] == (480, 256)
        with open(os.path.join(r, 'link_summary.csv')) as f:
            assert sum(1 for _ in f) - 1 == 480
    except Exception as e:
        bad.append((d, repr(e)[:80]))
print(f'{len(dirs)} scenes checked, {len(bad)} bad')
for b in bad[:20]:
    print(' BAD:', b)
EOF
```

### PC.4 成本（pilot 100 实测回填 @4090×2，2026-11-28）

| 项 | 估算 | 实测 |
|---|---|---|
| scenes 阶段（CPU） | 秒级/场景 | ~1s/场景 |
| env_pc 阶段（CPU） | ~10s/场景 | ~1–2s/场景 |
| rt solveA（人体关） | pilot 实测 | **1.08±0.10s**（480RX 分块 8×64@1e6；max/min 1.5x） |
| rt 人体场景（+480 次逐快照 solve） | 估 5–15 min/场景 @GPU | **实测 ≈16.2±1.0s（human-loop+post）；单场景全程 ≈21s，100 场景双卡 17min**（原估算高估 ~20×） |
| 磁盘 | ~6–10 MB/场景 → 1000 ≈ 6–10GB，5000 ≈ 30–50GB | dataset ≈9MB/场景 + 几何 ≈0.16MB/场景（区间内） |

推算：**1000 场景 ≈3h、5000 场景 ≈15h**（双卡）。调参入口（`--config <yaml>` 覆盖
GenConfig，未知键报错）：`samples_per_src`（默认 1e6）、`rt_rx_chunk`（solveA 分块，
显存吃紧调小）、`n_snapshots`（480）、`traj_speed_m_s`、`pc_full_points/pc_fps_points`、
`out_root`。**正式库保持默认值**，与 10rooms 口径可比。
GPU 利用率低（峰值偶到 50%、常态 <10%）是该负载的正常形态：短促 GPU solve 与 CPU 段
（逐快照 XML 写盘/mitsuba 解析/drjit 读回同步/numpy 合成/写盘）交替；~14GB 常驻显存是
Dr.Jit 分配缓存，不代表持续计算。

### PC.5 已知事实与陷阱（合并清单，务必读完；标"已修复"的为 2026-11-28 落地）

环境/后端：
1. **GPU 计算能力 ≥7.5**：V100（7.0）被静默跳过 → 自检 ① 必须先过。
2. **LLVM**：drjit 需要 `libllvm17`（15 编不了 fmaximum 指令；12 被拒）。缺 LLVM 时
   症状可能是 import 无限空转而非报错——先装再排障。
3. **conda run 缓冲**：始终用 `--no-capture-output` + `python -u`，否则 tqdm/日志看不到
   实时进度。

数据口径（用户决策，勿擅改）：
4. **cir 不落盘**（2026-11-27）：csi 来自频域合成（内存系数），link_summary 是逐快照
   标量元数据——两者数值"对不上"是正常的，本就是两路。
5. **LOS/人体遮挡标签为几何判定**（2026-11-28 重定义）：`los`/`los_without_human` =
   2D 线段-凸多边形精确相交 + z 区间重叠（家具 ≤0.9m、门楣 ≥2.1m 按高度天然排除，只有
   贯通全高的隔墙/隔断/圆柱参与）；`los_blocked_by_human` = 视线穿过人体躯干核心
   （`human_sight_core_frac` × min(横向,前后)半径——10rooms 口径 UE 本体在人体轮廓内
   （0.14m 旁偏 < 肩宽 0.21m），"线段触及人体"会误判全部视线）。旧实现 min-tau≈d/c
   （2ns 容差）会把穿墙折射（墙附加时延 ~0.4ns）与掠射反射误标 LOS，pilot 实测 37%。
   link_summary 三列同名同构，语义以此为准。
6. **人体实现**：sionna-rt 2.2 的 `paths.interactions` 只给物体索引（不给路径折点坐标），
   人体遮挡 = **场景内 mesh 逐快照 solve**（人体开时该场景 solve 成本 ≈ 1+480 次）。
   `run.json` 的 `human_blockage.blocked_snapshots` 与 link_summary 的
   `los_blocked_by_human` 列是遮挡是否生效的第一判据。
7. **多普勒**：`path_doppler_hz` 不在本数据集口径里（到达向不可得），link_summary 无此
   列，run.json 有说明——不要"补"。
8. **normalize_delays=False** 才保留绝对到达相位（TOF 信息）；True 只留相对时延。
9. **子载波频点自己构造**（居中式 f_k = fc + (k−(n−1)/2)·Δf）；**天线间距单位是波长**
   （0.5 = 半波长）；**材质 thickness 与几何厚度一致**（0.1m），否则折射损耗口径错。

管线行为：
10. **确定性**：PathSolver `deterministic=True` + 每场景独立 seed；同 scene_id 重跑产物
    逐位一致；各阶段产物已存在即 skip，`--overwrite` 才强制重做，断点续跑安全。
11. **PlanarArray 必须关键字传参**（位置参数会 TypeError）。
12. **`--bandwidth` 只切主口径**，另一口径自动同 solve 写出（勿改 cfg.delta_f_hz 来
    "切换"，会污染另一口径——已修复为 preset 查表，但别绕过）。
13. **solveA 显存 ∝ rx 数 × samples_per_src**（480 全批量 @1e6 在 24GB 卡实测 OOM）：
    按 `rt_rx_chunk`（默认 64）分块，deterministic+seed 下分块与全批量**逐位等价**
    （冒烟 bitwise 校验过）；块内逐 chunk 先 `_truncate` 到 26 槽再拼接。
14. **人体 solve 必须传含人体 mesh 的场景对象**（曾因闭包误用无人体场景致 blocked 恒 0、
    人体场景 CSI 与人体关逐位相同——已修复 `_solve(sc_i, 1)` 并以 5 场景验证）。
15. **人体内联 bsdf id 必须唯一**（曾与场景 wood 家具重名 → Mitsuba duplicate ID 崩溃；
    已修复：id 加 `_human` 后缀）。
16. **已修复的历史口径 bug**：家具材质比例曾反转（79% metal，文档口径 80% wood）；
    折线轨迹顶点曾恒 3（现为 3-4 顶点 = 2-3 段）。**旧代码生成的场景几何/CSI 与现版
    不可比，需重生成。**
17. **human-frac 按 scene id 分带**：判据 `(seed % 1000) < frac*1000` 且
    `seed = 3407*1000003 + id` → id 0-778 全关、779-979 全开（每 1000 连号恰 20%）。
    按 1000 一片分片时占比正确；取任意连续子集时人体占比可能是 0% 或 100%。
18. **房间尺寸（2026-11-28 用户决策）**：`room_size_range_m = (7.0, 15.0)`——w,d 逐场景
    独立 U(7,15)m、h 固定 3m（yaml 置 null 可回固定 10×10 模式）。

数据集纪律（沿用旧 §8，仍有效）：
19. **PNG 是数据集资产、张量是派生**——不要把张量当数据集 floorplan（rooms10 布局中
    floorplan 仅 QC 可选）。
20. **Radio Map 仅可视化**（用户约定），不要写进数据集或喂模型。
21. **划分纪律**：同一场景的样本永不跨 train/test（合成数据重演 D53"相邻样本跨 split"
    教训没有借口）。

### PC.6 交付回传

生成 + QC 全部通过后，回传训练机：

```bash
# ① 必交付：dataset 层（消费层入口）+ rt_timing.csv
tar czf sionna_dataset.tar.gz -C sionna_dataset dataset rt_timing.csv
# ② 建议一并交付：scenes/（几何源——训练侧"干净几何 vs 传感几何"消融要用）
tar czf sionna_scenes.tar.gz -C sionna_dataset scenes
# ③ 校验：文件数与字节数清单
find sionna_dataset/dataset -name '*.npz' | wc -l    # 期望 = 场景数 × 4（env_pc+sidecar+csi×2 口径）
find sionna_dataset/dataset -name 'link_summary.csv' | wc -l   # 期望 = 场景数
du -sh sionna_dataset/*
```

训练机消费方式：`pc_csi_pretrain.py --sionna-root <解压路径>/dataset`（与 rooms10 数据
混合或单独训练均可；消费层只读 `csi.npz + link_summary.csv + sidecar_pc.npz +
env_pc.npz`，契约见 `XRF55_HAR/csi_jepa/pc_data.py`）。

**回报内容**：pilot/正式的 rt_timing.csv 均值、QC 的 bad 场景清单（应为 0）、最终场景数、
人体场景占比（`grep -l human_blockage dataset/*/radio/run.json | wc -l`）。

---

## 0. 速览（deprecated DLoc 布局速览，仅留档；主线见 §Phase PC）

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
| 频段 | 28 GHz mmWave | **3.5 GHz**（Phase PC rooms10 口径；旧 DLoc 布局为 2.4 GHz ISM） |
| 子载波 | 48 @1.8kHz | **256 @30kHz**（10rooms 同款主口径 + 100MHz wide 副口径） |
| 天线 | BS 16×4 UPA | **单 TX-SISO**（TX 高 2.12m，UE 高 1.25m） |
| 场景 | Blender 单间 | **程序化**（无需 Blender）：外壳 w,d∈U(7,15)m/h=3m + 1-2 面贯通带门洞隔墙 + 0-3 隔墙 + 0-2 圆柱 + 0-4 家具 |
| UE | 每场景 50 静态点 | **480 快照轨迹 @60Hz**（直线/折线/平滑随机游走） |
| 划分 | 按场景 2000/300/300 | 同纪律：场景级划分（`interp` / `loso`） |

---

## 2. 环境安装（背景阅读；执行以 §PC.1 的自检序列为准）

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

## 4. 我们的管线（deprecated DLoc 布局，留档；rooms10 主线见 §Phase PC）

### 4.1 目录与数据流（deprecated）

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

### 4.2 命令（deprecated）

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

### 4.3 场景配方的随机化明细（deprecated DLoc 布局）

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

### 4.4 floorplan（数据集资产）规范（deprecated；rooms10 中仅 QC 可选）

逐条执行 `floorplan_dataset/FLOORPLAN_SPEC.md`：SCALE 16px/m、画布 1024×464（64×29m 全局，
与 DLoc 同框）、内容左下角锚定 MARGIN=1m、语义调色板（free #ffffff / occ #1a1a1a /
unknown #c9c9cf、网格 #d9d9de 只画自由区、AP 红点 #dc2626 Ø0.5m 白描边）、**一图一 AP**
、PNG 内嵌 metadata + `manifest.csv`（§6 字段全）。门楣（z≥2.1m）不画——floorplan 是
1m 高接收视角的障碍图。`world_origin` 恒 (-1,-1)。

张量（预处理，非资产）：PNG → 精确逆采样 native 0.1m → ch1 occ（0.2m max-pool）/
ch2 SDF（native EDT、tanh(sdf/1m) mean-pool）/ ch3 unknown（画布补白 + >1m 启发式）/
ch4 AP 高斯（σ=0.5m，训练侧加载时按坐标现算）。数学与 `csi_jepa/floorplan.py` 的
DLocRasterBuilder 同款，张量形状 (160,320) 与 DLoc 完全一致 → SceneEncoder 可直接换语料。

### 4.5 缓存格式（与 dloc_cache 逐位同构，deprecated）

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
- 并行：单场景内多 UE/快照已批处理（rooms10：solveA 分块批 + 人体逐快照）；场景间按
  GPU 分片（`--start/--num-scenes`）近线性加速。
- 路径数上不去/太慢的旋钮：`samples_per_src`（默认 1e6）、`max_depth`（默认 3）、
  `diffraction`（默认关——开一阶绕射耗时 ×3-10，留作对照实验）、`rt_rx_chunk`
  （solveA 分块，显存吃紧调小）。

---

## 6. 存储预算（deprecated DLoc 估算；rooms10 实测见 §PC.4）

每样本 32×234×4B ≈ 75KB；50 样本/场景 ≈ 3.75MB；**2600 场景 ≈ 10GB CSI**，
张量 ~0.6MB/场景（compressed ≈1.6GB 总），场景 XML+PLY ~1MB/场景。pilot 100 场景 <0.5GB。
磁盘紧张时把 `sionna_dataset` 放大容量盘并软链回仓库根（不 move 原始数据的约定不变，
这是新产物）。

---

## 7. 时间估算（deprecated DLoc 布局估算；rooms10 实测见 §PC.4）

| 阶段 | 设备 | 单场景 | pilot 100 | full 2600 |
|---|---|---|---|---|
| 场景生成（XML+PLY） | CPU | <1s | ~1min | ~30min |
| floorplan PNG + 张量 | CPU | <1s | ~2min | ~45min |
| RT solve（无绕射，depth 3） | GPU | ~1-3s | ~3-5min | ~1-2.5h |
| RT solve（+一阶绕射） | GPU | ~3-10s | ~8-17min | ~3-8h |
| cfr + 写缓存 | GPU+盘 | <0.5s | ~1min | ~20min |

sanity_check 会打印单场景实测并外推 pilot 耗时——把实测值回填到本表。

---

## 8. 常见坑（DLoc 时期清单，留档；主线陷阱以 §PC.5 为准）

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

## 9. 训练侧接入（deprecated DLoc 接入方式，留档）

> Phase PC 主线的训练侧消费 = `csi_jepa/pc_data.py`（PCSceneCorpus/PCDataset，
> rooms10 与 sionna 两库同构目录一套代码；CSI 统一 (T,2,256) re/im；
> interp/loso 划分），见 PROJECT_LOG §12.3。

`csi_jepa/sionna_data.py` 提供与 DLoc 接口对齐的 `SionnaCorpus`（动态场景数；
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

- [x] 用户：建 env `sionna` + 实装 `sionna-rt==2.2.0`（生成机 4090×2，2026-11-28 就绪）
- [x] sanity_check 通过（pilot v4：4/4 场景全 PASS，NMSE/轨迹/标签全绿）
- [x] pilot 100 场景（2026-11-28 完成，双卡 17 min，全套 QC 通过，实测回填 §PC.4）
- [ ] 批量 1000–5000 场景启动（规模与参数待用户确认；实测 1000 ≈3h、5000 ≈15h 双卡）
- [ ] Phase PC bugfix/口径变更提交 git 固化（rt_runner/config/scene_generator/
      trajectory/sanity_check + 本文档）
- [ ] PROJECT_LOG §12 誊录（生成机实测 + bug 修复清单 + pilot 记录）
- [ ] floorplan QC PNG 补渲染（可选，单进程 `--stages floorplans`）
- [ ] 训练机侧：pilot 数据回传后 pc_csi_pretrain 冒烟（§PC.6 消费方式）

## 11. 运行记录

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-11-13 | plan 批准：Sionna 合成数据集（MUSE-FM 方法复现 + WiFi 化）；用户决策：独立 env `sionna`、floorplan 只能是 PNG、Radio Map 仅可视化、维护本文档 | sionna_gen/ 管线 + 训练侧 sionna_data.py + 本文档落地 |
| 2026-11-27 | Phase PC 转向 rooms10 对齐（PROJECT_LOG §12）；训练机 V100 无兼容 GPU，CPU llvm 冒烟 5s/场景全链路通过 | 布局切换 + 轨迹/人体/点云/双带宽落地 |
| 2026-11-28 | 生成机（4090×2）pilot 执行：修复 6 个 bug——①solveA 480RX 全批量 OOM→`rt_rx_chunk=64` 分块（bitwise 等价验证）②人体 solve 闭包误用无人体场景（blocked 恒 0）③家具材质 wood/metal 比例反转（79% metal→80% wood）④人体内联 bsdf 与 wood 家具重名 duplicate ID 崩溃 ⑤折线顶点恒 3（→3-4）⑥几何相交 `_seg_poly_t` 裁剪上下界写反；LOS 改几何判定（旧 min-tau 2ns 法 37% 误标穿墙/掠射反射）| pilot v4 100 场景 17min，全套 QC 绿（0 坏场景、独立对拍 0 不一致、tau 交叉 100% 吻合、4×sanity PASS），实测回填 §PC.4 |
| 2026-11-28 | 房间尺寸多样性（用户决策）：w,d 逐场景 U(7,15)m、h=3m 固定；500 场景元数据审计（尺寸均匀、小房间轨迹不退化）| 旧 10×10 固定尺寸产物清除重生成 |
| 2026-11-28 | SIONNA_HANDOFF.md 并入本文档（§Phase PC）后删除；此后唯一维护本文档 | — |
