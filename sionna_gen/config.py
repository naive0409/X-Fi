"""GenConfig：管线全部参数（单点真相，yaml 覆盖）。

Phase PC（2026-11-27）：布局切换为 rooms10 对齐（layout='rooms10'，新主线）——
3.5GHz、单 TX-SISO、UE 轨迹 480 快照@60Hz、可选人体遮挡、cir/csi/run.json 输出
（与 3D_dataset/dataset_10rooms 逐键同构）。旧 DLoc 对齐（layout='dloc'，
2.412GHz、1-4 AP 2x2 UPA、静态 UE 点）deprecated：显式 yaml 覆盖 +
`legacy_dloc.yaml` 可复现旧默认，管线代码保留未删。
带宽双口径（--bandwidth 覆盖 delta_f_hz）：narrow = 256x30kHz = 7.68MHz
（10rooms 同款）；wide = 256x390.625kHz ≈ 100MHz（张量形状不变，仅频谱口径变）。
"""
import os
from dataclasses import dataclass, asdict, fields

import yaml

# deprecated（layout='dloc' 专用）
N_AP_SLOTS = 4          # 与 DLoc 缓存/模型对齐的 AP 槽位数
N_SUBCARRIERS = 234
BAND_W = 26             # 234 = 26 x 9（与 build_dloc_cache / tokenizer patch_t 一致）
N_BANDS = N_SUBCARRIERS // BAND_W
ROWS_PER_AP = 8         # (4 ant) x (re, im) 交错行——FPCSIJEPA tokenizer 约定

# rooms10 对齐常量（与 dataset_10rooms run.json/cir.npz 逐键同源）
N_PATH_SLOTS = 26                   # cir 径槽位（10rooms 同款）
CIR_MAX_PATHS = N_PATH_SLOTS
BANDWIDTH_PRESETS = {
    # 'narrow' = dataset_10rooms 同款（SCS 30kHz）；'wide' = 100MHz（形状不变）
    'narrow': 30e3,
    'wide': 390.625e3,
}
# 人体 8 段轮廓 [z_m, 横向半径_m, 前后半径_m]（dataset_10rooms run.json 逐值同源）
HUMAN_PROFILE = (
    (0.00, 0.10, 0.07), (0.14, 0.12, 0.08), (0.85, 0.16, 0.10),
    (1.05, 0.19, 0.12), (1.37, 0.23, 0.12), (1.49, 0.10, 0.085),
    (1.63, 0.10, 0.095), (1.78, 0.04, 0.04),
)


@dataclass
class GenConfig:
    # ---- 布局（Phase PC 主线 = rooms10；'dloc' deprecated）
    layout: str = 'rooms10'          # 'rooms10' | 'dloc'
    # ---- RF / CSI（rooms10 对齐；fc/SCS 与 dataset_10rooms run.json 一致）
    fc_hz: float = 3.5e9             # 3.5 GHz（10rooms 同款）
    n_subcarriers: int = 256         # 10rooms 同款（含 DC）
    delta_f_hz: float = 30e3         # SCS 30kHz -> 7.68MHz（'wide' 预设由 --bandwidth 覆盖）
    normalize_delays: bool = False  # 保留绝对到达相位（TOF 信息）；消融可开
    # ---- 天线（PlanarArray；间距单位 = 波长倍数。rooms10 = 单 TX/单 RX 各 1 天线 SISO）
    tx_rows: int = 1                # rooms10：单天线 iso（10rooms 的 TX 无阵列）
    tx_cols: int = 1
    tx_spacing_wl: float = 0.5
    tx_pattern: str = 'iso'
    tx_polarization: str = 'V'
    rx_rows: int = 1                # UE 单天线全向
    rx_cols: int = 1
    rx_spacing_wl: float = 0.5
    rx_pattern: str = 'iso'
    rx_polarization: str = 'V'
    # ---- 设备布置（rooms10：单 TX 近墙角高 2.12m；UE 沿轨迹高 1.25m）
    ap_height_m: float = 2.12       # rooms10：TX 安装高（10rooms run.json tx z=2.12）
    ue_height_m: float = 1.25       # UE 高（10rooms 同款，背在身上的接收机）
    n_ap_min: int = 1               # rooms10：恒 1 个 TX（n_ap_weights 退役）
    n_ap_max: int = 1
    n_ap_weights: tuple = (1.0,)    # deprecated（layout='dloc' 用）
    n_ue: int = 50                  # deprecated（layout='dloc' 静态 UE 点数）
    # ---- 轨迹（Isaac 替代：解析轨迹；10rooms = 480 快照@60Hz、UE ≈0.77 m/s）
    n_snapshots: int = 480
    snapshot_rate_hz: float = 60.0
    traj_speed_m_s: float = 0.8     # 名义速率 m/s（实际由折线长度/时长归一）
    traj_clearance_m: float = 0.2   # 轨迹点距障碍/外墙的最小净空
    sensor_time_offset_s: float = 0.0  # sensor_time = capture_time + offset（10rooms 有 IMU 会话偏移，本库无需）
    # ---- 人体遮挡（用户决策：保留 + config 开关；轮廓逐值同 10rooms run.json）
    human_occluder: bool = True
    human_offset_m: float = 0.14    # 人体中心与 UE 的水平偏移（10rooms 恒 0.140m）
    human_height_m: float = 1.78
    human_material: str = 'wood'    # 人体近似材质（理想吸收体以后处理路径剪枝实现）
    human_profile: tuple = HUMAN_PROFILE
    snr_tiers_db: tuple = (0, 10, 20, 30)   # estimated_csi 的 4 档 SNR（10rooms 同款）
    # ---- 环境点云（替代 Isaac lidar：mesh 表面采样；每场景一张静态图）
    pc_full_points: int = 40960     # 全分辨率表面采样点数
    pc_fps_points: int = 2048       # FPS 下采样（模型输入规模）
    # ---- 射追（PathSolver；主线不开绕射——开销 3-10 倍，留开关做对照）
    max_depth: int = 3
    samples_per_src: int = 1_000_000
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True         # 穿墙（thin-wall 模型）
    diffraction: bool = False
    synthetic_array: bool = True
    # ---- 场景配方（MUSE-FM：10x10x3m、地板 concrete、外围墙 brick、0-3 隔墙 +
    #      0-2 圆柱全 marble；扩展：1-2 面分房隔墙（带门洞）+ 0-4 件家具 wood/metal）
    room_w_m: float = 10.0
    room_d_m: float = 10.0
    room_h_m: float = 3.0
    wall_thickness_m: float = 0.1   # RT 几何实际厚度（thin-box，材质 thickness 同值）
    n_divider_min: int = 1          # 分房隔墙（轴对齐、开门洞）
    n_divider_max: int = 2
    n_partition_max: int = 3        # 追加随机朝向大理石隔墙（0-3，MUSE-FM 配方）
    n_cylinder_max: int = 2         # 大理石圆柱（0-2，MUSE-FM 配方）
    n_furniture_max: int = 4        # 家具级低障碍（0-4，wood/metal）
    door_width_m: float = 0.9
    door_height_m: float = 2.1
    # ---- floorplan 渲染（FLOORPLAN_SPEC.md 标准，全局常量永不 resize）
    scale_px_per_m: int = 16
    canvas_px: tuple = (1024, 464)  # 64 x 29 m 全局画布
    margin_m: float = 1.0
    native_res_m: float = 0.1       # 几何光栅化原生分辨率（DLoc 掩码同款）
    cell_m: float = 0.2             # 张量格元（0.1m 的 2x 池化）
    patch_m: float = 4.0            # raster patch（20 格元）
    canvas_m: tuple = (64.0, 29.0)
    unknown_free_dist_m: float = 1.0
    # ---- 数据集与随机
    seed_base: int = 3407
    out_root: str = 'sionna_dataset'  # 生成产物根（scenes/floorplan/tensors/cache/preview）

    def meta_dict(self):
        return asdict(self)


def load_config(path=None):
    "默认配置 + yaml 逐键覆盖（未知键报错，防拼写静默错参）。"
    cfg = GenConfig()
    if path:
        with open(path) as f:
            over = yaml.safe_load(f) or {}
        known = {f.name for f in fields(GenConfig)}
        unk = set(over) - known
        if unk:
            raise KeyError(f'未知配置键: {sorted(unk)}')
        for k, v in over.items():
            setattr(cfg, k, v)
    return cfg


def scene_seed(cfg, scene_id):
    "每场景独立确定性种子（场景几何/设备布置/RT 同源）。"
    return int(cfg.seed_base) * 1_000_003 + int(scene_id)


def out_dirs(cfg):
    """产物目录布局。

    layout='rooms10'：scenes/<几何源文件> + dataset/scene_<id>/{env_pc.npz,
    radio/{cir,csi,link_summary,run.json}}（与 3D_dataset/dataset_10rooms/room<k>/
    {lidar->env_pc, radio/} 同构——消费层一套代码读两库）+ deprecated cache/。
    layout='dloc'：原布局不变（deprecated）。
    """
    root = cfg.out_root
    if cfg.layout == 'rooms10':
        return {
            'root': root,
            'scenes': os.path.join(root, 'scenes'),   # scene_<id>/{scene.xml, meshes/, metadata.json}
            'dataset': os.path.join(root, 'dataset'),  # scene_<id>/{env_pc.npz, radio/}（消费层入口）
            'floorplan_ds': os.path.join(root, 'floorplan', 'SIONNA'),  # QC 可选（FLOORPLAN_SPEC）
            'tensors': os.path.join(root, 'tensors'),  # deprecated（raster 管线）
            'cache': os.path.join(root, 'cache'),      # deprecated（dloc 同构）
            'preview': os.path.join(root, 'preview'),  # 仅可视化（radio map 等，不入数据集）
        }
    return {
        'root': root,
        'scenes': os.path.join(root, 'scenes'),
        'floorplan_ds': os.path.join(root, 'floorplan', 'SIONNA'),
        'tensors': os.path.join(root, 'tensors'),
        'cache': os.path.join(root, 'cache'),
        'preview': os.path.join(root, 'preview'),
    }
