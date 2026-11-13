"""GenConfig：管线全部参数（单点真相，yaml 覆盖）。

用户决策（2026-11-13 plan 批准）：WiFi 化适配（2.4GHz ISM、234 子载波 ≈40MHz、
子载波结构日后可换——改本文件即可）；场景 = MUSE-FM 配方 + 适度扩展（1-2 面
内隔墙分房 + 门洞 + 家具）；数据集 floorplan 只能是 PNG（FLOORPLAN_SPEC 标准），
4 通道张量只是从 PNG 派生的预处理；Radio Map 仅可视化。
"""
import os
from dataclasses import dataclass, asdict, fields

import yaml

N_AP_SLOTS = 4          # 与 DLoc 缓存/模型对齐的 AP 槽位数
N_SUBCARRIERS = 234
BAND_W = 26             # 234 = 26 x 9（与 build_dloc_cache / tokenizer patch_t 一致）
N_BANDS = N_SUBCARRIERS // BAND_W
ROWS_PER_AP = 8         # (4 ant) x (re, im) 交错行——FPCSIJEPA tokenizer 约定


@dataclass
class GenConfig:
    # ---- RF / CSI（WiFi 化；换子载波结构只改这一段）
    fc_hz: float = 2.412e9          # WiFi 1 信道中心
    n_subcarriers: int = N_SUBCARRIERS
    delta_f_hz: float = 156.25e3    # 234 x 156.25kHz ≈ 36.6MHz（Nexmon 40MHz tone 结构）
    normalize_delays: bool = False  # 保留绝对到达相位（TOF 信息）；消融可开
    # ---- 天线（PlanarArray；间距单位 = 波长倍数）
    tx_rows: int = 2                # AP 2x2 UPA = 4 天线 -> 每 AP 8 行，与 DLoc 逐位同构
    tx_cols: int = 2
    tx_spacing_wl: float = 0.5
    tx_pattern: str = 'iso'
    tx_polarization: str = 'V'
    rx_rows: int = 1                # UE 单天线全向
    rx_cols: int = 1
    rx_spacing_wl: float = 0.5
    rx_pattern: str = 'iso'
    rx_polarization: str = 'V'
    # ---- 设备布置（MUSE-FM：BS 2.5m / UE 1m；AP 数随机 1-4，不足 4 补零 + valid_ap）
    ap_height_m: float = 2.5
    ue_height_m: float = 1.0
    n_ap_min: int = 1
    n_ap_max: int = 4
    n_ap_weights: tuple = (0.10, 0.15, 0.25, 0.50)   # 1/2/3/4 AP 的采样权重（偏 4-AP）
    n_ue: int = 50                  # 每场景 UE 采样数（= MUSE-FM 每场景样本数）
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
    "产物目录布局；floorplan 遵循 FLOORPLAN_SPEC §6（<root>/floorplan/<dataset>/<room>/）。"
    root = cfg.out_root
    d = {
        'root': root,
        'scenes': os.path.join(root, 'scenes'),           # scene_<id>/{scene.xml, meshes/, metadata.json}
        'floorplan_ds': os.path.join(root, 'floorplan', 'SIONNA'),  # <room>/scene_<id>_ap<k>.png + manifest.csv
        'tensors': os.path.join(root, 'tensors'),          # scene_<id>.npz（预处理产物，非数据集资产）
        'cache': os.path.join(root, 'cache'),              # CSI_scene_<id>.npy + sidecar_*.npz（dloc_cache 同构）
        'preview': os.path.join(root, 'preview'),          # 仅可视化（radio map 等，不入数据集）
    }
    return d
