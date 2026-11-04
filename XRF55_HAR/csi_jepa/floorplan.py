"""Floorplan 条件化模块（Phase B-v2 DLoc 试点，PROJECT_LOG §11；设计=已批准 plan）。

DLocRasterBuilder  掩码 PNG + AP 元数据 -> 4 通道度量张量（组合而非改动——A1 的
                   csi_jepa/model.py 零改动）：
                     ch1 占据（0.1m 掩码 max-pool 到 0.2m 格元）
                     ch2 SDF（native EDT、tanh(sdf/1m) 符号距离——亚格元墙距穿过降采样）
                     ch3 unknown（renderer 三值补零 + "距最近自由区 >1m 判未知"启发式）
                     ch4 AP 高斯图（σ=0.5m，逐 AP 求和截断；AP 身份由连续坐标 token 携带）
                   世界画布与 FLOORPLAN_SPEC/已渲染 PNG 同框（64x29m，内容左下角 +1m 锚定；
                   jacobs 窗口 (-10,-5.1)、atkinson (-5,-3.6)——WILD manifest 的 world_origin）。
SceneEncoder       Conv2d(4,dim,20)（4m patch = 20 格元）-> 16x8=128 raster tokens
                   （分解式 pos_row/pos_col，同 CSITokenizer 模式）+ 4 个 AP 连续坐标 token
                   （可学 per-slot query + MLP([x_m,y_m,valid])——回归友好米制框架）->
                   2xPreLNBlock + final LN = 132 场景 tokens。
OneHotSceneEncoder B2 对照：栅格几何 -> 场景 id embedding（广播到 raster 位置）；AP 连续坐标
                   token 保留（"id+坐标"与"几何"的差 = floorplan 的净贡献）。
FPCSIJEPA          组合封装：[scene 132; csi 36] = 168-token 单 encoder（MUSE-FM 可比拼接，
                   场景 tokens 保留到输出）；csi_only_view() 供 CSIEMATargetEncoder 做
                   CSI-only EMA 目标（场景分支只经 context 注意力获得梯度）。
P1Head             物理辅助：池化场景 tokens -> Linear(dim, n_scen*n_ap*n_band)，
                   预测逐 (场景, AP, 频带) 平均 |CSI|（无参数 LN 两侧在训练脚本内做）。
"""
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from csi_jepa.model import CSITokenizer
from jepa_modules import PreLNBlock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK_ROOT = os.path.join(REPO, 'floorplan_dataset', 'DLoc_pt_code', 'ref')

# 场景 -> (掩码 png, 掩码 x_min 米, 掩码 y_max 米)；行 0=y_max、列 0=x_min
# （make_wild_floorplans.py 的权威映射与坐标约定，逐字节沿用）
SCENARIO_MASK = {
    'jacobs_July28': ('jacobs_default', -9.0, 12.0),
    'jacobs_July28_2': ('jacobs_default', -9.0, 12.0),
    'jacobs_Aug16_1': ('jacobs_aug16_1', -9.0, 12.0),
    'jacobs_Aug16_3': ('jacobs_aug16_3', -9.0, 12.0),
    'jacobs_Aug16_4_ref': ('jacobs_aug16_4_ref', -9.0, 12.0),
    'July16': ('atk_July22_1_ref', -4.0, 7.5),
    'July18': ('atk_July22_1_ref', -4.0, 7.5),
    'July22_2_ref': ('atk_July22_2_ref', -4.0, 7.5),
}
# WILD manifest 的 world_origin（内容包围盒左下角 + 1m 锚定）
BUILDING_WINDOW = {'jacobs': (-10.0, -5.1), 'atkinson': (-5.0, -3.6)}
CELL_M = 0.2                     # 格元（native 0.1m 掩码的 2x 整数池化）
PATCH_M = 4.0                    # raster patch = 20 格元（固定物理尺寸，与 16px/m 可换算）
CANVAS_M = (64.0, 29.0)          # 全局画布（宽, 高），FLOORPLAN_SPEC 渲染标准同框
UNKNOWN_FREE_DIST_M = 1.0        # 启发式：阻挡且距最近自由区 >1m -> unknown（可消融）


def remove_marker_noise(occ01, max_bbox=4, max_px=6):
    """清除游离标记噪点（源 PNG 内 3x3 '+' 等非建筑结构；逐字节复用
    make_wild_floorplans.py 的清理逻辑——仅清 ≤max_bbox x max_bbox、≤max_px 的孤立块）。"""
    from scipy import ndimage
    lab, n = ndimage.label(occ01)
    out = occ01.copy()
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if len(ys) <= max_px and h <= max_bbox and w <= max_bbox:
            out[ys, xs] = 0
    return out


def _pool2(x, mode):
    "0.1m -> 0.2m 的 2x 池化（H,W 均偶数）。"
    h, w = x.shape
    x = x[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2)
    if mode == 'max':
        return x.max(axis=(1, 3))
    return x.mean(axis=(1, 3))


def _place(canvas, sub, r0, c0):
    "把子阵放进画布（越界部分裁掉——掩码窗口设计上不越界，防御性处理）。"
    h, w = sub.shape
    H, W = canvas.shape
    r1, c1 = min(r0 + h, H), min(c0 + w, W)
    canvas[r0:r1, c0:c1] = sub[: r1 - r0, : c1 - c0]


class DLocRasterBuilder:
    """29 张 floorplan 常驻内存（global: (4,160,320) float32 ~0.8MB/张；content: (4,100,200)）。
    canvas='global' 64x29m 全局画布；'content' 40x20m 紧凑画布（消融：空白 token 代价）。"""

    def __init__(self, scenarios, canvas='global'):
        from PIL import Image
        self.canvas_mode = canvas
        if canvas == 'global':
            self.win = {b: (ox, oy, ox + CANVAS_M[0], oy + CANVAS_M[1])
                        for b, (ox, oy) in BUILDING_WINDOW.items()}
            self.H = int(CANVAS_M[1] / CELL_M / 20 + 0.999) * 20   # 145 -> 160（pad unknown）
            self.W = int(CANVAS_M[0] / CELL_M)                     # 320
        else:  # content：内容 bbox 左下角外推 2m 的 40x20m 窗
            self.win = {b: (BOLD[b][0] - 2.0, BOLD[b][1] - 2.0,
                            BOLD[b][0] + 38.0, BOLD[b][1] + 18.0)
                        for b in BUILDING_WINDOW}
            self.H, self.W = 100, 200
        self.tables = {}
        for sc in scenarios:
            mask_png, mx0, my1 = SCENARIO_MASK[sc]
            raw = np.array(Image.open(os.path.join(MASK_ROOT, mask_png + '.png')).convert('L'))
            free_src = raw > 128                                   # 源 255=自由，0=阻挡/未知
            occ_src = (~free_src).astype(np.uint8)
            occ_src = remove_marker_noise(occ_src)
            b = 'jacobs' if sc.startswith('jacobs') else 'atkinson'
            ox, oy, _, _ = self.win[b]
            nat_h, nat_w = int(CANVAS_M[1] / 0.1), int(CANVAS_M[0] / 0.1)
            r0 = int(round((oy + CANVAS_M[1] - my1) / 0.1))        # 掩码行 0 = y_max
            c0 = int(round((mx0 - ox) / 0.1))
            free = np.zeros((nat_h, nat_w), dtype=bool)
            _place(free, free_src.astype(bool), r0, c0)
            occ = np.zeros((nat_h, nat_w), dtype=bool)
            _place(occ, occ_src.astype(bool), r0, c0)
            covered = np.zeros((nat_h, nat_w), dtype=bool)
            covered[r0:r0 + free_src.shape[0], c0:c0 + free_src.shape[1]] = True

            from scipy import ndimage
            d_free = ndimage.distance_transform_edt(~free) * 0.1   # 非自由格到最近自由区(米)
            d_occ = ndimage.distance_transform_edt(free) * 0.1     # 自由格到最近阻挡(米)
            sdf = np.where(free, d_occ, -d_free)
            unknown = (~covered) | (occ & (d_free > UNKNOWN_FREE_DIST_M))

            ch_occ = _pool2(occ.astype(np.float32), 'max')
            ch_unk = _pool2(unknown.astype(np.float32), 'max')
            ch_sdf = _pool2(np.tanh(sdf / 1.0), 'mean')
            self.tables[sc] = {'occ': ch_occ, 'unk': ch_unk, 'sdf': ch_sdf,
                               'n_cells': (nat_h, nat_w), 'off': (r0, c0)}

    def grid_hw(self):
        return self.H // int(PATCH_M / CELL_M), self.W // int(PATCH_M / CELL_M)

    def build(self, scenario, ap_centroids, valid_ap):
        "(4, H, W) float32：occ/unknown/sdf 通道 pad 到 (H,W)，AP 高斯直接在 0.2m 格上画。"
        t = self.tables[scenario]
        H, W = self.H, self.W
        out = np.zeros((4, H, W), dtype=np.float32)
        oh, ow = t['occ'].shape
        out[0, :oh, :ow] = t['occ']
        out[1, :oh, :ow] = t['unk']
        out[2, :oh, :ow] = t['sdf']
        # ch4：canvas 网格坐标（row0=y_max），高斯 sigma=0.5m=2.5 格
        yy, xx = np.mgrid[0:H, 0:W]
        ox, oy, _, y_max = self.win[
            'jacobs' if scenario.startswith('jacobs') else 'atkinson']
        for ap in range(len(valid_ap)):
            if not valid_ap[ap]:
                continue
            x, y = ap_centroids[ap]
            col = (x - ox) / CELL_M
            row = (y_max - y) / CELL_M
            g = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * (0.5 / CELL_M) ** 2))
            out[3] = np.clip(out[3] + g, 0, 1)
        return out

    def ap_window_origin(self, scenario):
        b = 'jacobs' if scenario.startswith('jacobs') else 'atkinson'
        return self.win[b][:2]


# content 画布模式的楼级内容锚（x_min/y_min 米——§11 审计值）
BOLD = {'jacobs': (-9.0, -4.0), 'atkinson': (-4.0, -2.5)}


class SceneEncoder(nn.Module):
    """栅格 -> tokens：Conv2d(4,dim,patch) + 分解式 pos_row/pos_col + AP 连续坐标 token
    （可学 per-slot query + MLP([x_m,y_m,valid])）-> 2xPreLNBlock -> final LN。"""

    def __init__(self, grid_hw, patch, dim=256, depth=2, num_heads=4, ffn_mult=4.0,
                 n_ap_slots=4):
        super(SceneEncoder, self).__init__()
        gh, gw = grid_hw
        self.gh, self.gw, self.patch = gh, gw, patch
        self.patch_embed = nn.Conv2d(4, dim, patch, stride=patch)
        self.pos_row = nn.Parameter(torch.zeros(1, gh, 1, dim))
        self.pos_col = nn.Parameter(torch.zeros(1, 1, gw, dim))
        self.ap_query = nn.Parameter(torch.zeros(1, n_ap_slots, dim))
        self.ap_mlp = nn.Linear(3, dim)                       # [x_m, y_m, valid]
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, int(dim * ffn_mult)) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_row, std=0.02)
        nn.init.trunc_normal_(self.pos_col, std=0.02)
        nn.init.trunc_normal_(self.ap_query, std=0.02)

    def forward(self, raster, ap_centroids, valid_ap):
        "raster (B,4,H,W) 原始值；ap_centroids (B,4,2) 米；valid_ap (B,4) bool。"
        h = self.patch_embed(raster)                          # (B, dim, gh, gw)
        h = h.flatten(2).transpose(1, 2)                      # (B, gh*gw, dim) row-major
        pos = (self.pos_row + self.pos_col).flatten(1, 2)     # (1, gh*gw, dim)
        tokens = h + pos
        coords = torch.cat([ap_centroids, valid_ap.float().unsqueeze(-1)], dim=-1)  # (B,4,3)
        ap_tokens = self.ap_query + self.ap_mlp(coords)       # (B,4,dim)
        tokens = torch.cat([tokens, ap_tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm_out(tokens)


class OneHotSceneEncoder(nn.Module):
    """B2 对照：栅格几何 -> 场景 id embedding（广播到 raster token 位置）；AP 连续坐标
    token 保留。'id + AP 坐标' 能给出而栅格不能给的 = 0；差值即 floorplan 几何的净贡献。"""

    def __init__(self, n_scenarios, grid_hw, dim=256, depth=2, num_heads=4, ffn_mult=4.0,
                 n_ap_slots=4):
        super(OneHotSceneEncoder, self).__init__()
        gh, gw = grid_hw
        self.n_raster = gh * gw
        self.scen_emb = nn.Embedding(n_scenarios, dim)
        self.ap_query = nn.Parameter(torch.zeros(1, n_ap_slots, dim))
        self.ap_mlp = nn.Linear(3, dim)
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, int(dim * ffn_mult)) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.scen_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.ap_query, std=0.02)

    def forward(self, raster, ap_centroids, valid_ap, scenario_id):
        h = self.scen_emb(scenario_id).unsqueeze(1).expand(-1, self.n_raster, -1)
        coords = torch.cat([ap_centroids, valid_ap.float().unsqueeze(-1)], dim=-1)
        tokens = torch.cat([h, self.ap_query + self.ap_mlp(coords)], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm_out(tokens)


class _CSIOnlyView(nn.Module):
    """CSI-only 前向视图（与 FPCSIJEPA 共享子模块引用）——专供 CSIEMATargetEncoder
    deepcopy 出场景无关的 EMA 目标分支：目标分支 forward = 完整 CSI、无掩码、无场景。"""

    def __init__(self, tokenizer, norm_in, blocks, norm_out):
        super(_CSIOnlyView, self).__init__()
        self.tokenizer, self.norm_in = tokenizer, norm_in
        self.blocks, self.norm_out = blocks, norm_out

    def forward(self, csi):
        h = self.norm_in(self.tokenizer(csi))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_out(h)


class FPCSIJEPA(nn.Module):
    """[scene; csi] 单 encoder 封装。scene_mode: 'raster'（主）/ 'onehot'（对照）/ 'none'。"""

    def __init__(self, dim=256, depth=8, num_heads=4, ffn_dim=1024, dropout=0.0,
                 scene_mode='raster', grid_hw=(8, 16), raster_patch=20, n_scenarios=8,
                 n_ap_slots=4, csi_repr='reim'):
        super(FPCSIJEPA, self).__init__()
        self.scene_mode = scene_mode
        csi_channels = 32 if csi_repr == 'reim' else 16       # reim: 8 re/im 行/AP；mag: 4 幅度行/AP
        self.csi_repr = csi_repr
        patch_c = 8 if csi_repr == 'reim' else 4              # 两模式网格均 (4 AP, 9 频带)=36
        self.tokenizer = CSITokenizer(csi_channels, 234, patch_c, 26, dim,
                                      input_norm='channel_std')
        if scene_mode == 'raster':
            self.scene = SceneEncoder(grid_hw, raster_patch, dim, n_ap_slots=n_ap_slots)
        elif scene_mode == 'onehot':
            self.scene = OneHotSceneEncoder(n_scenarios, grid_hw, dim,
                                            n_ap_slots=n_ap_slots)
        else:
            self.scene = None
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.n_scene_tokens = 0 if scene_mode == 'none' else (
            grid_hw[0] * grid_hw[1] + n_ap_slots)
        self.csi_grid = (4, 9)                                # reim 与 mag 均 (4 AP, 9 频带)
        self.num_tokens = self.n_scene_tokens + 36
        self._csi_view = _CSIOnlyView(self.tokenizer, self.norm_in, self.blocks,
                                      self.norm_out)

    def csi_only_view(self):
        "共享子模块引用的 CSI-only 视图（CSIEMATargetEncoder 由此 deepcopy+EMA）。"
        return self._csi_view

    def to_mag(self, csi):
        "csi_repr='mag' 消融：8 re/im 行 -> 4 幅度行（AP 组 8->4，网格不变）。"
        x = csi.reshape(csi.shape[0], 4, 4, 2, 234)
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        return mag.reshape(csi.shape[0], 16, 234)

    def forward(self, csi, token_mask=None, scene_tokens=None, raster=None,
                ap_centroids=None, valid_ap=None, scenario_id=None):
        """token_mask: (B, n_csi_tokens) bool（只作用于 CSI 位置）-> (B, num_tokens, dim)。
        scene_tokens 直通入口：批内唯一场景的 tokens（预训练脚本按唯一场景分组前向后
        gather，避免 512 份重复栅格进显存）。"""
        if self.csi_repr == 'mag':
            csi = self.to_mag(csi)
        c = self.norm_in(self.tokenizer(csi))
        if token_mask is not None:
            c = torch.where(token_mask.unsqueeze(-1), self.mask_token, c)
        if self.scene is not None:
            if scene_tokens is None:
                if self.scene_mode == 'onehot':
                    s = self.scene(raster, ap_centroids, valid_ap, scenario_id)
                else:
                    s = self.scene(raster, ap_centroids, valid_ap)
            else:
                s = scene_tokens
            x = torch.cat([s, c], dim=1)
        else:
            x = c
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class P1Head(nn.Module):
    """物理辅助头：池化场景 tokens -> 逐 (AP, 频带) 平均 |CSI|（36 = 4*9）。
    场景身份不在输出槽位里（那会退化成查表、one-hot 也能过）——不同 floorplan 必须
    经由场景表征映射到不同的统计。无参数 LN 两侧在训练脚本内施加（D17/D30 纪律）。"""

    def __init__(self, dim=256, n_ap_slots=4, n_bands=9):
        super(P1Head, self).__init__()
        self.fc = nn.Linear(dim, n_ap_slots * n_bands)

    def forward(self, scene_tokens):
        "scene_tokens: (B, S, dim) 场景位置切片 -> (B, 36)。"
        return self.fc(scene_tokens.mean(dim=1))
