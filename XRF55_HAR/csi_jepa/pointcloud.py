"""点云条件化模型（Phase PC；组合而非改动——csi_jepa/model.py 与 floorplan.py 零改动）。

PointPatchTokenizer  环境点云 -> K 个 patch token（Point-MAE 式连续 patch，跳过
                     dVAE——dVAE 留二期消融）：FPS 中心 + kNN 分组（非学习，逐场景
                     预计算缓存，与 DLocRasterBuilder 的"栅格常驻内存"同模式）->
                     Mini-PointNet（共享 MLP + max-pool）per-patch embed；位置编码 =
                     度量式 MLP(patch 中心 xyz 按 bbox 归一化)（跨场景迁移友好，与
                     AP 连续坐标 token 同哲学）。geo_dropout（Sonata）：训练期以
                     概率 p 把中心坐标置零——防绝对位置记忆捷径（Phase B 查表教训
                     的"预防版"）。
OneHotPatchEncoder   查表对照：场景 id embedding 广播到 K 个 patch 位置 + 度量式
                     位置编码（"id+坐标"能给的 = patch 内容几何给不了的差值）。
PCSIJEPA             [pc tokens K; csi tokens 8] 单 encoder（结构镜像 FPCSIJEPA，
                     MUSE-FM 可比拼接、场景 tokens 保留到输出）；CSI tokenizer
                     (2,256) patch(2,32) -> grid (1,8)；csi_only_view() 供
                     CSIEMATargetEncoder 做场景无关 EMA 目标；full_view() 供
                     target_mode='dual'（Sonata 式：EMA 目标含 pc 位置——对照消融，
                     重开 D47"场景恒定输出"风险，默认不用）。
P1Head / P2Head      物理辅助：池化 pc tokens -> 逐 (TX,频带) mean|CSI|（8 维，
                     槽位=8 防场景查表）/ 逐 TX 路径统计 (3)。无参数 LN 两侧在
                     训练脚本内施加（D17/D30 纪律）。
"""

import numpy as np
import torch
from torch import nn

from csi_jepa.model import CSITokenizer
from jepa_modules import PreLNBlock

# ---------------------------------------------------------------- patch 预计算
def build_scene_patches(xyz, n_patches=64, patch_size=32, seed=3407):
    """(N,3) 点云 -> dict(centers (K,3), patches (K,S,3), center_norm (K,3))。

    FPS 选中心（确定性首点 = 距质心最远点，避免逐场景随机）、kNN 分组、中心坐标
    按点云 bbox 归一化到 [-1,1]（跨场景共享同一编码空间）。numpy 预计算、
    场景级缓存（点云是静态量，与 DLocRasterBuilder 同模式）。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    n = len(xyz)
    k = min(n_patches, n)

    # FPS：首点 = 距质心最远（确定性、无 rng）
    c0 = int(np.argmax(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)))
    sel = np.empty(k, dtype=np.int64)
    sel[0] = c0
    dist = np.linalg.norm(xyz - xyz[c0], axis=1)
    for i in range(1, k):
        j = int(np.argmax(dist))
        sel[i] = j
        dist = np.minimum(dist, np.linalg.norm(xyz - xyz[j], axis=1))
    centers = xyz[sel]
    # kNN 分组：每中心取 patch_size 近邻（含中心自身；点数 < patch_size 时循环补）
    d2 = ((xyz[None, :, :] - centers[:, None, :]) ** 2).sum(-1)     # (K, N)
    idx = np.argsort(d2, axis=1)[:, :patch_size]
    patches = xyz[idx]                                              # (K, S, 3)
    center_norm = (2.0 * (centers - lo) / span - 1.0)               # [-1,1]^3
    return {'centers': centers.astype(np.float32),
            'patches': patches.astype(np.float32),
            'center_norm': center_norm.astype(np.float32)}


class PatchProvider:
    """场景级 patch 缓存（key -> dict）；dict 直接喂 tokenizer（torch 侧再 to GPU）。"""

    def __init__(self, n_patches=64, patch_size=32):
        self.n_patches, self.patch_size = n_patches, patch_size
        self._cache = {}

    def get(self, key, xyz):
        if key not in self._cache:
            seed = (hash(key) % (2 ** 31)) + 17
            self._cache[key] = build_scene_patches(
                xyz, self.n_patches, self.patch_size, seed=seed)
        return self._cache[key]


# ---------------------------------------------------------------- 模型件
class PointPatchTokenizer(nn.Module):
    """patch (K,S,3) -> token (K,dim)：Mini-PointNet + 度量式位置编码。"""

    def __init__(self, dim=256, geo_dropout=0.0):
        super().__init__()
        self.geo_dropout = float(geo_dropout)
        self.pnet = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, dim))     # Mini-PointNet
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))   # 度量式 pos（Sonata
            # 几何 dropout 的作用点：训练期把 center_norm 置零 -> pos 退化为纯偏置）
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, patches, center_norm):
        # patches (B,K,S,3) 绝对坐标；center_norm (B,K,3) 已归一化
        rel = patches - center_norm.unsqueeze(2)                 # patch 内局部坐标
        h = self.pnet(rel).max(dim=2).values                     # (B,K,dim)
        cn = center_norm
        if self.training and self.geo_dropout > 0:
            drop = (torch.rand(cn.shape[0], 1, 1, device=cn.device)
                    < self.geo_dropout).float()
            cn = cn * (1.0 - drop)                               # 整样本置零
        return self.norm_out(h + self.pos_mlp(cn))


class OneHotPatchEncoder(nn.Module):
    """查表对照（BG3 对应物）：场景 id embedding 广播到 K 位置 + 同款度量 pos。
    'id + 中心坐标' 给不了 patch 内容几何——差值即点云内容的净贡献。"""

    def __init__(self, n_scenarios, dim=256):
        super().__init__()
        self.emb = nn.Embedding(n_scenarios, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm_out = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.emb.weight, std=0.02)

    def forward(self, center_norm, scenario_id):
        h = self.emb(scenario_id).unsqueeze(1).expand(-1, center_norm.shape[1], -1)
        return self.norm_out(h + self.pos_mlp(center_norm))


class _CSIOnlyView(nn.Module):
    """CSI-only 前向视图（共享子模块引用）——CSIEMATargetEncoder deepcopy 目标：
    目标分支 forward = 完整 CSI、无掩码、无场景（Phase B 验证设计）。"""

    def __init__(self, tokenizer, norm_in, blocks, norm_out):
        super().__init__()
        self.tokenizer, self.norm_in = tokenizer, norm_in
        self.blocks, self.norm_out = blocks, norm_out

    def forward(self, csi):
        h = self.norm_in(self.tokenizer(csi))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_out(h)


class _FullView(nn.Module):
    """[pc; csi] 全 token 视图（target_mode='dual' 专用；共享子模块引用）。
    EMA 目标含 pc 位置 -> 被掩 pc token 也有潜空间预测目标（Sonata 式）。"""

    def __init__(self, tokenizer, scene, patch_net, norm_in, blocks, norm_out):
        super().__init__()
        self.tokenizer = tokenizer
        self.scene, self.patch_net = scene, patch_net
        self.norm_in, self.blocks, self.norm_out = norm_in, blocks, norm_out

    def forward(self, csi, patches, center_norm, scenario_id=None):
        c = self.norm_in(self.tokenizer(csi))
        if self.scene is not None:                             # onehot
            s = self.scene(center_norm, scenario_id)
        elif self.patch_net is not None:                       # pc
            s = self.patch_net(patches, center_norm)
        else:
            s = None
        if s is not None:
            c = torch.cat([s, c], dim=1)
        for blk in self.blocks:
            c = blk(c)
        return self.norm_out(c)


class PCSIJEPA(nn.Module):
    """[pc K; csi 8] 单 encoder。scene_mode: 'pc'（主）/ 'onehot'（查表对照）/ 'none'。

    n_scene_tokens = K（pc/onehot）或 0（none）；CSI grid = (1, 8) = 8 tokens
    （patch_c=2 覆盖 re/im 两行，patch_t=32 -> 8 频带；P1 的 8 维与之一一对应）。"""

    def __init__(self, dim=256, depth=8, num_heads=4, ffn_dim=1024, dropout=0.0,
                 scene_mode='pc', n_scenarios=8, n_patches=64, csi_channels=2,
                 csi_time_steps=256, geo_dropout=0.0):
        super().__init__()
        assert scene_mode in ('pc', 'onehot', 'none')
        self.scene_mode = scene_mode
        self.tokenizer = CSITokenizer(csi_channels, csi_time_steps,
                                      patch_c=csi_channels, patch_t=32, dim=dim,
                                      input_norm='channel_std')
        if scene_mode == 'pc':
            self.patch_net = PointPatchTokenizer(dim, geo_dropout)
            self.scene = None
        elif scene_mode == 'onehot':
            self.patch_net = None
            self.scene = OneHotPatchEncoder(n_scenarios, dim)
        else:
            self.patch_net = None
            self.scene = None
        self.geo_dropout = float(geo_dropout)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.n_patches = n_patches
        self.n_scene_tokens = 0 if scene_mode == 'none' else n_patches
        self.csi_grid = (1, 8)
        self.num_tokens = self.n_scene_tokens + 8
        self._csi_view = _CSIOnlyView(self.tokenizer, self.norm_in, self.blocks,
                                      self.norm_out)
        self._full_view = (_FullView(self.tokenizer, self.scene, self.patch_net,
                                     self.norm_in, self.blocks, self.norm_out)
                           if scene_mode != 'none' else None)

    def csi_only_view(self):
        return self._csi_view

    def full_view(self):
        assert self._full_view is not None, 'full_view 需要 scene_mode != none'
        return self._full_view

    def forward(self, csi, token_mask=None, patches=None, center_norm=None,
                scenario_id=None, scene_tokens=None):
        """token_mask: (B, 8) bool 只作用于 CSI 位置 -> (B, num_tokens, dim)。

        scene_tokens 直通入口（与 FPCSIJEPA 同模式）：调用方按唯一场景分组编码后
        gather 传入，避免批内重复栅格/点云编码（脚本侧 forward_scene_grouped）。"""
        c = self.norm_in(self.tokenizer(csi))
        if token_mask is not None:
            c = torch.where(token_mask.unsqueeze(-1), self.mask_token, c)
        if scene_tokens is not None:
            s = scene_tokens
        elif self.scene_mode == 'pc':
            s = self.patch_net(patches, center_norm)
        elif self.scene_mode == 'onehot':
            s = self.scene(center_norm, scenario_id)
        else:
            s = None
        x = torch.cat([s, c], dim=1) if s is not None else c
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class P1Head(nn.Module):
    """物理辅助 P1：池化 pc tokens -> 逐 (TX, 频带) mean|CSI|（1x8=8 维）。
    输出槽位是频带而非场景——场景身份必须经场景表征传导（防查表，Phase B 教训）。
    无参数 LN 两侧在训练脚本内施加。"""

    def __init__(self, dim=256, n_bands=8):
        super().__init__()
        self.fc = nn.Linear(dim, n_bands)

    def forward(self, scene_tokens):
        "scene_tokens: (B, S, dim) pc 位置切片 -> (B, 8)。"
        return self.fc(scene_tokens.mean(dim=1))


class P2Head(nn.Module):
    """物理辅助 P2：池化 pc tokens -> 逐 TX 路径统计回归 (3,)=
    [mean n_paths, median first_delay_ns, mean sum_gain_db]（绝对量纲差异大 ->
    目标做标准化后在标准化域回归，训练脚本内施加）。"""

    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, scene_tokens):
        return self.fc(scene_tokens.mean(dim=1))


class CenterAuxHead(nn.Module):
    """Point-JMAE 式中心点辅助：被掩 patch 位置预测其中心坐标（归一化域，L2）。"""

    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, tokens):
        "tokens: (B, K, dim) pc 位置切片 -> (B, K, 3)。"
        return self.fc(tokens)
