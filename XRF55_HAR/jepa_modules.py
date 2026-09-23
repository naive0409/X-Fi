"""JEPA modules (JEPAREADME §4.2): Predictor + EMATargetEncoder.

Role in the project (JEPAREADME §3):
- Predictor g_phi: light 4-layer pre-LN transformer mapping the context representation
  z_cm (B,32,512) to the predicted target representation z_hat (B,32,512).
- EMATargetEncoder f_theta_bar: momentum copy of the online linear_projector + X_Fusion
  ONLY (the feature extractor is frozen and shared by reference on both branches,
  JEPAREADME §3.3-2 / §5.2 trap 4). Parameters and float buffers are EMA-updated after
  every optimizer step; the branch stays in eval() so its BatchNorm layers use their own
  (EMA-tracked) running statistics and are never double-updated (trap 3, human-confirmed
  decision to keep BatchNorm).
Momentum schedule follows I-JEPA: m rises from base 0.996 to 1.0 along a cosine over
training progress.
"""
import copy
import math

import torch
from torch import nn


class PreLNBlock(nn.Module):
    "Pre-LN transformer block: x + Attn(LN(x)); x + FFN(LN(x))."

    def __init__(self, dim, num_heads, ffn_dim, dropout=0.0):
        super(PreLNBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class Predictor(nn.Module):
    """g_phi: z_cm (B,32,512) -> z_hat (B,32,512) (JEPAREADME §4.2).

    Adds a learnable input marker embedding (distinguishing 'this is an inference input')
    and a learnable positional embedding, then `depth` pre-LN transformer blocks and a
    final LayerNorm.
    """

    def __init__(self, dim=512, num_tokens=32, depth=4, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Predictor, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.input_marker = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, ffn_dim, dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.input_marker, std=0.02)

    def forward(self, z_cm):
        x = z_cm + self.pos_embed + self.input_marker
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class AuxModalityHeads(nn.Module):
    """Cross-modal auxiliary prediction heads (JEPACHANGES §4-14): decode each ABSENT
    modality's projected features out of the aggregate prediction z_hat.

    Deliberately Linear (512->512): the aux objective is 'make the aggregate representation
    LINEARLY decodable into absent-modality features', matching the linear-probe evaluation.
    Kept OUTSIDE X_Fi_Backbone/EMATargetEncoder so no existing state_dict changes; stored
    under a separate 'aux_state' checkpoint key. ~0.79M params.
    """

    def __init__(self, dim=512, num_modalities=3):
        super(AuxModalityHeads, self).__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_modalities)])

    def forward(self, z_hat, absent_idxs):
        "z_hat: (B,32,512) -> [head_m(z_hat) for m in absent_idxs], each (B,32,512)."
        return [self.heads[m](z_hat) for m in absent_idxs]


class EMATargetEncoder(nn.Module):
    """f_theta_bar: EMA copy of the online linear_projector + X_Fusion.

    The frozen feature extractor is NOT copied — it is held by reference so both branches
    run the identical frozen weights (JEPAREADME §3.3-2 / §5.2 trap 4). The target branch
    input is always the full, unmasked, all-modalities data (JEPAREADME §5.1 step 5).
    """

    def __init__(self, online_backbone, base_momentum=0.996):
        super(EMATargetEncoder, self).__init__()
        self.linear_projector = copy.deepcopy(online_backbone.linear_projector)
        self.X_Fusion_block = copy.deepcopy(online_backbone.X_Fusion_block)
        self.feature_extractor = online_backbone.feature_extractor  # shared reference
        self.out_norm = nn.LayerNorm(512)  # own copy of the backbone's final LN; EMA-tracked
        for p in self.parameters():
            p.requires_grad_(False)
        self.base_momentum = base_momentum
        self.eval()

    def train(self, mode=True):
        "Guard: the EMA branch always stays in eval() (JEPAREADME §5.2 trap 3)."
        return super(EMATargetEncoder, self).train(False)

    @torch.no_grad()
    def update(self, online_backbone, progress=0.0):
        """EMA-update all params and buffers of the two copied modules.

        progress: training progress in [0, 1]; momentum goes base_momentum -> 1.0 on a
        cosine (I-JEPA). Returns the momentum used (for logging).
        """
        m = 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0
        self._ema_module(self.linear_projector, online_backbone.linear_projector, m)
        self._ema_module(self.X_Fusion_block, online_backbone.X_Fusion_block, m)
        self._ema_module(self.out_norm, online_backbone.out_norm, m)
        return m

    @torch.no_grad()
    def sync_buffers_from(self, online_backbone):
        """Copy the BatchNorm running buffers from the online branch into this copy.

        Called once after the BN warmup, BEFORE any optimizer step: at that point the EMA
        weights are still identical to the online weights, so the running stats should match
        too. Without this, the target branch would start normalizing with the (0, 1) init
        stats and drift toward the true stats for hundreds of steps (moving-target noise).
        """
        for ema_b, online_b in zip(self.linear_projector.buffers(),
                                   online_backbone.linear_projector.buffers()):
            ema_b.copy_(online_b)
        for ema_b, online_b in zip(self.X_Fusion_block.buffers(),
                                   online_backbone.X_Fusion_block.buffers()):
            ema_b.copy_(online_b)

    @torch.no_grad()
    def _ema_module(self, ema_mod, online_mod, m):
        for ema_p, online_p in zip(ema_mod.parameters(), online_mod.parameters()):
            ema_p.mul_(m).add_(online_p.detach(), alpha=1.0 - m)
        for ema_b, online_b in zip(ema_mod.buffers(), online_mod.buffers()):
            if torch.is_floating_point(ema_b):
                ema_b.mul_(m).add_(online_b, alpha=1.0 - m)
            else:  # integer buffers, e.g. BatchNorm num_batches_tracked
                ema_b.copy_(online_b)

    def forward_with_parts(self, mmwave_data, wifi_data, rfid_data):
        """Like forward(), but also returns the per-modality projected features (before the
        fusion) — [p_mm, p_wifi, p_rfid], each (B, 32, 512). The main-loss computation is
        numerically IDENTICAL to forward() (cat(parts, dim=1) ≡ linear_projector.forward on
        the full modality list); `parts` feed the cross-modal auxiliary prediction loss."""
        modality_list = [True, True, True]
        feature_list = self.feature_extractor(mmwave_data, wifi_data, rfid_data, modality_list)
        parts = self.linear_projector.forward_per_modality(feature_list)
        projected = torch.cat(parts, dim=1)
        z_tgt = self.out_norm(self.X_Fusion_block.forward_backbone(projected, modality_list))
        return z_tgt, parts

    def forward(self, mmwave_data, wifi_data, rfid_data):
        return self.forward_with_parts(mmwave_data, wifi_data, rfid_data)[0]
