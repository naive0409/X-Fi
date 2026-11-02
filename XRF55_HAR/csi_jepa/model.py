"""csi_jepa model family (PROJECT_LOG §9 Phase A1).

CSITokenizer    raw CSI (270, 1000) -> structured patch tokens on a (grid_c, grid_t)
                grid with a FACTORIZED learnable positional embedding (link-axis +
                time-axis), so a dataset with a different link count/subcarrier count
                only changes the grid shape (Phase C MM-Fi adapter), not the design.
CSIJEPAEncoder  tokenizer + pre-LN transformer body (reuses jepa_modules.PreLNBlock)
                + learnable mask_token replacement (same pattern as X_Fi_Backbone)
                + final LayerNorm (the D17 scale anchor, built in from step 0).
                No BatchNorm anywhere -> no BN warmup / buffer sync / eval guards.
CSIEMATargetEncoder EMA copy of the ENTIRE encoder (tokenizer included — unlike
                rounds 1-5 there is no frozen extractor, so everything is EMA'd);
                permanently eval(); EMA math shared with jepa_modules via ema_module().
"""
import copy

import torch
from torch import nn

from jepa_modules import PreLNBlock, ema_module


class CSITokenizer(nn.Module):
    """Raw CSI (B, C, T) -> tokens (B, grid_c*grid_t, dim).

    patch = (patch_c, patch_t) with C % patch_c == 0 and T % patch_t == 0; the channel
    axis is grouped so that one token spans exactly the subcarriers of ONE RX link
    (audit_wifi_layout.py: 270 = 9 links x 30 subcarriers, amplitude-only), and the time
    axis is grouped into fixed spans (100 samples = 0.1 s — the empirically decorrelated
    unit). Positional embedding is factorized: pos_c (grid_c) + pos_t (grid_t), summed —
    a different link count reuses the same design via interpolation instead of a
    full-sized square pos embed.
    """

    def __init__(self, channels, time_steps, patch_c, patch_t, dim, input_norm='none'):
        super(CSITokenizer, self).__init__()
        assert channels % patch_c == 0, f'C={channels} not divisible by patch_c={patch_c}'
        assert time_steps % patch_t == 0, f'T={time_steps} not divisible by patch_t={patch_t}'
        self.grid_c = channels // patch_c
        self.grid_t = time_steps // patch_t
        self.input_norm_mode = input_norm
        if input_norm == 'channel_std':  # stats live INSIDE the model as buffers (trap 9:
            # the dataloader stays raw); set via set_input_stats() before training
            self.register_buffer('inp_mean', torch.zeros(1, 1, channels, 1))
            self.register_buffer('inp_std', torch.ones(1, 1, channels, 1))
        self.patch_embed = nn.Conv2d(
            1, dim, kernel_size=(patch_c, patch_t), stride=(patch_c, patch_t))
        self.pos_c = nn.Parameter(torch.zeros(1, self.grid_c, 1, dim))
        self.pos_t = nn.Parameter(torch.zeros(1, 1, self.grid_t, dim))
        nn.init.trunc_normal_(self.pos_c, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)

    def set_input_stats(self, mean, std):
        "Install corpus-computed per-channel mean/std (input_norm='channel_std' only)."
        assert self.input_norm_mode == 'channel_std'
        self.inp_mean.copy_(mean.reshape(self.inp_mean.shape))
        self.inp_std.copy_(std.reshape(self.inp_std.shape).clamp(min=1e-6))

    def forward(self, x):
        "x: (B, C, T) raw CSI -> (B, grid_c*grid_t, dim), c-major token order (c*T + t)."
        if self.input_norm_mode == 'channel_std':
            x = (x - self.inp_mean) / self.inp_std
        h = self.patch_embed(x.unsqueeze(1))                    # (B, dim, grid_c, grid_t)
        h = h.flatten(2).transpose(1, 2)                        # (B, grid_c*grid_t, dim), c-major
        pos = self.pos_c + self.pos_t                           # (1, grid_c, grid_t, dim)
        return h + pos.flatten(1, 2)


class CSIJEPAEncoder(nn.Module):
    "Tokenizer + transformer body + mask_token + final LN; forward(x, token_mask=None)."

    def __init__(self, channels=270, time_steps=1000, patch_c=30, patch_t=100,
                 dim=256, depth=8, num_heads=4, ffn_dim=1024, dropout=0.0,
                 input_norm='none'):
        super(CSIJEPAEncoder, self).__init__()
        self.tokenizer = CSITokenizer(channels, time_steps, patch_c, patch_t, dim,
                                      input_norm)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [PreLNBlock(dim, num_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)  # D17 scale anchor — parameter-free LN semantics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.num_tokens = self.tokenizer.grid_c * self.tokenizer.grid_t

    def apply_token_mask(self, tokens, token_mask):
        "tokens: (B, L, dim); token_mask: (B, L) bool, True = replace with mask_token."
        return torch.where(token_mask.unsqueeze(-1), self.mask_token, tokens)

    def forward(self, x, token_mask=None):
        h = self.norm_in(self.tokenizer(x))
        if token_mask is not None:
            h = self.apply_token_mask(h, token_mask)
        for blk in self.blocks:
            h = blk(h)
        return self.norm_out(h)


class CSIEMATargetEncoder(nn.Module):
    """EMA copy of the WHOLE online encoder (tokenizer included).

    Differs from rounds-1-5 EMATargetEncoder in one structural way: there is no frozen
    feature extractor to share by reference — the extractor IS the trainable tokenizer —
    so the target branch EMA-copies every parameter. No BatchNorm -> no buffer sync.
    Permanently eval() (no buffers to drift, but keeps dropout semantics deterministic).
    """

    def __init__(self, online_encoder, base_momentum=0.996):
        super(CSIEMATargetEncoder, self).__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for p in self.parameters():
            p.requires_grad_(False)
        self.base_momentum = base_momentum
        self.eval()

    def train(self, mode=True):
        "Guard: the EMA branch always stays in eval()."
        return super(CSIEMATargetEncoder, self).train(False)

    @torch.no_grad()
    def update(self, online_encoder, progress=0.0):
        """EMA-update all params; progress in [0,1] drives cosine momentum base->1 (I-JEPA)."""
        import math
        m = 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0
        ema_module(self.encoder, online_encoder, m)
        return m

    def forward(self, x):
        "Full unmasked input -> target latents (B, L, dim). Call under torch.no_grad()."
        return self.encoder(x)
