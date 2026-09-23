"""X-Fi backbone for JEPA pretraining (JEPAREADME §4.1).

Copy of X_Fi.py with two targeted changes; every other line is kept identical (JEPACHANGES):
  1) X_Fusion gains forward_backbone(feature, modality_list) -> z_cm (B,32,512); the original
     forward() is re-expressed on top of it with identical behaviour.
  2) feature_extrator takes a configurable backbone_root and loads the .pt weights with
     map_location='cpu' (the released archives hold CUDA tensors; the caller moves the whole
     model to its device afterwards, e.g. model.to(device)).
New main class X_Fi_Backbone: frozen feature_extractor (eval, shared by reference with the
EMA target branch) -> linear_projector -> optional token-level mask (learnable mask_token
replacement, JEPAREADME §3.4 / §5.2 trap 2) -> X_Fusion up to z_cm. No classification head.
Used by jepa_pretrain.py / jepa_downstream.py.
"""
import os

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from backbone_models.mmWave.ResNet import *

"map each modality num of features to 32 "
" Use modality fusion transforemr"

class mmwave_feature_extractor(nn.Module):
    def __init__(self, mmwave_model):
        super(mmwave_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(mmwave_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 32

class wifi_feature_extractor(nn.Module):
    def __init__(self, wifi_model):
        super(wifi_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(wifi_model.children())[:-2])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x
    # shape: B, 512, 4

class rfid_feature_extractor(nn.Module):
    def __init__(self, rfid_model):
        super(rfid_feature_extractor, self).__init__()
        self.part = nn.Sequential(*list(rfid_model.children())[:-3])
    def forward(self, x):
        x = self.part(x).view(x.size(0), 512, -1)
        x = x.permute(0, 2, 1)
        return x 
    # shape: B, 512, 5




class feature_extrator(nn.Module):
    def __init__(self, backbone_root='./backbone_models'):
        super(feature_extrator, self).__init__()

        # CHANGE vs X_Fi.py: configurable backbone_root (was a hard-coded relative path) and
        # map_location='cpu' so building works on CPU-only machines too; caller does .to(device)
        mmwave_model = torch.load(os.path.join(backbone_root, 'mmWave/mmwave_ResNet18.pt'), map_location='cpu')
        mmwave_extractor = mmwave_feature_extractor(mmwave_model)
        mmwave_extractor.eval()

        wifi_model = torch.load(os.path.join(backbone_root, 'WIFI/wifi_ResNet18.pt'), map_location='cpu')
        wifi_extractor = wifi_feature_extractor(wifi_model)
        wifi_extractor.eval()

        rfid_model = torch.load(os.path.join(backbone_root, 'RFID/rfid_ResNet18.pt'), map_location='cpu')
        rfid_extractor = rfid_feature_extractor(rfid_model)
        rfid_extractor.eval()

        self.mmwave_extractor = mmwave_extractor
        self.wifi_extractor = wifi_extractor
        self.rfid_extractor = rfid_extractor

    def forward(self, mmwave_data, wifi_data, rfid_data, modality_list):
        if sum(modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            real_feature_list = []
            if modality_list[0] == True:
                mmwave_feature = self.mmwave_extractor(mmwave_data)
                # print("mmwave pre-trained model loaded")
                "shape b x 32 x 512"
                real_feature_list.append(mmwave_feature)
            if modality_list[1] == True:
                wifi_feature = self.wifi_extractor(wifi_data)
                # print("wifi pre-trained model loaded")
                "shape b x 3 x 512"
                real_feature_list.append(wifi_feature)
            if modality_list[2] == True:
                rfid_feature = self.rfid_extractor(rfid_data)
                # print("rfid pre-trained model loaded")
                "shape b x 5 x 512"
                real_feature_list.append(rfid_feature)
        
        return real_feature_list


class linear_projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_projector, self).__init__()
        '''Conv 1d layer for each modality'''
        self.mmwave_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.wifi_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(4, 32),
            nn.ReLU()
        )
        self.rfid_linear_projection = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(5, 32),
            nn.ReLU()
        )
    def forward_per_modality(self, feature_list):
        """Per-modality projected features from a FULL feature_list (all 3 modalities, in
        [mmWave, WiFi, RFID] order): returns [p_mm, p_wifi, p_rfid], each (B, 32, 512).
        Numerically identical to slicing the concatenated `forward` output; used by the
        cross-modal auxiliary prediction loss (JEPACHANGES §4-14). `forward` is untouched."""
        return [self.mmwave_linear_projection(feature_list[0].permute(0, 2, 1)).permute(0, 2, 1),
                self.wifi_linear_projection(feature_list[1].permute(0, 2, 1)).permute(0, 2, 1),
                self.rfid_linear_projection(feature_list[2].permute(0, 2, 1)).permute(0, 2, 1)]

    def forward(self, feature_list, modality_list):
        # example:
        # feature_list = [wifi_feature, rfid_feature]
        # modality_list = [False, True, True]
        feature_flag = 0
        for i in range(len(modality_list)):
            if modality_list[i] == True:
                if i == 0:
                    mmwave_feature = feature_list[feature_flag]
                    # print("mmwave_feature shape", mmwave_feature.shape)
                elif i == 1:
                    wifi_feature = feature_list[feature_flag]
                    # print("wifi_feature shape", wifi_feature.shape)
                elif i == 2:
                    rfid_feature = feature_list[feature_flag]
                    # print("rfid_feature shape", rfid_feature.shape)
                feature_flag += 1
            else:
                continue
        if sum (modality_list) == 0:
            raise ValueError("At least one modality should be selected")
        else:
            projected_feature_list = []
            if modality_list[0] == True:
                projected_feature_list.append(self.mmwave_linear_projection(mmwave_feature.permute(0, 2, 1)))
            if modality_list[1] == True:
                projected_feature_list.append(self.wifi_linear_projection(wifi_feature.permute(0, 2, 1)))
            if modality_list[2] == True:
                projected_feature_list.append(self.rfid_linear_projection(rfid_feature.permute(0, 2, 1)))
            projected_feature = torch.cat(projected_feature_list, dim=2).permute(0, 2, 1)
            "projected_feature shape: B, 32*n, 512"
            # if modality_list[3] == True:
            #     feature_shape = projected_feature.shape
            #     new_xyz = selective_pos_enc(lidar_points, feature_shape[1])
            #     # print("new_xyz shape", new_xyz.shape)
            #     'new_xyz shape: B, 32, 3'
            #     pos_enc = self.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)
            #     # print("pos_enc shape", pos_enc.shape)
            #     'pos_enc shape: B, 32, 512'
            #     # pos_enc_repeat = pos_enc.repeat(1, sum(modality_list), 1)
            #     projected_feature += pos_enc
            # else:
            #     pass
        return projected_feature

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 512, num_heads = 8, dropout = 0.0):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        out = self.pool(out)
        return out

class qkv_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(qkv_Attention,self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qkv):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class kv_projection(nn.Module):
    def __init__(self, dim, expension = 2):
        super(kv_projection,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * expension),
            nn.LayerNorm(dim * expension),
            nn.ReLU(),
            nn.Linear(dim * expension, dim)
        )
        # self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
    
    def forward(self, x):
        x = self.MLP(x)
        # q = self.to_q(self.norm(x))
        k = self.to_k(self.norm(x))
        v = self.to_v(self.norm(x))
        kv = [k, v]
        return kv

class cross_modal_transformer(nn.Module):
    def __init__(self, num_feature=32, max_num_modality=4, dim_expansion=2, emb_size = 512, num_heads = 8, dropout=0.3):
        super(cross_modal_transformer,self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.ffw = FeedForward(emb_size, emb_size*dim_expansion, dropout)
        self.pool = nn.AdaptiveAvgPool2d((32,None))
    
    def forward(self, feature_embedding, modality_list):
        feature_embedding_ = self.attention(feature_embedding) + self.pool(feature_embedding)
        out_feature_embedding = self.ffw(feature_embedding_) + feature_embedding_
        return out_feature_embedding

class fusion_transformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, dropout):
        super(fusion_transformer,self).__init__()
        self.mutihead_attention = qkv_Attention(dim, num_heads, dim_heads, dropout)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
    
    def forward(self, feature_embedding, kv):
    # def forward(self, x, modality_list):
        qkv = (feature_embedding, kv[0], kv[1])
        x = self.mutihead_attention(qkv) + feature_embedding
        new_feature_embedding = self.feed_forward(x) + x
        # x = self.mutihead_attention(x) + x
        # x = self.feed_forward(x) + x
        return new_feature_embedding


class cross_attention_transformer_block(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, dim_heads, num_modality, dropout):
        super(cross_attention_transformer_block,self).__init__()
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_modality):
            self.transformer_layers.append(fusion_transformer(dim, hidden_dim, num_heads, dim_heads, dropout))

    def forward(self, feature_embedding, kv_list, modality_list):
        "each layer fusion"
        transformer_layer_idx = 0
        feature_idx_ = 0
        features_list = []
        for layer in self.transformer_layers:
            # print("transformer_layer_idx", transformer_layer_idx)
            # print("feature_idx_", feature_idx_)
            if modality_list[transformer_layer_idx] == True:
                # print(qkv_list[feature_idx_][1:])
                new_feature_embedding = layer(feature_embedding, kv_list[feature_idx_])
                features_list.append(new_feature_embedding)
                transformer_layer_idx += 1
                feature_idx_ += 1
            else:
                transformer_layer_idx += 1
        feature_embedding = torch.cat(features_list, dim=1)
        return feature_embedding

class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27):
        super(classification_Head,self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.norm(x)
        x = self.fc(x)
        # x = x.view(x.size(0), 17, 3)
        return x
    
class X_Fusion(nn.Module):
    def __init__(self, num_modalities, dim, qkv_hidden_expansion, hidden_dim, num_feature, num_heads, dim_heads, model_depth, dropout, num_classes):
        super(X_Fusion,self).__init__()
        self.kv_layers = nn.ModuleList([])
        self.cross_attention_transformer = nn.ModuleList([])
        for _ in range(num_modalities):
            self.kv_layers.append(kv_projection(dim, qkv_hidden_expansion))
        # for _ in range(model_depth):
        #     self.transformer_layers.append(fusion_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities))
        self.cross_attention_transformer = cross_attention_transformer_block(dim, hidden_dim, num_heads, dim_heads, num_modalities, dropout)
        self.cross_modal_transformer = cross_modal_transformer(num_feature, num_modalities, qkv_hidden_expansion, dim, num_heads, dropout)
        self.depth = model_depth
        # self.qkv_layers: [qkv_layer_1, qkv_layer_2, ...]
        # self.transformer_layers: [
        #     [mod1_transformer_layer_1, mod2_transformer_layer_1, ...], 
        #     [mod1_transformer_layer_2, mod2_transformer_layer_2, ...],
        #     ...
        # ]
        # self.FFD = FeedForward(dim, hidden_dim)
        self.classification_head = classification_Head(dim, num_classes)

    def forward_backbone(self, feature, modality_list):
        "JEPA exit (JEPAREADME §4.1): the original forward minus the classification head."
        num_modalities = sum(modality_list)
        feature_list = list(feature.chunk(num_modalities, dim = 1))

        kv_list = []
        kv_layer_idx = 0
        feature_idx = 0
        for kv_layer in self.kv_layers:
            if modality_list[kv_layer_idx] == True:
                kv = kv_layer(feature_list[feature_idx])
                kv_list.append(kv)
                feature_idx += 1
                kv_layer_idx += 1
            else:
                kv_layer_idx += 1
                continue
        "output kv_list: [[k_1, v_1], [k_2, v_2], ...]"
        feature_embedding = self.cross_modal_transformer(feature, modality_list)
        "feature_embedding shape: B, 32, 512 (averaged sum of all features)"

        for i in range(self.depth):
            # print("dominant_q", dominant_q.shape)
            # print("qkv_list", qkv_list)
            # print("modality_list", modality_list)
            "feature_embedding shape: B, 32, 512"
            feature_embedding = self.cross_attention_transformer(feature_embedding, kv_list, modality_list)
            "feature_embedding shape: B, 32*n, 512"
            feature_embedding = self.cross_modal_transformer(feature_embedding, modality_list)
            "feature_embedding shape: B, 32, 512"
        # for transformer_layer in self.transformer_layers:
        #     dominant_q = transformer_layer(dominant_q, qkv_list, modality_list)

        return feature_embedding

    def forward(self, feature, modality_list):
        x = self.classification_head(self.forward_backbone(feature, modality_list))
        return x
        


class X_Fi(nn.Module):
    def __init__(self, model_depth, num_classes):
        super(X_Fi, self).__init__()
        self.feature_extractor = feature_extrator()
        self.linear_projector = linear_projector(512, 512)
        self.X_Fusion_block = X_Fusion(
            num_modalities = 3,
            dim = 512,
            qkv_hidden_expansion = 2,
            hidden_dim = 512,
            num_feature = 32,
            num_heads = 8,
            dim_heads = 64,
            model_depth = model_depth,
            dropout = 0.,
            num_classes = num_classes
        )
    def forward(self,  mmwave_data, wifi_data, rfid_data, modality_list):
        feature_list = self.feature_extractor(mmwave_data, wifi_data, rfid_data, modality_list)
        projected_features = self.linear_projector(feature_list, modality_list)
        out = self.X_Fusion_block(projected_features, modality_list)
        return out


class X_Fi_Backbone(nn.Module):
    """JEPA context encoder f_theta (JEPAREADME §3.2): frozen feature_extractor + linear_projector
    + X_Fusion up to z_cm. Reuses every submodule of X_Fi with identical internals; differences:
    - no classification head (z_cm is the output);
    - optional token-level mask applied AFTER linear_projector by replacing masked tokens with a
      learnable mask_token (JEPAREADME §5.2 trap 2) — token count stays 32 per modality, so the
      chunk() logic inside X_Fusion is untouched;
    - feature_extractor is frozen and kept in eval() so both branches (online / EMA target) see
      identical, deterministic features (JEPAREADME §3.3-2, §5.2 traps 3/4).

    token_mask: optional bool tensor (B, num_modalities, 32), True = masked; slices of absent
    modalities are ignored. forward returns z_cm (B, 32, 512).
    """
    def __init__(self, model_depth, num_classes=55, backbone_root='./backbone_models'):
        super(X_Fi_Backbone, self).__init__()
        self.feature_extractor = feature_extrator(backbone_root)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        self.linear_projector = linear_projector(512, 512)
        self.X_Fusion_block = X_Fusion(
            num_modalities = 3,
            dim = 512,
            qkv_hidden_expansion = 2,
            hidden_dim = 512,
            num_feature = 32,
            num_heads = 8,
            dim_heads = 64,
            model_depth = model_depth,
            dropout = 0.,
            num_classes = num_classes
        )
        # learnable mask token, broadcast as (B,1,512) (JEPAREADME §5.2 trap 2)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 512) * 0.02)
        # final LayerNorm on z_cm (CHANGE vs X_Fi.py): anchors the representation scale.
        # X_Fusion ends with un-normalized residuals; without this, the JEPA objective lets
        # the online/EMA representations grow without bound and the loss explodes (observed:
        # z scale drifted to ~7 while the predictor lagged -> SmoothL1 rose monotonically).
        # I-JEPA avoids this because its ViT encoders/predictor end in LayerNorm.
        self.out_norm = nn.LayerNorm(512)

    def train(self, mode=True):
        "Guard: the shared frozen extractor stays in eval() even when the backbone is set to train."
        super(X_Fi_Backbone, self).train(mode)
        self.feature_extractor.eval()
        return self

    def apply_token_mask(self, projected_feature, modality_list, token_mask):
        "projected_feature: (B, 32*n_present, 512); token_mask: (B, num_modalities, 32) bool"
        offset = 0  # only present modalities are packed contiguously, in modality order
        chunks = []
        for i, present in enumerate(modality_list):
            if present:
                seg = projected_feature[:, offset * 32:(offset + 1) * 32, :]
                m = token_mask[:, i, :].unsqueeze(-1)  # (B,32,1) -> broadcast against (1,1,512)
                chunks.append(torch.where(m, self.mask_token, seg))
                offset += 1
        return torch.cat(chunks, dim=1)

    def forward(self, mmwave_data, wifi_data, rfid_data, modality_list, token_mask=None):
        feature_list = self.feature_extractor(mmwave_data, wifi_data, rfid_data, modality_list)
        projected_features = self.linear_projector(feature_list, modality_list)
        if token_mask is not None:
            projected_features = self.apply_token_mask(projected_features, modality_list, token_mask)
        z_cm = self.X_Fusion_block.forward_backbone(projected_features, modality_list)
        return self.out_norm(z_cm)



