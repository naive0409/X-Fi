# README：基于 X-Fi 仓库实现 JEPA 预训练的多模态感知基础模型（X-Fi-JEPA Baseline）

> **给 coding agent 的话**：本文档是自包含的实现规格。你不需要任何外部上下文——所有背景、代码事实、设计决策、验收标准都在本文档中。请通读全文后再动手。文档中所有对 X-Fi 代码的引用（文件、类名、行号）均已对照仓库真实代码核实。

---

## 0. 工作方式约定（最高优先级，必须遵守）

1. **环境用 conda 管理**。conda 环境已由人工创建，**名称为 `xfi`**——你的一切命令都用 `-n xfi`，**不得自行创建/重命名/删除环境**；若发现环境缺依赖，报告并给出安装命令，由人工执行。仓库代码实际依赖超出 requirements.txt 的包（代码里 `import pandas、h5py、cv2、yaml`，见 `XRF55_Dataset.py` 头部）——若环境中缺失，把补装命令（`conda run -n xfi pip install ...` 形式）写入 PROJECT_LOG.md 交人工执行。`environment.yml`（`conda env export -n xfi` 导出）仍需作为交付物之一（§9）。不得使用 pip/venv 直装到系统环境。
2. **你不运行训练。你的职责是把代码和命令准备好，训练一律由人工执行。** 所有训练/评估命令写清"完整命令行 + 每个参数的含义"，不要自行启动任何长时训练。允许你运行的是：轻量验证（python -c 的 shape 检查、单 batch 前向、1-2 个 iteration 的冒烟测试）——这些可以跑，目的是确认代码正确，不是训练。
3. **训练命令用 `conda run` 形式给出**，保证在指定环境中执行且输出可捕获。格式模板：
   ```bash
   conda run -n xfi --no-capture-output python jepa_pretrain.py --dataset <XRF55路径> --config configs/jepa_baseline.yaml
   ```
   每条命令后面附参数表。写 `--no-capture-output`（否则 conda run 会缓冲 tqdm/日志输出，看不到实时进度）。
4. **维护一个 `PROJECT_LOG.md` 文件**（项目文档，持续更新，作为交付物），包含且不限于：
   - **命令手册**：每条需要人工运行的命令 + 逐参数解释 + 预期运行时长与输出
   - **项目结构**：新增/修改文件的树状说明（每个文件一行职责描述）
   - **数据集说明**：XRF55 的组织方式、本管线读取哪些部分、预处理与归一化约定
   - **模型结构**：JEPA 版模型的模块树（含与原 X-Fi 的差异标注）
   - **开发日志**：你做了什么决定、遇到了什么问题、如何解决（简明列表）
   每完成一个任务清单项（§4）就更新一次，不要最后补写。

---

## 1. 项目背景与目标

### 1.1 我们要做什么

构建一个**室内多模态人体感知基础模型**（Foundation Model, FM）：

- **输入**：多模态传感器数据（WiFi CSI、毫米波雷达、RFID 等，后续会加入摄像头等）
- **核心模态**：WiFi CSI（我们实验室用 Intel 5300 网卡采集，采集链路只能导出 CSI，**拿不到原始 IQ**）
- **预训练方式**：自监督（无标签），采用 **JEPA**（Joint-Embedding Predictive Architecture）目标
- **下游任务**：HAR（人体活动识别）、姿态估计等**理解类任务**
- **关键能力**：训练一次后，任意模态子集（含单模态）都能独立工作——即**模态缺失鲁棒性**

### 1.2 实现策略：为什么不重写、而是改 X-Fi

**X-Fi**（ICLR 2025, NTU MARS Lab, arXiv:2410.10167）是本领域唯一与我们目标严格对齐的开源工作，它已解决最难的部分：

| X-Fi 已提供               | 作用                                    |
| :--------------------- | :------------------------------------ |
| 各模态特征提取器（预训练 ResNet18） | 统一把不同模态映射到 token 序列                   |
| `linear_projector`     | 把各模态可变数量的 token 统一到 32 tokens × 512 维 |
| `X_Fusion`（跨模态融合）      | 模态不变融合：任意模态子集都能前向，缺失模态直接跳过            |
| `modality_list` 机制     | 以布尔列表声明"本样本哪些模态在场"，缺失即子集变小            |

**JEPA 部分没有可直接复用的开源多模态实现**（JEPA-MSAC arXiv:2603.29796 未开源；Wireless World Model arXiv:2603.25216 代码仓库已无法访问）。但 JEPA 所需的组件（EMA 目标编码器、预测器、潜空间损失、掩码策略）都是**标准件**，参考实现为 Meta 的 **I-JEPA**（github.com/facebookresearch/ijepa，可访问）。

因此本项目的实现 = **X-Fi 骨架（融合与模态处理，尽量不改） + JEPA 四件套（新写，参考 I-JEPA）**。

### 1.3 一句话概括改动

> 把 X-Fi 的"监督分类训练"替换为"JEPA 潜空间预测预训练"；X-Fi 的融合机制（X_Fusion）与模态处理（feature_extractor / linear_projector）**保持原样**；新增预测器、EMA 目标编码器、潜空间损失、两级掩码。下游仍用 X-Fi 的分类头做线性探测评估。

---

## 2. X-Fi 仓库现状（已核实的代码事实）

### 2.1 目录结构

```
X-Fi/
├── MMFi_HAR/          # MM-Fi 数据集 HAR 任务：4 模态 = rgb, depth, mmwave, lidar
├── MMFi_HPE/          # MM-Fi 数据集 HPE（姿态估计）任务：同上 4 模态
├── XRF55_HAR/         # XRF55 数据集 HAR 任务：3 模态 = mmwave, wifi, rfid ←★唯一含 WiFi CSI 的版本
├── requirements.txt   # scipy 1.9.1, numpy 1.23.0, opencv 4.5.5, tqdm, einops 0.4.0
├── LICENSE            # Apache 2.0
└── README.md          # 官方说明（数据与预训练权重下载方式）
```

> ⚠️ **关键事实**：**MMFi_HAR 与 MMFi_HPE 的代码中没有任何 WiFi/CSI 分支**（模态为 rgb/depth/mmwave/lidar）。含 CSI 的只有 `XRF55_HAR/`。我们的工作以 WiFi CSI 为核心，因此 **一切改动以 `XRF55_HAR/` 为基础**。MM-Fi 数据集如需使用，需自行为其补 WiFi 分支（见 §7.3）。

### 2.2 `XRF55_HAR/X_Fi.py` 的关键结构与行号（共 423 行）

| 组件 | 类名 | 位置 | 说明 |
|:--|:--|:--|:--|
| 模态特征提取器 | `mmwave_feature_extractor` / `wifi_feature_extractor` / `rfid_feature_extractor` | L26-56 | 加载预训练 ResNet18（`torch.load('./backbone_models/...pt')`，**相对路径，必须从 XRF55_HAR 目录运行**），输出 `B × 512 × N_tokens`。**注意 token 数不同**：mmwave=32, wifi=**4**, rfid=5 |
| 提取器容器 | `feature_extrator` | L59-100 | `forward(mmwave_data, wifi_data, rfid_data, modality_list)`，按 `modality_list` 布尔值决定哪些模态前向，返回 feature 列表 |
| 统一投影 | `linear_projector` | L103-170 | 每模态一个分支：`Conv1d(512,512,1)+BN+ReLU+Linear(N_tokens,32)+ReLU`，把可变 token 数统一为 32。输出沿 token 维拼接为 `B × 32n × 512`（n=在场模态数）。**L160-172 有被注释掉的 LiDAR 位置编码**（XRF55 版无 LiDAR，未启用） |
| 注意力件 | `MultiHeadAttention` / `qkv_Attention` / `FeedForward` | L173-248 | 标准 transformer 组件，含 `AdaptiveAvgPool2d((32,None))` 池化 |
| KV 投影 | `kv_projection` | L250-270 | `LayerNorm+MLP`，每模态专属，输出 `[k, v]` |
| 跨模态变换器 | `cross_modal_transformer` | L272-282 | 自注意力 + 池化 → 输出统一表征 **z_cm（`B × 32 × 512`）** |
| 交叉注意力块 | `cross_attention_transformer_block` / `fusion_transformer` | L284-324 | z_cm 作 query，逐在场模态做 cross-attention，输出沿 token 维拼接回 `B × 32n × 512` |
| 分类头 | `classification_Head` | L326-338 | mean pooling (32 tokens→1) + LayerNorm + Linear(512, num_classes) |
| **融合主体** | `X_Fusion` | L340-409 | `forward(feature, modality_list)`：① `feature.chunk(sum(modality_list), dim=1)` 切回各模态 → ② 在场模态各过专属 `kv_projection` 生成 KV（缺失模态直接跳过，**无零填充**）→ ③ `cross_modal_transformer(feature)` 生成 z_cm → ④ `depth` 轮循环（cross-attention → 重新聚合回 z_cm）→ ⑤ 分类头输出 |
| 主模型 | `X_Fi` | L411-423 | `forward(mmwave_data, wifi_data, rfid_data, modality_list)`：feature_extractor → linear_projector → X_Fusion_block → **分类结果** |

### 2.3 训练代码事实（`XRF55_HAR/run.py` + `utils.py`）

- 模型实例化：`X_Fi(model_depth=5, num_classes=55)`（XRF55 有 55 类动作）
- 优化器：AdamW，lr=1e-4，**只训练 `linear_projector` 和 `X_Fusion_block` 的参数**——模态特征提取器是冻结的预训练权重
- 损失：CrossEntropyLoss（监督分类）
- epochs=100，batch_size=16（训练）/32（测试），种子 3407
- **模态子集采样发生在 collate 阶段、逐样本进行**：`utils.py` 的 `generate_none_empth_modality_list()` 以固定概率独立决定每个模态是否在场——**wifi 90% 保留、mmwave 50%、rfid 60%**，全 False 时重采样。`random.seed(epoch)` 保证按 epoch 可复现
- `validate_all.py`：在**所有模态组合**上评估（`multi_test` 函数按组合分别统计精度）

### 2.4 数据事实（`XRF55_Dataset.py`）

- 数据组织：`<root>/train_data|test_data/{RFID,WiFi,mmWave}/SceneN/.../*.npy`，同一索引下三模态文件名一一对应（replace 互换）
- 每样本：`wifi_data`（np.load 直接读）、`rfid_data`、`mmwave_data`（reshape 为 `17×256×128`）、label（从文件名解析，0-indexed）
- 预训练模态权重下载：官方 README 的 Google Drive 链接，`.pt` 文件放入 `XRF55_HAR/backbone_models/{mmWave,WIFI,RFID}/`

---

## 3. X-Fi-JEPA 模型定义

### 3.1 总体架构

```
                        ┌─────────────────────────────────────────────┐
                        │  在线分支（online branch，参与梯度更新）          │
 输入(被掩码后的子集) ──→ │  feature_extractor(冻结) → linear_projector   │
                        │  → X_Fusion(截至 z_cm，去掉分类头)              │──→ 上下文表征 z_cm (B,32,512)
                        └─────────────────────────────────────────────┘         │
                                                                                ↓
                                                                        预测器 predictor（新写）
                                                                                ↓
                                                                        预测的目标表征 ẑ (B,32,512)
                                                                                │
                        ┌─────────────────────────────────────────────┐         │  潜空间损失
                        │  目标分支（target branch，无梯度）               │         │  (smooth L1)
 输入(完整，无掩码) ──→  │  feature_extractor(同一冻结权重，共享)          │         │
                        │  → EMA(linear_projector + X_Fusion)          │──→ 目标表征 z_tgt (B,32,512)
                        │      ↑ EMA 滑动平均，只前向不反传               │         │
                        └─────────────────────────────────────────────┘─────────┘
```

### 3.2 组件对应关系（术语表，便于与论文对照）

| 本文档术语      | JEPA 论文术语（英文原名）                                         | X-Fi 代码中的对应物                                                                       | 状态          |
| :--------- | :------------------------------------------------------ | :--------------------------------------------------------------------------------- | :---------- |
| 在线分支       | context encoder $f_\theta$（online encoder）              | `feature_extractor`（冻结）+ `linear_projector` + `X_Fusion` 截至输出 z_cm                 | 复用          |
| 上下文表征 z_cm | context representation                                  | `X_Fusion.forward` 中 `cross_modal_transformer` 产出的 `feature_embedding`（`B×32×512`） | 复用（需暴露该中间量） |
| 预测器        | predictor $g_\phi$                                      | **X-Fi 无对应物**（X-Fi 从 z_cm 直接接分类头）                                                  | **新写**      |
| 目标编码器      | target encoder $f_{\bar\theta}$（EMA / momentum encoder） | **X-Fi 无对应物**。= `linear_projector + X_Fusion` 的 EMA 副本（**不含**特征提取器，见 §3.4）         | **新写**      |
| 目标表征 z_tgt | target representation                                   | 与 z_cm 同样的计算，但权重来自 EMA 副本、输入为完整（无掩码）数据                                             | 机制复用        |
| 潜空间损失      | latent prediction loss                                  | X-Fi 是监督 CrossEntropy；**替换**为 smooth L1                                            | 替换          |
| 任务头        | lightweight task head / linear probe                    | `classification_Head`                                                              | 复用（下游阶段用）   |

### 3.3 三个关键设计决策（不可偏离）

1. **在线分支中，模态特征提取器保持冻结（沿用 X-Fi 的预训练 .pt 权重），可训练部分 = `linear_projector` + `X_Fusion` + `predictor`。** 理由：复用 X-Fi 官方发布的单模态预训练权重（否则小数据上从零训 ResNet18 会欠拟合），且与 X-Fi 原始训练设定（优化器只含 projector+X_Fusion）保持一致，对照实验才干净。
2. **EMA 只覆盖 `linear_projector + X_Fusion`（+predictor 不需要，predictor 只在在线分支）。** 因为特征提取器两侧完全相同且冻结，对它做 EMA 是无操作，白白多占显存——目标分支直接**共享**同一份冻结权重即可。
3. **目标分支的输入是完整数据（不掩码）；上下文分支的输入是被掩码的数据。** 预训练目标：从被掩码的部分模态/词元，推断完整输入的表征。这是 I-JEPA 的标准设定。

### 3.4 掩码策略（两级）

**模态级掩码（主机制）**：
- 复用 X-Fi 的 `modality_list` 布尔机制，但**采样逻辑从 collate 阶段移到训练循环内**（见 §5.2 陷阱 1）
- 概率设定：wifi（CSI，核心模态）保留概率 **0.9**；mmwave 0.5；rfid 0.5；保证至少一个模态在场；**记录每种子集的采样频率**，确保 CSI-only 子集出现频率不低于 ~15%（弱模态不能是长尾）

**词元级掩码（次机制，从简）**：
- 对在场模态的 32 个 token，以掩码率 **0.25** 随机置为不可见（从进入 `X_Fusion` 前的序列中删除）
- **不要过度设计**：wifi 只有 4 个源 token（投影后 32 个），mmwave 32 个、rfid 5 个——词元级掩码的物理意义有限，模态级掩码才是主力。此设计参考 WiFi-JEPA（arXiv:2607.11064）的链路掩码思想，但因我们 token 数小，不做复杂的结构化掩码
- 注意：删除 token 后序列变短，`X_Fusion` 的 chunk 逻辑假设每模态 32 token——**词元掩码在 `linear_projector` 之后、`chunk` 之前实现**，把被掩 token 从该模态的 32 token 中去掉并记住每个模态的实际 token 数（详见 §5.2 陷阱 2 的实现建议）

**固定种子**：掩码生成用独立的 `numpy RandomState` / `torch.Generator`，每个样本记录其掩码（便于复现与调试）。

### 3.5 损失函数

```
L = SmoothL1Loss( predictor(z_cm), z_tgt.detach() ) [+ 可选的方差正则]
```

- 对 `B×32×512` 的**完整 token 级表征**计算损失（不是 mean pooling 后的向量），让每个 token 位置都有学习信号
- `z_tgt` 必须 `detach()`（目标分支无梯度）
- **防塌缩监控（必做）**：每个 epoch 统计 `z_tgt` 的 token 级标准差（对 batch 维）。若标准差持续趋近 0，说明表征塌缩。I-JEPA 靠 EMA 机制防止塌缩，通常够用；若监控发现塌缩，再加 VICReg 式方差-协方差正则（权重 1.0/0.01，先不加，留作故障开关）

---

## 4. 具体实现任务清单

在 `XRF55_HAR/` 下进行。建议新建文件，**尽量不改动原文件**（保留 X-Fi 原始行为做对照）：

### 4.1 新建 `X_Fi_backbone.py`（从 `X_Fi.py` 改造）

把 `X_Fi.py` 整体复制为新文件，做两处修改：

1. **`X_Fusion` 增加暴露 z_cm 的出口**：`forward(feature, modality_list)` 在 depth 循环结束后、调用 `classification_head` 之前返回 z_cm。推荐做法：给 `X_Fusion.forward` 加参数 `return_zcm=False`；或新增方法 `forward_backbone(feature, modality_list) -> z_cm`。
2. **主类 `X_Fi` 增加 backbone 模式**：
   ```python
   class X_Fi_Backbone(nn.Module):
       # 复用 X_Fi 的全部子模块；区别：
       # - 不含/不使用 classification_head
       # - forward(mmwave_data, wifi_data, rfid_data, modality_list, token_mask=None) -> z_cm (B,32,512)
       # - token_mask: 可选，词元级掩码（见 §3.4），在 linear_projector 之后应用
   ```
   其余（feature_extractor 加载、linear_projector、X_Fusion 内部逻辑）**逐行保持**。

### 4.2 新建 `jepa_modules.py`（JEPA 四件套）

```python
class Predictor(nn.Module):
    # 轻量 transformer 或 MLP：
    # 推荐：4 层 transformer encoder（dim=512, heads=8, ffn=2048, pre-LN）+ 输出 LayerNorm
    # 输入 z_cm (B,32,512)（可加可学习的 mask token embedding 以区分"这是推断"）
    # 输出 ẑ (B,32,512)

class EMATargetEncoder(nn.Module):
    # __init__(online_projector, online_fusion, momentum=0.996):
    #   deepcopy linear_projector 和 X_Fusion（不含特征提取器！特征提取器直接共享引用）
    #   所有参数 requires_grad=False
    # update(online): 每个训练 step 后调用
    #   ema_p = m*ema_p + (1-m)*online_p   对 linear_projector 和 X_Fusion 的所有浮点参数与 buffer
    # momentum 调度（参考 I-JEPA）：m 从 0.996 余弦升到 1.0（随训练进度）
    # forward: 与 online 相同的接口，输出 z_tgt (B,32,512)
```

### 4.3 新建 `jepa_pretrain.py`（预训练主脚本）

职责：
1. 构建 `X_Fi_Backbone`（在线）、`EMATargetEncoder`、`Predictor`
2. 加载 XRF55 数据（`XRF55_Dataset`，**忽略 label**——仅作预训练）
3. 训练循环（见 §5.1）
4. 保存 checkpoint：`{online_state, ema_state, predictor_state, optimizer_state, epoch, config}`

### 4.4 新建 `jepa_downstream.py`（下游评估脚本）

1. 加载预训练 checkpoint，冻结 `linear_projector + X_Fusion`（EMA 分支弃用）
2. 新建随机初始化的 `classification_Head(512, 55)`（或轻量 MLP 头）接在 z_cm 后
3. 用 label 做监督训练（**只训头**，即线性探测 linear probe；可另设一个"微调"模式解冻 projector+X_Fusion，lr 降到 1e-5）
4. 评估：调用/改造 `validate_all.py` 的 `multi_test` 逻辑，**逐模态子集报告精度，并计算 worst-subset（最差子集）精度**——这是我们的必报指标（X-Fi 原版只报平均，我们要补齐子集维度）

### 4.5 新建 `jepa_eval.sh` / `configs/`

把超参集中到 config（掩码率、EMA 动量、lr、epochs、子集采样概率），不要散落在代码里。

---

## 5. 训练流程

### 5.1 阶段 1：JEPA 预训练（无标签）

每个 batch 的步骤：

```
1. 取 batch（三模态数据 + label[忽略]）
2. 采样模态级掩码 → context_modality_list（按 §3.4 概率）
3. 采样词元级掩码 → token_mask（对 context 中在场的模态，掩码率 0.25）
4. z_cm = online(mmwave, wifi, rfid, context_modality_list, token_mask)
5. with torch.no_grad():
       z_tgt = ema_target(mmwave, wifi, rfid, [True,True,True], token_mask=None)  # 完整输入
   # 注意：目标分支的 modality_list 恒为全 True、无词元掩码
6. ẑ = predictor(z_cm)
7. loss = SmoothL1(ẑ, z_tgt.detach())
8. loss.backward(); optimizer.step()   # 优化：linear_projector + X_Fusion + predictor
9. ema_target.update(online)
```

超参起点（参考 I-JEPA 与 X-Fi 的量级，允许 agent 微调并在日志中记录）：

| 项 | 值 |
|:--|:--|
| optimizer | AdamW（params: linear_projector + X_Fusion + predictor），lr=5e-4，wd=0.05 |
| lr schedule | 余弦退火 + 5 epoch 线性 warmup |
| epochs | 100（XRF55 train split） |
| batch_size | 16（与 X-Fi 一致；显存不够可降） |
| EMA momentum | 0.996 → 1.0 余弦调度 |
| 词元掩码率 | 0.25 |
| 模态保留概率 | wifi 0.9 / mmwave 0.5 / rfid 0.5 |

### 5.2 已知陷阱（每条都有真实代码依据，务必处理）

1. **模态采样的位置错了会破坏复现性**：X-Fi 原版在 `collate_fn_padd`（utils.py L19-47）里逐样本随机生成 `modality_list`，靠 `random.seed(epoch)` 复现。JEPA 需要每个样本有**两套**掩码（context 的模态+词元掩码、target 的全可见），collate 里做不了。**把所有掩码采样移到训练循环内**，用独立于 epoch 种子的 generator，并把每个样本的掩码写入日志（首 epoch 抽样记录即可）。
2. **`chunk` 假设每模态 token 数相等**：`X_Fusion.forward` L388 `feature.chunk(sum(modality_list), dim=1)` 假设拼接序列按模态顺序、每模态恰好 32 token。若实现词元掩码时直接删 token，chunk 会错位。**推荐实现**：词元掩码不物理删除，而是把被掩 token 的特征替换为可学习的 `mask_token`（B,1,512 广播）——这样 token 数恒为 32，chunk 逻辑完全不用动，且与 predictor 的"推断被掩位置"语义一致。被掩 token 不应参与自注意力的 key/value？——**不需要**，mask token 参与注意力是 MAE/I-JEPA 常规做法，保持简单。
3. **BatchNorm 的 train/eval 不一致**：`linear_projector` 内有 `BatchNorm1d`（L107 等三处）。在线分支在 train 模式、EMA 目标分支如果也处于 train 模式，BN 的 running stats 会被双份更新且两支统计量漂移。**处理**：EMA 分支整体 `.eval()`（只影响 BN/dropout，不影响 EMA 参数语义）；或更干净——新代码中把 projector 的 BN 换成 LayerNorm（在 `X_Fi_backbone.py` 中改，注明与原版的差异）。
4. **EMA 不要 deepcopy 特征提取器**：见 §3.4 决策 2。特征提取器两个分支共享同一实例（其权重永不更新，`.eval()` 冻结 BN）。
5. **相对路径**：`feature_extrator.__init__` 里 `torch.load('./backbone_models/...')` 是相对 cwd 的路径。所有脚本必须从 `XRF55_HAR/` 目录运行，或在代码中改为绝对/配置化路径（推荐改，注明）。
6. **modality_list 全 False**：`generate_none_empth_modality_list` 会重采样保证非空；我们自己采样时同样必须保证 context 至少一个模态在场（target 恒全 True，无此问题）。
7. **`multi_test` 的统计口径**：原版 `validate_all.py`/`utils.py` 的 `multi_test` 按 6 种组合分别累计精度。改造时注意它原来没有 worst-subset 概念——新增。另外其精度是 batch 粒度平均（不是样本粒度），评估大集合时改成分样本累计更准。
8. **随机种子一致性**：X-Fi 用 seed 3407（模型初始化）+ `random.seed(epoch)`（模态采样）+ `val_random_seed`（验证）。我们的脚本应统一用一个 config 里的 seed 派生：模型初始化、掩码 generator、数据顺序三者分开。
9. **XRF55 的 wifi 数据形状**：`wifi_feature_extractor` 期望能被 ResNet18(1D) 处理并输出 `B×512×4`。任何数据预处理改动（如归一化）都会破坏预训练权重的输入分布——**不要动数据加载，保持 `np.load` 原样**。

### 5.3 阶段 0：先行校验（半天量级，必做）

在写 JEPA 之前，先让原样跑通 X-Fi 监督版，确认环境与数据无误。**命令由你准备好写入 PROJECT_LOG.md，由人工运行**（见 §0 约定 2/3）：

```bash
# 人工运行；参数 --dataset：XRF55 数据集根目录（含 train_data/ 与 test_data/）
conda run -n xfi --no-capture-output python run.py --dataset <XRF55数据集路径>
```

你的职责：确认环境依赖完整（可先跑一个仅 CPU 的 import + 模型构建检查）、命令正确写入 PROJECT_LOG.md。训练与验收由人工汇报：损失正常下降、100 epoch 后验证精度与论文量级一致（HAR ~80% 区间）。环境有问题时优先修复环境，不要带病开工。

### 5.4 阶段 2：下游评估

- **对比组（全部必须）**：
  - A. X-Fi 原版监督训练（阶段 0 的模型）
  - B. X-Fi-JEPA 预训练 + 线性探测（我们的方法）
  - C. 随机初始化 projector+X_Fusion + 线性探测（JEPA 是否真学到东西的下界）
  - D.（可选）同骨架 MAE（把 JEPA 损失换成"重建被掩 token 的原始输入"）——留接口，不强制
- **报告口径（必改）**：每个模态子集（mmwave / wifi / rfid / mmwave+wifi / mmwave+rfid / wifi+rfid / 全部）分别报告精度；**额外报告 worst-subset**。
- **验收标准（预训练成功的判据）**：
  1. 预训练损失稳定下降、无 NaN
  2. z_tgt token 级标准差不塌缩（>0.1 量级，持续监控）
  3. **B 的平均精度显著优于 C**（差值 <2% 视为 JEPA 没学到东西，需排查）
  4. B 在弱模态子集（如 rfid-only / wifi-only）上相对 C 的提升**大于**其在全模态上的提升（说明预训练表征对子集泛化有贡献）

---

## 6. 数据准备

1. 下载 XRF55：github.com/aiotgroup/XRF55-repo（官方链接在 X-Fi 的 README 中）
2. 下载预训练模态权重：X-Fi README 中 Google Drive 链接，`.pt` 放入 `XRF55_HAR/backbone_models/{mmWave,WIFI,RFID}/`
3. 目录约定：数据放 `X-Fi/Data/XRF55_Dataset/`（与官方 README 一致）
4. 预训练阶段不使用 label；下游阶段使用 train/test split（`XRF55_Dataset` 的 `is_train` 参数）

> MM-Fi 数据集（github.com/ybhbingo/MMFi_dataset）暂缓：其官方 X-Fi 代码没有 WiFi 分支，若后续需要（扩大预训练数据），需为 `MMFi_HAR/X_Fi.py` 补 WiFi 分支后再接入同一套 JEPA 管线。这不在本次任务范围内，但架构上不要把改动写死在 XRF55 上（模块化，见 §4）。

---

## 7. 代码风格与工程约定

- **Python 3.10 / PyTorch 2.1.1**（与 requirements.txt 一致）；用 `einops`（仓库已依赖）
- 新文件放 `XRF55_HAR/` 下；**不修改原 `X_Fi.py`/`utils.py`/`run.py`**（它们是对照基线）；确需改动公共逻辑时复制到新文件
- 所有新超参进 config 文件；日志用 tqdm + print（沿用仓库风格）即可，但要记录：每 epoch 损失、z_tgt 标准差、各模态子集的采样频率
- checkpoint 保存：在线/EMA/predictor 三份 state_dict 分开存
- 每个新文件头部写 3-5 行 docstring：该文件在整体方案中的角色（对应本文档 §几）

## 8. 与后续工作的接口（本次不实现，但不要堵死）

以下是我们研究计划中的后续创新点，本次 baseline **不做**，但实现时保持模块化以便插入：

1. **场景先验通道**：floor plan + 单 AP 位置编码为额外 token 加入 `X_Fusion`（做法：modality_list 加一位恒 True 的"场景先验模态"，新增一个 `scene_encoder` 产出 `B×32×512` 的场景 token，走同一条 linear_projector→X_Fusion 通路）。届时 `num_modalities` 从 3 改 4。
2. **位置编码来源替换**：X-Fi 的位置编码绑定 LiDAR（XRF55 版中已注释掉，见 §2.2 L160-172）。未来位置信息将由场景先验生成——请保留 `linear_projector` 中被注释的 pos_enc 代码原样，不要清理它。
3. **坐标回归头**：未来加回归任务（定位），头结构与分类头并列。
4. **按部署配置采样**：未来模态保留概率将由部署场景的配置文件驱动（而非固定 0.5/0.9）——把 §3.4 的采样概率做成 config 可读的表。

---

## 9. 交付物清单

| 交付物                                    | 说明                                                   |
| :------------------------------------- | :--------------------------------------------------- |
| `XRF55_HAR/X_Fi_backbone.py`           | 改造后的骨干（暴露 z_cm；BN 处理按 §5.2 陷阱 3）                     |
| `XRF55_HAR/jepa_modules.py`            | Predictor + EMATargetEncoder                         |
| `XRF55_HAR/jepa_pretrain.py`           | 预训练脚本（阶段 1）                                          |
| `XRF55_HAR/jepa_downstream.py`         | 下游评估脚本（阶段 2，含逐子集 + worst-subset 报告）                  |
| `XRF55_HAR/configs/jepa_baseline.yaml` | 全部超参                                                 |
| `environment.yml`                      | conda 环境定义（§0 约定 1，含代码实际依赖的 pandas/h5py/opencv/yaml） |
| `PROJECT_LOG.md`                       | 项目文档（§0 约定 4：命令手册/项目结构/数据集说明/模型结构/开发日志，持续更新）         |
| `JEPACHANGES.md`                       | 一页变更说明：列出与原 X-Fi 的每一处差异及理由（供人工审查）                    |
| 训练命令清单                                 | 所有需人工运行命令的 conda run 形式（收录在 PROJECT_LOG.md，§0 约定 3）  |
| 预训练 checkpoint + 训练日志                  | 含损失曲线与 z_tgt 标准差曲线（由人工运行产出）                          |
| 对比实验表                                  | §5.4 的 A/B/C 组 × 6 子集精度矩阵 + worst-subset（由人工运行产出）    |

## 10. 参考实现与论文（按用途）

| 资源                                        | 用途                                                                  |
| :---------------------------------------- | :------------------------------------------------------------------ |
| X-Fi 仓库本体（本仓库）                            | 骨架全部复用；`XRF55_HAR/` 为基准目录                                           |
| I-JEPA（github.com/facebookresearch/ijepa） | EMA 更新与 momentum 调度、predictor 结构、损失的标准写法（`utils/` 下有 momentum 相关工具） |
| X-Fi 论文 arXiv:2410.10167                  | X_Fusion 机制理解（KV 在 depth 轮迭代中保持不变，z_cm 每轮重新聚合）                      |
| I-JEPA 论文 arXiv:2301.08243                | JEPA 预训练配方                                                          |
| WiFi-JEPA arXiv:2607.11064（仅论文，无代码）       | CSI 分词与链路掩码思想参考（我们已简化，见 §3.4）                                       |
| CSI-JEPA arXiv:2605.14171（仅论文）            | CSI 上 JEPA 掩码设计参考                                                   |

---

*文档版本：v1.1（2026-09-17）。所有代码行号基于 commit 时的 `XRF55_HAR/X_Fi.py`（423 行）核实。若仓库文件与本文档描述不符，以仓库实际代码为准并在 JEPACHANGES.md 中记录差异。*
