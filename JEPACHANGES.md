# JEPACHANGES — 与原版 X-Fi 的逐条差异说明

> 对照基准：commit 时的 `XRF55_HAR/` 原始代码。原则（JEPAREADME §7）：**不修改任何原文件**（`X_Fi.py`、`utils.py`、`run.py`、`validate_all.py`、`XRF55_Dataset.py` 均保持原样，作为对照基线 A）；所有新逻辑放新文件。

## 1. 新增文件一览

| 文件 | 职责 | 对应规格 |
|:--|:--|:--|
| `XRF55_HAR/X_Fi_backbone.py` | `X_Fi.py` 的副本 + 定点改造（见 §2） | §4.1 |
| `XRF55_HAR/jepa_modules.py` | Predictor + EMATargetEncoder | §4.2 |
| `XRF55_HAR/jepa_pretrain.py` | 阶段 1 预训练主脚本 | §4.3, §5.1 |
| `XRF55_HAR/jepa_downstream.py` | 阶段 2 下游评估（方法可切换） | §4.4, §5.4 |
| `XRF55_HAR/configs/jepa_baseline.yaml` | 预训练全部超参 | §4.5 |
| `XRF55_HAR/configs/eval_methods.yaml` | 下游/探测超参 | §4.5 |
| `XRF55_HAR/configs/eval_methods_warmstart.yaml` | B' 热启动变体配置（backbone_lr 1e-4，其余同默认） | §4.5, D28 |
| `XRF55_HAR/configs/eval_methods_mirror.yaml` | B' 镜像 A' 配方变体（恒定 lr 1e-4 / wd 0.01 / 无 val 留出 / last-model；唯一差异=JEPA 初始化） | §4.5, D28 |
| `XRF55_HAR/configs/jepa_aux.yaml` | 预训练变体：跨模态辅助损失（λ0.5/ramp5/LN both）+ WiFi keep 0.7 | §4-14, D30 |
| `XRF55_HAR/configs/jepa_aux_w09.yaml` | R3.4 消融变体：辅助损失 + 原 WiFi keep 0.9 | §4-14, D30 |
| `XRF55_HAR/jepa_eval.sh` | A/B/B'/C 对比实验一键脚本 | §5.4 |
| `XRF55_HAR/make_xrf55_split.py` | XRF55 train/test 符号链接划分树（官方 14/6 试验划分） | §6 |
| `XRF55_HAR/run_xfi_baseline.py` | 原版 X-Fi 自训的忠实包装：复刻 `run.py` 设置（数据/种子/模型/`utils.har_train`），仅把产物按 `--run-name` 归档到 `xfi_baseline_runs/<run名>/`；`run.py`/`utils.py` 仍零改动 | §1.3 |
| `environment.yml` | conda 环境定义（`conda env export -n xfi` 导出） | §0-1 |
| `PROJECT_LOG.md` / `JEPACHANGES.md` | 项目文档 / 本文档 | §0-4, §9 |

## 2. `X_Fi_backbone.py` 相对 `X_Fi.py` 的差异（逐条）

1. **`X_Fusion` 增加 `forward_backbone(feature, modality_list) -> z_cm`**（§4.1-1）。原 `forward()` 改写为 `classification_head(forward_backbone(...))`，数值行为完全一致。
2. **`feature_extrator` 骨干路径参数化 + `map_location='cpu'`**（§5.2 陷阱 5）。原版是写死的相对路径且依赖 torch.load 默认设备；发布的三份 `.pt` 内部是 CUDA 张量（CPU 机器上构建会直接报错），统一 `map_location='cpu'` 后由调用方 `.to(device)`。默认值 `./backbone_models` 与原版路径语义相同。
3. **新主类 `X_Fi_Backbone`**（§4.1-2）：复用全部子模块（`feature_extractor`/`linear_projector`/`X_Fusion` 内部逐行未动）；不含分类头（`X_Fusion.classification_head` 仍被构造但不使用、不进优化器，保证 state_dict 键与原版兼容）；`forward(mmwave, wifi, rfid, modality_list, token_mask=None) -> z_cm (B,32,512)`。
4. **词元级掩码 = 可学习 `mask_token` 替换**（陷阱 2 的推荐做法）：在 `linear_projector` 之后、`chunk` 之前，把被掩 token 的 512 维特征替换为广播的 `(1,1,512)` 可学习参数；**不物理删除 token**，每模态恒 32 token，`X_Fusion` 的 chunk 逻辑零改动。掩码布尔张量形状 `(B,3,32)`，只对在场模态生效。
5. **冻结 extractor 钉在 eval()**：`X_Fi_Backbone.train()` 被重写为"整体 train 后强制 `feature_extractor.eval()`"。理由：① extractor 是在线/EMA 两分支共享的同一实例（决策 2），两侧必须看到完全相同的特征；② 原版 `har_train` 的 `model.train()` 会让冻结 extractor 的 BN 统计量在训练中漂移（权重冻结但统计量更新），而其验证/评估用的是 eval 模式 —— 我们统一钉在 eval，使预训练特征与 released 权重的推理行为一致。**这是与原版训练动态的一个刻意差异**。
6. 被注释的 LiDAR 位置编码（原 L159-170）**原样保留**（§8-2 要求）。
7. 顶部 import 裁剪为本文件实际所需（原版的 `scipy.io/csv/tqdm/yaml/...` 属遗留 import；`backbone_models.mmWave.ResNet` 的 star-import 保留，torch 反序列化需要该模块可导入）。

## 3. `jepa_modules.py`

1. **Predictor**：4 层 pre-LN transformer（dim 512 / heads 8 / ffn 2048，`nn.MultiheadAttention`），输入加可学习位置编码与"推断标记" embedding，输出 LayerNorm；`(B,32,512) -> (B,32,512)`（§4.2）。
2. **EMATargetEncoder**：只 deepcopy `linear_projector + X_Fusion`；**特征提取器按引用共享**（决策 2 / 陷阱 4），不复制不 EMA；参数与浮点 buffer 每步 EMA，整数 buffer（BN `num_batches_tracked`）直接拷贝；momentum 按 I-JEPA 余弦 `0.996 -> 1.0`（随训练进度）；分支整体**永久 eval()**（陷阱 3 的人工确认方案：保留 BatchNorm，EMA 侧用自己的 EMA 统计量，绝不双更新）。
3. 目标分支输入恒为完整数据、全模态、无词元掩码（§5.1 step 5）。

## 4. `jepa_pretrain.py` 的关键实现决定

1. **掩码采样全部移入训练循环**（陷阱 1）：模态级掩码每 batch 采一个 `modality_list`（与原版 `collate_fn_padd` 的机制一致——原版代码实际是**每 batch 一个列表**，README §2.3 的"逐样本"指的是按 batch 的 collate 调用；我们保留该机制，逐样本粒度由词元掩码承担）；全 False 重采样（陷阱 6）；独立 `np.random.RandomState(seed+1)`（陷阱 8：模型初始化/数据顺序用 torch 全局种子 3407，掩码独立）；首 epoch 前 3 步打印掩码记录。
2. **损失**：`SmoothL1(z_hat, z_tgt.detach())`，作用于完整 `(B,32,512)` token 级表征（§3.5）。
3. **优化器**：AdamW lr 5e-4 / wd 0.05，参数 = projector + X_Fusion（**排除未使用的 classification_head**）+ predictor + mask_token；5 epoch 线性 warmup + 余弦退火（§5.1 表）。
4. **监控**（§3.5 必做）：每 epoch 记录损失、`z_tgt` token 级标准差（对 batch 维；<0.1 告警）、各模态子集采样频率、实际词元掩码率、当前 EMA momentum；写入 `jepa_checkpoints/train_log.csv`（损失/标准差曲线数据源）。
5. **checkpoint**：`online_state / ema_state / predictor_state / optimizer_state / scheduler_state` 分开存于 `last.pth`（每 epoch 覆盖）与 `epoch_XXX.pth`（每 10 epoch）。单文件约 400 MB（含 optimizer 状态与两份骨干引用），100 epoch 全程磁盘占用约 4-5 GB。支持 `--resume`。
6. 非 finite loss 跳过该 batch 并告警（沿用原版打印 nan 的精神，但不让 NaN 进入权重）。
7. DataLoader 加载器选项 config 化（`data.pin_memory` / `data.prefetch_factor` / `persistent_workers`）：pin_memory **默认关闭**——实测本机 batch≥256 时 pin 线程锁页分配失败（"CUDA error: invalid argument"，与 GPU 显存无关，见 PROJECT_LOG D16）；本工作负载关闭 pinning 几乎无损。
8. **两分支输出终归一化 `out_norm = LayerNorm(512)`**（训练稳定性关键修复，2026-09-18）：原版 `X_Fusion` 输出无任何终归一化，JEPA 目标下表征尺度存在正反馈通路（在线分支权重增长 → EMA 侧因 BN running stats 滞后而输出尺度被放大 → 编码器被进一步推大），实测 z 尺度漂移至 ~7、SmoothL1 单调上升（PROJECT_LOG D17）。在线 `X_Fi_Backbone` 与 EMA `EMATargetEncoder` 的 forward 末端各加一个 LayerNorm（EMA 侧参与 EMA 更新），与 I-JEPA 的 ViT final-norm 设计对齐。**注意**：state_dict 新增 `out_norm.*` 键，与修复前的旧 checkpoint 不兼容（旧 ckpt 无保留价值，已归档为 `jepa_checkpoints_b512_prefix/`）。
9. **BN warmup + buffer 同步**：开训前跑 `bn_warmup_batches`（默认 50）个前向 batch（train 模式、全模态、无优化器步），使 projector 的 BN running stats 收敛，再整体拷贝进 EMA 分支（`EMATargetEncoder.sync_buffers_from`）——否则 eval 态的目标分支从 (0,1) 初始统计量出发、以 EMA 时间常数 ~250 步缓慢漂移，前若干 epoch 的 z_tgt 是"移动靶"。
10. **weight decay 排除 ndim≤1 参数**（BN γ/β、LayerNorm、所有 bias、mask_token）：标准做法，避免衰减归一化层参数。
11. **`--run-name` 按 run 归档 + TensorBoard**（2026-09-19，人工需求）：两个脚本各 run 的产物进独立命名目录（缺省当前时间到秒），互不覆盖；预训练逐 step/逐 epoch 指标、下游探测逐 epoch 指标与最终各子集精度均写入 TB 事件文件（`tensorboard` 为可选依赖，未安装时优雅降级为 CSV/JSON 照常）；每次 run 附 config 快照便于区分实验。
12. **few-shot 协议 `--label-fraction`**（2026-09-20）：`jepa_downstream.py` 与 `run_xfi_baseline.py` 支持按类分层抽取训练标签比例（`jepa_pretrain.py::stratified_subset_indices`，同 seed 下不同比例嵌套）；下游在 val 划分之后抽样（val/test 零泄漏），原版包装的候选池排除同一 val 集 → 各对比方法见到相同的有标签样本；results.json 新增 `label_fraction/n_train_labeled/n_val`。
13. **`finetune.schedule` 配置键 + 配方镜像变体**（2026-09-20）：`jepa_downstream.py` 支持 `schedule: none`（恒定 lr；此前 `warmup_epochs: 0` 会退化为纯 cosine）——新增 `configs/eval_methods_warmstart.yaml`（backbone_lr 1e-4 热启动）与 `configs/eval_methods_mirror.yaml`（对齐 `utils.har_train` 配方：恒定 lr 1e-4/wd 0.01/无 val 留出/last-model），使 B'−A' 对比只隔离 JEPA 初始化这一个变量。
14. **跨模态辅助预测损失 + WiFi 掩码表变体**（2026-09-21）：预训练新增辅助目标——对 context 中**缺席**的模态 m，用线性头 `aux_m(z_hat)` 预测 EMA 分支的 per-modality 投影特征（`linear_projector.forward_per_modality` + `EMATargetEncoder.forward_with_parts`，与主损失数值等价、零额外抽取成本）；目标与预测均过**无参数 LayerNorm**（无 affine → 不进 EMA，结构上杜绝 D17 的尺度正反馈）；损失 `λ·SmoothL1`，λ=0.5 线性 ramp 5 epochs，config 块 `aux_loss`（weight/ramp_epochs/normalize）。`AuxModalityHeads`（3×Linear，~0.79M）置于 backbone/EMA 之外，checkpoint 新增顶层 `aux_state`（`online_state/ema_state` 键不变，旧 ckpt 下游加载与续训兼容）。新变体 `configs/jepa_aux.yaml` 同时把 WiFi keep 0.9→**0.7**（WiFi 缺席 batch 10%→30%，针对"无 WiFi 聚合"能力缺失——B 探测 mmWave+RFID 塌陷与双侧低标签 WiFi 塌陷的首要假设）。

## 5. `jepa_downstream.py` 的关键实现决定

1. **四挡方法切换**（人工需求）：`xfi_supervised`（A，直接评估，不训练）/ `jepa_linear_probe`（B）/ `jepa_finetune`（B'）/ `random_init`（C 下界）。全部走**同一评估器**，口径可比。
2. **评估器**：7 个固定子集逐一评估，**样本粒度**累计正确数（修正原版 `multi_test` 的 batch 粒度平均，陷阱 7），额外输出 **worst-subset** 与 macro 平均，结果存 JSON（`jepa_eval_results/`）。
3. 探测/微调训练镜像原版监督协议：每 batch 采一个 modality_list（概率表在 config，默认沿用原版 wifi .9 / mmwave .5 / rfid .6）、CE loss。
4. 线性探测时骨干**整体 eval()**（特征确定、BN 用 running stats）；微调时 projector+X_Fusion 回 train()（extractor 仍被守卫钉在 eval）。
5. **torch.load shim**：脚本内将 `torch.load` 默认 `map_location` 设为所选 device——因为原版 `X_Fi.py` 与发布 checkpoint 都假定默认设备（CUDA 存档）。仅运行时行为，不改仓库文件。
6. 预训练 checkpoint 的 `online_state` 直接 `load_state_dict(strict=True)` 进 `X_Fi_Backbone`（键完全匹配，因为模块结构与键名未改）。

## 6. 数据侧

1. **`make_xrf55_split.py`**：X-Fi 代码期望 `{train,test}_data/{模态}/SceneN/SceneN/*.npy`，而官方下载数据是按场景平铺、无划分的。按 XRF55 官方 `split_train_test.py` 的规则（每动作 20 试验，trial≤14 训练 / ≥15 测试；X-Fi 论文 §5.1 声明沿用原划分）生成**纯符号链接树** `data/XRF55_Dataset_split/`，原始数据（`/mnt/DataDrive164/wr/xrf55`）一字节不动。
2. **`backbone_models/RFID/rfid_ResNet18.pt -> RFID_ResNet18.pt` 符号链接**：代码要求小写文件名而发布权重是大写（Linux 大小写敏感，原样会 FileNotFoundError）。
3. 原始数据经由 `data/XRF55_Dataset` 单一符号链接寻址；挪盘只需更新该链接。

## 7. 未改动清单（对照完整性）

`X_Fi.py`、`utils.py`、`run.py`、`validate_all.py`、`XRF55_Dataset.py`、`requirements.txt`（人工已注释 opencv）、`backbone_models/*` 网络定义 —— 全部保持原样。基线 A 的三种产出途径：未改动的 `run.py`+`validate_all.py`（原版原样）、`jepa_downstream.py --method xfi_supervised`（统一评估器）、`run_xfi_baseline.py`（忠实包装，仅产物归档方式不同，见 §1）。

## 8. 已知限制 / 后续接口

- 场景先验模态（§8-1）：`modality_list` 机制与 `X_Fi_Backbone` 的 token 掩码按 `(B, num_modalities, 32)` 设计，加第 4 模态只需扩 kv_layers/projection 分支与 config 概率表，无需改 JEPA 组件。
- MAE 对照组（§5.4-D）：接口已留（Predictor 与骨干解耦），未实现。
- `X_Fusion.classification_head` 在骨干中存在但闲置（为保证 state_dict 兼容），已从所有优化器/EMA 路径排除。
