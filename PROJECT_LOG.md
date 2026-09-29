# PROJECT_LOG — X-Fi-JEPA 多模态感知基础模型 Baseline

> 依据 `JEPAREADME.md`（v1.1, 2026-09-17）实现。本文档为持续更新的项目文档：命令手册 / 项目结构 / 数据集说明 / 模型结构 / 开发日志。
> 约定：所有训练由人工执行（命令见 §1）；agent 只准备代码并做轻量验证（冒烟测试等，也遵守单卡/空闲卡约定）。

## 0. 当前状态速览

| 项 | 状态 |
|:--|:--|
| 环境（xfi, torch 2.1.1+cu121） | ✅ 依赖齐（pandas 2.3.3 / h5py 3.16.0 / pyyaml 6.0.3）；`environment.yml` 已导出 |
| 数据 | ✅ 原始数据在 `/mnt/DataDrive164/wr/xrf55`（固定不动）；✅ 划分符号链接树 `data/XRF55_Dataset_split/` 已建好并验证 |
| 阶段 0（监督基线 A） | ✅ 完结：数据管线经官方 ckpt 验收（§1.3a，全模态 89.5%）；**A 组定为官方发布模型**（人工决策 2026-09-19，D22），§1.3 自训版降为可选对照 |
| 阶段 1（JEPA 预训练） | ✅ 完成：batch 512 / 100 epochs / 约 2 小时，结果检查通过（D19），产物 `jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/` |
| 阶段 2（下游对比 A_off/A'/B/B'/C） | ✅ 完成并汇总（§6 对比表 + D25/D27）：判据 ③✓（B−C macro +45.4pp）、判据 ④ WiFi✓/RFID✗；**核心对比 B' vs A'（自训原版）macro −1.0pp、WiFi-only +3.8pp** |
| 性能提升第一轮（B' 配方修正 + few-shot） | ✅ 完成（§7 清单 + D29）：**mirror 配方下 B' 追平原版**（macro 0.7171 ≥ 0.7162，worst +1.7pp），采纳为默认 finetune 配方；few-shot 10% 标签下 JEPA 仅 +0.4pp——SSL 优势未显现，归因与下轮方案见 D29 |
| 性能提升第二轮（辅助损失 + WiFi0.7 掩码） | ✅ 完成（§7 第二轮表 + D31）：**总目标达成——R3.3 macro 0.7426 超过 A'（+2.6pp）与 A_off（+2.2pp），全模态 0.8964 全组最高，WiFi-only +10.1pp**；剩余短板 worst-subset（RFID-only） |
| 性能提升第四轮（①22000 数据扩展 + ②步数预算） | ✅ 完成（§7 第四轮表 + D38）：**R4.6 macro 0.7634 创新高——vs A' 7/7 全胜（+4.7pp）、vs A_off +4.3pp 且 worst-subset 0.4386 首次反超（RFID-only 短板攻克）、WiFi-only 0.7395（vs A_off +17.9pp）**；步数假说证实（b64 +1.8pp） |
| 性能提升第五轮（因子合并 run） | ✅ 完成（§7 第五轮 + D39）：**因子不可加**——22000 语料 + b64 合并 run macro 0.7266，比两个单因子都差；**最终配方定格 = pretrain_aux_22000 (epoch_100) + mirror 微调（b512）= macro 0.7634**；b64 路线证伪封存；待办=标签效率曲线 + 探针选点（可选） |
| 性能提升第三轮（R4 均衡练习 + 块掩码） | ✅ 完成（§7 第三轮表 + D35）：R4.3 macro 0.7410 与 R3.3 打平（判定门未过）——**两轮配方收敛于同一平台（~0.74），"仅预训练融合层 + 同分布数据"的天花板已到**；轮巡/块掩码按设计工作但无增量；下一轮候选（数据扩展/步数预算/解冻末层/消融）等人工决策，见 D35 |
| 服务器 | 3× Tesla V100-32GB **公用**：先 `nvidia-smi` 挑空闲卡，`CUDA_VISIBLE_DEVICES=<N>` 指定，一次一张卡 |

---

## 1. 命令手册（需人工运行的命令）

> 所有命令都在 `XRF55_HAR/` 目录下执行（骨干 .pt 相对路径）。
> **GPU 约定**：先 `nvidia-smi` 看空闲卡，命令前加 `CUDA_VISIBLE_DEVICES=<N>`（脚本内部固定用"卡 0"，该前缀把它映射到第 N 张物理卡）。骨干 .pt 是 CUDA 存档，构建模型时就会占卡，务必带前缀。
> **时长（V100 实测）**：batch 16 时预训练 ~1.7-2.5 s/it（100 epochs 约 2-3 天）；**batch 256（pin_memory=false）稳态 ~1-1.5 s/it，100 epochs 约 2-2.5 小时**（推荐，已默认关闭 pin_memory，见 D16）。⚠️ 加大 batch 后建议把 `optim.lr` 从 5e-4 提到 1e-3（5e-4 按 README 的 batch 16 调配）；批量改动请在 JEPACHANGES.md 记一笔。下游线性探测同理（`configs/eval_methods.yaml`，冻结特征下 30 epoch 通常已够）。
> **run 命名约定（2026-09-19 新增）**：训练/评估脚本都支持 `--run-name`（缺省=当前时间精确到秒）；每次 run 的全部产物（checkpoint/CSV/tensorboard/config 快照/results.json）存放在各自的 run 目录下，**互不覆盖**。

### 1.0 安装依赖

```bash
conda run -n xfi pip install pandas h5py pyyaml   # ✅ 已完成（2026-09-18）
conda run -n xfi pip install tensorboard          # ✅ 已完成（2026-09-19 人工安装，2.21.0；TB 记录已启用）
```

### 1.1 数据准备（✅ 已由 agent 完成，备查/重建）

```bash
conda run -n xfi python make_xrf55_split.py
```

| 部分 | 含义 |
|:--|:--|
| `make_xrf55_split.py`（agent 新写） | 按官方划分（每动作 20 试验，trial≤14 训练 / ≥15 测试）生成 `data/XRF55_Dataset_split/{train,test}_data/{RFID,WiFi,mmWave}/SceneN/SceneN/*.npy` 纯符号链接树（train 46200 + test 19800 链接 = 22000 样本×3 模态），不复制不移动原始数据 |
| 幂等性 | 重复运行跳过已有链接 |

### 1.2 数据抽检（✅ 已完成）

train 15400 / test 6600；实测形状 **WiFi (270, 1000)**、**RFID (23, 148)**、**mmWave (17, 256, 128)**；标签 0–54。

### 1.3a 阶段 0 快速验收：官方 checkpoint 过我们的数据管线（几分钟，先跑）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python validate_all.py --dataset ../data/XRF55_Dataset_split --pt_weights ./pre-trained_weights/XRF55_har_checkpoint.pt
```

| 部分 | 含义 |
|:--|:--|
| `--pt_weights .../XRF55_har_checkpoint.pt` | X-Fi 官方发布的 XRF55 HAR 模型（已就位）。用它在自建 test split 上评估 |
| 验收意义 | 7 组合精度与论文量级一致（全模态 ~80% 区间）⇒ 数据组织+划分与官方训练口径对齐；偏差大先排查数据 |

### 1.3 阶段 0 主命令：原版 X-Fi 监督训练（自训基线 A；正式对比的 A 组此前用官方 ckpt，自训版用于对照，见 D22/D26）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python run_xfi_baseline.py --run-name <run名>
# few-shot 变体（10% 标签，标签池与 JEPA few-shot 组完全一致，见 D28）：
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python run_xfi_baseline.py --label-fraction 0.1 --run-name A_xfi_lab10
```

| 部分 | 含义 |
|:--|:--|
| `run_xfi_baseline.py`（2026-09-19 新增的忠实包装） | **完全复刻原版 `run.py` 的训练设置**（同数据/`collate_fn_padd`/`X_Fi(model_depth=5, num_classes=55)`/种子 3407 位置/`utils.har_train` AdamW@1e-4/100 epochs/batch 16），仅把产物目录从共享的 `./pre-trained_weights/` 改为 `xfi_baseline_runs/<run名>/`。`run.py`/`utils.py` 原文件零改动 |
| `--run-name <run名>` | 同 §1.6 约定，缺省当前时间到秒；run 目录内含 `args.yaml` 快照 + `checkpoint_<时间戳>.pth`（原版行为：训练结束存 last model） |
| `--label-fraction <f>` | **（2026-09-20 新增）** few-shot 协议：按类分层抽取 f 比例的训练标签；候选池排除标准 5% val（seed 3407+7），抽样 seed 3407+11——与下游 JEPA few-shot 组**完全相同的有标签样本** |
| `--epochs / --batch-size / --lr` | 可覆盖（默认 100/16/1e-4 = 原版协议值）；**改了就偏离协议 A**，需记录 |
| `--dry-run` | 只构建数据/模型并打印输出路径，不训练 |
| 覆盖安全性 | **不会覆盖任何东西**：`har_train` 的 checkpoint 文件名带秒级时间戳（`utils.py` L89-90），且新包装按 run 分目录，与官方 ckpt、JEPA 产物互不干扰 |
| 预期输出 | 每 epoch Accuracy/Loss，每 5 epoch 测试集验证并打印 best（原版行为：只在 stdout 记录，不存 best ckpt）；时长 ~2-3 天（batch 16 全标签）；lab10 约 30 分钟（batch 512） |
| 验收 | 损失正常下降、无 NaN；§1.4（把 `--pt_weights` 指向 run 目录里的 checkpoint）全模态精度 ~80% 区间，与官方 ckpt（§1.3a 89.5%）对照 |

原版原始调用方式保留备查（产物混在 `pre-trained_weights/`、仅时间戳区分，不推荐）：
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python run.py --dataset ../data/XRF55_Dataset_split
```

### 1.4 阶段 0 收尾：评估自训 checkpoint（~7 分钟）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python validate_all.py --dataset ../data/XRF55_Dataset_split --pt_weights ./pre-trained_weights/checkpoint_<时间戳>.pth
```

### 1.6 阶段 1 主命令：JEPA 预训练（~2-3 天）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_pretrain.py --config configs/jepa_baseline.yaml --run-name <run名>
```

| 部分 | 含义 |
|:--|:--|
| `--config configs/jepa_baseline.yaml` | 全部超参所在：seed 3407；batch 16；AdamW lr 5e-4 / wd 0.05，5 epoch warmup + 余弦；100 epochs；EMA 0.996→1.0 余弦；模态保留概率 [mmWave .5, WiFi .9, RFID .5]；词元掩码率 0.25；predictor 4 层 |
| `--run-name <run名>` | **（2026-09-19 新增）** 本次训练的唯一命名，手动指定便于区分实验（如 `b512_lr5e-4_tokenmask0.25`）；缺省为当前时间精确到秒。该 run 的全部产物（checkpoint、train_log.csv、tensorboard、config 快照）都存放在 `jepa_checkpoints/<run名>/` 下，**不同 run 互不覆盖** |
| `--dataset <路径>` | （可选）覆盖 config 里的数据根目录，默认即 `../data/XRF55_Dataset_split` |
| `--resume <ckpt>` | （可选）从断点续训，需同时传同一个 `--run-name`，如 `--resume ./jepa_checkpoints/<run名>/last.pth --run-name <run名>` |
| `--limit-train-batches N` / `--epochs N` | 调试用，正式训练不传 |
| 预期输出 | 开训前先打印 `BN warmup: 50 forward-only batches ...`（约半分钟）和 train/val 划分（默认 13860/1540）。每 epoch：JEPA loss、z_tgt std（<0.1 会告警塌缩）、词元掩码率、7 种子集采样频率、EMA momentum、**val: loss / explained var**（留出 10% 训练集上的确定性 SSL 验证：全模态、无词元掩码）；val loss 创新低时打印 `[best] ... best_val.pth updated`。产物（都在 `<run名>/` 内）：`train_log.csv`（含 val_loss/val_expl_var 列）+ tensorboard（Loss/train_step、Loss/train_epoch、Metrics/ztgt_std、Optim/lr、Optim/ema_momentum、Sampling/freq_*、**Val/loss、Val/explained_var**）+ `config.yaml` 快照 + **`best_val.pth`（验证 SSL 损失最低的权重——⚠️ 仅作诊断产出，见 D34：该判据偏向低信息量检查点，下游请用 `epoch_100.pth`）** + `last.pth`（run 内每 epoch 覆盖，供续训）+ `epoch_XXX.pth`（每 10 epoch 长期保留；单 ckpt 约 400MB）。健康曲线参考 D18/D19；**收敛判据建议看解释方差而非绝对 loss**（D19） |
| 验收（§5.4 判据 1/2） | 损失稳定下降、无 NaN；z_tgt std 不塌缩（>0.1 量级） |

### 1.7 阶段 2 命令：下游评估（按方法分条跑；或用 §1.8 一键）

```bash
# A 组：原版监督模型 = X-Fi 官方发布 checkpoint（人工决策 2026-09-19，避免重复原版训练；自训对照可选，跑 §1.3 后替换路径即可）
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_downstream.py --method xfi_supervised --dataset ../data/XRF55_Dataset_split --xfi-weights ./pre-trained_weights/XRF55_har_checkpoint.pt

# B 组：JEPA 预训练 + 线性探测（只训分类头）；--run-name 建议注明所用预训练 ckpt
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_downstream.py --method jepa_linear_probe --dataset ../data/XRF55_Dataset_split --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name B_probe_ep100

# B' 组：JEPA 预训练 + 微调（projector+X_Fusion lr 1e-5，头 lr 1e-4）
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_downstream.py --method jepa_finetune --dataset ../data/XRF55_Dataset_split --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name Bp_ft_ep100

# C 组：随机初始化 + 线性探测（下界；无需 checkpoint）
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_downstream.py --method random_init --dataset ../data/XRF55_Dataset_split --run-name C_random_init
```

| 部分 | 含义 |
|:--|:--|
| `--method` | 方法开关（人工需求"方便切换"）：`xfi_supervised` / `jepa_linear_probe` / `jepa_finetune` / `random_init`，四组共用同一评估器，口径一致 |
| **配方变体与 few-shot（2026-09-20 新增，D28）** | ① `--config configs/eval_methods_warmstart.yaml`：B' 热启动变体（backbone_lr 1e-4，其余同默认）；② `--config configs/eval_methods_mirror.yaml`：镜像 A' 配方（backbone_lr 1e-4 恒定、wd 0.01、无调度器、无 val 留出、last-model）——与 A' 唯一差异=JEPA 初始化；③ `--label-fraction 0.1`（或 0.05/0.01）：few-shot 协议，从去 val 的 train 池按类分层抽标签（seed 3407+11），与 `run_xfi_baseline.py --label-fraction` 完全同标签样本；results.json 记录 `label_fraction/n_train_labeled/n_val` |
| **best/last 权重（2026-09-19 新增）** | 探测/微调从 train 内部按标签分层留出 10% 做验证（不碰 test，无泄漏），每 `eval_every`（默认 1）epoch 按 `eval.selection_metric`（默认 `acc_all`，可选 macro_avg/worst_subset）评估验证集；**`best.pth` = 验证指标最优权重**，`last.pth` = 最终权重，均存 `jepa_eval_results/<run名>/`；结束后 test 上**同时报告 best 与 last 两套精度**，results.json 的 headline `accuracy` = best（`accuracy_last_model` 并列给出） |
| `--pretrained` | `jepa_pretrain.py` 产出的 checkpoint（读其中 `online_state`；`last.pth` 或 `epoch_XXX.pth`） |
| `--xfi-weights` | A 组用的完整 `X_Fi` checkpoint（官方发布版或 §1.3 自训版） |
| 探测/微调超参 | `configs/eval_methods.yaml`（epochs / lr / warmup / 探测期模态采样概率等） |
| 预期输出 | 7 子集逐项精度（样本粒度）+ **worst-subset** + macro 平均；JSON 存 `jepa_eval_results/<run名>/results.json`（含 TB 事件与 config 快照） |
| 时长（batch 512，V100 估算，以实测为准） | A 组仅评估 ~1 分钟；B/C 探测 100 epoch 各 ~50-70 分钟（`linear_probe.epochs: 30` 则 ~20 分钟）；B' 微调 100 epoch ~1.5-2 小时 |
| ⚠️ 口径提醒 | 对比表里 A 组数字必须来自本统一评估器（`--method xfi_supervised`，样本粒度+worst-subset）；§1.3a `validate_all.py` 的 89.5% 是 batch 粒度口径，仅作管线验收，不进对比表 |

### 1.8 阶段 2 一键脚本（替代 §1.7 逐条）

```bash
# 先编辑脚本头部：GPU_ID（空闲卡）、XFI_CKPT（§1.3 产出的 checkpoint 文件名）、JEPA_CKPT
bash jepa_eval.sh
```

| 部分 | 含义 |
|:--|:--|
| 脚本内变量 | `GPU_ID` / `DATASET` / `JEPA_CKPT`（已指向归档的 `epoch_100.pth`）/ `XFI_CKPT`（已默认官方发布 ckpt，A 组决策见 D22），改好即依次跑 A→B→B'→C 四组 |
| 产出 | `jepa_eval_results/` 下 4 份 JSON，即 §5.4 对比表的数据源（7 子集 × worst-subset） |

### 1.9 查看训练曲线（TensorBoard，装完 §1.0 的 tensorboard 后可用）

```bash
conda run -n xfi tensorboard --logdir jepa_checkpoints --port 6006
# 本机浏览器访问 http://<服务器IP>:6006 ；如端口被占换一个；远程访问可加 --bind_all（公用服务器慎用）或用 SSH 隧道
```

| 部分 | 含义 |
|:--|:--|
| `--logdir jepa_checkpoints` | 一次挂载全部预训练 run（每个 run 一个子目录，TB 里可勾选对比）；下游评估改 `--logdir jepa_eval_results` |
| 曲线面板 | Loss/train_step（逐 step）、Loss/train_epoch、Metrics/ztgt_std（塌缩监控，警戒线 0.1）、Metrics/token_mask_rate、Optim/lr、Optim/ema_momentum、Sampling/freq_*（7 子集采样频率）；下游为 Loss/train、Accuracy/train 与各子集 Accuracy |

### 1.5 模型构建冒烟测试（✅ agent 已通过，备查）

```bash
conda run -n xfi python -c "from X_Fi import X_Fi; X_Fi(model_depth=5, num_classes=55); print('model build OK')"
```

---

## 2. 项目结构

```
X-Fi/
├── JEPAREADME.md / PROJECT_LOG.md / JEPACHANGES.md   # 规格 / 项目文档 / 差异说明
├── README.md            # 官方说明（未动）
├── environment.yml      # ★ conda env export -n xfi（含 pandas/h5py/pyyaml/torch 等）
├── requirements.txt     # 人工注释掉 opencv（未提交改动，保持原样）
├── data/
│   ├── XRF55_Dataset -> /mnt/DataDrive164/wr/xrf55   # 原始数据符号链接（固定不动）
│   └── XRF55_Dataset_split/                          # ★ agent 生成的划分符号链接树（66000 链接）
└── XRF55_HAR/           # ← 唯一工作目录
    ├── X_Fi.py / run.py / utils.py / validate_all.py / XRF55_Dataset.py   # 原版，未动（对照基线）
    ├── X_Fi_backbone.py        # ★ X_Fi.py 副本+改造：forward_backbone 暴露 z_cm；X_Fi_Backbone（token_mask、冻结 extractor 钉 eval、mask_token）
    ├── jepa_modules.py         # ★ Predictor（4 层 pre-LN transformer）+ EMATargetEncoder（只 EMA projector+X_Fusion，extractor 共享引用，eval 钉死）
    ├── jepa_pretrain.py        # ★ 阶段 1：掩码入训练循环、SmoothL1 token 级、EMA 更新、防塌缩监控、checkpoint/resume、CSV 日志
    ├── jepa_downstream.py      # ★ 阶段 2：四挡方法切换（A/B/B'/C）+ 统一 7 子集 worst-subset 样本粒度评估
    ├── make_xrf55_split.py     # ★ 数据划分符号链接树生成（官方 14/6 试验划分）
    ├── run_xfi_baseline.py     # ★ 原版 X-Fi 自训包装（复刻 run.py 设置 + --run-name/--label-fraction，§1.3）
    ├── configs/eval_methods_warmstart.yaml # ★ B' 热启动变体（backbone_lr 1e-4）
    ├── configs/eval_methods_mirror.yaml    # ★ B' 镜像 A' 配方变体（恒定 lr/wd0.01/无val）
    ├── configs/jepa_aux.yaml               # ★ 预训练变体：跨模态辅助预测损失 + WiFi keep 0.7（D30）
    ├── configs/
    │   ├── jepa_baseline.yaml  # ★ 预训练全部超参（含模态保留概率表，§8-4 接口）
    │   └── eval_methods.yaml   # ★ 下游方法/探测超参
    ├── jepa_eval.sh            # ★ A/B/B'/C 对比一键脚本
    ├── backbone_models/        # 预训练骨干 .pt（RFID 小写名为 agent 加的符号链接）
    └── pre-trained_weights/XRF55_har_checkpoint.pt   # 官方发布模型（管线验收 + A 组参考）
# 运行时生成：XRF55_HAR/jepa_checkpoints/<run名>/（预训练 ckpt+CSV+TB）、XRF55_HAR/jepa_eval_results/<run名>/（评估 JSON+TB）、
#            XRF55_HAR/xfi_baseline_runs/<run名>/（自训 A：args.yaml + checkpoint_<时间戳>.pth）
```

---

## 3. 数据集说明（XRF55）

- **原始组织**（`/mnt/DataDrive164/wr/xrf55`，只读）：`SceneN/SceneN/{RFID,WiFi,mmWave}/<subject>_<action>_<trial>.npy`。
  - 文件名三段：subject（共 20 人：Scene1 11 人 + 其余各 3 人）_ action（01–55 = label）_ trial（01–20）。
  - 规模：22000 样本（Scene1 12100 + 其余各 3300）。
- **官方划分**（XRF55 `split_train_test.py`；X-Fi 论文 §5.1 声明沿用原划分）：每动作 20 试验中 **trial 01–14 → train（15400）、15–20 → test（6600）**；由 `make_xrf55_split.py` 以符号链接实现。
- **代码侧布局**（`XRF55_Dataset.py` 期望）：`<root>/{train_data,test_data}/{RFID,WiFi,mmWave}/SceneN/SceneN/*.npy`；同索引三模态文件名一一对应；label = 文件名第 2 段 − 1。
- **样本形状**（实测）：WiFi `(270, 1000)` float64 → **4 tokens**；RFID `(23, 148)` float64 → **5 tokens**；mmWave `(17, 256, 128)` float32 → **32 tokens**（均 512 维，token 数由骨干结构决定）。
- **预处理**：无（裸 `np.load`，collate 转 float32）。⚠️ 不要给数据加载加任何归一化（陷阱 9）。
- 数据本体永不移动；`data/XRF55_Dataset` 符号链接是唯一入口。

## 4. 模型结构

### 4.1 原版 X-Fi（已逐行核实）

```
X_Fi(model_depth=5, num_classes=55)                          # 总参数 32.94M
├── feature_extractor (=feature_extrator, 20.55M, 冻结)
│   ├── mmwave_feature_extractor  # children()[:-2] → (B,32,512)
│   ├── wifi_feature_extractor    # children()[:-2] → (B,4,512)
│   └── rfid_feature_extractor    # children()[:-3] → (B,5,512)
├── linear_projector   # 每模态 Conv1d(512,512,1)+BN+ReLU+Linear(N_tok,32)+ReLU → (B,32n,512)
│   # ⚠️ 被注释的 LiDAR pos_enc 原样保留
└── X_Fusion_block     # kv_layers(3×) → cross_modal_transformer 聚合 z_cm (B,32,512)
    │                  # → depth=5 轮 [逐在场模态 cross-attention → 拼回 → 重新聚合 z_cm]（KV 每轮不变）
    └── classification_head: mean+LayerNorm+Linear(512,55)
```

训练配置（`run.py`/`utils.py`）：AdamW lr=1e-4 只训 projector+X_Fusion；CE；100 epochs；batch 16/32；种子 3407；模态子集 collate 阶段逐 batch 采样（wifi .9/mm .5/rfid .6）。

### 4.2 X-Fi-JEPA（✅ 已实现，四文件见 §2）

```
online:  X_Fi_Backbone = extractor(冻结, eval 钉死, 与 EMA 共享同一实例)
         → linear_projector → [token_mask: 被掩 token 换可学习 mask_token(1,1,512) 广播]
         → X_Fusion.forward_backbone → z_cm (B,32,512)
target:  EMATargetEncoder = deepcopy(projector + X_Fusion)，参数+浮点 buffer 每步 EMA，
         momentum 0.996→1.0 余弦（I-JEPA）；整体永久 eval()（保留 BN 的人工确认方案）；
         输入恒全模态完整数据、无词元掩码
predictor: 4 层 pre-LN transformer (dim512/heads8/ffn2048) + 位置编码 + 推断标记 + 输出 LN
loss:    SmoothL1(predictor(z_cm), z_tgt.detach())，完整 token 级 (B,32,512)
optim:   AdamW lr 5e-4 / wd 0.05（projector + X_Fusion[去分类头] + predictor + mask_token），
         5 epoch warmup + 余弦；EMA 在每步 optimizer.step() 之后更新
mask:    模态级每 batch 一个列表（与原版机制一致；概率 wifi .9 / mm .5 / rfid .5，非空保证）
         + 词元级逐样本 (B,3,32)，率 0.25（mask_token 替换，chunk 逻辑零改动）
监控:    每 epoch 损失 / z_tgt token 级 std（对 batch 维，<0.1 告警）/ 子集采样频率 / 实际掩码率 / momentum
```

下游（`jepa_downstream.py`）：A=原版 X_Fi+完整 ckpt 仅评估；B=预训练 ckpt 冻结骨干+新分类头（classification_Head 结构，mean+LN+Linear）；B'=解冻 projector+X_Fusion（lr 1e-5）+头（lr 1e-4）；C=随机骨干+探测。评估统一 7 子集样本粒度 + worst-subset + macro，JSON 落盘。

---

## 5. 开发日志

### 2026-09-17（阶段 0：仓库核实 + 环境校验）

- **D1 仓库事实核实**：五个原版文件与 JEPAREADME §2 逐条一致。
- **D2 环境缺失依赖**：缺 pandas/h5py/pyyaml（只 import 不调用，但缺 yaml 时模型无法构建）。报告并给命令，人工执行。
- **D3 权重是 CUDA 存档**：torch.load 无 map_location 时在 CPU 机器会挂；新代码统一显式处理。
- **D4 数据目录小写 `data/`**（人工确认）。
- **D5 输入形状反推**（数据未到时）：WiFi in_ch=270、RFID in_ch=23、mmWave in_ch=17；token 数 32/4/5 与文档一致。
- **D6 RFID 权重文件名大小写不匹配（已修复）**：符号链接 `rfid_ResNet18.pt -> RFID_ResNet18.pt`（人工确认保留）。
- **D7 CPU 冒烟通过**：构建 + 7 子集前向 + 参数量（32.94M/12.39M/20.55M）。

### 2026-09-18（数据管线打通 + 阶段 0 解锁 + JEPA 代码实现）

- **D8 依赖复验**：人工安装后 `torch/pandas/h5py/yaml` 全部通过。
- **D9 数据布局不匹配诊断（阶段 0 首跑 num_samples=0 的根因）**：真实数据按 `SceneN/SceneN/{模态}` 平铺、无划分，与 `XRF55_Dataset.py` 期望结构不符 → glob 落空。数据本身完好（22000 样本×3 模态）。
- **D10 官方划分确认 + 符号链接方案**：XRF55 官方 `split_train_test.py` = trial≤14 训练/≥15 测试，X-Fi 论文声明沿用 → 新写 `make_xrf55_split.py`，生成 66000 个符号链接（train 15400/test 6600），不动 /mnt。实测形状 WiFi (270,1000)、RFID (23,148)、mmWave (17,256,128)；原版数据类+collate+X_Fi 真实数据端到端前向通过。`--dataset` 统一为 `../data/XRF55_Dataset_split`。
- **D11 人工新要求收录**：方法可切换（`--method` 四挡）；公用 3×V100 礼仪；发现官方 `XRF55_har_checkpoint.pt` 已就位 → 新增 §1.3a 管线验收命令。
- **D12 JEPA 代码实现（阶段 1+2 全部完成）**：`X_Fi_backbone.py`（复制+定点改造，见 JEPACHANGES §2）、`jepa_modules.py`、`jepa_pretrain.py`、`jepa_downstream.py`、两份 config、`jepa_eval.sh`、`environment.yml`、`JEPACHANGES.md`。原版五个文件零改动。
- **D13 实现期修掉的 bug**：① `jepa_pretrain.py` 子集频率统计对名字 `'all'` 做 KeyError 的查表（改为显式分支）；② `jepa_downstream.py` finetune 优化器参数组 lr 归属错误（projector 组误用 head_lr）；③ `--limit-eval-batches` 未生效（评估器补 max_batches 上限）；④ 草稿残留的死代码/未用 import 清理。
- **D14 冒烟测试结果（GPU，均为 1-2 iteration 轻量验证，测试后产物已删除）**：
  - 预训练全链路：24.99M 可训练参数（projector+X_Fusion[去头]+predictor+mask_token）；两级掩码正常记录（实际词元掩码率 ≈0.24）；loss 1.78 有限；**z_tgt std 0.389**（远高于 0.1 塌缩线）；EMA 首步 momentum 0.996 正常（曾疑为 0.998，查明是 `--epochs 1` 把 total_steps 压成 2 所致的测试假象，公式正确）；checkpoint/CSV 正常落盘。
  - 下游四路径：`random_init`（随机头精度 0.0182 ≈ 1/55，评估器统计口径正确）；`xfi_supervised`（官方 ckpt 严格加载成功，3 batch 精度 0.23-0.67 远超随机，口径兼容）；`jepa_linear_probe` 与 `jepa_finetune`（ckpt strict 加载 + 探测/微调循环 + 评估均跑通）。
- **D15 速度实测与建议**：V100 batch16 预训练 ~1.7-2.5 s/it → 100 epochs 约 2-3 天；探测同量级。已在 §1 前言给出提速旋钮（调大 batch_size / 减少探测 epochs / 断点续训 `--resume`）。正式训练时长请以人工实测为准。
- **D16 batch 256 崩溃排查（人工实测发现，已修复）**：现象——batch 128 正常（5.1GB/32GB），batch 256 在第一个 batch 崩于 DataLoader **pin_memory 线程** `CUDA error: invalid argument`。根因：pin 线程为在途 batch 锁定页内存（batch 256 时单 mmWave 张量 544MiB、三模态 ~850MiB/batch，4 workers×prefetch 2 ≈ 8 个在途 ≈ 6.8GB 锁定内存）；本机 `ulimit -l` 仅 64MiB，大块锁页分配（cudaHostAlloc，受 IOMMU/驱动限制）失败即报该误导性错误——与 GPU 显存无关。**修复**：`pin_memory` 改为 config 项且默认 false（`data.pin_memory`），并新增 `data.prefetch_factor`、`persistent_workers`；H2D 传输仅 ~85ms/batch 且与计算重叠，关闭几乎无损。验证：batch 256 + pin_memory=false 冒烟 3 迭代通过，稳态 ~1-1.5 s/it → 100 epochs 约 2-2.5 小时（较 batch16 提速 ~20 倍）。**附带提醒**：① lr 5e-4 是按 README 的 batch 16 配的，batch 128-256 建议改 `optim.lr: 1.0e-3`（偏差记录在案）；② batch 级模态采样在 batch 256 时每 epoch 仅 61 次子集抽取（batch16 时 963 次），子集频率的 epoch 间方差变大；③ **磁盘紧张**：/mnt 数据盘 100% 满（余 10.8GB，本管线只读不写，暂无影响），仓库分区 98% 满（余 53GB，够放 ~4-5GB checkpoint，但请留意），必要时调大 `log.save_every_epochs` 减少定期 checkpoint。
- **D17 预训练 loss 单调上升排查（batch 512，人工实测发现，已修复）**：人工 batch 512 训练 11 epoch，loss 1.59→3.91 单调上升（增速递减）、z_tgt std 稳定在 0.37-0.41。排查（用人工留下的 checkpoint 实测）：① 在线/EMA 分支 BN running stats 严重错位（在线 mmWave var 4.7e-5 vs EMA 0.275）；② **conv 权重范数从初始化 ~1.4 膨胀到 13**；③ z_cm/z_tgt 实测尺度 ~6-7（max 21.7）而 predictor 输出仅 1.25 → 逐元素误差 ~5，SmoothL1 处线性段 ≈3.9，与日志吻合。**根因**：`X_Fusion` 输出无终归一化 + 目标分支 BN 用滞后的 running stats 归一化，构成尺度正反馈（在线权重↑ → EMA 侧输出尺度↑ → 编码器被推得更大）；I-JEPA 无此问题因其 ViT/预测器末端均有 LayerNorm。此前 z_tgt std 监控测的是跨样本方差，测不出尺度漂移，故未报警。**修复三件套**：① 在线/EMA 两分支 forward 末端加 `out_norm=LayerNorm(512)`（EMA 侧参与 EMA 更新；state_dict 新增 `out_norm.*` 键，**与旧 ckpt 不兼容**，旧 ckpt 已归档 `jepa_checkpoints_b512_prefix/`）；② 新增 `bn_warmup_batches: 50`——开训前 50 个前向 batch（train 模式、全模态、无优化器步）使 BN running stats 收敛后整体同步进 EMA 分支（`sync_buffers_from`），消除冷启动移动靶；③ weight decay 排除 ndim≤1 参数（BN/LN/bias/mask_token）。
- **D18 修复验证（agent 代跑，batch 512 / lr 5e-4 / 5 epoch，产物已清理）**：loss **0.199 → 0.075 → 0.055 → 0.048 → 0.044** 单调下降（对照修复前同期 1.59 / 1.79 / 2.27 / 2.70 / 3.04 单调上升）；z_tgt std 0.192→0.160 缓降且未塌缩（0.16 >> 0.1 告警线；且 loss 0.044 高于"只预测均值"的下限 ~0.014，说明 predictor 学的是非平凡映射）。⚠️ 该验证传了 `--epochs 5`，EMA/cosine 日程被压缩（momentum 5 epoch 内升到 ~1.0），仅用于验证趋势；**正式训练必须不带 `--epochs`、按 config 的 100 epochs 跑**。1.3a 快速验收（人工跑）确认数据管线与官方 checkpoint 口径对齐：全模态 89.5% / mmWave 84.0% / mmWave+RFID 86.4% 等，与论文量级一致。
- **D19 阶段 1 正式训练完成检查（人工跑，batch 512，100 epochs，2026-09-18 13:53-15:52，约 2 小时）**：全部 100 epoch 完成、无 NaN；lr warmup+余弦、EMA momentum 0.996→1.0、词元掩码率 0.25、子集采样频率均符合 config。**loss 呈 U 形（0.199→min 0.018@ep9→0.13@ep40→0.096@ep100）的判定**：同期 ztgt_std 0.14→0.94——表征跨样本多样性（信息量）持续增长使预测任务绝对难度上升，绝对 loss 上升不代表退化；尺度锚定指标**解释方差**（1−2·loss/std²）从 ~0 单调升至 **0.78** 并平台。checkpoint 实测：out_norm γ≈1.04（无尺度漂移，D17 的病根未复发）、z_cm/z_tgt absmean ~0.82、std ~1.04（两支一致）、`epoch_100.pth` 可 strict 加载进下游管线（2-batch 探测冒烟通过，产物已清理）。**结论：预训练健康可用，进入阶段 2**。提示：绝对 loss 曲线不适合做阶段 1 收敛判据，以解释方差为准（后续脚本可考虑直接记录该指标）。
- **D20 按 run 归档 + TensorBoard（人工需求，2026-09-19）**：① `jepa_pretrain.py`/`jepa_downstream.py` 新增 `--run-name`（缺省当前时间精确到秒），run 的全部产物进独立目录（预训练 `jepa_checkpoints/<run名>/`：config.yaml 快照 + train_log.csv + TB 事件 + last.pth + epoch_XXX.pth；下游 `jepa_eval_results/<run名>/`：config.yaml + results.json + TB 事件），跨 run 零覆盖；`--resume` 需带同一 `--run-name`。② TB 记录：预训练逐 step loss + 逐 epoch loss/ztgt_std/掩码率/lr/momentum/子集频率；下游探测逐 epoch loss/acc + 最终各子集精度。tensorboard 包缺失，已按约定给出安装命令（§1.0），脚本做了优雅降级（未装时 CSV 照记、仅 TB 关闭）。③ 2026-09-18 的正式预训练产物已归档至 `jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/`（§1.7 的 `--pretrained` 路径已同步更新）。两条路径均冒烟通过（未装 TB 的降级分支）。
- **D21 下游 batch 提升（人工需求，2026-09-19）**：`eval_methods.yaml` 的 `train_batch_size` 16→256、`test_batch_size` 32→256。要点：① test 批次是纯推理，BN 在 eval 模式 + 精度按样本粒度累计 → **批次大小对精度零影响**，纯提速；② train 批次影响 B/B'/C 三组的训练协议，**三组共用同一 config，数值自动保持可比**（对比的是表征质量，口径统一即可；与 A 组监督协议本就不同，不影响结论解读）；③ 注意 batch 级模态子集采样在 batch 256 时每 epoch 仅 61 次抽取（同 D16 提示，100 epoch 总量 6100 次足够）；④ probe 的 head lr 1e-3 在 batch 256 下仍然稳妥（线性探测对该超参不敏感），若头训练曲线过早平台可试 2e-3；⑤ 本下游 DataLoader 未开 pin_memory，D16 的锁页问题不适用。batch 256 + 真实 epoch_100 ckpt 冒烟通过（GPU0 空闲卡）。
- **D22 A 组协议决策 + 下游 batch 512（人工决策，2026-09-19）**：① **A 组 = X-Fi 官方发布 checkpoint**（`pre-trained_weights/XRF55_har_checkpoint.pt`）——人工不愿重复原版 2-3 天监督训练；该 ckpt 即论文原协议（batch 16 / 种子 3407 / 原模态采样）的产物，且 §1.3a 已验证其在我们自建划分上全模态 89.5%、与论文量级一致，作为"全监督上限参照"成立。§1.3 自训版降为可选（将来想复核作者协议时再跑）。② **读表注意**：核心科学对比是 **B vs C**（同探测协议：JEPA 表征 vs 随机初始化表征）；A 的训练协议与探测不同，作上限参照解读。③ A 组进对比表的数字以 `jepa_downstream --method xfi_supervised` 的统一评估器为准（样本粒度 + worst-subset）；§1.3a validate_all 的 89.5% 是 batch 粒度口径、仅作管线验收，两者不混用。④ 下游 batch 人工提至 **512** 并验证可跑（train+test；eval_methods.yaml 注释已同步）。⑤ `jepa_eval.sh` 的 `XFI_CKPT` 默认已指向官方 ckpt、`JEPA_CKPT` 指向归档的 `epoch_100.pth`。
- **D23 best 权重机制（人工需求"要 validation 上性能最好的权重"，2026-09-19）**：① 人工已自行安装 **tensorboard**（事件文件确认生成，降级分支不再触发）。② **阶段 2（探测/微调）**：train 内按标签分层留出 10%（13860/1540，从文件名解析标签、零数据加载）作验证集——**不碰 test，无选择泄漏**；每 `eval_every`（默认 1）epoch 按 `eval.selection_metric`（默认 acc_all）评验证集，**best.pth = 验证最优权重**（探测 ~0.3MB、微调 ~50MB），`last.pth` = 最终权重；结束在 test 上同时报告 best 与 last 两套精度，results.json 的 headline `accuracy` = best、`accuracy_last_model` 并列。注意：B/C 因留出 10% 少用 1540 个训练样本，相对官方 A（全量训练）是保守侧偏差，读表时知悉。③ **阶段 1（预训练）**：无标签无分类精度，"best"定义为**留出集上 SSL 验证损失最低**（确定性协议：全模态、无词元掩码；Val/loss 与 Val/explained_var 进 CSV/TB）→ `best_val.pth`；`--resume` 会恢复 best 基线继续比较。④ 两路径冒烟通过（GPU0）。原 9-18 归档 run 无 val 列/无 best_val.pth（当时机制未存在），其 epoch_100 继续有效。
- **D24 归档 run 的最优预训练权重甄别 + val_split 降至 0.05（人工需求，2026-09-19）**：① 对 `20260918_135302_b512_lr5e-4_100ep` 的全部 10 个快照做事后确定性 SSL 验证（留出 5%=770 样本、分层、seed 3407+7，与训练脚本同一规则）。结果：**epoch_100.pth 最优**（val_loss 0.01171、expl_var 0.978；epoch_090 0.01200/0.9776、epoch_080 0.01335/0.9751 次之；epoch_030-040 最差 ~0.034，与 D19 的 loss U 形一致——早期表征信息量低所以"好预测"，中期表征快速丰富导致任务变难，末期 predictor 追上）。val 曲线后 5 个快照单调改善，epoch_100 也是 I-JEPA 惯例的自然选择（EMA 编码器最收敛）。**结论：阶段 2 用 epoch_100.pth**（= last.pth 同一状态）。注意 ep80/90/100 差距很小（0.0133/0.0120/0.0117），若想进一步确认可对 epoch_090 做对照探测（可选）。② `val_split` 两处 config 均 0.1 → **0.05**（人工要求；预训练 train/val = 14630/770，下游同），减少 B/C 的训练数据让渡。
- **D25 阶段 2 四组结果汇总与 §5.4 验收（2026-09-19，人工执行，结果目录 `jepa_eval_results/{20260919_101005, B_probe_ep100, Bp_ft_ep100, C_random_init}`）**：对比表见 §6。判定：**判据 ③ 通过**（B−C macro **+45.4pp**，远超 2pp 门槛——JEPA 预训练确学到可迁移表征）；**判据 ④ 部分通过**（WiFi-only +54.9pp > 全模态 +41.4pp ✓；RFID-only 仅 +6.7pp ✗——C 组随机融合在 RFID-only 已达 26.3%，冻结 RFID 提取器特征可被随机融合近似线性读出、天花板效应）。亮点：**B' 的 WiFi-only 0.6005 > A 0.5606**——JEPA 预训练+微调在核心模态单模态上超过全监督官方模型。口径备注：A/B 于 best/last 机制上线前运行（B 为 last-model 口径、A 为官方 ckpt 直接评估），B'/C 为 best-by-val 口径（C 的 best=last 数值相同）；如需严格同口径可重跑 B（~1h），不影响判据结论。C 组并非纯随机水平（mmWave≈1.7% 机会水平，但 RFID 系 26-29%）：冻结提取器的信息经随机融合仍部分可达，C 是"随机融合+冻结特征"的下界而非机会水平下界——这使判据 ③ 的通过更保守、更可信。
- **D26 自训 A 组的 run 包装（人工需求，2026-09-19）**：人工决定自训原版 X-Fi 作 A 组对照。① **覆盖安全结论**：原版 `run.py`/`utils.py` 的 checkpoint 文件名带秒级时间戳（`utils.py` L89-90），每次运行都是新文件——**不会覆盖**官方 ckpt 或任何现有产物；问题是所有 run 混在 `pre-trained_weights/` 仅靠时间戳区分。② 新增 **`run_xfi_baseline.py`**：忠实包装（完全复刻 `run.py` 的数据/种子位置/模型/`utils.har_train` 调用链，原文件零改动），产物按 `--run-name` 归档到 `xfi_baseline_runs/<run名>/`（含 `args.yaml` 快照），支持 `--epochs/--batch-size/--lr`（默认=原版协议值 100/16/1e-4）与 `--dry-run`；与原版一致只存 last model（官方 ckpt 的产出惯例）。③ `--dry-run` 冒烟通过（GPU0）。④ 建议跑法：保持默认协议值与官方 ckpt 可比；训完用 §1.4（`--pt_weights` 指向 run 目录内 checkpoint）对照 §1.3a 的 89.5%。
- **D27 自训 A' 完成并纳入对比（2026-09-20，人工执行）**：`run_xfi_baseline.py --run-name A_xfi_original_b512`，产物 `xfi_baseline_runs/A_xfi_original_b512/checkpoint_2026-09-19_20:45:21.pth`；A 组重跑于 `jepa_eval_results/A_xfi/`。**协议备注**：人工实际用了 batch 512 + lr 1e-4（原版协议 batch 16），属协议变体但 macro 0.7162 与官方 ckpt 0.7207 仅差 0.5pp——复现达标。**三组核心阅读**：① A' vs A_off：复现校验（−0.5pp macro）✓；② **B' vs A'：主对比**——JEPA+微调 vs 原版全监督，macro −1.0pp、全模态 −1.4pp、WiFi-only **+3.8pp**；③ B vs C：表征增益下界（+45.4pp，D25）。B（纯探测）vs A' 不作方法级对比（协议不同：线性头 vs 全量监督），仅作参考。§6 表已加 A' 列与 B'−A' 差值列。
- **D28 性能提升第一轮：B' 配方修正 + few-shot 协议（人工确认范围与成功标准后实施，2026-09-20）**：背景——B' vs A' 落后 1.0pp，规划 agent（读仓库核实）发现 **B'−A' 混杂 4 个配方变量**：`utils.har_train` 实为 AdamW **wd=0.01（默认值）+ 恒定 lr（无调度器）+ 全量 15400 + last-model**，而 B' 是 backbone_lr 1e-5（几乎不动 JEPA 骨干）+ wd 0.05 + warmup/cosine + 14630 + best-by-val。实施：① `jepa_downstream.py` 支持 `finetune.schedule: none`（恒定 lr；注意 `warmup_epochs: 0` 在旧代码里会退化成纯 cosine 而非恒定）；② 两个配置变体 `eval_methods_warmstart.yaml`（backbone_lr 1e-4）与 `eval_methods_mirror.yaml`（对齐 A' 全部配方，唯一差异=JEPA 初始化）；③ few-shot 协议：`--label-fraction`（0.1/0.05/0.01）三脚本贯通——`jepa_pretrain.py::stratified_subset_indices`（同 seed 嵌套抽样）、下游在 val 划分后抽样（val/test 零泄漏）、`run_xfi_baseline.py` 用同一候选池（去 val）与 seed → **各方法见到完全相同的有标签样本**（lab10 = 每类 27 个、共 1485，冒烟确认 27/27）。④ 三项冒烟通过；人工 A' checkpoint（`A_xfi_original_b512`）未受影响。**实验矩阵与判定门**：R1.1 warmstart / R1.2 mirror（对照 A' macro 0.7162：≥0.7162 即"同配方更好初始化"成立并采纳为默认 finetune 配方；介于 0.7061-0.7162 补 lr 5e-5/3e-5 扫描；仍低则升级下一轮）+ few-shot 对（A_xfi_lab10 vs Bp_ft_lab10，预期 SSL 领先扩大）。下一轮备选（agent 已出方案）：跨模态辅助预测损失（EMA per-modality 目标 + 无参数 LN 锚尺度 + backbone 外的 aux 头）、WiFi keep 0.9→0.7 掩码表（B 探测 mmWave+RFID 塌陷的首要假设）、train+test 无标签预训练、300 epochs。

---

## 6. 阶段 2 对比实验结果（§5.4 交付，2026-09-19）

Test set（6600 样本）上的样本粒度精度；A/B/B'/C 定义见 §4.2，结果目录见 D25。

| 模态子集 | A_off 官方监督 | A' 自训(b512) | B JEPA+探测 | B' JEPA+微调 | C 随机+探测 | **B'−A'** | **B−C** |
|:--|:--|:--|:--|:--|:--|:--|:--|
| mmWave | **0.8402** | 0.8108 | 0.7947 | 0.8097 | 0.0174 | −0.1pp | +77.7pp |
| WiFi | 0.5606 | 0.5623 | 0.5880 | **0.6005** | 0.0394 | **+3.8pp** | +54.9pp |
| RFID | **0.4242** | 0.3917 | 0.3297 | 0.3738 | 0.2626 | −1.8pp | +6.7pp |
| mmWave+WiFi | 0.8820 | **0.8909** | 0.8655 | 0.8612 | 0.0409 | −3.0pp | +82.5pp |
| mmWave+RFID | **0.8630** | 0.8345 | 0.5702 | 0.8371 | 0.2688 | +0.3pp | +30.1pp |
| WiFi+RFID | 0.5800 | **0.6365** | 0.5329 | 0.5876 | 0.2855 | −4.9pp | +24.7pp |
| 全模态 | **0.8947** | 0.8870 | 0.7041 | 0.8730 | 0.2897 | −1.4pp | +41.4pp |
| **worst-subset** | **0.4242** | 0.3917 | 0.3297 | 0.3738 | 0.0174 | −1.8pp | +31.2pp |
| **macro 平均** | **0.7207** | 0.7162 | 0.6264 | 0.7061 | 0.1720 | **−1.0pp** | **+45.4pp** |

### §5.4 验收判据

1. 预训练损失稳定下降、无 NaN：✅（D18/D19；绝对 loss U 形已有解释，判据以解释方差为准，0 → 0.78）
2. z_tgt std 不塌缩：✅（0.14 → 0.94 平台，远离 0.1 警戒线；out_norm γ≈1.04 无尺度漂移）
3. B 显著优于 C（<2pp 视为失败）：✅ **macro +45.4pp**，全部 7 子集均为正（+6.7 ~ +82.5pp）
4. 弱模态子集提升 > 全模态提升：**部分成立**——WiFi-only +54.9pp > 全模态 +41.4pp ✅；RFID-only +6.7pp < +41.4pp ❌（C 组天花板效应，见 D25）

### 结论与备注

- **JEPA 预训练学到了可迁移的多模态表征**：同探测协议下 macro 领先随机初始化 45.4pp，且在全部子集为正（B−C 列）。
- **核心对比 B' vs A'（JEPA+微调 vs 原版全监督）**：macro **−1.0pp**（0.7061 vs 0.7162）、全模态 −1.4pp——JEPA 管线以无标签预训练达到接近全监督的性能；**WiFi-only 反超 +3.8pp**（0.6005 vs 0.5623），与"以 WiFi CSI 为核心模态"的动机吻合；原版在 mmWave+WiFi（+3.0pp）、WiFi+RFID（+4.9pp）组合上占优。
- **自训复现校验 A' vs A_off**：macro −0.5pp（0.7162 vs 0.7207），全模态 −0.8pp——自训复现了官方模型水平。⚠️ 协议备注：A' 实际用了 **batch 512 + lr 1e-4**（原版协议为 batch 16；run 名 `A_xfi_original_b512`，`args.yaml` 有记录），属协议变体，但复现质量达标；如需严格协议版可再跑 batch 16（~2-3 天，可选）。
- **B（纯探测）在含 RFID 的多模态子集上明显偏低**（mmWave+RFID 0.5702 < mmWave-only 0.7947）：冻结融合的表征对这些组合不是线性可分的，微调后恢复（0.8371）——线性探测低估了表征质量，符合预期。
- 口径：A'/B'/C 为 best/last 机制上线后运行（A' 为 ckpt 直接评估=last model，原版行为；B'/C=best-by-val）；B 与 A_off 为机制上线前运行（B=last-model 口径）。可选：用当前脚本重跑 B 以严格同口径（~1h）。
- 判据 4 的 RFID 项未过：C 在 RFID 系子集已有 26-29%（冻结 RFID 提取器特征经随机融合仍可部分线性读出），不是机会水平（1/55≈1.8%），提升空间本就有限。

---

## 7. 性能提升第一轮实验清单（2026-09-20 起，进行中）

目标与成功标准（方案见 D28 与 plans/clever-beaming-sundae.md）：**B' 追平 A'（macro ±1pp，A' 基准 0.7162 / worst 0.3917 / WiFi-only 0.5623）**，并产出 few-shot 10% 标签下的"原版 vs JEPA"对比。所有 run 遵守：单卡、先 `nvidia-smi`、`--run-name` 归档、统一评估器口径。

### 实验清单与状态

| ID | run 名 | 目的 | 关键参数 | 预计时长 | 状态 | macro / worst / all（回填） |
|:--|:--|:--|:--|:--|:--|:--|
| R1.1 | `Bp_ft_mirror` | B' 镜像 A' 配方：恒定 lr 1e-4 / wd 0.01 / 无 val / last-model，**唯一差异 = JEPA 初始化** | `--config configs/eval_methods_mirror.yaml --method jepa_finetune --pretrained .../epoch_100.pth` | ~2h | ✅ 完成 | **0.7171 / 0.4082 / 0.8888** |
| R1.2 | `Bp_ft_warmstart` | B' 热启动：backbone_lr 1e-4，其余同默认 B' 配方（wd 0.05/cosine/val 0.05/best-by-val） | `--config configs/eval_methods_warmstart.yaml --method jepa_finetune --pretrained .../epoch_100.pth` | ~2h | ✅ 完成 | 0.6801 / 0.3588 / 0.8812 |
| R2.1 | `A_xfi_lab10` | 原版 X-Fi @ 10% 标签从零训练（few-shot 对照；忠实协议 batch 16） | `run_xfi_baseline.py --label-fraction 0.1`（93 it/epoch） | ~30min | ✅ 完成（agent 补跑统一评估） | 0.5537 / 0.1718 / 0.7742 |
| R2.2 | `Bp_ft_lab10` | JEPA 预训练权重 @ 10% 标签微调（mirror 配方） | `jepa_downstream.py --method jepa_finetune --label-fraction 0.1 --config configs/eval_methods_mirror.yaml --pretrained .../epoch_100.pth` | ~20min | ✅ 完成 | 0.5578 / 0.1692 / 0.7873 |
| R2.3（可选） | `B_probe_lab10` | JEPA 表征 @ 10% 标签线性探测 | `--method jepa_linear_probe --label-fraction 0.1` | ~15min | 未跑 | — |
| R2.4（可选） | `C_probe_lab10` | 随机表征 @ 10% 标签线性探测（下界） | `--method random_init --label-fraction 0.1` | ~15min | 未跑 | — |
| R1.3/1.4（条件） | `Bp_ft_lr5e-5` / `Bp_ft_lr3e-5` | R1 判定门已由 R1.1 通过，未触发 | — | — | 取消 | — |

### 第二轮：跨模态辅助预测损失 + WiFi 0.7 掩码表（2026-09-21 新增，方案与实现见 D30）

| ID | run 名 | 目的 | 关键参数 | 预计时长 | 状态 | macro / worst / all（回填） |
|:--|:--|:--|:--|:--|:--|:--|
| R3.1 | `pretrain_aux_w07` | 预训练：辅助损失 λ0.5（预测缺失模态 EMA 投影特征）+ WiFi keep 0.7 | `jepa_pretrain.py --config configs/jepa_aux.yaml --run-name pretrain_aux_w07` | ~2h | ✅ 完成 | val_loss 0.0192 / val_expl_var 0.9634（主 SSL 指标略降，见 D31 归因）；aux_loss 0.637→0.329 持续下降 |
| R3.2 | `B_probe_aux` | 新表征探测（重点看 mmWave+RFID 是否恢复 ≥0.75） | `jepa_downstream.py --method jepa_linear_probe --pretrained .../pretrain_aux_w07/epoch_100.pth` | ~1h | ✅ 完成 | 0.6156 / 0.3494 / 0.6620 |
| R3.3 | `Bp_ft_mirror_aux` | 新表征 + mirror 配方微调（对照 A' 0.7162 / 0.3917） | `jepa_downstream.py --method jepa_finetune --config configs/eval_methods_mirror.yaml --pretrained .../pretrain_aux_w07/epoch_100.pth` | ~2h | ✅ 完成 | **0.7426 / 0.4141 / 0.8964** |
| R3.4（可选） | `pretrain_aux_w09` | 消融：辅助损失 + 原 WiFi keep 0.9（分离掩码表与辅助损失的贡献） | 同 R3.1 但改回 0.9 | ~2h | 未排期 | — |

### 第二轮判定结果（2026-09-22，详细分析见 D31）

| 模态子集 | A_off 官方 | A' 自训 | 第一轮 mirror | **R3.3 aux** | C 下界 | R3.3−A' |
|:--|:--|:--|:--|:--|:--|:--|
| mmWave | **0.8402** | 0.8108 | 0.8152 | 0.8309 | 0.0174 | +2.0pp |
| WiFi | 0.5606 | 0.5623 | 0.5561 | **0.6630** | 0.0394 | **+10.1pp** |
| RFID | **0.4242** | 0.3917 | 0.4082 | 0.4141 | 0.2626 | +2.2pp |
| mmWave+WiFi | 0.8820 | 0.8909 | 0.8773 | **0.8920** | 0.0409 | +1.0pp |
| mmWave+RFID | **0.8630** | 0.8345 | 0.8464 | 0.8426 | 0.2688 | +0.8pp |
| WiFi+RFID | 0.5800 | 0.6365 | 0.6282 | **0.6589** | 0.2855 | +2.2pp |
| 全模态 | 0.8947 | 0.8870 | 0.8888 | **0.8964** | 0.2897 | +0.9pp |
| **worst-subset** | **0.4242** | 0.3917 | 0.4082 | 0.4141 | 0.0174 | +2.2pp |
| **macro 平均** | 0.7207 | 0.7162 | 0.7171 | **0.7426** | 0.1720 | **+2.6pp** |

**总目标达成**：R3.3 macro 0.7426 超过 A'（+2.6pp）与 A_off（+2.2pp），全模态 0.8964 为全部组别最高；最大收益在 WiFi-only（+10.1pp）与 WiFi+RFID（vs A_off +7.9pp）——正是掩码表与辅助损失的直接靶点。剩余短板：worst-subset（RFID-only 0.4141）仍略低于 A_off（0.4242）。

### 第二轮完整命令（R3.1-R3.3 已按此执行，留档供复现；R3.4 为消融备选）

```bash
cd ~/dev/X-Fi/XRF55_HAR
# R3.1 预训练：辅助损失 λ0.5 + WiFi keep 0.7（~2h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_pretrain.py \
    --config configs/jepa_aux.yaml --run-name pretrain_aux_w07
# R3.2 探测（~1h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_linear_probe --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_aux_w07/epoch_100.pth --run-name B_probe_aux
# R3.3 微调：mirror 配方（~2h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --config configs/eval_methods_mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_aux_w07/epoch_100.pth --run-name Bp_ft_mirror_aux
# R3.4（可选消融，未排期）：辅助损失 + 原 WiFi keep 0.9，分离两项改动的贡献（~2h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_pretrain.py \
    --config configs/jepa_aux_w09.yaml --run-name pretrain_aux_w09
```

| 部分 | 含义 |
|:--|:--|
| `--config configs/jepa_aux.yaml` | 第二轮预训练变体：`aux_loss`（weight 0.5 / ramp_epochs 5 / normalize both）+ `mask.modality_keep_probs [0.5, 0.7, 0.5]`（WiFi 0.9→0.7）；其余同 `jepa_baseline.yaml`（batch 512 / lr 5e-4 / 100 epochs / EMA 0.996→1.0 / val_split 0.05） |
| `--config configs/eval_methods_mirror.yaml` | 第一轮胜出的微调配方（恒定 lr 1e-4 / wd 0.01 / 无 val 留出 / last-model，见 D28/D29） |
| `--pretrained .../pretrain_aux_w07/epoch_100.pth` | R3.1 产出的预训练权重（epoch_100 与 last 等价，与 D24 的甄别结论一致） |
| `--run-name` | 分别产出 `jepa_checkpoints/pretrain_aux_w07/`、`jepa_eval_results/{B_probe_aux, Bp_ft_mirror_aux}/`，产物互不覆盖 |
| 训练中监控 | TB/CSV 的 `Loss/aux_train_epoch` 应持续下降（实测 0.637→0.329，未收敛）；`Val/explained_var` 与主任务指标解耦（本轮 0.9634 < 基线 0.978，但下游 +2.6pp，见 D31——下游为准） |

### 第三轮：R4 均衡练习 + 时序块掩码（2026-09-27 新增，方案与实现见 D33）

针对导师意见①（各模态组合都要提升）的根因 ②（练习分布不均）与 ③（掩码无结构）：**子集 7 路轮巡**（每组合等频练习，替代 iid 抽取——keep_probs 在此配置下退役）+ **时序连续块掩码**（连续 4 token × 2 块 = 25%，替代均匀随机 token 掩码，JEPA-MSAC/WiFi-JEPA 手段）；保留辅助损失与 WiFi0.7 表被轮巡取代。

| ID | run 名 | 目的 | 关键参数 | 预计时长 | 状态 | macro / worst / all（回填） |
|:--|:--|:--|:--|:--|:--|:--|
| R4.1 | `pretrain_r4_balanced` | 预训练：均衡子集轮巡 + 时序块掩码 + aux | `jepa_pretrain.py --config configs/jepa_aux_r4.yaml --run-name pretrain_r4_balanced` | ~2h | ✅ 完成 | val_loss 0.0181 / expl_var 0.9653（与健康基线同形）；轮巡生效（freq_* 见 D35）；aux 0.640→0.309 |
| R4.2 | `B_probe_r4` | 探测：对照 R3.2（macro 0.6156），判定门 macro>0.6264 且 mmWave+RFID 恢复 | `jepa_downstream.py --method jepa_linear_probe --pretrained .../pretrain_r4_balanced/epoch_100.pth` | ~1h | ✅ 完成 | 0.6016 / 0.3489 / 0.6498 |
| R4.3 | `Bp_ft_mirror_r4` | 微调 mirror 配方：判定门 macro>0.7426（超 R3.3）且 7 子集全部 ≥ A' | `jepa_downstream.py --method jepa_finetune --config configs/eval_methods_mirror.yaml --pretrained .../pretrain_r4_balanced/epoch_100.pth` | ~2h | ✅ 完成 | **0.7410 / 0.4133 / 0.8930** |

### 第三轮判定结果（2026-09-28，详细分析见 D35）

| 模态子集 | A_off 官方 | A' 自训 | R3.3 aux（第二轮） | **R4.3 均衡+块掩码** | R4.3−A' | R4.3−A_off |
|:--|:--|:--|:--|:--|:--|:--|
| mmWave | **0.8402** | 0.8108 | 0.8309 | 0.8323 | +2.2pp | −0.8pp |
| WiFi | 0.5606 | 0.5623 | 0.6630 | **0.6606** | **+9.8pp** | **+10.0pp** |
| RFID | **0.4242** | 0.3917 | 0.4141 | 0.4133 | +2.2pp | −1.1pp |
| mmWave+WiFi | 0.8820 | 0.8909 | **0.8920** | 0.8870 | −0.4pp | +0.5pp |
| mmWave+RFID | **0.8630** | 0.8345 | 0.8426 | 0.8570 | +2.3pp | −0.6pp |
| WiFi+RFID | 0.5800 | 0.6365 | **0.6589** | 0.6435 | +0.7pp | **+6.4pp** |
| 全模态 | 0.8947 | 0.8870 | **0.8964** | 0.8930 | +0.6pp | −0.2pp |
| **worst-subset** | **0.4242** | 0.3917 | 0.4141 | 0.4133 | +2.2pp | −1.1pp |
| **macro 平均** | 0.7207 | 0.7162 | **0.7426** | 0.7410 | **+2.5pp** | **+2.0pp** |

**判定**：R4 = R3.3（macro 差 0.16pp，噪声内打平）——**判定门"macro>0.7426"未过，probe 门也未过（0.6016 < 0.6264；mmWave+RFID 0.5753 仍未恢复）**。两轮不同预训练配方收敛到同一平台（macro ~0.74），表明**"仅预训练融合层（12.4M）+ 同分布数据"的天花板已到**。轮巡与块掩码的价值：消除了练习不均的方差源（原则性保留为默认），但未带来额外性能。

### 第三轮完整命令（复制即用）

```bash
cd ~/dev/X-Fi/XRF55_HAR
# R4.1 预训练（~2h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_pretrain.py \
    --config configs/jepa_aux_r4.yaml --run-name pretrain_r4_balanced
# R4.2 探测（~1h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_linear_probe --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_r4_balanced/epoch_100.pth --run-name B_probe_r4
# R4.3 微调 mirror 配方（~2h）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --config configs/eval_methods_mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_r4_balanced/epoch_100.pth --run-name Bp_ft_mirror_r4
```

| 部分 | 含义 |
|:--|:--|
| `--config configs/jepa_aux_r4.yaml` | R4 预训练变体：`mask.subset_schedule: balanced`（7 路轮巡，每 epoch 内洗牌顺序；`modality_keep_probs` 在此模式下退役）+ `mask.token_mode: block`、`mask.block_len: 4`（连续 4 token × 2 块 = 25% 覆盖）+ 保留 `aux_loss`（λ0.5/ramp5/LN both） |
| 判定门 | R4.2：macro>0.6264 且 mmWave+RFID ≥0.75；R4.3：macro>0.7426 且 7 子集全部 ≥ A'（0.7162 基准）与 A_off（逐子集对照 §6 表） |
| 训练中监控 | CSV 的 `freq_*` 列应≈各 1/7（0.143）——这是轮巡生效的直接证据；`token_mask_rate` ≈0.25；aux 持续下降 |

### 第四轮：① 数据扩展（22000 语料）+ ② 微调步数预算（2026-09-27 已实现/配置，待跑）

| ID | run 名 | 目的 | 关键参数 | 预计时长 | 状态 | macro / worst / all（回填） |
|:--|:--|:--|:--|:--|:--|:--|
| R4.4 | `pretrain_aux_22000` | ① 数据扩展：R3.3 配方（aux+WiFi0.7）+ train+test 无标签语料 22000（+43%） | `jepa_pretrain.py --config configs/jepa_aux_22000.yaml --run-name pretrain_aux_22000` | ~2.9h | ✅ 完成 | val_loss 0.0142 / expl_var 0.9734（**优于 w07 的 0.0192/0.9634**——数据杠杆在 SSL 层亦生效） |
| R4.5 | `B_probe_aux22000` | 新语料表征探测 | `--method jepa_linear_probe --pretrained .../pretrain_aux_22000/epoch_100.pth` | ~1h | ✅ 完成 | **0.7251** / **0.5041** / 0.7509（探针门通过；vs 只用 train 语料的 0.6156 = **+11pp**） |
| R4.6 | `Bp_ft_mirror_aux22000` | 新语料 + mirror 微调：**总判定** vs R3.3（0.7426）与 A'（0.7162）/A_off（0.7207） | `--method jepa_finetune --config configs/eval_methods_mirror.yaml --pretrained .../pretrain_aux_22000/epoch_100.pth` | ~2h | ✅ 完成 | **0.7634 / 0.4386 / 0.8918** |
| R5.1 | `Bp_ft_mirror_b64` | ② 步数预算：mirror 配方 batch 512→64（3.1K→24.1K 步），检验 mmWave 系差距是否步数问题 | `--method jepa_finetune --config configs/eval_methods_b64mirror.yaml --pretrained .../pretrain_aux_w07/epoch_100.pth` | ~8h | ✅ 完成 | 0.7350 / 0.3965 / 0.8902 |

判定门：**R4.6 = 0.7634 ≥ 0.7426 → 通过（+2.1pp over R3.3）**；R4.5 探针门通过（0.7251 > 0.6264）；R5.1 步数假说**证实**（详见 D38）。

### 第四轮判定结果（2026-09-28，完整分析见 D38）

| 模态子集 | A_off 官方 | A' 自训 | R3.3 第二轮 | R5.1 b64(旧pretrain) | **R4.6 22000语料** | R4.6−A' | R4.6−A_off |
|:--|:--|:--|:--|:--|:--|:--|:--|
| mmWave | **0.8402** | 0.8108 | 0.8309 | 0.8170 | 0.8395 | +2.9pp | −0.1pp |
| WiFi | 0.5606 | 0.5623 | 0.6630 | 0.6677 | **0.7395** | **+17.7pp** | **+17.9pp** |
| RFID | 0.4242 | 0.3917 | 0.4141 | 0.3965 | **0.4386** | **+4.7pp** | +1.4pp |
| mmWave+WiFi | 0.8820 | 0.8909 | 0.8920 | 0.8998 | **0.9088** | +1.8pp | **+2.7pp** |
| mmWave+RFID | **0.8630** | 0.8345 | 0.8426 | 0.8042 | 0.8368 | +0.2pp | −2.6pp |
| WiFi+RFID | 0.5800 | 0.6365 | 0.6589 | 0.6698 | **0.6885** | **+5.0pp** | **+10.9pp** |
| 全模态 | **0.8947** | 0.8870 | **0.8964** | 0.8902 | 0.8918 | +0.5pp | −0.3pp |
| **worst-subset** | 0.4242 | 0.3917 | 0.4141 | 0.3965 | **0.4386** | **+4.7pp** | **+1.4pp** |
| **macro 平均** | 0.7207 | 0.7162 | 0.7426 | 0.7350 | **0.7634** | **+4.7pp** | **+4.3pp** |

**总目标复核**：vs A' = **7/7 子集全胜**（首次全扫描）；vs A_off = 4 胜（WiFi +17.9 / WiFi+RFID +10.9 / mmWave+WiFi +2.7 / RFID +1.4）2 平（mmWave −0.1 / 全模态 −0.3）1 负（mmWave+RFID −2.6）；**worst-subset 0.4386 首次超过 A_off（0.4242）——最后一块短板（RFID-only）已攻克**。macro 0.7634 创全部组别新高。

### 第五轮（组合实验）：两个 winning 因子尚未合并

R4.6（22000 语料 + b512/3.1K 步）与 R5.1（旧语料 + b64/24.1K 步）各自验证了一个因子；**合并 run** 是当前最直接的增量来源：

| ID | run 名 | 组合 | 预计时长 | 状态 |
|:--|:--|:--|:--|:--|
| R5.2 | `Bp_ft_mirror_b64_aux22000` | `--config configs/eval_methods_b64mirror.yaml --pretrained ./jepa_checkpoints/pretrain_aux_22000/epoch_100.pth --run-name Bp_ft_mirror_b64_aux22000`（其余同 §7 第四轮命令块） | ~8h | ✅ 完成 |

### 第五轮判定结果（2026-09-29，分析见 D39）

| | macro | worst | all | WiFi | mmWave+RFID |
|:--|:--|:--|:--|:--|:--|
| R4.6（22000 语料 + b512/3.1K 步） | **0.7634** | **0.4386** | **0.8918** | **0.7395** | **0.8368** |
| R5.2（22000 语料 + b64/24.1K 步） | 0.7266 | 0.3991 | 0.8645 | 0.6889 | 0.7818 |
| R5.1（旧语料 + b64/24.1K 步） | 0.7350 | 0.3965 | 0.8902 | 0.6677 | 0.8042 |

**判定：两因子不可加——R5.2 比两个单因子都差（vs R4.6 −3.7pp、vs R5.1 −0.8pp），R4.6 仍是冠军。** 机理（D39）：24.1K 步的持续 CE 适配会**侵蚀** 22000-aux 表征的结构（类灾难性遗忘），而对旧表征反而有益——步数预算的"最优值"依赖于预训练表征的丰富度，不是越大越好。b64 路线就此证伪封存。

### 第四轮完整命令（复制即用）

```bash
cd ~/dev/X-Fi/XRF55_HAR
# ① 数据扩展
# R4.4
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_pretrain.py \
    --config configs/jepa_aux_22000.yaml --run-name pretrain_aux_22000
# R4.5
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_linear_probe --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_aux_22000/epoch_100.pth --run-name B_probe_aux22000
# R4.6
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --config configs/eval_methods_mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_aux_22000/epoch_100.pth --run-name Bp_ft_mirror_aux22000
# ② R5.1 步数预算（~8h，可与 ① 的下游并行跑另一张卡）
CUDA_VISIBLE_DEVICES=<卡N> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --config configs/eval_methods_b64mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name Bp_ft_mirror_b64
# R5.2 Bp_ft_mirror_b64_aux22000
CUDA_VISIBLE_DEVICES=0 conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --config configs/eval_methods_b64mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/pretrain_aux_22000/epoch_100.pth --run-name Bp_ft_mirror_b64_aux22000
```

| 部分 | 含义 |
|:--|:--|
| `configs/jepa_aux_22000.yaml` | ① 的预训练变体：R3.3 配方（aux + keep [0.5,0.7,0.5]）+ `data.include_test_split: true`（corpus 21230 train / 770 val，val 仅从 train 部分划）；**协议披露**：test 输入无标签参与预训练，写论文时注明 |
| `configs/eval_methods_b64mirror.yaml` | ② 的微调变体：mirror 配方 + `data.train_batch_size: 64`（步数 3.1K→24.1K） |
| `--pretrained`（②） | 步数实验的初始化用**原 R3.3 同款预训练权重**（20260918 归档 run），保证与 R3.3/Bp_ft_mirror 只差 batch——单变量对照 |

## 8. R6/R7 计划（2026-09-27，本轮只计划、不改码——人工指示）

### 8.1 R6：从论文照片复原 4 个 Scene 的 floorplan（可行性调查结论）

> **范围修正（2026-09-27 人工指示）**：floorplan = 房间**平面图**（非 3D 模型），只含**墙、柱**等房间结构，**不含家具**；设备布局只关注**收集 CSI 的设备（WiFi TX/RX Intel 5300）**，其他模态设备不管；**只关注平面位置，不含高度**。此修正大幅简化复原：设备布局 = "矩形感知区四角的 4 台 Intel 5300 位置 + 朝向"（论文 §3.1：1 TX + 3 RX 位于矩形四角，笔记本高 1.2m 的高度信息弃用），Scene 3 照片的红色标注含笔记本间距离 → 矩形边长可直接恢复。房间多边形只描墙体与立柱（Scene 4 有立柱），跳过衣柜/窗帘等家具。



**信息盘点（读 3643543.pdf 实测）**：Fig. 2（p21:6）为 4 张场景照片（含设备标签：mmWave 雷达/WiFi TX+RX/RFID 天线/RIS 128×128 面板）；图注与 §3.1 明确两点关键事实——**① 设备相对摆放与朝向跨 4 个场景保持一致；② 相对摆放（设备间距离）在 Scene 3 照片上用红色标注**。此外论文给出设备高度（WiFi 笔记本 1.2m、RFID 天线-标签阵 40cm）。Scene 4 有立柱，Scene 1 有衣柜/窗帘等家具（房间外壳差异的主体）。

**可行性结论**：
- ✅ **设备布局图（场景先验通道最需要的部分）——高置信可行**：以 Scene 3 的距离标注为约束做三角化/MDS，重建设备 2D 相对坐标 + 朝向；由于布局跨场景一致，Scene 3 的标定可直接服务 4 个场景；
- ⚠️ **房间外壳（墙体多边形）——中置信可行**：从照片手绘近似多边形（单视角、广角畸变、无绝对尺度 → 误差 ±0.5~1m）；以 Scene 3 标定比例 + 门/窗/立柱做参照可收敛到"布局指纹"级别；
- ❌ **精确建筑级按比例 floorplan——不可行**（照片无法提供全屋尺度标定）。

**复原方法（建议流程）**：
1. Scene 3 照片提取红色距离标注 → 以一台 mmWave 雷达为原点，MDS/最小二乘三角化全部设备 2D 坐标 + 朝向（**共享布局**，一次重建四场景通用）；
2. 四张照片手绘房间多边形（含立柱/家具/门窗），以第 1 步的比例标定对齐 → 4 份近似 floorplan（矢量：墙体线段列表）；
3. 产出格式：每场景 JSON `{room_polygon: [...], devices: [{type, xy, yaw}]}`；编码方案 A=64×64 占用/语义栅格 + 设备位置热图 → 小 CNN；方案 B=多边形+设备坐标 → 坐标 MLP/点云编码器（JEPAREADME §8-1 的预留接口：场景先验作为第 4 模态走同一 projector→X_Fusion 通路）。

**工作量**：人工手绘+标注 ~0.5-1 天；编码器实现 1-2 天。

**实验设计要点（成色保障）**：必须对照 **one-hot scene-id 基线**（4 类，零成本）——若 floorplan 先验不显著优于 one-hot，说明模型只在做场景查表，创新点不成立；另做 leave-one-scene-out 与布局扰动鲁棒性实验。风险评估：几何误差 ±0.5m 对"场景指纹"用途可接受，对精确定位类任务不可接受。

### 8.2 R7：MM-Fi 多数据集/多任务下游评测

> **M2 格式审计结果（2026-09-27 完成，数据已链接 `data/MMFi_Dataset -> /mnt/DataDrive164/wr/MMFi_Dataset`）**：结构 = E01-E04 环境 × S01-S10 受试者 × A01-A27 动作 × 每动作一段记录（实测 E01/S02/A14 为 297 帧）× 逐帧文件。**WiFi CSI**：每帧 `.mat` 含 `CSIamp (3, 114, 10)` + `CSIphase (3, 114, 10)` float64——3 天线 ✓（与 XRF55 天线数一致）但 **114 子载波 × 10 包**，与 Intel 5300 的 270 通道（30 子载波）**不同构**，需输入适配层（如包平均 + 114→30 子载波重采样，或 1×1 Conv 342→270 通道适配）；**mmWave**：每帧 `.bin` 仅 920 字节的 TI 原始 ADC（IWR6843 与 XRF55 同款雷达但采集配置不同）——需 MM-Fi 官方预处理流程重建 Range-Doppler/Angle 图再重采样到 (17,256,128)，工作量大于 WiFi 路径；**标签红利**：`ground_truth.npy (297, 17, 3)` 为逐帧 17 关键点 3D 人体骨架（Kinect）——**多任务扩展（HPE）的数据是现成的**，验证了 §8.2 原判断。帧级 vs XRF55 样本级（1 秒切片）的时序聚合策略需设计（均匀帧池化 / 滑窗 + 投票）。


**重要事实修正**：XRF55 论文 Table 1（p21:4）记录 **MM-Fi 数据集本身包含 Wi-Fi CSI**（模态：WiFi/mmWave/RGB+D/LiDAR，40 subjects）——此前"MM-Fi 无 WiFi 模态"的判断只对 X-Fi 官方仓库的**代码**成立（MMFi_HAR/MMFi_HPE 无 WiFi 分支），对**数据集**不成立。⇒ 多数据集 WiFi 评测可行，且能保留我们最关心的 WiFi 通道。

**方案（分里程碑）**：
- M1 数据获取（人工）：MM-Fi 官方 GitHub（ybhbingo/MMFi_dataset）按模态/任务下载子集（全集大；先下 WiFi CSI + mmWave + 动作标签）；
- M2 格式审计（关键风险）：核对 MM-Fi 的 WiFi CSI 设备/协议/维度（MM-Fi 论文与 XRF55 的 Intel 5300 30 子载波是否同构）与 mmWave 数据布局（XRF55 为 (17,256,128) 距离-多普勒/角度拼接图；MM-Fi 若为原始 ADC/其他 FFT 布局则不兼容，需重预处理或放弃 mmWave 只做 WiFi）；
- M3 新加载器：`MMFi_Dataset.py`（新文件，镜像 `XRF55_Dataset` 接口；**不改动** XRF55_Dataset.py）；类空间 MM-Fi HAR = 27 类 → `classification_Head(512, 27)`；
- M4 评测协议（两档）：① **迁移探测**——冻结全部，只训头（检验跨数据集表征迁移，JEPA 的主战场）；② **微调**——projector+X_Fusion+头（mirror 配方）；对照组 = 原版 X-Fi 同数据同预算训练（scratch 对照）；
- M5 判定：JEPA 初始化 vs scratch 在 MM-Fi 上的宏平均/逐子集差；预期 WiFi 子集优势最大（XRF55 预训练已含 WiFi 表征）；
- M6 多任务扩展（后续）：MM-Fi 支持 HPE/定位等，README §7-4 的任务头并列接口可承接。

**风险**：WiFi 设备/协议差异（Intel 5300 vs MM-Fi 所用网卡）导致 CSI 不可直接同构——若差异大，退化为"仅 mmWave 跨数据集"或需域自适应；MM-Fi 受试者划分协议（S1 测试）需遵循其论文。

### 8.3 待人工决策

① R6 手绘标注是否开工（~0.5-1 天人工）；② R7 M1 数据下载是否启动；③ 第四轮（§7）与 R6/R7 的执行顺序。

### 判定门

1. **R1 门**（对照 A' macro 0.7162）：**R1.1 mirror = 0.7171 ≥ 0.7162 → 通过**（worst +1.7pp、mmWave+RFID +1.2pp；幅度在噪声边缘但方向一致）。**采纳 mirror 配方为后续 finetune 默认**。R1.2 为否定性结果：wd 0.05 + cosine 在 lr 1e-4 下**损害**预训练特征（WiFi 0.4108、best-by-val 停在 epoch 12/100）——JEPA 初始化的微调对配方敏感，恒定低 wd 是正确配方。
2. **R2 门**：`Bp_ft_lab10` vs `A_xfi_lab10` = macro +0.4pp、all +1.3pp、worst −0.3pp——**JEPA 仅微弱领先，未出现 SSL 预期的大幅优势**（两侧标签样本相同，已核实）。归因见 D29。

### 完整命令（复制即用；`<卡N>` 先 nvidia-smi 挑空闲卡）

```bash
cd ~/dev/X-Fi/XRF55_HAR
# R1.1 / R1.2（双卡并行）
CUDA_VISIBLE_DEVICES=<卡1> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --config configs/eval_methods_mirror.yaml --method jepa_finetune \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name Bp_ft_mirror
CUDA_VISIBLE_DEVICES=<卡2> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --config configs/eval_methods_warmstart.yaml --method jepa_finetune \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name Bp_ft_warmstart
# R2.1（第三张卡）
CUDA_VISIBLE_DEVICES=<卡3> conda run -n xfi --no-capture-output python run_xfi_baseline.py \
    --label-fraction 0.1 --run-name A_xfi_lab10
# R2.2（等卡1空出）
CUDA_VISIBLE_DEVICES=<卡1> conda run -n xfi --no-capture-output python jepa_downstream.py \
    --method jepa_finetune --label-fraction 0.1 --config configs/eval_methods_mirror.yaml \
    --dataset ../data/XRF55_Dataset_split \
    --pretrained ./jepa_checkpoints/20260918_135302_b512_lr5e-4_100ep/epoch_100.pth --run-name Bp_ft_lab10
```

### 结果回填约定

每个 run 产出 `jepa_eval_results/<run名>/results.json`（headline `accuracy` = best-by-val；mirror 变体无 val 时为 last-model，JSON 的 `headline` 字段会注明）。跑完把 run 目录名发给 agent，或直接把 macro/worst/all 三格填进上表；全部完成后由 agent 汇总升级 §6/§7 对比表并写 D29 结论。
- **D29 性能提升第一轮结果与结论（2026-09-21，人工执行，结果目录 `jepa_eval_results/{Bp_ft_mirror, Bp_ft_warmstart, Bp_ft_lab10, A_xfi_lab10}`）**：
  - **R1（全标签）判定门通过**：R1.1 mirror（JEPA 初始化 + A' 配方）macro **0.7171 ≥ A' 0.7162**（worst 0.4082 vs 0.3917、mmWave+RFID 0.8464 vs 0.8345），"同配方、更好初始化"方向成立但幅度在噪声边缘；**采纳 mirror 配方为默认 finetune 配方**。R1.2 warmstart（同为 lr 1e-4 但 wd 0.05 + cosine + best-by-val）macro 跌至 0.6801、WiFi 0.4108、val 最优停在 epoch 12/100——**否定性结果：预训练特征对微调配方敏感，高 wd + 余弦调度会侵蚀 JEPA 表征；恒定 lr + 低 wd（A' 式配方）才是正确配对**。这也解释了最初 Bp_ft_ep100（lr 1e-5）落后的原因：不是 JEPA 表征差，而是适应预算不对等。
  - **R2（10% 标签 few-shot）**：Bp_ft_lab10 macro 0.5578 vs A_xfi_lab10 0.5537（+0.4pp）、all +1.3pp、worst −0.3pp——**JEPA 仅微弱领先，SSL 大幅优势未出现**。两侧有标签样本完全一致（同池同 seed 已核实）。归因分析：① 本架构中真正携带学习的是**作者预训练的冻结 extractor**（20.55M/32.94M），A/B 两组共享——原版"从零训练"实为"冻结特征上训练融合层"，并非真正的冷启动；JEPA 只额外预训练了 12.4M 的融合层；② 预训练语料与微调数据同分布同量，无数据杠杆；③ 1485 个有标签样本对 12.4M 融合层的适配已足够充分。两侧 WiFi-only 一致塌陷至 ~17%（全标签时 55-60%）——**WiFi CSI 在低标签下最脆弱**，是下轮改进的首要靶点。
  - **下一轮建议**（按优先级，方案已在 D28 备案）：① 跨模态辅助预测损失 + WiFi keep 0.9→0.7 掩码表（同时针对融合层表征质量与 WiFi 脆弱性，预训练 ~2h/次）；② 预训练数据扩展（train+test 无标签 22000）；③ 若仍无突破，考虑部分解冻 extractor 末层（架构偏离，需人工决策）；④ few-shot 曲线补 1%/5% 两点（探针口径，~1h）以完整刻画标签效率。
  - **状态更新（2026-09-23）**：① 已完成（D30/D31 即其结果）；② 数据扩展——fork 会话曾认领，**后经其用户指示取消认领、归还待办池**（其分支转产小组会汇报文档，jepa_pretrain.py 未被其改动）；③④ 待人工决策。
- **D30 第二轮实现：跨模态辅助预测损失 + WiFi 0.7 掩码表（2026-09-21，人工批准实施）**：动机——D29 表明 JEPA 只预训练了 12.4M 融合层且"无 WiFi 聚合"能力缺失（B 探测 mmWave+RFID 塌陷、双侧 WiFi-only 低标签塌至 17%）。实现：① `linear_projector.forward_per_modality`（`forward` 零改动）；② `EMATargetEncoder.forward_with_parts`（**与 forward 数值等价已断言 allclose**，parts 原本就在 forward 内部计算后被丢弃——零额外抽取成本）；③ `AuxModalityHeads`：3×Linear(512,512)（~0.79M），**有意用线性头**——目标是"聚合表征可被线性解码出缺失模态特征"，与线性探测评测同构，且头难以替代表征作弊；放在 backbone/EMA 之外 → `online_state/ema_state` 键零变化（已验证新 ckpt 下游 strict 加载 OK），checkpoint 新增顶层 `aux_state`；④ 损失：对 context 中**缺席**模态，SmoothL1(aux_m(z_hat), LN(ema_parts[m]))，目标与预测都过**无参数 LayerNorm**（无 affine → 不进 EMA、结构上免疫 D17 尺度正反馈）；λ=0.5 线性 ramp 5 epochs；~17.5% 全模态 batch 无辅助项；⑤ 掩码表 [0.5, **0.7**, 0.5]：WiFi 缺席 batch 从 10%→30%，同时加密辅助信号；⑥ CSV/TB 增 `aux_loss` 列；`--resume` 对旧 ckpt 容忍缺失 `aux_state`。冒烟：aux 0.642（有限、量级合理）、val 流程正常、config 变体 `configs/jepa_aux.yaml`（λ0.5/ramp5/LN both/WiFi0.7）。实验行见 §7 第二轮表；判定门：探测恢复 mmWave+RFID ≥0.75、微调 ≥A' 且 worst >0.4082。
- **D31 第二轮结果与结论（2026-09-22，人工执行，结果目录 `jepa_checkpoints/pretrain_aux_w07`、`jepa_eval_results/{B_probe_aux, Bp_ft_mirror_aux}`）**：
  - **总目标达成：JEPA 首次总体超过原版 X-Fi**。R3.3（辅助损失+WiFi0.7 预训练 → mirror 微调）macro **0.7426**，超 A' +2.6pp、超 A_off +2.2pp；全模态 **0.8964** 为全部组别历史最高；7 子集中 6 个超过 A'、5 个超过 A_off。最大收益 WiFi-only **0.6630**（vs A' +10.1pp、vs A_off +10.2pp）与 WiFi+RFID 0.6589（vs A_off +7.9pp）——与两项改动的靶点（WiFi 脆弱性、缺失模态推断）精确对应。
  - **判定门一过一不过，且解耦有信息量**：R3.3 门过（macro/worst 双达标）；R3.2 探测门未过（0.6156 < 0.6264，mmWave+RFID 0.5585 未恢复）——辅助损失塑造的是"微调后可利用"的跨模态结构而非线性可读结构，probe 与 finetune 解耦（probe −1.1pp、finetune +2.6pp）。含义：评测 JEPA 预训练价值应以微调为准，线性探测会低估（与 §6 备注一致）。
  - **主 SSL 指标略降但下游更优**：R3.1 的 val_loss 0.0192 / expl_var 0.9634 弱于基线 run 的 0.0117 / 0.978——辅助目标占用了部分表征容量；但换来的跨模态结构在微调后净收益 +2.6pp。"SSL 验证指标更好 ≠ 下游更好"，后续判定以下游为准、SSL 指标仅作健康监控。aux_loss 0.637→0.329 仍在下降（未收敛，加长训练或仍有空间）。
  - **剩余短板与下轮候选**：① worst-subset（RFID-only 0.4141）仍低于 A_off 0.4242——RFID 是最后未攻克的子集；② R3.4 消融（aux+w0.9）分离掩码表与辅助损失的贡献；③ 数据扩展（train+test 22000，~2.9h）；④ aux λ 扫描（1.0）或 aux 收敛后加长预训练。均未排期，等人工决策。
- **D32 导师反馈与改进路线图（2026-09-23，小组会）**：导师三点意见——① R3.3 相比原版增益不够（期望**各任务、各模态组合全面提升**，目前仅特定组合有优势）；② 基础模型需**多任务/多数据集**评价（目前仅 XRF55 HAR）；③ **floorplan 未引入**（创新点之一，后续工作）。参考论文与可借鉴点映射：CSI-JEPA（信道变化强度**感知掩码**；冻结+轻量适配器 7 任务 +10.64pp/标签节省 98%）；WiFi-JEPA（**整链路结构化掩码**迫跨链路推理；其"视觉原生 SSL 目标不及从零训练"的反例与我们观察一致）；JEPA-MSAC（统一词元空间 + **时序块掩码**；冻结骨干+轻量头多任务）；LatentWave（**逐通道 patch + 随机通道采样**；反对低层重建目标）；WWM/WirelessJEPA 已排除（点云射线追踪/原始 IQ）。**根因分析**（为什么未全面胜出）：① 预训练的"练习分布"不均衡——模态子集按 iid 采样，组合练习次数方差大，部分组合欠训练；② 掩码无结构信息——均匀随机 token 掩码 + 固定模态概率，未利用 CSI 时序结构与信道信息量（论文的核心手段）；③ 步数预算不对等——A_off 为 batch16×100ep=96K 步，我们 batch512=3.1K 步，mmWave 系子集的劣势与 A'（同为 b512）一致，提示步数因素；④ 数据无杠杆——预训练语料=训练集本身；⑤ 评价维度单一（仅 HAR 全标签）。**分轮路线**：R4 均衡练习+结构化掩码（子集分层轮巡替代 iid、时序块掩码、保留 aux）→ R5 微调步数预算（batch 64/16 mirror）+ 适配器对照 → R6 floorplan/场景先验通道（JEPAREADME §8-1 预留接口，创新点）→ R7 标签效率曲线（1%/5%/10%/25%）与多任务扩展（顾问意见②的正面回应）。
- **D33 第三轮实现：R4 均衡练习 + 时序块掩码（2026-09-27，人工批准"一项一项来做"）**：针对 D32 根因 ②③。① `mask.subset_schedule: balanced`——子集 7 路轮巡（每 epoch 内 mask_rng 洗牌顺序，batch i 取 `SUBSET_BOOLS[cycle_order[i % 7]]`），**各组合等频练习**（替代 iid 抽取，`modality_keep_probs` 在此模式退役）；② `mask.token_mode: block` + `block_len: 4`——连续 4 token × 2 块（round(32×0.25/4)=2 块，不重叠起点）替代均匀 iid token 掩码（JEPA-MSAC 时序块 / WiFi-JEPA 结构化掩码手段），块起点 `mask_rng.permutation(29)[:2]`；③ aux 损失保留（均衡轮巡使每个缺席模态的辅助目标获得等频练习——与轮巡协同）。**人工决策记录**：R4 先行、R5 等 R4 结果；**R6 受阻——XRF55 未提供 floorplan**（数据事实，已核实），floorplan 式场景先验不可行，场景 id 版先验存疑待议；**R7 重心改定义为"在多样的数据集和/或多样的任务上做下游微调"**（目前仅 XRF55+HAR，不满足基础模型评价——需新数据/新任务获取决策，如 XRF55 定位任务需位置标签核实）。冒烟（3+3 batch × 2 epoch）：轮巡频率每组合 33-67%（6 batch/7 组合的预期分布）、块掩码 rate 0.237≈0.25、aux 0.640 正常。实验行见 §7 第三轮表 + 完整命令。
- **D34 checkpoint 选择分析：为什么下游用 epoch_100 而非 best_val.pth（2026-09-27，人工提问触发）**：人工问为何命令都用 epoch_100。用 pretrain_aux_w07 的数据回答——**SSL 验证指标存在"选择偏向低信息量检查点"的缺陷**：val_loss 于 ep13 最低（0.00715）、val_expl_var 于 ep13 最高（0.9858），best_val.pth 因此记录的是 ep12/13 权重；但同期 z_tgt 跨样本 std 仅 ~0.19（ep100 为 0.91）——低 val loss = 表征"容易预测"= 跨样本信息量低，而下游判别力恰恰需要跨样本信息。R3.3 从 ep100 出发取得 +2.6pp，佐证末段丰富表征才是正确选择（EMA momentum→1.0，末段编码器最平滑收敛）。**结论：① 下游默认 epoch_100（= last.pth）；② best_val.pth 降级为诊断产出（记录"表征何时最可预测"，不是推荐检查点）；③ 可选实证（~1h）：对 ep12 的 best_val 跑探针对比 ep100（预期明显更差）。**待办池新增：checkpoint 选择判据修正（如 val loss + 跨样本 std 组合，或小探针选点）。另：20260918 归档 run 无 best_val 机制，D24 事后甄别即为其 epoch_100 的依据——与本次结论一致。
- **D35 第三轮结果与结论（2026-09-28，人工执行，结果目录 `jepa_checkpoints/pretrain_r4_balanced`、`jepa_eval_results/{B_probe_r4, Bp_ft_mirror_r4}`）**：
  - **机制验证**：轮巡生效——7 子集抽取严格等频（freq 列恒定，如 mmWave+RFID 恒 0.276/0.31，all 恒 0.138-0.172）；块掩码 rate 0.237≈0.25；无塌缩（z_tgt std 0.19→0.90）；主 SSL val 曲线同形（min 0.00775@ep9 → 0.0181@ep100，D34 的选点问题在 R4 依旧存在）。
  - **判定门**：R4.3 macro 0.7410 vs 门 0.7426——**未过，但与 R3.3 打平（−0.16pp，噪声内）**；6/7 子集 ≥ A'（mmWave+WiFi −0.4pp 为唯一例外）。probe 门未过（0.6016 < 0.6264，mmWave+RFID 0.5753）。
  - **核心结论：天花板到达**。两轮独立配方（R3.3 aux+wifi0.7 vs R4 轮巡+块掩码）微调后 macro 收敛于 0.741-0.743、probe 收敛于 0.60-0.62——**"仅预训练融合层（12.4M 参数）+ 同分布 15400 样本"这一设定的性能上限已到**。轮巡与块掩码按设计工作（消除练习不均、结构化掩码）但未转化为增量，说明信息瓶颈不在掩码策略，而在：① 可学习的参数盘子太小（extractor 20.55M 冻结）；② 数据无杠杆（预训练语料 = 训练集）。
  - **对导师意见①的现状回答**：R4.3 vs A' = 全 7 子集中 6 个领先（macro +2.5pp）；vs A_off = macro +2.0pp，WiFi 系大幅领先（+10.0/+6.4pp），mmWave 系与 RFID 系仍小幅落后（−0.6 ~ −1.1pp）——"部分组合有优势、部分组合略差"的格局未变。
  - **下一轮候选（等人工决策，均需新代码或新数据）**：① 数据扩展——train+test 无标签 22000（方案在 D28/D31 备案，未实现）；② 部分解冻 extractor 末层微调（架构偏离协议，需人工拍板）；③ R3.4 消融（配置/命令已就绪）；④ few-shot 曲线补 1%/5%（探针口径 ~1h）；⑤ 微调步数预算实验（batch 64 mirror，~8h，检验 mmWave 系差距是否步数问题——A_off 96K 步 vs 我们 3.1K 步）。
- **D36 第四轮实现 + R6/R7 计划（2026-09-27，人工批准"①+② 并行，R6/R7 只计划"）**：① **数据扩展已实现并冒烟通过**：`jepa_pretrain.py` 新增 `data.include_test_split`（ConcatDataset，索引空间 train[0,15400)+test[15400,22000)），**val 始终只从 train 部分划**（test 数据不触碰 val/test 标签语义）；冒烟实测 corpus 21230 train / 770 val，aux 块正常；协议披露已写进日志行与 config 注释（test 输入无标签参与预训练）。变体 `configs/jepa_aux_22000.yaml`（R3.3 配方 + 22000 语料，单变量对照 R3.1）。② **步数预算实验配置**：`configs/eval_methods_b64mirror.yaml`（mirror 配方 + batch 512→64，步数 3.1K→24.1K，单变量对照 Bp_ft_mirror 与 A_off 的 96K 步）。第四轮实验表与完整命令见 §7（R4.4-R4.6 + R5.1）。**R6 调查结论**（读 3643543.pdf）：Fig.2 四场景照片 + Scene 3 红色相对距离标注 + 图注"设备相对摆放跨场景一致"——设备布局图高置信可复原（MDS/三角化），房间外壳中置信（手绘近似 ±0.5-1m），精确建筑级不可行；**但场景先验通道需要的正是"设备布局 + 粗房间指纹"，可行性成立**；方案/编码/实验设计（含 one-hot 对照防"场景查表"退化）见 §8.1；前置工作为人工手绘标注 ~0.5-1 天。**R7 计划**（§8.2）：重要事实修正——XRF55 论文 Table 1 记录 **MM-Fi 含 WiFi CSI**（此前"MM-Fi 无 WiFi"仅对 X-Fi 官方仓库代码成立）；六里程碑方案（M1 下载→M2 格式审计→M3 加载器→M4 两档协议→M5 判定→M6 多任务）；关键风险 = WiFi 设备/协议差异与 mmWave 数据布局。§8.3 列了三个待人工决策点。
- **D37 R6 范围修正 + R7 审计完成（2026-09-27，人工指示）**：① **R6 范围收窄**（人工指示）：floorplan = 2D 平面图（墙+柱，无家具）；设备布局只含 CSI 收集设备（WiFi TX/RX Intel 5300，矩形四角 4 台）、只记平面位置。简化后设备布局先验 = "感知矩形四角 + 朝向"（Scene 3 标注直接给出边长），复原难度进一步下降。② **R7 M2 审计完成**（数据已链接 `data/MMFi_Dataset -> /mnt/DataDrive164/wr/MMFi_Dataset`）：结构 E01-E04 × S01-S10 × A01-A27 × 逐帧文件；WiFi CSI 每帧 `.mat` = `CSIamp/CSIphase (3,114,10)`——**3 天线一致但 114 子载波×10 包，与 Intel 5300（30 子载波）不同构**，需输入适配（包平均+子载波重采样或 342→270 Conv）；mmWave 每帧 `.bin` 920 字节 TI 原始 ADC（IWR6843 同款雷达、不同采集配置）→ 需官方预处理重建热图后重采样，工作量大于 WiFi 路径；**ground_truth.npy (297,17,3) = 逐帧 17 关键点 3D 骨架** → HPE 多任务数据现成。③ R7 里程碑更新：M1/M2 完成；M3 加载器 + 时序聚合策略（帧池化/滑窗）为下一步实现项（等人工下令）；M4 协议两档不变。
- **D38 第四轮结果与结论（2026-09-28，人工执行，结果目录 `jepa_checkpoints/pretrain_aux_22000`、`jepa_eval_results/{B_probe_aux22000, Bp_ft_mirror_aux22000, Bp_ft_mirror_b64}`）**：
  - **① 数据杠杆证实（本轮最大收益）**：22000 语料使 SSL 验证指标本身改善（0.0142/0.9734 优于 train-only 的 0.0192/0.9634），下游三处跳变——probe macro 0.6156→**0.7251**（+11pp，探针门首次通过）；微调 macro 0.7426→**0.7634**；**vs A' 7/7 子集全胜**、vs A_off 4胜2平1负、**worst-subset 0.4386 首次超过 A_off（RFID-only 短板攻克）**。WiFi-only **0.7395**（vs A_off +17.9pp）创单子集最大领先。
  - **② 步数假说证实（作为混杂因子）**：同预训练同配方，batch 512→64（步数 ×8）使 macro 0.7171→0.7350（+1.8pp）、WiFi 0.5561→0.6677（+11.2pp）——**此前所有微调数字都被 3.1K 步的低适应预算压低了**（含 D25-D27 的 B' 系列）。但 mmWave vs A_off 仍 −2.3pp（24K 步未完全追平 96K 步）→ mmWave 的剩余差距 = 步数 + 其他因素（可能含 batch 统计量效应）。
  - **两个因子的量级**：数据缩放 +2.1pp > 步数 +1.8pp（各自相对控制组），且**尚未合并**——R5.2（aux22000 预训练 + b64 微调）已排入 §7 第五轮，预期 ≥0.76。
  - **mmWave+RFID 仍是唯一未过的子集**（R4.6 0.8368 vs A_off 0.8630，−2.6pp；且 b64 反而降到 0.8042）——该组合对步数负敏感，归因待查（可能 mmWave+RFID 组合依赖更充分的监督收敛）。
  - **文档与叙事**：§6/§7 主表后续以 R4.6 为 JEPA 代表组；论文叙事三支柱=① 数据杠杆（+43% 无标签语料 → 全面提升）、② 步数/预算敏感性（SSL 初始化需配对预算才公平）、③ 标签效率曲线（待补 1%/5% 探针点）。D34 的"SSL val 指标偏向低信息检查点"在 22000 语料下依旧成立（val min @ep8）。
- **D39 第五轮结果：因子不可加，R4.6 仍为冠军（2026-09-29，人工执行，结果目录 `jepa_eval_results/Bp_ft_mirror_b64_aux22000`）**：合并 run（22000 语料 + b64/24.1K 步微调）macro **0.7266**——比两个单因子都差（R4.6 0.7634 / R5.1 0.7350），全模态跌至 0.8645、WiFi 跌至 0.6889、mmWave+RFID 0.7818。**机理**：持续 24.1K 步的恒定 lr CE 适配会侵蚀 22000-aux 表征的结构（对 SSL 预训练结构的"过写"），而对旧 w07 表征反而有益（R5.1 +1.8pp）——**微调步数的最优值随预训练表征的丰富度变化，丰富表征需要更短的适应预算**。b64 路线就此证伪封存。**最终配方定格：pretrain_aux_22000（epoch_100）+ mirror 微调（b512 / 恒定 lr 1e-4 / wd 0.01 / 100 epochs / last-model）= macro 0.7634，vs A' 7/7 全胜（+4.7pp）、vs A_off macro +4.3pp 且 worst-subset 反超**。后续若再压榨：只剩探针选点（epoch_XXX 之间挑，~1h/点）、λ 扫描、或 extractor 部分解冻（需人工拍板架构偏离）。标签效率曲线（1%/5% 探针，R4.6 预训练）为论文核心图，建议优先补。
