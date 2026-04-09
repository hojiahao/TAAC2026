# UniRec：面向 TAAC 2026 的统一推荐模型

**KDD Cup 2026 - Tencent UNI-REC Challenge**

> 面向大规模推荐的序列建模与特征交互统一化

---

## 概述

UniRec 将**序列建模**与**特征交互**统一到单一同构可堆叠骨干网络中，用于 **CVR 预测**（AUC of ROC）。融合 6 项研究工作的创新：

| 来源 | 融合的创新 |
|------|-----------|
| OneTrans (WWW 2026) | 统一分词 + 混合参数化 (S-tokens共享 / NS-tokens独立) |
| InterFormer (Meta 2025) | Feature Cross Layer (骨干前显式特征交互) |
| HSTU 2.0 (Meta 2026) | SiLU Gated Attention + Semi-Local Attention + Attention Truncation + MoT |
| DIN (阿里 2018) | Target-Aware Interest Extraction (目标item查询用户历史) |
| Kimi (2026) | Block Attention Residuals (深度维度选择性信息检索) |
| 本项目 | Hybrid Attention Mask (NS全双向 + S区SLA + target全可见) |

---

## 模型架构

```
输入: user_features (57字段, 4种类型) + item_features (16字段, 3种类型)
      + action_seq (200步) + content_seq (200步) + item_seq (200步)

Stage 1: 统一分词 [OneTrans]
  每个特征字段 → embed(feat_d) → per-feature proj(D) = 73个特征token
  每种序列每步 → concat所有特征 → proj(D) = 600个序列token

Stage 2: 骨干前预处理
  2a. Feature Cross Layer [InterFormer]: 73个NS-token做self-attention
  2b. Target-Aware Interest [DIN]: item_pool查询600个S-token → interest_token
  2c. Target Fusion [HSTU]: MLP([user_pool; item_pool; u*i]) → target_token
  2d. MoT [HSTU 2.0]: 3条序列独立Transducer → 门控融合 → mot_token

Stage 3: 统一序列组装
  [73 feat | 200 action | 200 content | 200 item | mot | interest | target]
  总计 ~677 tokens, 全部D维 + segment_embedding + field_embedding

Stage 4: 同构可堆叠骨干 (Unified Block × N)
  每个 Unified Block:
    ① RMSNorm (Pre-Norm)
    ② SiLU Gated Attention [HSTU 2.0]:
       - S-tokens: 共享 Q/K/V/U
       - NS-tokens: 独立 Q/K/V/U
       - A = SiLU(Q·K^T/sqrt(d)) · HybridMask
       - HybridMask: NS↔NS全双向 + S区SLA + target全可见 [本项目]
    ③ Block Attention Residuals [Kimi]:
       - 每层: pseudo-query w_l (D维, 初始化零)
       - Block内: 标准残差累加
       - Block间: softmax attention over block summaries
    ④ RMSNorm → ⑤ SwiGLU FFN (S共享/NS独立) → ⑥ Residual

  Full层 × N1: 处理全部677 tokens
  Truncated层 × N2: 只处理最近L'个S-tokens [HSTU 2.0 Attention Truncation]

Stage 5: CVR 预测头
  target输出 → MLP → sigmoid → P(conversion)
  Loss: CombinedAUCLoss = (1-α)·WeightedBCE + α·PairwiseBPR
```

---

## 创新点与比赛奖项

| 创新点 | 描述 | 奖项目标 |
|--------|------|---------|
| Hybrid Attention Mask | NS区全双向 + S区Semi-Local + target全可见 | 统一模块创新奖 $45K |
| Block AttnRes in RecSys | 首次在推荐backbone中应用Attention Residuals | 统一模块创新奖 $45K |
| SiLU + 混合参数化 | HSTU 2.0 SiLU gating + OneTrans混合参数化 | 统一模块创新奖 $45K |
| 骨干前DIN | 统一backbone前注入target-aware信号 | AUC排行榜 |
| CombinedAUCLoss | BCE + PairwiseBPR直接优化AUC | AUC排行榜 |
| Scaling Law验证 | XS/S/M/L四规模AUC vs FLOPs幂律曲线 | 扩展规律创新奖 $45K |

---

## 工程优化

### 训练优化

| 优化 | 效果 |
|------|------|
| BF16混合精度 (PyTorch 2.x API) | 速度=FP16, 无overflow, 不需GradScaler |
| Gradient Checkpointing | 省50-70%激活显存 |
| torch.compile | 10-30%训练加速 |
| Muon+AdamW混合优化器 | Muon管Linear(省45%显存), AdamW管Embedding |
| Warmup+Cosine LR (手写, 各optimizer独立) | 稳定收敛 |
| DDP 2卡训练 | torchrun启动, 速度翻倍 |
| Block AttnRes | 等价1.25x计算量的Scaling优势 |

### 推理优化

| 优化 | 效果 |
|------|------|
| Attention Truncation | 深层只处理近L'个token, 省38% FLOPs |
| Semi-Local Attention | O((K1+K2)·L) 替代 O(L²) |
| Attention Mask缓存 | 相同L只构建1次, 后续查缓存 |
| SDPA自动FlashAttention | SoftmaxAttention自动dispatch到FlashAttention2 |
| torch.compile max-autotune | 推理专用编译优化 |
| 推理batch翻倍 | 无梯度→可用训练2倍batch |
| BlockAttnRes轻量化 | Last-token summary O(N·D), 非O(N·L·D) |

### 数据IO优化

| 优化 | 效果 |
|------|------|
| Dataset预解析行 | 避免iloc随机访问, 加速3-5x |
| persistent_workers + prefetch | worker跨epoch存活 + 预取4个batch |
| GPU端累积评估 | 最后一次性转CPU, 减少CUDA同步 |
| 时间切分验证集 | 同文件自动90/10按时间切, 零数据泄漏 |
| Vocab动态扫描+pickle | 基数按实际数据设, 推理复用训练词表 |

### GPU显存优化

| 消耗项 | 未优化 | 优化后 | 手段 |
|--------|--------|--------|------|
| 模型权重 | 332MB | 166MB | BF16 |
| 激活值 | ~2GB | ~600MB | Gradient Checkpointing |
| 优化器状态 | 664MB | 400MB | Muon单buffer |
| Attention mask | 1.8MB/层×N | 1.8MB (缓存) | Mask Cache |
| BlockAttnRes | O(N·L·D) | O(N·D) | Last-token summary |

---

## 数据格式

| 字段 | 类型 | 描述 |
|------|------|------|
| `user_id` | str | 用户标识 |
| `item_id` | int64 | 物品/广告标识 |
| `timestamp` | int64 | 曝光时间戳 (Unix) |
| `user_feature` | List[Dict] | 57个用户特征 (int_value/int_array/float_array/mixed) |
| `item_feature` | List[Dict] | 16个物品特征 (int_value/int_array/float_value) |
| `seq_feature` | Dict | 3条异构序列: action_seq(fid 19-28), content_seq(fid 40-48), item_seq(fid 29-39,49) |
| `label` | List[Dict] | action_type: 1=点击, 2=转化; action_time: 行为时间戳 |

---

## Scaling 配置

| 规模 | D | Heads | Full+Trunc层 | feat_d | 参数量 | 适用场景 |
|------|---|-------|-------------|--------|--------|---------|
| XS | 64 | 2 | 1+1 | 16 | ~40M | Demo验证 |
| S | 128 | 4 | 2+2 | 32 | ~83M | 第一轮 (100万) |
| M | 256 | 8 | 3+5 | 48 | ~138M | 第二轮 (1000万) |
| L | 512 | 16 | 4+8 | 64 | ~263M | 第二轮 + 2卡DDP |

---

## 项目结构

```
TAAC2026/
├── README.md                  # 本文档
├── EDA_FINDINGS.md            # EDA分析结果与比赛策略
├── config.py                  # 配置 + SCALING_CONFIGS (XS/S/M/L)
├── data/
│   ├── dataset.py             # 数据集 + 时间切分 + 预解析行 + DataLoader优化
│   └── feature_processor.py   # 特征解析 + 动态词表 + pickle序列化
├── model/
│   ├── unified_model.py       # UniRec完整架构 (BlockAttnRes + 全部Stage)
│   ├── modules.py             # SiLU/Softmax Attention + Mask缓存 + SDPA
│   ├── tricks.py              # RoPE + SessionEncoder + EnhancedBlock
│   ├── mot.py                 # MoT (3分支 + Gate/Attention/Concat融合)
│   ├── heads.py               # CombinedAUCLoss + FocalLoss + WeightedBCE
│   └── optimizer.py           # Muon (Newton-Schulz) + AdamW混合
├── train.py                   # 训练 (单卡/DDP + BF16 + warmup+cosine)
├── evaluate.py                # AUC评估 (GPU端累积)
├── inference.py               # 推理 (compile + batch翻倍)
├── scaling_study.py           # Scaling Law实验 (XS/S/M/L)
├── configs_round1.py          # 第一轮配置 (100万样本)
├── configs_round2.py          # 第二轮配置 (1000万样本)
├── eda_analysis.py            # EDA分析 (21项分析, 12张图)
└── run.sh                     # 启动脚本
```

---

## 快速开始

```bash
# Demo训练 (1000条, 自动按时间90/10切分)
uv run python train.py

# 2卡DDP训练
torchrun --nproc_per_node=2 train.py

# Scaling Law实验
python scaling_study.py --data sample_data.parquet --sizes XS S M L

# 推理
python inference.py
```

---

## 参考文献

1. OneTrans: Unified Feature Interaction and Sequence Modeling (ByteDance, WWW 2026)
2. InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction (Meta, 2025)
3. HyFormer: Revisiting Sequence Modeling and Feature Interaction in CTR Prediction (ByteDance, 2026)
4. HSTU 2.0: Bending the Scaling Law Curve in Large-Scale Recommendation (Meta, 2026)
5. Attention Residuals (Kimi Team, 2026)
6. DIN: Deep Interest Network for Click-Through Rate Prediction (Alibaba, 2018)
7. GR4AD: Generative Recommendation for Large-Scale Advertising (Kuaishou, 2026)
