# UniRec：面向 TAAC 2026 的统一推荐模型

**KDD Cup 2026 - Tencent UNI-REC Challenge**

> 面向大规模推荐的序列建模与特征交互统一化

---

## 概述

UniRec 是一个将**序列建模**与**特征交互**统一到单一同构可堆叠骨干网络中的推荐架构，用于 **CVR 预测**。它融合了 6 项研究成果的创新：

| 来源 | 融合的创新 |
|------|-----------|
| OneTrans (WWW 2026) | Auto-Split Tokenizer + 混合参数化 (S共享/NS独立) |
| InterFormer (Meta 2025) | Feature Cross Layer (显式特征交互) |
| HSTU 2.0 (Meta 2026) | SiLU Gated Attention + Semi-Local Attention + Attention Truncation + MoT |
| DIN (阿里 2018) | Target-Aware Interest Extraction (目标item查询用户历史) |
| Kimi (2026) | Block Attention Residuals (深度维度的选择性信息检索) |
| 本项目 | Hybrid Attention Mask (NS全双向 + S区SLA + target全可见) |

---

## 架构设计

```
输入: user_features (57字段) + item_features (16字段)
      + action_seq (200步) + content_seq (200步) + item_seq (200步)

Stage 1: 统一分词 [OneTrans]
  每个特征字段 → embed(feat_d) → per-feature proj(D) = 73个特征token
  每种序列每步 → concat所有特征 → proj(D) = 600个序列token

Stage 2: 骨干前预处理
  2a. Feature Cross Layer [InterFormer]: 73个NS-token做self-attention
  2b. Target-Aware Interest [DIN]: item_pool查询600个S-token
  2c. Target Fusion [HSTU]: MLP([user_pool; item_pool; u*i])
  2d. MoT [HSTU 2.0]: 3条序列独立Transducer → 门控融合

Stage 3: 统一序列组装
  [73 feat | 200 action | 200 content | 200 item | mot | interest | target]
  总计 ~677 tokens，全部D维

Stage 4: 同构可堆叠骨干 (Unified Block × N)
  每个 Unified Block:
    ① RMSNorm (Pre-Norm)                               [OneTrans]
    ② Mixed SiLU Gated Attention:
       S-tokens: 共享 Q/K/V/U                           [OneTrans]
       NS-tokens: 独立 Q/K/V/U                          [OneTrans]
       注意力: A = SiLU(Q·K^T/sqrt(d)) · Mask           [HSTU 2.0]
       掩码: NS↔NS全双向 + S区SLA + target全可见         [本项目]
    ③ Block Attention Residuals                          [Kimi]
    ④ RMSNorm
    ⑤ Mixed SwiGLU FFN (S共享/NS独立)                    [OneTrans]
    ⑥ Block Attention Residuals

  全序列层 × N1: 处理全部677 tokens                      [HSTU 2.0]
  截断层 × N2: 只处理最近L'个S-tokens                     [HSTU 2.0]

Stage 5: CVR 预测头
  target输出 → MLP → sigmoid → P(conversion)
  损失: CombinedAUCLoss = 0.5·BCE + 0.5·PairwiseBPR
```

---

## 创新点

| 创新点 | 描述 | 对应奖项 |
|--------|------|---------|
| Hybrid Attention Mask | 首次在统一架构中设计NS区全双向+S区SLA+target全可见的混合掩码 | 统一模块创新奖 |
| Block AttnRes in RecSys | 首次将注意力残差应用于推荐系统的统一骨干 | 统一模块创新奖 |
| SiLU + 混合参数化 | 将HSTU 2.0的SiLU gating与OneTrans的混合参数化结合 | 统一模块创新奖 |
| 骨干前DIN | 在统一骨干之前注入target-aware信号 | AUC排行榜 |
| Scaling Law验证 | XS/S/M/L四规模验证AUC vs FLOPs幂律关系 | 扩展规律创新奖 |

---

## 数据格式

| 字段 | 类型 | 描述 |
|------|------|------|
| `user_id` | str | 用户标识 |
| `item_id` | int64 | 物品/广告标识 |
| `timestamp` | int64 | 曝光时间戳 (Unix) |
| `user_feature` | List[Dict] | 57个用户特征 (int_value/int_array/float_array/mixed) |
| `item_feature` | List[Dict] | 16个物品特征 (int_value/int_array/float_value) |
| `seq_feature` | Dict | 3个子序列: action_seq(10特征), content_seq(9特征), item_seq(12特征) |
| `label` | List[Dict] | action_type: 1=点击, 2=转化; action_time: 行为时间戳 |

---

## 项目结构

```
TAAC2026/
├── README.md
├── config.py                  # 配置 + SCALING_CONFIGS (XS/S/M/L)
├── data/
│   ├── dataset.py             # 数据集 + 按时间切分 + DataLoader
│   └── feature_processor.py   # 特征解析 + 词表构建
├── model/
│   ├── unified_model.py       # UniRec: 完整架构 (含BlockAttnRes)
│   ├── modules.py             # SiLU Attention + SLA + Hybrid Mask
│   ├── tricks.py              # RoPE + Session Encoder + Enhanced Block
│   ├── mot.py                 # Mixture of Transducers
│   ├── heads.py               # CombinedAUCLoss (BCE + PairwiseBPR)
│   └── optimizer.py           # Muon + AdamW 混合优化器
├── train.py                   # 训练 (单卡 / 2卡DDP)
├── evaluate.py                # AUC of ROC 评估
├── inference.py               # 推理 + 提交文件生成
├── scaling_study.py           # 多规模Scaling Law实验
├── configs_round1.py          # 第一轮配置 (100万样本)
├── configs_round2.py          # 第二轮配置 (1000万样本)
├── eda_analysis.py            # 数据探索分析
├── run.sh                     # 启动脚本
├── papers/                    # 论文阅读笔记
└── sample_data.parquet        # 样本数据 (1000条)
```

---

## 快速开始

```bash
# Demo训练 (1000条样本，自动按时间90/10切分)
uv run python train.py

# 2卡DDP训练
torchrun --nproc_per_node=2 train.py

# Scaling Law实验
python scaling_study.py --data sample_data.parquet --sizes XS S M L

# 推理
python inference.py
```

---

## Scaling 配置

| 规模 | D | Heads | 层数 | 参数量 | 适用场景 |
|------|---|-------|------|--------|---------|
| XS | 64 | 2 | 2 | ~30M | Demo验证 |
| S | 128 | 4 | 4 | ~60M | 第一轮 (100万) |
| M | 256 | 8 | 8 | ~100M | 第二轮 (1000万) |
| L | 512 | 16 | 12 | ~210M | 第二轮 + 2卡DDP |

---

## 参考文献

1. InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction (Meta, 2025)
2. HyFormer: Revisiting Sequence Modeling and Feature Interaction in CTR Prediction (ByteDance, 2026)
3. OneTrans: Unified Feature Interaction and Sequence Modeling (ByteDance, WWW 2026)
4. HSTU 2.0: Bending the Scaling Law Curve in Large-Scale Recommendation (Meta, 2026)
5. Attention Residuals (Kimi Team, 2026)
6. DIN: Deep Interest Network for Click-Through Rate Prediction (Alibaba, 2018)
