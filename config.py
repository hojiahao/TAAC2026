"""
TAAC 2026 - Tencent UNI-REC Challenge (KDD Cup 2026)
UniRec: Unified Sequence Modeling + Feature Interaction for CVR Prediction
Metric: AUC of ROC
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    # ── User Features (57 total) ──
    user_sparse_feature_ids: List[int] = field(default_factory=lambda: [
        1, 3, 4, 50, 51, 52, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 76, 80, 82,
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    ])
    user_array_feature_ids: List[int] = field(default_factory=lambda: [5, 18, 53, 54, 67, 74])
    user_dense_feature_ids: List[int] = field(default_factory=lambda: [68, 81])
    user_dense_dims: List[int] = field(default_factory=lambda: [256, 320])
    user_mixed_feature_ids: List[int] = field(default_factory=lambda: [69, 70, 71, 72, 73, 83, 84, 85])

    # ── Item Features (16 total) ──
    item_sparse_feature_ids: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 75, 77, 78, 79])
    item_array_feature_ids: List[int] = field(default_factory=lambda: [14])
    item_float_feature_ids: List[int] = field(default_factory=lambda: [17])

    # ── Sequence Feature IDs ──
    action_seq_feature_ids: List[int] = field(default_factory=lambda: [19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    content_seq_feature_ids: List[int] = field(default_factory=lambda: [40, 41, 42, 43, 44, 45, 46, 47, 48])
    item_seq_feature_ids: List[int] = field(default_factory=lambda: [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49])

    default_vocab_size: int = 100000
    max_array_len: int = 16

    # 时间戳特征 (不做embedding, 用连续编码, EDA验证: fid=28,29,41)
    timestamp_feature_ids: List[int] = field(default_factory=lambda: [28, 29, 41])
    # 高基数特征的hash bucket上限 (防止vocab爆炸, 正式数据可能有百万级unique值)
    max_hash_bucket: int = 200000


@dataclass
class ModelConfig:
    # ── Unified Backbone ──
    embedding_dim: int = 128            # D: token dimension
    num_heads: int = 4                  # H: attention heads (head_dim = D/H = 32)
    feature_embedding_dim: int = 32     # feat_d: per-feature embedding dim before proj
    dropout: float = 0.1
    attention_type: str = "silu"        # "silu" (HSTU-style) or "softmax"

    # ── Semi-Local Attention [HSTU 2.0] ──
    use_semi_local_attention: bool = True
    global_window_size: int = 128       # K2: must >= n_feature_tokens (73)
    local_window_size: int = 128        # K1: local window for S-tokens

    # ── Attention Truncation [HSTU 2.0] ──
    use_attention_truncation: bool = True
    full_seq_layers: int = 2            # N1: layers processing full sequence
    truncated_seq_layers: int = 2       # N2: layers processing only recent L' S-tokens
    truncated_seq_len: int = 200        # L': how many recent S-tokens to keep in truncated layers

    # ── MoT [HSTU 2.0] ──
    use_mot: bool = True
    mot_layers: int = 2
    mot_merge: str = "gate"             # "concat", "gate", "attention"

    # ── Block Attention Residuals [Kimi] ──
    attn_res_block_size: int = 2        # layers per block (total_layers / block_size = num blocks)

    # ── Prediction Head ──
    head_hidden_dim: int = 256

    # ── Tricks Toggle ──
    use_tricks: bool = False            # True: RoPE in Enhanced Block (session encoder disabled)


@dataclass
class TrainConfig:
    # ── Data ──
    train_data_path: str = "data/train.parquet"
    val_data_path: str = "data/val.parquet"
    test_data_path: str = "data/test.parquet"
    max_action_seq_len: int = 200
    max_content_seq_len: int = 200
    max_item_seq_len: int = 200

    # ── Training ──
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # ── Optimizer ──
    optimizer: str = "muon"             # "muon": Muon(Linear)+AdamW(Embed/bias), "adamw": pure AdamW
    adam_betas: tuple = (0.9, 0.98)

    # ── Loss ──
    loss_type: str = "auc"              # "auc" (BCE+PairwiseBPR) / "bce" / "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    conversion_weight: float = 3.0
    auc_loss_weight: float = 0.5
    auc_margin: float = 0.5

    # ── System & Efficiency ──
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    compile_model: bool = True
    prefetch_factor: int = 4

    # ── Logging ──
    log_interval: int = 100
    eval_interval: int = 1000
    save_dir: str = "checkpoints"
    experiment_name: str = "unirec_v1"


@dataclass
class Config:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ═══════════════════════════════════════════════════
# Scaling Law Configs
# ═══════════════════════════════════════════════════

SCALING_CONFIGS = {
    "XS": {
        "embedding_dim": 64, "num_heads": 2, "feature_embedding_dim": 16,
        "full_seq_layers": 1, "truncated_seq_layers": 1,
        "head_hidden_dim": 64, "mot_layers": 1,
        "global_window_size": 128, "attn_res_block_size": 1,
    },
    "S": {
        "embedding_dim": 128, "num_heads": 4, "feature_embedding_dim": 32,
        "full_seq_layers": 2, "truncated_seq_layers": 2,
        "head_hidden_dim": 128, "mot_layers": 2,
        "global_window_size": 128, "attn_res_block_size": 2,
    },
    "M": {
        "embedding_dim": 256, "num_heads": 8, "feature_embedding_dim": 48,
        "full_seq_layers": 3, "truncated_seq_layers": 5,
        "head_hidden_dim": 256, "mot_layers": 2,
        "global_window_size": 128, "attn_res_block_size": 2,
    },
    "L": {
        "embedding_dim": 512, "num_heads": 16, "feature_embedding_dim": 64,
        "full_seq_layers": 4, "truncated_seq_layers": 8,
        "head_hidden_dim": 512, "mot_layers": 3,
        "global_window_size": 128, "attn_res_block_size": 3,
    },
}


def get_scaling_config(size: str, base_config: Config = None) -> Config:
    config = base_config or Config()
    for key, value in SCALING_CONFIGS[size].items():
        setattr(config.model, key, value)
    return config
