"""
Round 2 config: 10,000,000 samples, A100/V100 32GB+ GPU.
Larger model, push scaling law, maximize AUC.
"""
from config import Config


def get_round2_config() -> Config:
    config = Config()

    # ── Model: large, scaling law exploration ──
    config.model.embedding_dim = 256
    config.model.num_heads = 8
    config.model.num_layers = 9          # 总层数
    config.model.dropout = 0.05          # 数据量大→dropout可以小
    config.model.attention_type = "silu"

    config.model.use_semi_local_attention = True
    config.model.global_window_size = 128
    config.model.local_window_size = 256

    config.model.use_attention_truncation = True
    config.model.full_seq_layers = 3     # HSTU 2.0论文推荐配比 3+6
    config.model.truncated_seq_layers = 6
    config.model.truncated_seq_len = 200

    config.model.feature_embedding_dim = 64  # 更大的特征embedding
    config.model.use_mot = True
    config.model.mot_layers = 3          # MoT分支也加深
    config.model.mot_merge = "gate"

    config.model.head_hidden_dim = 512
    config.model.use_tricks = True

    # ── Sequence lengths: longer for more history ──
    config.train.max_action_seq_len = 200
    config.train.max_content_seq_len = 300
    config.train.max_item_seq_len = 200

    # ── Training ──
    config.train.epochs = 2              # 数据量大, 2 epoch够
    config.train.batch_size = 512
    config.train.learning_rate = 8e-4    # 大batch→稍降lr
    config.train.weight_decay = 1e-5
    config.train.warmup_steps = 2000     # ~39000总步, warmup ~5%
    config.train.max_grad_norm = 1.0

    config.train.optimizer = "muon"
    config.train.loss_type = "auc"
    config.train.conversion_weight = 3.0
    config.train.auc_loss_weight = 0.5
    config.train.auc_margin = 0.5

    # ── Efficiency ──
    config.train.bf16 = True
    config.train.fp16 = False
    config.train.gradient_checkpointing = True
    config.train.compile_model = True
    config.train.num_workers = 8         # 数据量大→更多worker
    config.train.prefetch_factor = 4
    config.train.gradient_accumulation_steps = 2  # 等效batch=1024

    # ── Logging ──
    config.train.log_interval = 500
    config.train.eval_interval = 2000
    config.train.save_dir = "checkpoints_round2"

    return config


# Estimated:
#   params: ~350M
#   GPU memory: ~20-24GB (with grad checkpoint)
#   steps/epoch: 19531 (10M / 512)
#   total steps: 39062 (2 epochs)
#   training time: ~8-12 hours on A100
