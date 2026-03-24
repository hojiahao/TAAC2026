"""
Round 1 config: 1,000,000 samples, T4 16GB GPU.
Conservative model, fast iteration, validate ideas.
"""
from config import Config


def get_round1_config() -> Config:
    config = Config()

    # ── Model: medium size, fit in 16GB with grad checkpoint ──
    config.model.embedding_dim = 128
    config.model.num_heads = 4
    config.model.num_layers = 6          # 总层数
    config.model.dropout = 0.1
    config.model.attention_type = "silu"

    config.model.use_semi_local_attention = True
    config.model.global_window_size = 64
    config.model.local_window_size = 128

    config.model.use_attention_truncation = True
    config.model.full_seq_layers = 2     # 前2层处理全序列
    config.model.truncated_seq_layers = 4  # 后4层只处理最近128token

    config.model.feature_embedding_dim = 32
    config.model.use_mot = True
    config.model.mot_layers = 2
    config.model.mot_merge = "gate"

    config.model.head_hidden_dim = 256
    config.model.use_tricks = True       # RoPE + Session

    # ── Sequence lengths ──
    config.train.max_action_seq_len = 150
    config.train.max_content_seq_len = 150
    config.train.max_item_seq_len = 150

    # ── Training ──
    config.train.epochs = 3
    config.train.batch_size = 256
    config.train.learning_rate = 1e-3
    config.train.weight_decay = 1e-5
    config.train.warmup_steps = 500      # ~12000总步, warmup ~4%
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
    config.train.num_workers = 4
    config.train.prefetch_factor = 4

    # ── Logging ──
    config.train.log_interval = 200
    config.train.eval_interval = 1000
    config.train.save_dir = "checkpoints_round1"

    return config


# Estimated:
#   params: ~90M
#   GPU memory: ~6-8GB (with grad checkpoint)
#   steps/epoch: 3906 (1M / 256)
#   total steps: 11718 (3 epochs)
#   training time: ~2-3 hours on T4
