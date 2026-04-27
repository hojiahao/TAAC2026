"""
Scaling Law Study for TAAC 2026.

Trains the SAME unified architecture at 4 scales (XS/S/M/L),
plots AUC vs FLOPs to verify recommendation scaling law.

Usage:
    python scaling_study.py --data data/train.parquet
    python scaling_study.py --data sample_data.parquet --epochs 3
"""

import argparse
import json
import os
import time
import torch

from config import Config, SCALING_CONFIGS, get_scaling_config
from data.dataset import build_dataloaders
from model.unified_model import UnifiedRecModel
from model.heads import build_loss
from model.optimizer import build_optimizer
from train import get_lr_scale, set_lr, compute_loss
from evaluate import evaluate


def estimate_flops(config, seq_len=675):
    """Rough FLOPs estimate per sample (forward pass)."""
    D = config.model.embedding_dim
    H = config.model.num_heads
    L = seq_len
    n_layers = config.model.num_layers

    # Attention: 4*L*D^2 (QKV proj) + 2*L^2*D (attn) per layer
    attn_flops = n_layers * (4 * L * D * D + 2 * L * L * D)
    # FFN: 3 * L * D * 4D per layer (SwiGLU)
    ffn_flops = n_layers * 3 * L * D * 4 * D
    # Total per sample
    return attn_flops + ffn_flops


def train_one_scale(size: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_scaling_config(size)
    config.train.train_data_path = args.data
    config.train.val_data_path = args.data
    config.train.batch_size = args.batch_size
    config.train.epochs = args.epochs
    config.train.warmup_steps = args.warmup
    config.train.compile_model = False
    config.train.num_workers = 0 if args.data == "sample_data.parquet" else 4
    config.train.save_dir = f"checkpoints_scaling/{size}"

    print(f"\n{'='*60}")
    print(f"  Scaling Study: size={size}")
    print(f"  D={config.model.embedding_dim}, H={config.model.num_heads}, "
          f"L={config.model.num_layers}, feat_d={config.model.feature_embedding_dim}")
    print(f"{'='*60}")

    train_loader, val_loader, vocab = build_dataloaders(config)
    model = UnifiedRecModel(config, vocab.get_vocab_sizes()).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = estimate_flops(config)

    print(f"  Params: {n_params:,}")
    print(f"  FLOPs/sample: {flops:,.0f}")

    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * config.train.epochs
    global_step = 0
    best_auc = 0.0

    model.train()
    t0 = time.time()
    for epoch in range(config.train.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            loss = compute_loss(logits, batch, criterion)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            set_lr(optimizer, get_lr_scale(global_step, total_steps, config.train.warmup_steps))

    train_time = time.time() - t0

    # Final evaluation
    metrics = evaluate(model, val_loader, device)
    best_auc = metrics["auc"]
    print(f"  → AUC={best_auc:.4f}  time={train_time:.1f}s")

    return {
        "size": size,
        "params": n_params,
        "flops_per_sample": flops,
        "auc": best_auc,
        "train_time": train_time,
        "config": {
            "embedding_dim": config.model.embedding_dim,
            "num_heads": config.model.num_heads,
            "num_layers": config.model.num_layers,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sample_data.parquet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--sizes", nargs="+", default=["XS", "S", "M", "L"])
    args = parser.parse_args()

    results = []
    for size in args.sizes:
        result = train_one_scale(size, args)
        results.append(result)

    # Save results
    os.makedirs("checkpoints_scaling", exist_ok=True)
    with open("checkpoints_scaling/scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("SCALING LAW RESULTS")
    print("=" * 70)
    print(f"{'Size':<6} {'Params':>12} {'FLOPs':>15} {'AUC':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']:<6} {r['params']:>12,} {r['flops_per_sample']:>15,.0f} "
              f"{r['auc']:>8.4f} {r['train_time']:>7.1f}s")

    print("\nResults saved to checkpoints_scaling/scaling_results.json")
    print("Plot with: python plot_scaling.py")


if __name__ == "__main__":
    main()
