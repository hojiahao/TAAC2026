"""
TAAC 2026 Training Script.
Task: CVR Prediction | Metric: AUC of ROC
Model: Unified Sequence + Feature Interaction (HSTU-style backbone)
Supports: single GPU / 2-GPU DDP
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import random
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from config import Config
from data.dataset import TAACDataset, build_dataloaders
from model.unified_model import UnifiedRecModel
from model.heads import build_loss
from model.optimizer import build_optimizer as build_optimizer_v2
from evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: Config):
    """Route to Muon or AdamW based on config."""
    return build_optimizer_v2(model, config)


import math


def get_lr_scale(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.01) -> float:
    """
    Returns a multiplier in [min_lr_ratio, 1.0].
    Warmup: linear 0→1 over warmup_steps.
    Then cosine decay: 1→min_lr_ratio over remaining steps.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, scale: float):
    """
    Scale each param_group's lr by multiplier, relative to its own initial lr.
    Muon(lr=0.02) and AdamW(lr=1e-3) each get scaled proportionally.
    """
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def compute_loss(logits, batch, criterion):
    """CVR prediction loss. Single task, single output."""
    return criterion(logits, batch["label"], batch.get("action_type"))


def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank):
    return rank == 0


def train(config: Config):
    rank, local_rank, world_size = setup_ddp()
    set_seed(config.train.seed + rank)
    use_ddp = world_size > 1

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device(config.train.device)
    else:
        device = torch.device("cpu")

    # ── Data ──
    if is_main_process(rank):
        print("=" * 60)
        print("TAAC 2026 - Unified Rec Model Training")
        mode = "Enhanced (RoPE+Session)" if config.model.use_tricks else "Baseline"
        print(f"  Mode: {mode}")
        print(f"  Optimizer: {config.train.optimizer}")
        print(f"  GPUs: {world_size}")
        print("=" * 60)

    train_loader, val_loader, vocab = build_dataloaders(config)
    vocab_sizes = vocab.get_vocab_sizes()

    # DDP: replace DataLoader sampler
    if use_ddp:
        from data.dataset import TAACDataset
        from torch.utils.data import DataLoader
        train_ds = train_loader.dataset
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        nw = config.train.num_workers
        train_loader = DataLoader(
            train_ds, batch_size=config.train.batch_size, sampler=train_sampler,
            num_workers=nw, pin_memory=True, drop_last=True,
            persistent_workers=nw > 0, prefetch_factor=config.train.prefetch_factor if nw > 0 else None,
        )

    # ── Model ──
    model = UnifiedRecModel(config, vocab_sizes).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    if config.train.gradient_checkpointing:
        print("  Gradient checkpointing: ON (saves ~60% memory)")
    if config.train.compile_model and hasattr(torch, "compile"):
        print("  torch.compile: ON (compiling...)")
        model = torch.compile(model, mode="reduce-overhead")

    # ── Loss, Optimizer ──
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * config.train.epochs

    # BF16 > FP16 for training stability (no loss scaling needed)
    use_amp = config.train.fp16 or config.train.bf16
    amp_dtype = torch.bfloat16 if config.train.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=config.train.fp16 and not config.train.bf16)

    # ── Training Loop ──
    os.makedirs(config.train.save_dir, exist_ok=True)
    best_auc = 0.0
    global_step = 0
    log_history = []

    for epoch in range(config.train.epochs):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(batch)
                loss = compute_loss(logits, batch, criterion)
                loss = loss / config.train.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.train.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                # Warmup + cosine decay, proportional to each group's initial_lr
                scale = get_lr_scale(global_step, total_steps, config.train.warmup_steps)
                set_lr(optimizer, scale)

            epoch_loss += loss.item() * config.train.gradient_accumulation_steps

            # Logging (rank 0 only)
            if is_main_process(rank) and global_step > 0 and global_step % config.train.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [Epoch {epoch+1} | Step {global_step}] "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}")

            # Evaluation (rank 0 only)
            if is_main_process(rank) and global_step > 0 and global_step % config.train.eval_interval == 0:
                eval_model = model.module if use_ddp else model
                metrics = evaluate(eval_model, val_loader, device)
                print(f"  >>> Eval @ step {global_step}: "
                      f"AUC={metrics['auc']:.4f}  "
                      f"LogLoss={metrics['logloss']:.4f}  "
                      f"PosRate={metrics['pos_rate']:.3f}")

                log_entry = {"step": global_step, "epoch": epoch + 1, **metrics}
                log_history.append(log_entry)

                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    ckpt_path = os.path.join(config.train.save_dir, "best_model.pt")
                    state = eval_model.state_dict()
                    torch.save({
                        "model_state_dict": state,
                        "config": config,
                        "best_auc": best_auc,
                        "global_step": global_step,
                    }, ckpt_path)
                    print(f"  >>> New best AUC: {best_auc:.4f} (saved to {ckpt_path})")

                model.train()

        # End of epoch
        if is_main_process(rank):
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config.train.epochs} done. "
                  f"avg_loss={avg_loss:.4f}  time={epoch_time:.1f}s")

            eval_model = model.module if use_ddp else model
            metrics = evaluate(eval_model, val_loader, device)
            print(f"  Epoch {epoch+1} Eval: AUC={metrics['auc']:.4f}  "
                  f"LogLoss={metrics['logloss']:.4f}")
            log_history.append({"step": global_step, "epoch": epoch + 1, **metrics})

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                ckpt_path = os.path.join(config.train.save_dir, "best_model.pt")
                state = eval_model.state_dict()
                torch.save({
                    "model_state_dict": state,
                    "config": config,
                    "best_auc": best_auc,
                    "global_step": global_step,
                }, ckpt_path)
                print(f"  >>> New best AUC: {best_auc:.4f}")

    # ── Save training log ──
    if is_main_process(rank):
        log_path = os.path.join(config.train.save_dir, "train_log.json")
        with open(log_path, "w") as f:
            json.dump(log_history, f, indent=2)
        print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
        print(f"Logs saved to {log_path}")

    if use_ddp:
        dist.destroy_process_group()

    return best_auc


if __name__ == "__main__":
    config = Config()
    # Override for sample data smoke test
    config.train.train_data_path = "sample_data.parquet"
    config.train.val_data_path = "sample_data.parquet"
    config.train.batch_size = 32
    config.train.epochs = 3
    config.train.warmup_steps = 10         # demo数据只有93步，warmup别超过总步数的10%
    config.train.log_interval = 10
    config.train.eval_interval = 20
    train(config)
