"""
TAAC 2026 Inference - optimized for Angel ML Platform.
Key: torch.compile + GPU accumulation + efficient DataLoader.
"""

import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TAACDataset
from model.unified_model import UnifiedRecModel


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch)
        all_preds.append(torch.sigmoid(logits.squeeze(-1)))
    return torch.cat(all_preds).cpu().numpy()


def run_inference(config: Config):
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

    vocab_path = os.path.join(config.train.save_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    ckpt_path = os.path.join(config.train.save_dir, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab_sizes = vocab.get_vocab_sizes()

    model = UnifiedRecModel(config, vocab_sizes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded (AUC={checkpoint['best_auc']:.4f})")

    # torch.compile for inference speedup
    if config.train.compile_model and hasattr(torch, "compile"):
        print("Compiling for inference...")
        model = torch.compile(model, mode="max-autotune")

    test_ds = TAACDataset(
        config.train.test_data_path, config,
        vocab_builder=vocab, is_test=True,
    )
    nw = config.train.num_workers
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size * 2,  # inference可以用更大batch
        shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=config.train.prefetch_factor if nw > 0 else None,
    )

    print("Running inference...")
    preds = predict(model, test_loader, device)

    test_df = pd.read_parquet(config.train.test_data_path)
    submission = pd.DataFrame({
        "user_id": test_df["user_id"],
        "item_id": test_df["item_id"],
        "pcvr": preds,
    })
    output_path = os.path.join(config.train.save_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission: {output_path} ({len(submission)} rows)")


if __name__ == "__main__":
    config = Config()
    run_inference(config)
