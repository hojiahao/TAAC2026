"""
TAAC 2026 Inference Script.
Loads trained model, predicts CVR probabilities, outputs submission file.
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
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference, return CVR probabilities."""
    model.eval()
    all_preds = []
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch)
        probs = torch.sigmoid(logits.squeeze(-1))
        all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds)


def run_inference(config: Config):
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab_path = os.path.join(config.train.save_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load model
    ckpt_path = os.path.join(config.train.save_dir, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab_sizes = vocab.get_vocab_sizes()

    model = UnifiedRecModel(config, vocab_sizes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ckpt_path} (AUC={checkpoint['best_auc']:.4f})")

    # Inference optimization: torch.compile for speed
    if config.train.compile_model and hasattr(torch, "compile"):
        print("Compiling model for inference...")
        model = torch.compile(model, mode="max-autotune")

    # Load test data
    test_ds = TAACDataset(
        config.train.test_data_path, config,
        vocab_builder=vocab, is_test=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.train.num_workers,
    )

    # Predict
    print("Running inference...")
    preds = predict(model, test_loader, device)

    # Save submission
    test_df = pd.read_parquet(config.train.test_data_path)
    submission = pd.DataFrame({
        "user_id": test_df["user_id"],
        "item_id": test_df["item_id"],
        "pcvr": preds,
    })
    output_path = os.path.join(config.train.save_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")


if __name__ == "__main__":
    config = Config()
    run_inference(config)
