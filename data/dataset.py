"""
TAAC 2026 Dataset: Unified tokenization for sequence + feature interaction.

Each sample produces a unified token sequence:
    [user_sparse_tokens | item_sparse_tokens | action_seq | content_seq | item_seq]
All projected to the same embedding space by the model.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional

from data.feature_processor import (
    FeatureVocabBuilder,
    build_vocab_from_dataframe,
    parse_sparse_features,
    parse_array_features,
    parse_dense_features,
    parse_float_value_features,
    parse_seq_features,
)
from config import Config


class TAACDataset(Dataset):
    def __init__(
        self,
        parquet_path_or_df,
        config: Config,
        vocab_builder: Optional[FeatureVocabBuilder] = None,
        is_test: bool = False,
    ):
        if isinstance(parquet_path_or_df, pd.DataFrame):
            self.df = parquet_path_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_parquet(parquet_path_or_df)
        self.config = config
        self.is_test = is_test
        self.fc = config.feature
        self.tc = config.train

        if vocab_builder is None:
            self.vocab = build_vocab_from_dataframe(self.df)
        else:
            self.vocab = vocab_builder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── 1. User features (4 types) ──
        user_sparse = parse_sparse_features(
            row["user_feature"], self.fc.user_sparse_feature_ids, self.vocab,
        )
        user_array = parse_array_features(
            row["user_feature"], self.fc.user_array_feature_ids, self.vocab, self.fc.max_array_len,
        )
        user_dense = parse_dense_features(
            row["user_feature"], self.fc.user_dense_feature_ids, self.fc.user_dense_dims,
        )
        # mixed features: treat int_array part as array features
        user_mixed = parse_array_features(
            row["user_feature"], self.fc.user_mixed_feature_ids, self.vocab, self.fc.max_array_len,
        )

        # ── 2. Item features (3 types) ──
        item_sparse = parse_sparse_features(
            row["item_feature"], self.fc.item_sparse_feature_ids, self.vocab,
        )
        item_array = parse_array_features(
            row["item_feature"], self.fc.item_array_feature_ids, self.vocab, self.fc.max_array_len,
        )
        item_float = parse_float_value_features(
            row["item_feature"], self.fc.item_float_feature_ids,
        )

        # ── 3. Sequence features → (max_len, num_feats) each ──
        seq = row["seq_feature"]

        action_seq, action_len = parse_seq_features(
            seq["action_seq"],
            self.fc.action_seq_feature_ids,
            self.vocab,
            self.tc.max_action_seq_len,
        )
        content_seq, content_len = parse_seq_features(
            seq["content_seq"],
            self.fc.content_seq_feature_ids,
            self.vocab,
            self.tc.max_content_seq_len,
        )
        item_seq, item_len = parse_seq_features(
            seq["item_seq"],
            self.fc.item_seq_feature_ids,
            self.vocab,
            self.tc.max_item_seq_len,
        )

        # ── 4. Timestamp ──
        timestamp = np.array(row["timestamp"], dtype=np.int64)

        sample = {
            # User features (4 types)
            "user_sparse": torch.from_numpy(user_sparse),          # (n_us,)
            "user_array": torch.from_numpy(user_array),            # (n_ua, max_arr_len)
            "user_dense": torch.from_numpy(user_dense),            # (sum_dense_dims,)
            "user_mixed": torch.from_numpy(user_mixed),            # (n_um, max_arr_len)
            # Item features (3 types)
            "item_sparse": torch.from_numpy(item_sparse),          # (n_is,)
            "item_array": torch.from_numpy(item_array),            # (n_ia, max_arr_len)
            "item_float": torch.from_numpy(item_float),            # (n_if_float,)
            # Sequences
            "action_seq": torch.from_numpy(action_seq),            # (L_a, n_af)
            "action_seq_len": torch.tensor(action_len),
            "content_seq": torch.from_numpy(content_seq),          # (L_c, n_cf)
            "content_seq_len": torch.tensor(content_len),
            "item_seq": torch.from_numpy(item_seq),                # (L_s, n_sf)
            "item_seq_len": torch.tensor(item_len),
            # Meta
            "timestamp": torch.tensor(timestamp),
        }

        # ── 5. Label ──
        if not self.is_test:
            label_info = row["label"][0]
            action_type = label_info["action_type"]
            # CVR: conversion = 1, click-only = 0
            cvr_label = 1.0 if action_type == 2 else 0.0
            sample["label"] = torch.tensor(cvr_label, dtype=torch.float32)
            sample["action_type"] = torch.tensor(action_type, dtype=torch.long)

        return sample


def build_dataloaders(config: Config):
    """
    Build train/val dataloaders with shared vocabulary.

    Data splitting strategy:
      - If train_path == val_path: auto-split by timestamp (90% train / 10% val)
      - If different files: load each separately
    Time-based split avoids data leakage (train on past, validate on future).
    """
    train_path = config.train.train_data_path
    val_path = config.train.val_data_path

    if train_path == val_path:
        print("Same file for train/val → time-based split (90/10)...")
        full_df = pd.read_parquet(train_path)
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(full_df) * 0.9)
        train_df = full_df.iloc[:split_idx]
        val_df = full_df.iloc[split_idx:]
        print(f"  Train: {len(train_df)} samples (past)")
        print(f"  Val:   {len(val_df)} samples (future)")
    else:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")

    # Build vocab from training data only (no leakage from val/test)
    print("Building vocabulary from training data...")
    vocab = build_vocab_from_dataframe(train_df)
    vocab_sizes = vocab.get_vocab_sizes()
    print(f"  Vocab built: {len(vocab_sizes)} feature IDs")
    for fid in sorted(vocab_sizes.keys())[:10]:
        print(f"    feature_id={fid}: vocab_size={vocab_sizes[fid]}")
    print(f"    ... ({len(vocab_sizes)} total)")

    # Save vocab for inference
    os.makedirs(config.train.save_dir, exist_ok=True)
    vocab_path = os.path.join(config.train.save_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Vocab saved to {vocab_path}")

    # Create datasets
    train_ds = TAACDataset(train_df, config, vocab_builder=vocab)
    val_ds = TAACDataset(val_df, config, vocab_builder=vocab)

    nw = config.train.num_workers
    pf = config.train.prefetch_factor if nw > 0 else None
    pw = nw > 0  # persistent_workers: keep workers alive between epochs

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=pw,
        prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=pw,
        prefetch_factor=pf,
    )

    return train_loader, val_loader, vocab
