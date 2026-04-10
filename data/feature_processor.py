"""
Feature Processor: Parse nested dict/list features from parquet into tensors.

Unified Tokenization:
    user sparse features  → embedding lookup → [U1, U2, ..., Un]
    item sparse features  → embedding lookup → [I1, I2, ..., Im]
    action_seq features   → per-step embed   → [A1, A2, ..., At]
    content_seq features  → per-step embed   → [C1, C2, ..., Ct]
    item_seq features     → per-step embed   → [S1, S2, ..., St]

All tokens are projected to the same dimension D, forming a unified sequence:
    [U1..Un | I1..Im | A1..At | C1..Ct | S1..St | TARGET]
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def _default_vocab():
    return {"<PAD>": 0, "<UNK>": 1}


def _default_counter():
    return 2


class FeatureVocabBuilder:
    """Scan dataset to build vocabulary (value -> index) for each feature_id."""

    def __init__(self, max_hash_bucket: int = 200000, timestamp_fids: Optional[set] = None):
        self.vocabs: Dict[int, Dict] = defaultdict(_default_vocab)
        self.counters: Dict[int, int] = defaultdict(_default_counter)
        self.max_hash_bucket = max_hash_bucket
        self.timestamp_fids = timestamp_fids or set()

    def scan_features(self, feature_list: List[dict]):
        """Process one sample's feature list (user_feature or item_feature)."""
        for feat in feature_list:
            fid = feat["feature_id"]
            ftype = feat["feature_value_type"]
            if ftype == "int_value" and feat.get("int_value") is not None:
                val = int(feat["int_value"])
                if val not in self.vocabs[fid]:
                    self.vocabs[fid][val] = self.counters[fid]
                    self.counters[fid] += 1
            elif ftype == "int_array" and feat.get("int_array") is not None:
                for val in feat["int_array"]:
                    val = int(val)
                    if val not in self.vocabs[fid]:
                        self.vocabs[fid][val] = self.counters[fid]
                        self.counters[fid] += 1

    def scan_seq_features(self, seq_list):
        """Process one sub-sequence (action_seq / content_seq / item_seq)."""
        if isinstance(seq_list, np.ndarray):
            seq_list = seq_list.tolist()
        if not isinstance(seq_list, list):
            return
        for feat in seq_list:
            if not isinstance(feat, dict):
                continue
            fid = feat["feature_id"]
            # 跳过时间戳特征 (不做embedding, 用连续编码)
            if fid in self.timestamp_fids:
                continue
            if feat.get("int_array") is not None:
                arr = feat["int_array"]
                if hasattr(arr, "tolist"):
                    arr = arr.tolist()
                for val in arr:
                    val = int(val)
                    if val not in self.vocabs[fid]:
                        self.vocabs[fid][val] = self.counters[fid]
                        self.counters[fid] += 1

    def get_vocab_sizes(self) -> Dict[int, int]:
        sizes = {}
        for fid, vocab in self.vocabs.items():
            # Hash bucket cap: 防止1M/10M数据时vocab爆炸到百万级
            sizes[fid] = min(len(vocab), self.max_hash_bucket)
        return sizes

    def encode_value(self, fid: int, value) -> int:
        idx = self.vocabs[fid].get(int(value), 1)  # 1 = <UNK>
        # Hash bucketing: 超过上限的用hash映射
        vocab_size = len(self.vocabs[fid])
        if vocab_size > self.max_hash_bucket:
            idx = 2 + (hash(int(value)) % (self.max_hash_bucket - 2))  # 保留0=PAD, 1=UNK
        return idx


def build_vocab_from_dataframe(df: pd.DataFrame, max_hash_bucket: int = 200000,
                               timestamp_fids: Optional[set] = None) -> FeatureVocabBuilder:
    """Scan entire dataframe to build feature vocabularies."""
    builder = FeatureVocabBuilder(max_hash_bucket=max_hash_bucket,
                                  timestamp_fids=timestamp_fids or {28, 29, 41})
    for idx in range(len(df)):
        row = df.iloc[idx]
        builder.scan_features(row["user_feature"])
        builder.scan_features(row["item_feature"])
        seq = row["seq_feature"]
        builder.scan_seq_features(seq["action_seq"])
        builder.scan_seq_features(seq["content_seq"])
        builder.scan_seq_features(seq["item_seq"])
    return builder


def parse_sparse_features(
    feature_list: List[dict],
    target_fids: List[int],
    vocab_builder: FeatureVocabBuilder,
) -> np.ndarray:
    """Parse int_value features → index array (len(target_fids),)."""
    fid_to_val = {}
    for feat in feature_list:
        fid = feat["feature_id"]
        if feat.get("int_value") is not None:
            fid_to_val[fid] = int(feat["int_value"])

    result = np.zeros(len(target_fids), dtype=np.int64)
    for i, fid in enumerate(target_fids):
        if fid in fid_to_val:
            result[i] = vocab_builder.encode_value(fid, fid_to_val[fid])
    return result


def parse_array_features(
    feature_list: List[dict],
    target_fids: List[int],
    vocab_builder: FeatureVocabBuilder,
    max_len: int = 16,
) -> np.ndarray:
    """Parse int_array features → padded array (len(target_fids), max_len)."""
    fid_to_arr = {}
    for feat in feature_list:
        fid = feat["feature_id"]
        if fid in target_fids and feat.get("int_array") is not None:
            arr = feat["int_array"]
            if hasattr(arr, "tolist"):
                arr = arr.tolist()
            fid_to_arr[fid] = arr

    result = np.zeros((len(target_fids), max_len), dtype=np.int64)
    for i, fid in enumerate(target_fids):
        if fid in fid_to_arr:
            arr = fid_to_arr[fid][:max_len]
            for j, val in enumerate(arr):
                result[i, j] = vocab_builder.encode_value(fid, val)
    return result


def parse_dense_features(
    feature_list: List[dict],
    target_fids: List[int],
    dims: List[int],
) -> np.ndarray:
    """Parse float_array features → float vector (sum(dims),)."""
    fid_to_arr = {}
    for feat in feature_list:
        fid = feat["feature_id"]
        if fid in target_fids and feat.get("float_array") is not None:
            arr = feat["float_array"]
            if hasattr(arr, "tolist"):
                arr = arr.tolist()
            fid_to_arr[fid] = arr

    total_dim = sum(dims)
    result = np.zeros(total_dim, dtype=np.float32)
    offset = 0
    for fid, dim in zip(target_fids, dims):
        if fid in fid_to_arr:
            arr = fid_to_arr[fid][:dim]
            result[offset:offset + len(arr)] = arr
        offset += dim
    return result


def parse_float_value_features(
    feature_list: List[dict],
    target_fids: List[int],
) -> np.ndarray:
    """Parse float_value features → float array (len(target_fids),)."""
    fid_to_val = {}
    for feat in feature_list:
        fid = feat["feature_id"]
        if feat.get("float_value") is not None:
            fid_to_val[fid] = float(feat["float_value"])

    result = np.zeros(len(target_fids), dtype=np.float32)
    for i, fid in enumerate(target_fids):
        if fid in fid_to_val:
            result[i] = fid_to_val[fid]
    return result


def parse_seq_features(
    seq_data,
    target_fids: List[int],
    vocab_builder: FeatureVocabBuilder,
    max_len: int,
) -> Tuple[np.ndarray, int]:
    """
    Parse one sub-sequence into array of shape (max_len, len(target_fids)).
    Returns (padded_array, actual_length).
    Left-padding to align with causal attention.
    """
    if isinstance(seq_data, np.ndarray):
        seq_data = seq_data.tolist()
    if not isinstance(seq_data, list):
        return np.zeros((max_len, len(target_fids)), dtype=np.int64), 0

    # Build fid -> array mapping
    fid_to_arr = {}
    for feat in seq_data:
        if not isinstance(feat, dict):
            continue
        fid = feat["feature_id"]
        if fid in target_fids and feat.get("int_array") is not None:
            arr = feat["int_array"]
            if hasattr(arr, "tolist"):
                arr = arr.tolist()
            fid_to_arr[fid] = arr

    # Determine actual sequence length
    seq_len = 0
    for fid in target_fids:
        if fid in fid_to_arr:
            seq_len = max(seq_len, len(fid_to_arr[fid]))

    actual_len = min(seq_len, max_len)
    result = np.zeros((max_len, len(target_fids)), dtype=np.int64)

    for j, fid in enumerate(target_fids):
        if fid not in fid_to_arr:
            continue
        arr = fid_to_arr[fid]
        # Take the most recent `actual_len` items
        if len(arr) > max_len:
            arr = arr[-max_len:]
        # Left-padding: place at the end
        start = max_len - len(arr)
        for k, val in enumerate(arr):
            result[start + k, j] = vocab_builder.encode_value(fid, val)

    return result, actual_len
