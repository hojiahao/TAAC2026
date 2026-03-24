"""
Core building blocks for the Unified Recommendation Model.
Implements HSTU-style attention (SiLU gating) and ULTRA-HSTU innovations:
  - Semi-Local Attention (SLA)
  - Attention Truncation
  - Merged item+action embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SiLUAttention(nn.Module):
    """
    HSTU-style pointwise attention with SiLU gating.

    Forward pass (per layer):
        X = Norm(Z)
        U, Q, K, V = SiLU(Linear(X))
        A = (SiLU(Q @ K^T) * M) @ V
        Y = Linear(Norm(A) * U)
        Z = Y + Z
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Single projection for U, Q, K, V (4 * dim)
        self.qkuv_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_attn = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                           # (B, L, D)
        attn_mask: Optional[torch.Tensor] = None,   # (B, 1, L, L) or (B, H, L, L)
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project to U, Q, K, V with SiLU activation
        qkuv = F.silu(self.qkuv_proj(x))   # (B, L, 4D)
        q, k, v, u = qkuv.chunk(4, dim=-1)

        # Reshape to multi-head
        q = q.view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)
        u = u.view(B, L, H, d).transpose(1, 2)

        # Attention: SiLU(Q @ K^T) * Mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = F.silu(scores)  # (B, H, L, L)

        if attn_mask is not None:
            scores = scores * attn_mask

        attn_out = torch.matmul(scores, v)   # (B, H, L, d)
        attn_out = self.norm_attn(attn_out)

        # Gated output: Norm(A) * U
        attn_out = attn_out * u              # (B, H, L, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return out


class SoftmaxAttention(nn.Module):
    """Standard multi-head self-attention with softmax (baseline)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class PointwiseFeedForward(nn.Module):
    """FFN with SiLU activation (SwiGLU variant)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class UnifiedTransducerBlock(nn.Module):
    """
    Single HSTU-style transducer block:
        X = Norm(Z)
        Z = Attention(X) + Z
        Z = FFN(Norm(Z)) + Z
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_type: str = "silu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if attention_type == "silu":
            self.attn = SiLUAttention(dim, num_heads, dropout)
        else:
            self.attn = SoftmaxAttention(dim, num_heads, dropout)

        self.ffn = PointwiseFeedForward(dim, ffn_dim, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask: (1, 1, L, L)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def build_semi_local_mask(
    seq_len: int,
    global_window: int,
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Semi-Local Attention mask (ULTRA-HSTU), vectorized.
    Shape: (1, 1, L, L)
    """
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    causal = col_idx <= row_idx
    global_mask = col_idx < global_window
    local_mask = (row_idx - col_idx) < local_window

    mask = causal & (global_mask | local_mask)
    return mask.float().unsqueeze(0).unsqueeze(0)


def build_unified_attention_mask(
    seq_len: int,
    n_feature_tokens: int,
    global_window: int,
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Hybrid attention mask for unified architecture:

    Region layout: [features(0~F) | sequences(F~S) | special(S~L)]

    Rules:
      1. Feature↔Feature: FULL bidirectional (all feature crosses, replaces DCN)
      2. Sequence→Feature: all seq tokens can see ALL features (via global window)
      3. Sequence→Sequence: SLA (local_window + global_window, causal)
      4. Target→ALL: target token sees EVERY token (full attention to all)
      5. Padding: handled separately via padding_mask multiplication

    Shape: (1, 1, L, L)
    """
    F = n_feature_tokens
    L = seq_len
    row_idx = torch.arange(L, device=device).unsqueeze(1)
    col_idx = torch.arange(L, device=device).unsqueeze(0)

    # Start with causal SLA (same as before)
    causal = col_idx <= row_idx
    global_mask = col_idx < global_window
    local_mask = (row_idx - col_idx) < local_window
    mask = causal & (global_mask | local_mask)

    # Rule 1: Feature region (rows 0~F, cols 0~F) = full bidirectional
    mask[:F, :F] = True

    # Rule 4: Last 2 tokens (MoT + target) see EVERYTHING
    mask[-2:, :] = True

    return mask.float().unsqueeze(0).unsqueeze(0)
