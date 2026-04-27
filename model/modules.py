"""
Core building blocks with production-oriented efficiency optimizations.

Optimizations:
  1. FlashAttention/SDPA auto-detection for SoftmaxAttention
  2. Cached attention masks (avoid rebuild every forward)
  3. Vectorized Semi-Local + Hybrid mask building
  4. In-place operations where safe
  5. Triton SiLU Attention kernel (avoids O(L²) attention matrix materialization)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from functools import lru_cache

# Auto-detect Triton availability
from model.triton_kernels import TRITON_AVAILABLE
if TRITON_AVAILABLE:
    from model.triton_kernels import triton_silu_attention


# ── Attention mask cache (avoids rebuilding every forward pass) ──

_mask_cache: Dict[Tuple, torch.Tensor] = {}


def _get_or_build_mask(key, builder_fn, *args, **kwargs):
    """Cache masks by (type, seq_len, device) to avoid redundant computation."""
    if key not in _mask_cache:
        _mask_cache[key] = builder_fn(*args, **kwargs)
    return _mask_cache[key]


def clear_mask_cache():
    """Call when sequence length changes or at start of new epoch."""
    _mask_cache.clear()


# ── SiLU Gated Attention (HSTU-style) ──

class SiLUAttention(nn.Module):
    """
    HSTU-style pointwise attention with SiLU gating.
    Uses SiLU(Q@K^T) instead of softmax — allows negative attention weights.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkuv_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_attn = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        qkuv = F.silu(self.qkuv_proj(x))
        q, k, v, u = qkuv.chunk(4, dim=-1)

        q = q.view(B, L, H, d).transpose(1, 2)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)
        u = u.view(B, L, H, d).transpose(1, 2)

        if TRITON_AVAILABLE and x.is_cuda:
            # Triton kernel: fused Q@K^T + SiLU + mask + @V (no O(L²) materialization)
            attn_out = triton_silu_attention(q, k, v, attn_mask, scale=1.0 / math.sqrt(d))
        else:
            # PyTorch fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            scores = F.silu(scores)
            if attn_mask is not None:
                scores = scores * attn_mask
            attn_out = torch.matmul(scores, v)

        attn_out = self.norm_attn(attn_out)
        attn_out = attn_out * u
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.dropout(self.out_proj(attn_out))


class SoftmaxAttention(nn.Module):
    """Standard MHA with FlashAttention/SDPA auto-detection."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (auto-dispatches to FlashAttention/memory-efficient)
        if attn_mask is not None and attn_mask.dtype == torch.float32:
            # Convert multiplicative mask to additive for SDPA
            attn_bias = torch.where(attn_mask > 0.5, 0.0, float('-inf'))
        else:
            attn_bias = attn_mask

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# ── Feed-Forward Network (SwiGLU) ──

class PointwiseFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ── Unified Transducer Block ──

class UnifiedTransducerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1, attention_type="silu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SiLUAttention(dim, num_heads, dropout) if attention_type == "silu" else SoftmaxAttention(dim, num_heads, dropout)
        self.ffn = PointwiseFeedForward(dim, ffn_dim, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Mask Building (Cached + Vectorized) ──

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    key = ("causal", seq_len, str(device))
    if key not in _mask_cache:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        _mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
    return _mask_cache[key]


def build_semi_local_mask(seq_len, global_window, local_window, device):
    key = ("sla", seq_len, global_window, local_window, str(device))
    if key not in _mask_cache:
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        causal = col <= row
        global_m = col < global_window
        local_m = (row - col) < local_window
        _mask_cache[key] = (causal & (global_m | local_m)).float().unsqueeze(0).unsqueeze(0)
    return _mask_cache[key]


def build_unified_attention_mask(seq_len, n_feature_tokens, global_window, local_window, device):
    key = ("unified", seq_len, n_feature_tokens, global_window, local_window, str(device))
    if key not in _mask_cache:
        F_n = n_feature_tokens
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)

        causal = col <= row
        global_m = col < global_window
        local_m = (row - col) < local_window
        mask = causal & (global_m | local_m)

        # Feature region: full bidirectional
        mask[:F_n, :F_n] = True
        # Last 2 tokens (interest + target): see everything
        mask[-2:, :] = True

        _mask_cache[key] = mask.float().unsqueeze(0).unsqueeze(0)
    return _mask_cache[key]
