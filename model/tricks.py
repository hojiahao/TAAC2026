"""
Advanced modules for TAAC 2026 CVR prediction.

Implemented:
  1. RoPE (Rotary Position Embedding)
  2. FiLM (Feature-wise Linear Modulation) — click/conversion task decoupling
  3. ATTMatch (Attention Matching) — conversion-aware attention weighting
  4. Three-Level Session System — temporal hierarchy modeling
  5. Enhanced Attention Block — combines RoPE + ATTMatch + SiLU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# 1. RoPE — Rotary Position Embedding
# ═══════════════════════════════════════════════════════════════════

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: encode relative position via rotation in complex plane.
        q' = q * cos(theta) + rotate_half(q) * sin(theta)
        k' = k * cos(theta) + rotate_half(k) * sin(theta)
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (B, H, L, D)
            offset: position offset for KV cache
        """
        seq_len = q.shape[-2]
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ═══════════════════════════════════════════════════════════════════
# 2. FiLM — Feature-wise Linear Modulation
# ═══════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """
    FiLM(x) = gamma(task) * x + beta(task)
    Same backbone, different representations for click vs conversion.
    """

    def __init__(self, dim: int, num_tasks: int = 2):
        super().__init__()
        self.gamma = nn.Embedding(num_tasks, dim)
        self.beta = nn.Embedding(num_tasks, dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        task_idx = torch.tensor(task_id, device=x.device)
        return self.gamma(task_idx) * x + self.beta(task_idx)


class MultiTaskHead(nn.Module):
    """
    Multi-task head with FiLM modulation.

        backbone output → FiLM(task=click) → click_head → p(click)
        backbone output → FiLM(task=cvr)   → cvr_head   → p(conversion)

    Training: joint loss = alpha * L_click + beta * L_cvr
    Inference: only cvr output for AUC
    """

    def __init__(self, dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.film = FiLMLayer(dim, num_tasks=2)

        self.shared_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.click_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.cvr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x: (B, D)
        Returns: click_logits (B,1), cvr_logits (B,1)
        """
        x_click = self.film(x, task_id=0)
        click_logits = self.click_head(self.shared_mlp(x_click))

        x_cvr = self.film(x, task_id=1)
        cvr_logits = self.cvr_head(self.shared_mlp(x_cvr))

        return click_logits, cvr_logits


# ═══════════════════════════════════════════════════════════════════
# 3. ATTMatch — Conversion-Aware Attention Bias
# ═══════════════════════════════════════════════════════════════════

class ATTMatch(nn.Module):
    """
    Bias attention toward historically converted positions.
        attn_scores[i,j] += learnable_bias[h] * is_conversion[j]
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.conversion_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        attn_scores: torch.Tensor,
        conversion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            attn_scores: (B, H, L, L)
            conversion_mask: (B, L) — 1.0 at conversion positions
        """
        if conversion_mask is None:
            return attn_scores
        bias = conversion_mask.unsqueeze(1).unsqueeze(2)       # (B, 1, 1, L)
        head_bias = self.conversion_bias.view(1, -1, 1, 1)    # (1, H, 1, 1)
        return attn_scores + bias * head_bias


# ═══════════════════════════════════════════════════════════════════
# 4. Three-Level Session System
# ═══════════════════════════════════════════════════════════════════

class SessionEncoder(nn.Module):
    """
    Encode temporal hierarchy:
      Level 0 (short-term):  < 30 min
      Level 1 (mid-term):    30min ~ 24h
      Level 2 (long-term):   > 24h

    Output: per-position temporal embedding (B, L, D).
    """

    def __init__(self, dim: int, num_time_buckets: int = 64, num_session_levels: int = 3):
        super().__init__()
        self.num_time_buckets = num_time_buckets
        self.time_bucket_embedding = nn.Embedding(num_time_buckets, dim)
        self.session_embedding = nn.Embedding(num_session_levels, dim)
        self.periodic_proj = nn.Linear(4, dim)
        self.fusion = nn.Linear(3 * dim, dim)

    def _bucketize_time_diff(self, delta_t: torch.Tensor) -> torch.Tensor:
        delta_minutes = delta_t.float().clamp(min=1) / 60.0
        bucket_idx = (torch.log2(delta_minutes + 1)).long()
        return bucket_idx.clamp(0, self.num_time_buckets - 1)

    def _assign_session_level(self, delta_t: torch.Tensor) -> torch.Tensor:
        level = torch.zeros_like(delta_t, dtype=torch.long)
        level[delta_t > 1800] = 1
        level[delta_t > 86400] = 2
        return level

    def _periodic_encoding(self, timestamps: torch.Tensor) -> torch.Tensor:
        hour = ((timestamps % 86400) / 3600).float()
        day = ((timestamps / 86400) % 7).float()
        periodic = torch.stack([
            torch.sin(2 * math.pi * hour / 24),
            torch.cos(2 * math.pi * hour / 24),
            torch.sin(2 * math.pi * day / 7),
            torch.cos(2 * math.pi * day / 7),
        ], dim=-1)
        return self.periodic_proj(periodic)

    def forward(
        self,
        seq_timestamps: torch.Tensor,     # (B, L)
        current_timestamp: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        delta_t = (current_timestamp.unsqueeze(1) - seq_timestamps).clamp(min=0)

        time_emb = self.time_bucket_embedding(self._bucketize_time_diff(delta_t))
        session_emb = self.session_embedding(self._assign_session_level(delta_t))
        periodic_emb = self._periodic_encoding(seq_timestamps)

        return self.fusion(torch.cat([time_emb, session_emb, periodic_emb], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# 5. Enhanced Attention Block — RoPE + ATTMatch + SiLU
# ═══════════════════════════════════════════════════════════════════

class EnhancedSiLUAttention(nn.Module):
    """SiLU gated attention with RoPE and ATTMatch."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkuv_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_attn = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        self.att_match = ATTMatch(num_heads)

    def forward(self, x, attn_mask=None, conversion_mask=None):
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim

        qkuv = F.silu(self.qkuv_proj(x))
        q, k, v, u = qkuv.chunk(4, dim=-1)

        q = q.view(B, L, H, d).transpose(1, 2)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)
        u = u.view(B, L, H, d).transpose(1, 2)

        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = self.att_match(scores, conversion_mask)
        scores = F.silu(scores)

        if attn_mask is not None:
            scores = scores * attn_mask

        attn_out = self.norm_attn(torch.matmul(scores, v))
        attn_out = attn_out * u
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.dropout(self.out_proj(attn_out))


class EnhancedTransducerBlock(nn.Module):
    """Stackable block: Enhanced SiLU Attention + SwiGLU FFN."""

    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = EnhancedSiLUAttention(dim, num_heads, dropout, max_seq_len)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, conversion_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask, conversion_mask)
        h = self.norm2(x)
        x = x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x
