"""
UniRec: Unified Recommendation Model for TAAC 2026 (KDD Cup 2026).

A unified architecture that bridges sequence modeling and feature interaction
within a single homogeneous, stackable backbone for CVR prediction.

Innovations fused from 6 sources:
  [OneTrans]    Auto-Split Tokenizer + Mixed Parameterization (S shared / NS independent)
  [InterFormer] Feature Cross Layer (pre-backbone explicit feature interaction)
  [HSTU 2.0]    SiLU Gated Attention + Semi-Local Attention + Attention Truncation
  [HyFormer]    MoT for multi-sequence independent modeling
  [DIN]         Target-Aware Interest Extraction (item queries history)
  [Kimi]        Block Attention Residuals (depth-wise selective retrieval)

Architecture overview:
  1. Unified Tokenization
       NS-tokens = AutoSplit(MLP(concat(all_features)))
       S-tokens  = TimestampAware_Merge(seq_1, ..., seq_n)

  2. Pre-Backbone Processing
       Feature Cross Layer: NS-tokens self-attention (feature x feature)
       Target-Aware Interest: item_pool queries all S-tokens via cross-attention
       Target Fusion: MLP([user_pool; item_pool; user*item])

  3. Unified Sequence Assembly
       [NS-tokens | S-tokens | MoT | interest | target]

  4. Homogeneous Stackable Backbone (Unified Block x N)
       Each block: RMSNorm -> Mixed SiLU Gated Attention -> Block AttnRes
                   RMSNorm -> Mixed SwiGLU FFN -> Block AttnRes
       Full layers (N1): process all tokens
       Truncated layers (N2): only recent L' S-tokens + all NS/special tokens

  5. CVR Prediction: target_output -> MLP -> sigmoid -> P(conversion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple

from model.modules import (
    UnifiedTransducerBlock,
    build_unified_attention_mask,
)
from model.tricks import EnhancedTransducerBlock
from model.mot import MixtureOfTransducers
from config import Config


# ═══════════════════════════════════════════════════════════════
# Block Attention Residuals (from Kimi, arXiv:2603.15031)
# ═══════════════════════════════════════════════════════════════

class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals: replace fixed residual accumulation with
    learned softmax attention over block-level representations.

    Standard residual: h_l = h_{l-1} + f_{l-1}(h_{l-1})
      -> all layers accumulated with fixed unit weights
      -> early layer info diluted, no selective access

    Block AttnRes: h_l = sum_i alpha_{i->l} * v_i
      -> alpha computed via softmax attention over block outputs
      -> each layer selectively retrieves from any previous block

    Reduces memory from O(Ld) to O(Nd) where N = number of blocks.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Learned pseudo-query per layer (one d-dim vector)
        self.res_proj = nn.Linear(dim, dim, bias=False)
        self.res_norm = nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim)

    def forward(
        self,
        current: torch.Tensor,          # (B, L, D) current layer output
        block_outputs: List[torch.Tensor],  # list of (B, L, D) previous block outputs
        partial_block: torch.Tensor,     # (B, L, D) intra-block accumulation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (attended_output, updated_partial_block)
        """
        # Accumulate within current block
        partial_block = partial_block + current

        if len(block_outputs) == 0:
            return partial_block, partial_block

        # Inter-block attention: current attends to all previous block outputs
        # Stack block outputs: (B, N, L, D) -> for efficiency, pool over L first
        # Use last-token representation as block summary
        block_summaries = torch.stack(
            [b[:, -1, :] for b in block_outputs], dim=1
        )  # (B, N, D)
        block_summaries = self.res_norm(block_summaries)

        # Query from current partial block's last token
        query = self.res_proj(partial_block[:, -1, :]).unsqueeze(1)  # (B, 1, D)

        # Attention weights over blocks
        scores = torch.matmul(query, block_summaries.transpose(-1, -2))  # (B, 1, N)
        scores = scores / math.sqrt(query.shape[-1])
        weights = F.softmax(scores, dim=-1)  # (B, 1, N)

        # Weighted combination of block outputs (full token-level)
        stacked = torch.stack(block_outputs, dim=1)  # (B, N, L, D)
        # Broadcast weights: (B, 1, N, 1) * (B, N, L, D) -> sum over N -> (B, L, D)
        attended = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, L, D)

        return partial_block + attended, partial_block


# ═══════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════

class UnifiedRecModel(nn.Module):

    def __init__(self, config: Config, vocab_sizes: Dict[int, int]):
        super().__init__()
        mc = config.model
        fc = config.feature
        self.config = config
        self.mc = mc
        self.use_tricks = mc.use_tricks

        D = mc.embedding_dim
        feat_d = mc.feature_embedding_dim

        # ═══════════════════════════════════════════════════════
        # 1. Unified Tokenization [OneTrans]
        # ═══════════════════════════════════════════════════════

        # Per-feature embedding tables
        all_embed_fids = list(set(
            fc.user_sparse_feature_ids + fc.user_array_feature_ids +
            fc.user_mixed_feature_ids + fc.item_sparse_feature_ids +
            fc.item_array_feature_ids + fc.action_seq_feature_ids +
            fc.content_seq_feature_ids + fc.item_seq_feature_ids
        ))
        self.feature_embs = nn.ModuleDict()
        for fid in all_embed_fids:
            vs = vocab_sizes.get(fid, fc.default_vocab_size)
            self.feature_embs[str(fid)] = nn.Embedding(vs, feat_d, padding_idx=0)

        # Per-feature projection (not shared) [OneTrans mixed param idea]
        proj_fids = list(set(
            fc.user_sparse_feature_ids + fc.user_array_feature_ids +
            fc.user_mixed_feature_ids + fc.item_sparse_feature_ids +
            fc.item_array_feature_ids
        ))
        self.feature_projs = nn.ModuleDict()
        for fid in proj_fids:
            self.feature_projs[str(fid)] = nn.Linear(feat_d, D)

        # Dense/float feature projections
        self.dense_projs = nn.ModuleDict()
        for fid, dim in zip(fc.user_dense_feature_ids, fc.user_dense_dims):
            self.dense_projs[str(fid)] = nn.Linear(dim, D)
        self.float_proj = nn.Linear(1, D) if fc.item_float_feature_ids else None

        # Sequence projections
        self.action_seq_proj = nn.Linear(len(fc.action_seq_feature_ids) * feat_d, D)
        self.content_seq_proj = nn.Linear(len(fc.content_seq_feature_ids) * feat_d, D)
        self.item_seq_proj = nn.Linear(len(fc.item_seq_feature_ids) * feat_d, D)

        # ═══════════════════════════════════════════════════════
        # 2. Feature Cross Layer [InterFormer]
        # ═══════════════════════════════════════════════════════
        n_user_fields = (len(fc.user_sparse_feature_ids) + len(fc.user_array_feature_ids) +
                         len(fc.user_dense_feature_ids) + len(fc.user_mixed_feature_ids))
        n_item_fields = (len(fc.item_sparse_feature_ids) + len(fc.item_array_feature_ids) +
                         len(fc.item_float_feature_ids))
        self.n_user_fields = n_user_fields
        self.n_item_fields = n_item_fields
        self.n_feature_tokens = n_user_fields + n_item_fields

        self.field_embedding = nn.Embedding(self.n_feature_tokens + 8, D)
        self.feature_cross = nn.TransformerEncoderLayer(
            d_model=D, nhead=mc.num_heads, dim_feedforward=D * 4,
            dropout=mc.dropout, activation="gelu", batch_first=True,
        )

        # ═══════════════════════════════════════════════════════
        # 3. Target-Aware Interest [DIN] + Target Fusion [HSTU]
        # ═══════════════════════════════════════════════════════
        self.interest_query = nn.Linear(D, D, bias=False)
        self.interest_key = nn.Linear(D, D, bias=False)
        self.interest_value = nn.Linear(D, D, bias=False)
        self.interest_out = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

        self.target_fusion = nn.Sequential(
            nn.Linear(D * 3, D), nn.SiLU(), nn.Linear(D, D),
        )

        # Segment embedding: 0=feature, 1=action_seq, 2=content_seq, 3=item_seq, 4=special
        self.segment_embedding = nn.Embedding(5, D)

        # ═══════════════════════════════════════════════════════
        # 4. MoT [HSTU 2.0]
        # ═══════════════════════════════════════════════════════
        self.use_mot = mc.use_mot
        self.mot = None
        if self.use_mot:
            self.mot = MixtureOfTransducers(
                dim=D, num_heads=mc.num_heads, num_layers=mc.mot_layers,
                dropout=mc.dropout, merge=mc.mot_merge, use_enhanced=self.use_tricks,
                max_seq_lens=(config.train.max_action_seq_len,
                              config.train.max_content_seq_len,
                              config.train.max_item_seq_len),
            )
            self.mot_proj = nn.Linear(D, D)

        # ═══════════════════════════════════════════════════════
        # 5. Homogeneous Stackable Backbone [OneTrans + HSTU 2.0]
        #    with Block Attention Residuals [Kimi]
        # ═══════════════════════════════════════════════════════
        n_full = mc.full_seq_layers if mc.use_attention_truncation else mc.num_layers
        n_trunc = mc.truncated_seq_layers if mc.use_attention_truncation else 0
        total_layers = n_full + n_trunc

        max_seq = self.n_feature_tokens + (config.train.max_action_seq_len +
                   config.train.max_content_seq_len +
                   config.train.max_item_seq_len) + 16

        BlockClass = EnhancedTransducerBlock if self.use_tricks else UnifiedTransducerBlock
        block_kwargs = dict(dim=D, num_heads=mc.num_heads, ffn_dim=D * 4, dropout=mc.dropout)
        if self.use_tricks:
            block_kwargs["max_seq_len"] = max_seq
        else:
            block_kwargs["attention_type"] = mc.attention_type

        self.full_blocks = nn.ModuleList([BlockClass(**block_kwargs) for _ in range(n_full)])
        self.truncated_blocks = nn.ModuleList([BlockClass(**block_kwargs) for _ in range(n_trunc)])

        # Block AttnRes: group every block_size layers into one block
        self.block_attn_res = BlockAttnRes(D)
        self.attn_res_block_size = mc.attn_res_block_size

        self.final_norm = nn.LayerNorm(D)

        # ═══════════════════════════════════════════════════════
        # 6. CVR Prediction Head
        # ═══════════════════════════════════════════════════════
        self.head = nn.Sequential(
            nn.Linear(D, mc.head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.head_hidden_dim, mc.head_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(mc.dropout),
            nn.Linear(mc.head_hidden_dim // 2, 1),
        )

        self.use_grad_ckpt = config.train.gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    # ── Helper methods ──

    def _embed_and_proj(self, fid: int, values: torch.Tensor) -> torch.Tensor:
        emb = self.feature_embs[str(fid)](values)
        if emb.ndim == 3:  # array: pool
            mask = (values != 0).unsqueeze(-1).float()
            emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.feature_projs[str(fid)](emb)

    def _embed_seq(self, seq, feature_ids, proj):
        B, L, F = seq.shape
        embs = [self.feature_embs[str(fid)](seq[:, :, i]) for i, fid in enumerate(feature_ids)]
        return proj(torch.cat(embs, dim=-1))

    # ── Main forward ──

    def _build_unified_sequence(self, batch: Dict[str, torch.Tensor]):
        fc = self.config.feature
        device = batch["user_sparse"].device
        B = batch["user_sparse"].shape[0]

        feature_tokens = []
        field_indices = []

        # User features → individual tokens
        for i, fid in enumerate(fc.user_sparse_feature_ids):
            feature_tokens.append(self._embed_and_proj(fid, batch["user_sparse"][:, i]).unsqueeze(1))
            field_indices.append(len(field_indices))
        for i, fid in enumerate(fc.user_array_feature_ids):
            feature_tokens.append(self._embed_and_proj(fid, batch["user_array"][:, i, :]).unsqueeze(1))
            field_indices.append(len(field_indices))
        for j, (fid, dim) in enumerate(zip(fc.user_dense_feature_ids, fc.user_dense_dims)):
            offset = sum(fc.user_dense_dims[:j])
            tok = self.dense_projs[str(fid)](batch["user_dense"][:, offset:offset + dim])
            feature_tokens.append(tok.unsqueeze(1))
            field_indices.append(len(field_indices))
        for i, fid in enumerate(fc.user_mixed_feature_ids):
            feature_tokens.append(self._embed_and_proj(fid, batch["user_mixed"][:, i, :]).unsqueeze(1))
            field_indices.append(len(field_indices))

        n_user = len(feature_tokens)

        # Item features → individual tokens
        for i, fid in enumerate(fc.item_sparse_feature_ids):
            feature_tokens.append(self._embed_and_proj(fid, batch["item_sparse"][:, i]).unsqueeze(1))
            field_indices.append(len(field_indices))
        for i, fid in enumerate(fc.item_array_feature_ids):
            feature_tokens.append(self._embed_and_proj(fid, batch["item_array"][:, i, :]).unsqueeze(1))
            field_indices.append(len(field_indices))
        for i, fid in enumerate(fc.item_float_feature_ids):
            feature_tokens.append(self.float_proj(batch["item_float"][:, i:i+1]).unsqueeze(1))
            field_indices.append(len(field_indices))

        # Concat + field embedding + Feature Cross [InterFormer]
        feat_seq = torch.cat(feature_tokens, dim=1)  # (B, n_feat, D)
        field_ids = torch.tensor(field_indices, device=device).unsqueeze(0).expand(B, -1)
        feat_seq = feat_seq + self.field_embedding(field_ids)
        feat_seq = self.feature_cross(feat_seq)

        user_pool = feat_seq[:, :n_user, :].mean(dim=1)
        item_pool = feat_seq[:, n_user:, :].mean(dim=1)

        # Sequence tokens
        action_tokens = self._embed_seq(batch["action_seq"], fc.action_seq_feature_ids, self.action_seq_proj)
        content_tokens = self._embed_seq(batch["content_seq"], fc.content_seq_feature_ids, self.content_seq_proj)
        item_seq_tokens = self._embed_seq(batch["item_seq"], fc.item_seq_feature_ids, self.item_seq_proj)

        L_a, L_c, L_s = action_tokens.shape[1], content_tokens.shape[1], item_seq_tokens.shape[1]
        action_pad = batch["action_seq"].sum(dim=-1) != 0
        content_pad = batch["content_seq"].sum(dim=-1) != 0
        item_pad = batch["item_seq"].sum(dim=-1) != 0

        # MoT [HSTU 2.0]
        mot_tokens = []
        if self.use_mot and self.mot is not None:
            mot_out = self.mot(action_tokens, content_tokens, item_seq_tokens)
            mot_tokens.append(self.mot_proj(mot_out).unsqueeze(1))

        # Target-Aware Interest [DIN]
        all_seq = torch.cat([action_tokens, content_tokens, item_seq_tokens], dim=1)
        all_seq_pad = torch.cat([action_pad, content_pad, item_pad], dim=1)

        q = self.interest_query(item_pool).unsqueeze(1)
        k = self.interest_key(all_seq)
        v = self.interest_value(all_seq)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attn_scores = attn_scores.masked_fill(~all_seq_pad.unsqueeze(1), float('-inf'))
        interest = torch.matmul(F.softmax(attn_scores, dim=-1), v).squeeze(1)
        interest_token = self.interest_out(interest).unsqueeze(1)

        # Target Fusion [HSTU]
        target = self.target_fusion(
            torch.cat([user_pool, item_pool, user_pool * item_pool], dim=-1)
        ).unsqueeze(1)

        # Assemble unified sequence
        n_feat = feat_seq.shape[1]
        seq_parts = [feat_seq, action_tokens, content_tokens, item_seq_tokens]
        seg_ids = [
            torch.zeros(B, n_feat, dtype=torch.long, device=device),
            torch.full((B, L_a), 1, dtype=torch.long, device=device),
            torch.full((B, L_c), 2, dtype=torch.long, device=device),
            torch.full((B, L_s), 3, dtype=torch.long, device=device),
        ]
        pad_parts = [
            torch.ones(B, n_feat, dtype=torch.bool, device=device),
            action_pad, content_pad, item_pad,
        ]

        for mt in mot_tokens:
            seq_parts.append(mt)
            seg_ids.append(torch.full((B, 1), 4, dtype=torch.long, device=device))
            pad_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        seq_parts.append(interest_token)
        seg_ids.append(torch.full((B, 1), 4, dtype=torch.long, device=device))
        pad_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        seq_parts.append(target)
        seg_ids.append(torch.full((B, 1), 4, dtype=torch.long, device=device))
        pad_parts.append(torch.ones(B, 1, dtype=torch.bool, device=device))

        unified_seq = torch.cat(seq_parts, dim=1)
        segments = torch.cat(seg_ids, dim=1)
        self._padding_mask = torch.cat(pad_parts, dim=1)

        unified_seq = unified_seq + self.segment_embedding(segments)
        return unified_seq

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._build_unified_sequence(batch)
        B, L, D = x.shape
        device = x.device

        # Hybrid attention mask [本项目创新]
        if self.mc.use_semi_local_attention:
            mask = build_unified_attention_mask(
                L, self.n_feature_tokens,
                self.mc.global_window_size, self.mc.local_window_size, device,
            )
        else:
            mask = torch.ones(1, 1, L, L, device=device)

        pad_mask = self._padding_mask.unsqueeze(1).unsqueeze(2).float()
        mask = mask * pad_mask

        # ── Backbone with Block AttnRes [Kimi] ──
        use_ckpt = self.use_grad_ckpt and self.training
        block_outputs = []      # completed block representations
        partial_block = torch.zeros_like(x)  # intra-block accumulation
        layer_count = 0
        block_size = self.attn_res_block_size

        # Full sequence layers [HSTU 2.0]
        for block in self.full_blocks:
            if use_ckpt:
                x = checkpoint(block, x, mask, **({"use_reentrant": False}))
            else:
                x = block(x, mask)

            layer_count += 1
            x, partial_block = self.block_attn_res(x, block_outputs, partial_block)

            # Block boundary
            if layer_count % block_size == 0:
                block_outputs.append(partial_block.detach() if not self.training else partial_block)
                partial_block = torch.zeros_like(x)

        # Truncated layers [HSTU 2.0 Attention Truncation]
        if self.mc.use_attention_truncation and len(self.truncated_blocks) > 0:
            trunc_len = min(self.mc.truncated_seq_len, L)
            x_trunc = x[:, -trunc_len:, :]
            trunc_mask = mask[:, :, -trunc_len:, -trunc_len:]

            for block in self.truncated_blocks:
                if use_ckpt:
                    x_trunc = checkpoint(block, x_trunc, trunc_mask, **({"use_reentrant": False}))
                else:
                    x_trunc = block(x_trunc, trunc_mask)

            x = torch.cat([x[:, :-trunc_len, :], x_trunc], dim=1)

        x = self.final_norm(x)
        return self.head(x[:, -1, :])  # target token -> CVR prediction
