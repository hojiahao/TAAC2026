"""
Mixture of Transducers (MoT) — ULTRA-HSTU.

Instead of concatenating all 3 sequences into one long sequence,
MoT processes each sequence with its own lightweight transducer,
then fuses the outputs before the main backbone.

Data has 3 heterogeneous sequences:
    action_seq  — user behavior actions (short-term interaction)
    content_seq — content consumption (interest/preference)
    item_seq    — item interaction history (collaborative signal)

Architecture:
    action_seq  → [Transducer_A × N layers] → h_a (B, D)
    content_seq → [Transducer_C × N layers] → h_c (B, D)
    item_seq    → [Transducer_S × N layers] → h_s (B, D)
                           ↓
                   Fusion(h_a, h_c, h_s) → h_fused (B, D)

Benefits (from HSTU 2.0 paper):
    - 69% inference FLOP reduction vs single 16k sequence
    - Each transducer specializes in its own behavioral signal
    - Avoids cross-sequence interference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.modules import UnifiedTransducerBlock, build_causal_mask
from model.tricks import EnhancedTransducerBlock


class BranchTransducer(nn.Module):
    """Lightweight transducer for one sequence branch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        use_enhanced: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        BlockClass = EnhancedTransducerBlock if use_enhanced else UnifiedTransducerBlock
        block_kwargs = dict(dim=dim, num_heads=num_heads, ffn_dim=dim * 4, dropout=dropout)
        if use_enhanced:
            block_kwargs["max_seq_len"] = max_seq_len
        else:
            block_kwargs["attention_type"] = "silu"

        self.layers = nn.ModuleList([BlockClass(**block_kwargs) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args: x: (B, L, D)
        Returns: (B, D) — pooled sequence representation
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        # Last-token pooling (left-padded sequences → last token is most recent)
        return x[:, -1, :]


class GatedFusion(nn.Module):
    """
    Gated fusion of multiple branch outputs.
    gate_i = softmax(W_g @ h_i)
    h_fused = sum(gate_i * W_v @ h_i)
    """

    def __init__(self, dim: int, num_branches: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, num_branches, bias=False)
        self.value_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_branches)])

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args: branch_outputs: list of (B, D) tensors
        Returns: (B, D) fused representation
        """
        # Stack: (B, num_branches, D)
        stacked = torch.stack(branch_outputs, dim=1)

        # Compute gates from mean representation
        gate_input = stacked.mean(dim=1)           # (B, D)
        gates = F.softmax(self.gate_proj(gate_input), dim=-1)  # (B, num_branches)
        gates = gates.unsqueeze(-1)                # (B, num_branches, 1)

        # Apply value projections
        values = []
        for i, proj in enumerate(self.value_projs):
            values.append(proj(branch_outputs[i]))
        values = torch.stack(values, dim=1)        # (B, num_branches, D)

        return (gates * values).sum(dim=1)         # (B, D)


class AttentionFusion(nn.Module):
    """Cross-attention fusion: use a learnable query to attend over branch outputs."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        B = branch_outputs[0].shape[0]
        kv = torch.stack(branch_outputs, dim=1)    # (B, num_branches, D)
        q = self.query.expand(B, -1, -1)           # (B, 1, D)
        out, _ = self.cross_attn(q, kv, kv)        # (B, 1, D)
        return self.norm(out.squeeze(1))            # (B, D)


class ConcatFusion(nn.Module):
    """Simple concat + linear projection."""

    def __init__(self, dim: int, num_branches: int):
        super().__init__()
        self.proj = nn.Linear(dim * num_branches, dim)

    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(branch_outputs, dim=-1))


class MixtureOfTransducers(nn.Module):
    """
    MoT: 3 branch transducers + fusion layer.

    Replaces the approach of concatenating all sequences into one long sequence.
    Each branch processes its own sequence independently, then fuses.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        merge: str = "gate",
        use_enhanced: bool = False,
        max_seq_lens: tuple = (200, 200, 200),
    ):
        super().__init__()
        self.action_branch = BranchTransducer(
            dim, num_heads, num_layers, dropout, use_enhanced, max_seq_lens[0],
        )
        self.content_branch = BranchTransducer(
            dim, num_heads, num_layers, dropout, use_enhanced, max_seq_lens[1],
        )
        self.item_branch = BranchTransducer(
            dim, num_heads, num_layers, dropout, use_enhanced, max_seq_lens[2],
        )

        if merge == "gate":
            self.fusion = GatedFusion(dim, 3)
        elif merge == "attention":
            self.fusion = AttentionFusion(dim, num_heads)
        else:
            self.fusion = ConcatFusion(dim, 3)

    def forward(
        self,
        action_tokens: torch.Tensor,    # (B, L_a, D)
        content_tokens: torch.Tensor,   # (B, L_c, D)
        item_tokens: torch.Tensor,      # (B, L_s, D)
    ) -> torch.Tensor:
        """
        [Fix3] 3分支用CUDA stream并行 (L=1000时省16ms/step)
        """
        if action_tokens.is_cuda:
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                h_a = self.action_branch(action_tokens)
            with torch.cuda.stream(s2):
                h_c = self.content_branch(content_tokens)
            h_s = self.item_branch(item_tokens)  # default stream
            torch.cuda.current_stream().wait_stream(s1)
            torch.cuda.current_stream().wait_stream(s2)
        else:
            h_a = self.action_branch(action_tokens)
            h_c = self.content_branch(content_tokens)
            h_s = self.item_branch(item_tokens)
        return self.fusion([h_a, h_c, h_s])
