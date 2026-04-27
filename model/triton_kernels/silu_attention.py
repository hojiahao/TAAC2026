"""
Triton kernel for SiLU Gated Attention (HSTU 2.0).

Fuses: Q@K^T + SiLU activation + masking + @V into a single tiled kernel.
The full (B, H, L, L) attention matrix is NEVER materialized in HBM.

Forward (Triton):
    out = silu(Q @ K^T * scale) * mask @ V

Backward (PyTorch):
    Recomputes attention weights and computes dq, dk, dv via standard matmuls.
    With gradient checkpointing, backward only runs for one layer at a time.
"""

import torch
import triton
import triton.language as tl
import math


# ═══════════════════════════════════════════════════════════════
# Forward Kernel
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _silu_attn_fwd_kernel(
    # Tensor pointers
    Q, K, V, Mask, Out,
    # Q strides: (B, H, L, D)
    stride_qb, stride_qh, stride_ql, stride_qd,
    # K strides: (B, H, L, D)
    stride_kb, stride_kh, stride_kl, stride_kd,
    # V strides: (B, H, L, D)
    stride_vb, stride_vh, stride_vl, stride_vd,
    # Mask strides: (B, L, L) — head dim squeezed
    stride_mb, stride_ml, stride_mn,
    # Output strides: (B, H, L, D)
    stride_ob, stride_oh, stride_ol, stride_od,
    # Dimensions
    n_ctx,
    d_head: tl.constexpr,
    scale,
    # Flags
    HAS_MASK: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)   # batch
    pid_h = tl.program_id(1)   # head
    pid_m = tl.program_id(2)   # query row block

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d_head)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    if HAS_MASK:
        m_base = Mask + pid_b * stride_mb

    # Load Q tile: (BLOCK_M, d_head)
    q_ptrs = q_base + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    # Output accumulator — float32 for precision
    acc = tl.zeros((BLOCK_M, d_head), dtype=tl.float32)

    # ── Tiled loop over K/V sequence dimension ──
    for start_n in range(0, n_ctx, BLOCK_N):
        curr_n = start_n + offs_n
        valid_n = curr_n < n_ctx

        # Load K^T tile: (d_head, BLOCK_N)
        # Indexed as [d, n] so tl.dot(q, k) = Q @ K^T
        k_ptrs = k_base + offs_d[:, None] * stride_kd + curr_n[None, :] * stride_kl
        k = tl.load(k_ptrs, mask=valid_n[None, :], other=0.0)

        # Q @ K^T: (BLOCK_M, d_head) @ (d_head, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k) * scale

        # SiLU: x * sigmoid(x)
        silu_qk = qk * tl.sigmoid(qk)

        # Apply multiplicative attention mask
        if HAS_MASK:
            m_ptrs = m_base + offs_m[:, None] * stride_ml + curr_n[None, :] * stride_mn
            m_valid = (offs_m[:, None] < n_ctx) & valid_n[None, :]
            mask_val = tl.load(m_ptrs, mask=m_valid, other=0.0)
            silu_qk = silu_qk * mask_val

        # Load V tile: (BLOCK_N, d_head)
        v_ptrs = v_base + curr_n[:, None] * stride_vl + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_N) @ (BLOCK_N, d_head) -> (BLOCK_M, d_head)
        acc += tl.dot(silu_qk.to(v.dtype), v)

    # Store output (float32 acc -> bf16 output, auto cast)
    o_ptrs = o_base + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < n_ctx)


# ═══════════════════════════════════════════════════════════════
# Autotuned Launchers
# ═══════════════════════════════════════════════════════════════

_FWD_CONFIGS = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_warps=nw)
    for BM, BN, nw in [
        (32, 32, 4), (32, 64, 4), (64, 32, 4), (64, 64, 4),
        (64, 64, 8), (64, 128, 8), (128, 32, 4), (128, 64, 8),
    ]
]


def _launch_fwd(q, k, v, mask, scale, BLOCK_M=64, BLOCK_N=64):
    """Launch the forward SiLU attention kernel."""
    B, H, L, D = q.shape
    HAS_MASK = mask is not None

    out = torch.empty_like(q)

    if HAS_MASK:
        mask_3d = mask.squeeze(1) if mask.dim() == 4 else mask
        mb, ml, mn = mask_3d.stride()
    else:
        mask_3d = torch.empty(1, device=q.device)
        mb, ml, mn = 0, 0, 0

    grid = (B, H, triton.cdiv(L, BLOCK_M))

    _silu_attn_fwd_kernel[grid](
        q, k, v, mask_3d, out,
        *q.stride(), *k.stride(), *v.stride(),
        mb, ml, mn,
        *out.stride(),
        n_ctx=L, d_head=D, scale=scale,
        HAS_MASK=HAS_MASK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Autograd Wrapper
# ═══════════════════════════════════════════════════════════════

class _TritonSiLUAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        ctx.save_for_backward(q, k, v)
        ctx.mask = mask
        ctx.scale = scale
        return _launch_fwd(q, k, v, mask, scale)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        mask = ctx.mask
        scale = ctx.scale

        # Recompute attention weights (PyTorch — one layer at a time via grad checkpointing)
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale
        sig = torch.sigmoid(qk)
        p = qk * sig  # silu(qk)
        if mask is not None:
            p = p * mask

        # dV = P^T @ dO
        dv = torch.matmul(p.transpose(-2, -1).to(v.dtype), grad_output)

        # dP = dO @ V^T
        dp = torch.matmul(grad_output, v.transpose(-2, -1))

        # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig_prime = sig * (1.0 + qk * (1.0 - sig))
        dqk = dp * sig_prime * scale
        if mask is not None:
            dqk = dqk * mask

        # dQ = dQK @ K,  dK = Q^T @ dQK
        dq = torch.matmul(dqk.to(q.dtype), k)
        dk = torch.matmul(q.transpose(-2, -1), dqk.to(k.dtype))

        return dq, dk, dv, None, None


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def triton_silu_attention(q, k, v, mask=None, scale=1.0):
    """
    Memory-efficient SiLU attention using Triton.

    Forward never materializes the (B, H, L, L) attention matrix in HBM.

    Args:
        q: (B, H, L, D) query tensor
        k: (B, H, L, D) key tensor
        v: (B, H, L, D) value tensor
        mask: (B, 1, L, L) multiplicative attention mask, or None
        scale: 1/sqrt(d_head)

    Returns:
        out: (B, H, L, D) attention output
    """
    return _TritonSiLUAttnFunc.apply(q, k, v, mask, scale)
