"""Triton kernels for HSTU-style SiLU Gated Attention."""

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from .silu_attention import triton_silu_attention
