import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.autograd.function import FunctionCtx
import os

_ext = load(
    name="flash_attention_kernel",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float,
        Br: int,
        Bc: int
    ) -> torch.Tensor:
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        O, L = _ext.forward(Q, K, V, scale, Br, Bc)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.Br = Br
        ctx.Bc = Bc
        return O
    
    @staticmethod
    def backward(ctx: FunctionCtx, dO: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None
    ]:
        Q, K, V, O, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = _ext.backward(dO, Q, K, V, O, L, ctx.scale, ctx.Br, ctx.Bc)
        return dQ, dK, dV, None, None, None

class FlashAttentionModule(nn.Module):
    def __init__(self, Br: int, Bc: int) -> None:
        super().__init__()
        self.Br = Br
        self.Bc = Bc

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, S, d = K.shape
        assert K.shape == (B, H, S, d), [K.shape, (B, H, S, d)]
        assert V.shape == (B, H, S, d), [V.shape, (B, H, S, d)]
        assert Q.shape == (B, H, S, d), [Q.shape, (B, H, S, d)]

        O = FlashAttention.apply(Q, K, V, d ** -0.5, self.Br, self.Bc)
        assert O.shape == (B, H, S, d), [O.shape, (B, H, S, d)]
        return O
