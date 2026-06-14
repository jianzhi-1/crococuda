import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from typing import override, Any
import os

silu = load(
    name="SiLU",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "silu.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class SiLUFunction(torch.autograd.Function):
    @override
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = silu.forward(x)
        ctx.save_for_backward(x)
        return y
    
    @override
    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_x = silu.backward(grad_y.contiguous(), x)
        return grad_x
    
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SiLUFunction.apply(x)