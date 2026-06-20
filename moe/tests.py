from torch.utils.cpp_extension import load
moe_kernels = load(name="moe_kernels", sources=["./moe/experts_kernel.cu"], verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_naive import MoELayer

class Expert(nn.Module):
    def __init__(self, d_model: int, activation_cls: type[nn.Module]) -> None:
        super().__init__()
        self.d_model = d_model
        self.activation = activation_cls()
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape
        assert x.shape == (batch_size, self.d_model), [x.shape, (batch_size, self.d_model)]
        return self.W2(self.activation(self.W1(x)))


class MoEKernelised(nn.Module):
    def __init__(self, d_model: int, N: int, K: int, epsilon: float = 1e-5, activation_cls: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.K = K
        self.epsilon = epsilon
        self.Wg = nn.Linear(d_model, N)
        self.experts = nn.ModuleList([Expert(d_model, activation_cls) for _ in range(N)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape
        device = x.device
        dtype = x.dtype
        assert x.shape == (batch_size, self.d_model), [x.shape, (batch_size, self.d_model)]
        Wgx = self.Wg(x)
        noise = self.epsilon * torch.randn_like(Wgx) if self.training else 0
        unnormalised_gate_weights, expert_idx = torch.topk(F.softmax(Wgx + noise, dim=-1), self.K)
        expert_idx = expert_idx.to(torch.int32).contiguous()
        gate_weights = unnormalised_gate_weights / unnormalised_gate_weights.sum(dim=-1, keepdim=True)
        sorted_x = torch.zeros((batch_size * self.K, self.d_model), device=device, dtype=dtype)
        sorted_token_idx = torch.zeros((batch_size * self.K, ), device=device, dtype=torch.int32)
        sorted_gate = torch.zeros((batch_size * self.K,), device=device, dtype=dtype)
        expert_offsets = torch.zeros((self.N + 1, ), device=device, dtype=torch.uint32)
        # compute gate
        moe_kernels.gather(expert_idx, x, gate_weights, sorted_x, sorted_token_idx, sorted_gate, expert_offsets, self.N)
        # mat mul
        expert_out = torch.zeros((batch_size * self.K, self.d_model), device=device, dtype=dtype)

        offsets = expert_offsets.cpu()
        for i in range(self.N):
            s, t = offsets[i].item(), offsets[i + 1].item()
            if s == t: continue
            expert_out[s:t, :] = self.experts[i](sorted_x[s:t,:])
        out = torch.zeros((batch_size, self.d_model), device=device, dtype=dtype)

        moe_kernels.combine(expert_out, sorted_token_idx, sorted_gate, out)
        return out
    
if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    batch_size = 256
    d_model = 10
    n_experts, k = 4, 3
    with torch.device("cuda"):
        x = torch.randn(batch_size, d_model)
        moe = MoELayer(n_experts=n_experts, k=k, d_model=d_model)
        moe_kernelized = MoEKernelised(d_model=d_model, N=n_experts, K=k).eval()
        out_ref = moe(x)
        out_kernel = moe_kernelized(x)
    max_abs_err = (out_kernel - out_ref).abs().max().item()
    assert torch.all_close(out_kernel, out_ref, atol=1e-5, rtol=1e-4), f"{max_abs_err:.3e}"
    print("PASS")
