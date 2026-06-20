from torch.utils.cpp_extension import load
moe_kernels = load(name="moe_kernels", sources=["./moe/experts_kernel.cu"], verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_naive import MoELayer

class Expert(nn.Module):
    def __init__(self, D: int, activation_cls: type[nn.Module]) -> None:
        super().__init__()
        self.D = D
        self.activation = activation_cls()
        self.W1 = nn.Linear(D, D)
        self.W2 = nn.Linear(D, D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        return self.W2(self.activation(self.W1(x)))


class MoEKernelised(nn.Module):
    def __init__(self, D: int, N: int, K: int, epsilon: float = 1e-5, activation_cls: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.D = D
        self.N = N
        self.K = K
        self.epsilon = epsilon
        self.Wg = nn.Linear(D, N)
        self.experts = nn.ModuleList([Expert(D, activation_cls) for _ in range(N)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        device, dtype = x.device, x.dtype
        
        Wgx = self.Wg(x)
        noise = self.epsilon * torch.randn_like(Wgx) if self.training else 0
        unnormalised_weights, topk_indices = torch.topk(F.softmax(Wgx + noise, dim=-1), self.K)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        weights = unnormalised_weights / unnormalised_weights.sum(dim=-1, keepdim=True)
        expert_partitions_to_x = torch.zeros((B * self.K, self.D), device=device, dtype=dtype)
        expert_partitions_to_x_idx = torch.zeros((B * self.K, ), device=device, dtype=torch.int32)
        expert_partitions_to_weight = torch.zeros((B * self.K,), device=device, dtype=dtype)
        per_expert_offsets = torch.zeros((self.N + 1, ), device=device, dtype=torch.uint32)
        # compute gate
        moe_kernels.partition_by_expert(topk_indices, x, weights, expert_partitions_to_x, expert_partitions_to_x_idx, expert_partitions_to_weight, per_expert_offsets, self.N)
        # mat mul
        expert_partitions_to_output = torch.zeros((B * self.K, self.D), device=device, dtype=dtype)

        offsets = per_expert_offsets.cpu()
        for i in range(self.N):
            s, t = offsets[i].item(), offsets[i + 1].item()
            if s == t: continue
            expert_partitions_to_output[s:t, :] = self.experts[i](expert_partitions_to_x[s:t,:])
        out = torch.zeros((B, self.D), device=device, dtype=dtype)

        moe_kernels.reduce_from_expert_partitions(expert_partitions_to_output, expert_partitions_to_x_idx, expert_partitions_to_weight, out)
        return out
    
if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    B = 256
    D = 10
    N, K = 4, 3
    with torch.device("cuda"):
        x = torch.randn(B, D)
        moe = MoELayer(N=N, K=K, D=D)
        moe_kernelized = MoEKernelised(D=D, N=N, K=K).eval()
        out_ref = moe(x)
        out_kernel = moe_kernelized(x)
    max_abs_err = (out_kernel - out_ref).abs().max().item()
    assert torch.all_close(out_kernel, out_ref, atol=1e-5, rtol=1e-4), f"{max_abs_err:.3e}"
    print("PASS")
