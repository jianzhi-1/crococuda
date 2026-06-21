import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

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

class MoERef(nn.Module):
    """
    Module only used for eval and in CPU mode.
    """

    def __init__(self, D: int, N: int, K: int, activation_cls: type[nn.Module], sigma: float) -> None:
        assert K <= N, [K, N]
        super().__init__()
        self.D = D
        self.N = N
        self.K = K
        self.experts = nn.ModuleList([Expert(D, activation_cls) for _ in range(N)])
        self.Wg = nn.Linear(D, N)
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]

        # Routing
        Wgx = self.Wg(x)
        noise = torch.randn_like(Wgx) * self.sigma if self.training else 0.
        prob = F.softmax(Wgx + noise, dim=-1)
        unnormalized_weights, indices = torch.topk(prob, self.K)
        weights = unnormalized_weights / unnormalized_weights.sum(dim=-1, keepdim=True)

        # Weighted expert sum
        per_expert_input: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for b in range(B):
            for idx, k in enumerate(indices[b]):
                per_expert_input[k.item()].append(
                    (b, x[b:(b+1)], weights[b, idx])
                )
        token_indices = [b for n in range(self.N) for b, _, _ in per_expert_input[n]]
        
        expert_output_list: list[torch.Tensor] = []
        weights_list: list[torch.Tensor] = []
        for n in range(self.N):
            if len(per_expert_input[n]) == 0: continue
            per_expert_out = self.experts[n](torch.cat([input for _, input, _ in per_expert_input[n]], dim=0))
            expert_output_list.append(per_expert_out)
            weights_list.extend([weight for _, _, weight in per_expert_input[n]])
        
        token_indices_tensor = torch.tensor(token_indices, device=x.device, dtype=torch.long)
        expert_outputs = torch.cat(expert_output_list, dim=0)
        expert_weights = torch.stack(weights_list).unsqueeze(1)
        out = x.new_zeros((B, self.D)).index_add_(0, token_indices_tensor, expert_weights * expert_outputs)
        return out