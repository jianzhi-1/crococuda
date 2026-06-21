import torch
import torch.nn as nn
import torch.nn.functional as F

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
        weights = unnormalized_weights / unnormalized_weights.sum(axis=-1, keepdim=True)

        # Weighted expert sum
        per_expert_out: list[torch.Tensor] = []
        for b in range(B):
            cur_out = x.new_zeros((1, self.D))
            for idx, k in enumerate(indices[b]):
                cur_out += weights[b, idx] * self.experts[k.item()](x[b:(b+1)])
            per_expert_out.append(cur_out)
        out = torch.cat(per_expert_out, dim=0)
        return out
