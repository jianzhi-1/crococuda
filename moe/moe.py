import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, n_experts: int, k: int, d_model: int, epsilon: float = 1e-5, alpha: float = 1e-2):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.E = nn.Linear(d_model, n_experts * d_model)
        self.W = nn.Linear(d_model, n_experts)
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Expert function E: [n] -> R^d -> R^d
        # Weight function G: R^d -> R^n; G(x) = topk(σ(Wx + ε), k)
        # out = Σi G(x)i * Ei(x)
        batch_size, _ = x.shape
        assert x.shape == (batch_size, self.d_model), [x.shape, (batch_size, self.d_model)]
        Ex = self.E(x).view(-1, self.n_experts, self.d_model)
        Wx = self.W(x) + self.epsilon
        p = F.softmax(Wx, dim=-1)
        _, topk_indices = torch.topk(p, k=self.k)
        mask = torch.zeros_like(Wx).scatter_(-1, topk_indices, 1)
        f = torch.sum(mask, dim=0) / batch_size
        Gx_unnormalized = p * mask
        Gx = Gx_unnormalized / torch.sum(Gx_unnormalized, dim=-1, keepdim=True)
        out = torch.einsum("bn,bnd->bd", Gx, Ex)
        router_loss = self.alpha * self.n_experts * torch.sum(f * p.mean(dim=0))
        return out, router_loss
    
if __name__ == "__main__":
    batch_size = 256
    d_model = 10
    n_experts, k = 4, 3
    moe = MoELayer(n_experts=n_experts, k=k, d_model=d_model)
    x = torch.randn(batch_size, d_model)
    print(moe(x))
