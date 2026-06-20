import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, N: int, K: int, D: int, epsilon: float = 1e-5, alpha: float = 1e-2, activation_cls: type[nn.Module] = nn.SiLU):
        super().__init__()
        self.N = N
        self.D = D
        self.E = nn.Linear(D, N * D)
        self.experts = nn.ModuleList([nn.Linear(D, D) for _ in range(N)])
        self.activation = activation_cls()
        self.W = nn.Linear(D, N)
        self.epsilon = epsilon
        self.K = K
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Expert function E: [n] -> R^d -> R^d
        # Weight function G: R^d -> R^n; G(x) = topk(σ(Wx + ε), k)
        # out = Σi G(x)i * Ei(x)
        B, _ = x.shape
        assert x.shape == (B, self.D), [x.shape, (B, self.D)]
        Ex = self.activation(self.E(x).view(-1, self.N, self.D))
        Exx = torch.cat([self.experts[i](Ex[:, i, :]).unsqueeze(1) for i in range(self.N)], dim=1)
        Wx = self.W(x)
        noise = self.epsilon * torch.randn_like(Wx) if self.training else 0
        Wx_w_noise = Wx + noise
        p = F.softmax(Wx_w_noise, dim=-1)
        _, topk_indices = torch.topk(p, k=self.K)
        mask = torch.zeros_like(Wx_w_noise).scatter_(-1, topk_indices, 1)
        f = torch.sum(mask, dim=0) / B
        Gx_unnormalized = p * mask
        Gx = Gx_unnormalized / torch.sum(Gx_unnormalized, dim=-1, keepdim=True)
        out = torch.einsum("bn,bnd->bd", Gx, Exx)
        router_loss = self.alpha * self.N * torch.sum(f * p.mean(dim=0))
        return out, router_loss
    
if __name__ == "__main__":
    torch.manual_seed(42)
    B = 256
    D = 10
    N, K = 4, 3
    moe = MoELayer(N=N, K=K, D=D)
    x = torch.randn(B, D)
    print(moe(x))
