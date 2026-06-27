import torch
import torch.nn as nn
import torch.nn.functional as F

class MHATorch(nn.Module):
    def __init__(self, D: int, H: int) -> None:
        super().__init__()
        assert D % H == 0, [D, H]
        self.D = D
        self.H = H
        self.d = D // H
        self.W = nn.Linear(D, 3*D)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        kqv = self.W(x)

        k, q, v = kqv.split(self.D, dim=-1)
        q = q.view(B, S, self.H, self.d).transpose(1, 2)
        k = k.view(B, S, self.H, self.d).transpose(1, 2)
        v = v.view(B, S, self.H, self.d).transpose(1, 2)

        return F.scaled_dot_product_attention(
            q, k, v
        ).transpose(1, 2).view(B, S, self.D)
