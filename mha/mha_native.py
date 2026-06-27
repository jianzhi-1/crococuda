import torch
import torch.nn as nn
import torch.nn.functional as F

class MHANative(nn.Module):
    def __init__(self, D: int, H: int) -> None:
        super().__init__()
        assert D % H == 0, [D, H]
        self.D = D
        self.H = H
        self.d = D // H
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        K = self.Wk(x).view(B, S, self.H, self.d).transpose(1, 2).contiguous()
        Q = self.Wq(x).view(B, S, self.H, self.d).transpose(1, 2).contiguous()
        V = self.Wv(x).view(B, S, self.H, self.d).transpose(1, 2).contiguous()

        qkt = Q @ K.transpose(-1, -2)
        qkt = qkt / (self.d ** 0.5)
        att = F.softmax(qkt, dim=-1)
        out = (att @ V).transpose(1, 2)
        return out.reshape(B, S, self.D)
