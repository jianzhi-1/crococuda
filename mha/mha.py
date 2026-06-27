import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, D: int, H: int) -> None:
        assert D % H == 0, [D, H]
        super().__init__()
        self.D = D
        self.H = H
        self.d = D // H
        self.Wk = nn.Linear(D, D)
        self.Wq = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        assert x.shape == (B, S, self.D), [x.shape, (B, S, self.D)]
        K = self.Wk(x).view(B, S, self.H, self.d)
        Q = self.Wq(x).view(B, S, self.H, self.d)
        V = self.Wv(x).view(B, S, self.H, self.d)
        a = F.softmax(torch.einsum("bshd,bthd->bhst", Q, K) / (self.d ** 0.5), dim=-1)
        out = torch.einsum("bhst,bthd->bshd", a, V)
        return out.reshape(B, S, self.D)
    

def test_harness(seed: int) -> bool:
    import numpy as np
    rng = np.random.default_rng(seed)
    B = 1
    S = 3
    H = 2
    d = 2
    D = H*d

    mha = MHA(D, H)
    wk = [rng.standard_normal((d, d)).astype(np.float32) for _ in range(H)]
    wq = [rng.standard_normal((d, d)).astype(np.float32) for _ in range(H)]
    wv = [rng.standard_normal((d, d)).astype(np.float32) for _ in range(H)]
    with torch.no_grad():
        mha.Wk.weight.zero_()
        mha.Wq.weight.zero_()
        mha.Wv.weight.zero_()
        mha.Wk.bias.zero_()
        mha.Wq.bias.zero_()
        mha.Wv.bias.zero_()
        for i in range(H):
            s, e = i*d, (i + 1)*d
            mha.Wk.weight[s:e, s:e].copy_(torch.as_tensor(wk[i]).T)
            mha.Wq.weight[s:e, s:e].copy_(torch.as_tensor(wq[i]).T)
            mha.Wv.weight[s:e, s:e].copy_(torch.as_tensor(wv[i]).T)
            

    x = rng.standard_normal((B, S, H, d)).astype(np.float32)

    ks = [np.einsum("bsd,dl->bsl", x[:,:,i,:], wk[i]) for i in range(H)]
    vs = [np.einsum("bsd,dl->bsl", x[:,:,i,:], wv[i]) for i in range(H)]
    qs = [np.einsum("bsd,dl->bsl", x[:,:,i,:], wq[i]) for i in range(H)]

    a_unnormalized = [np.einsum("bsl,btl->bst", q, k) / (d ** 0.5) for q, k in zip(qs, ks, strict=True)]

    ats = [np.exp(a_unnorm) / np.sum(np.exp(a_unnorm), axis=-1, keepdims=True) for a_unnorm in a_unnormalized]

    out_uncat = [np.expand_dims(np.einsum("bst,btd->bsd", at, v), axis=2) for at, v in zip(ats, vs, strict=True)]
    out_np = torch.as_tensor(np.concat(tuple(out_uncat), axis=2)).reshape(B, S, D)

    out_torch = mha(torch.as_tensor(x.reshape(B, S, D)))
    assert torch.allclose(out_torch, torch.as_tensor(out_np), atol=2e-7), (out_torch - out_np).abs().max().item()
    return True


if __name__ == "__main__":
    for seed in range(10):
        assert test_harness(seed), f"test {seed} failed"
    print("PASSED")










