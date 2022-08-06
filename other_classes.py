import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class QuickFix(nn.Module):
    def __init__(self, dim, heads, fn):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.linear = nn.Linear(dim * heads, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.linear(self.fn(x, **kwargs))
