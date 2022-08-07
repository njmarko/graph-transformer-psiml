import torch
from torch_geometric.nn import GATv2Conv
from other_classes import *

class GATTransformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        # print(depth)
        self.layers = nn.ModuleList()
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, QuickFix(dim, heads, GATv2Conv(in_channels=dim, out_channels=dim, heads=heads, add_self_loops=False, dropout=0.1)))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))


    def forward(self, data):
        x, edge_index = data

        for attn, ff in self.layers:
            x = attn(x=x, edge_index=edge_index)
            x = ff(x)

        return x

