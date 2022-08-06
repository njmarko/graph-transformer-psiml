import torch
from other_classes import *
from GATTransformer_class import GATTransformer
from einops import rearrange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.dim = dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = GATTransformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        # print(p)
        # print(img.shape)
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # print(x.shape)
        x = x.to(device)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        cls_tokens = cls_tokens.to(device)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x_shape = x.shape
        edge_index = torch.ones((2, x.shape[1] ** 2 * x.shape[0]), dtype=torch.long)
        for i in range(x.shape[0] * x.shape[1]):
            for j in range(x.shape[1]):
                edge_index[0, i * x.shape[1] + j] = i
                edge_index[1, i * x.shape[1] + j] = x.shape[1] * (i // x.shape[1]) + j
        x = x.view((x.shape[0] * x.shape[1], x.shape[2]))
        edge_index = edge_index.to(device)

        x = self.transformer([x, edge_index])
        x = x.view(x_shape)

        x = self.to_cls_token(x[:, 0])
        x = x.to(device)

        out = self.mlp_head(x)

        return out
