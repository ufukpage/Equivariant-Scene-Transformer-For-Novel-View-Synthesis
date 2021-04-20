import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.module import Attention, PreNorm, FeedForward
import numpy as np

# from hamburger_pytorch import Hamburger

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super(ViViT, self).__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


class DeViT(nn.Module):
    def __init__(self, volume_size, patch_size, depth_size, dim=192, depth=4, heads=3,
                 in_channels=3, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super(DeViT, self).__init__()

        assert volume_size % patch_size == 0, 'Volume dimensions must be divisible by the patch size.'
        in_channels = volume_size if in_channels is None else in_channels
        self.volume_size = volume_size
        num_patches = (volume_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, depth_size, num_patches, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.depth_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c d (h p1) (w p2) -> b d (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.to_patch_embedding(x)
        b, d, n, _ = x.shape

        x += self.pos_embedding[:, :, :n]     # self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b de n d -> (b de) n d')
        x = self.space_transformer(x)
        # x = rearrange(x[: 0], '(b de) ... -> b de ...', b=b)

        x = self.depth_transformer(x)
        x = self.norm(x)
        return rearrange(x, '(b de) (h w) d -> b (de h w) d', b=b, h=self.volume_size )


if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    # model = ViViT(224, 16, 100, 16).cuda()
    model = DeViT(volume_size=32, patch_size=16, depth_size=16, depth=1, heads=1, dim=64).cuda()
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    # print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    
    