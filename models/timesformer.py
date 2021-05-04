import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(ScaleNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        n = torch.norm(x, dim = -1, keepdim = True).clamp(min = self.eps)
        return x / n * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = ScaleNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

from opt_einsum import contract
from entmax import entmax15

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # attention
        sim = contract('b i d, b j d -> b i j', q, k, backend='torch')
        attn = entmax15(sim, dim=-1)  # sim.softmax(dim=-1)
        out = contract('b i j, b j d -> b i d', attn, v, backend='torch')

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)


class DepthFormer(nn.Module):
    def __init__(self, image_size, patch_size, volume_depth, dim=192, depth=4, heads=3, channels=3, dim_head=64, attn_dropout = 0., ff_dropout = 0.):
        super(DepthFormer, self).__init__()

        num_patches = (image_size // patch_size) ** 2
        num_positions = volume_depth * num_patches
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            ScaleNorm(dim)
        )

    def forward(self, volume_feature):
        b, _, d, h, w, *_, device, p = *volume_feature.shape, volume_feature.device, self.patch_size
        # assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        volume = rearrange(volume_feature, 'b c d (h p1) (w p2) -> b (d h w) (p1 p2 c)', p1=p, p2=p)
        x = self.to_patch_embedding(volume)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (depth_attn, spatial_attn, ff) in self.layers:
            x = depth_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=d) + x
            x = ff(x) + x

        return x #self.to_out(x)


if __name__ == "__main__":
    model = DepthFormer(image_size=32, patch_size=1, volume_depth=32, channels=32, dim=64, depth=1).cuda()
    video = torch.randn(2, 32, 32, 32, 32).cuda() # (batch x  channels x depth x height x width)
    pred = model(video)  # (2, 10)
    print(pred.shape)