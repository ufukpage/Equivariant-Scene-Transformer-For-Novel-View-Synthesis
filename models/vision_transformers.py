import torch.nn as nn
from einops import rearrange
import torch

from vit_pytorch.efficient import ViT
#from vit_pytorch import ViT
from positional_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D, PositionalEncodingPermute3D


class ViewEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(ViewEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class FixedEmbedding(nn.Module):
    def __init__(self, dim):
        super(FixedEmbedding, self).__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim = 1, offset = 0):
        t = torch.arange(x.shape[seq_dim], device = x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class ViTransformer2DEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        transformer,
        dim=None,
        use_embeddings=True,
        pos_emd=False,
        channels=3
    ):
        super(ViTransformer2DEncoder, self).__init__()
        # super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = transformer.dim if dim is None else dim

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_to_embedding = nn.Linear(patch_dim, dim) if use_embeddings else None
        self.patch_size = patch_size
        # self.pia_pos_emb = Fixed3DPositionalEmbedding(dim)
        self.transformer = transformer

        if pos_emd:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()
        self.use_embeddings = use_embeddings

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        if self.use_embeddings:
            x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        if hasattr(self, 'pos_embedding'):
            x += self.pos_embedding[:, :n]

        x = self.transformer(x)
        x = self.norm(x)
        x = self.to_latent(x)
        # x = x.mean(dim=1)
        return x # torch.transpose(x, 1, 2)


class ViTransformer3DEncoder(nn.Module):
    def __init__(
        self,
        volume_size,
        patch_size,
        transformer,
        dim=None,
        pos_emd=False,
        channels=None,
        use_embeddings=True
    ):
        # super(ViTransformer3DEncoder, self).__init__(image_size=)
        super(ViTransformer3DEncoder, self).__init__()
        channels = volume_size if channels is None else channels
        dim = transformer.dim if dim is None else dim
        num_patches = (volume_size // patch_size) ** 3
        patch_dim = channels * patch_size ** 3
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        if pos_emd:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.use_embeddings = use_embeddings
        self.transformer = transformer
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()

    def forward(self, img, view_embedding=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
        # if self.use_embeddings:

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        if hasattr(self, 'pos_embedding'):
            x += self.pos_embedding[:, :n]

        if view_embedding:
            x += view_embedding

        x = self.transformer(x)
        x = self.norm(x)
        x = self.to_latent(x)
        # x = x.mean(dim=1)
        return x # torch.transpose(x, 1, 2)


from x_transformers import ViTransformerWrapper


class ViTransformer2DEncoderWrapper(ViTransformerWrapper):
    def __init__(
        self,
        image_size,
        patch_size,
        attn_layers,
        emb_dropout=0.,
        channel_size=3
    ):
        super(ViTransformer2DEncoderWrapper, self).__init__(image_size=image_size, patch_size=patch_size,
                                                     attn_layers=attn_layers, emb_dropout=emb_dropout)
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel_size * patch_size ** 2
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.patch_size = patch_size
        # self.pia_pos_emb = Fixed3DPositionalEmbedding(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        return x


class ViTransformer3DEncoderWrapper(ViTransformerWrapper):
    def __init__(
        self,
        volume_size,
        patch_size,
        attn_layers,
        emb_dropout=0.,
        use_embeddings=True
    ):
        super(ViTransformer3DEncoderWrapper, self).__init__(image_size=volume_size, patch_size=patch_size,
                                                     attn_layers=attn_layers, emb_dropout=emb_dropout)
        dim = attn_layers.dim
        num_patches = (volume_size // patch_size) ** 3
        patch_dim = volume_size * patch_size ** 3
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.use_embeddings = use_embeddings

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
        if self.use_embeddings:
            x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        return x


from x_transformers import ViTransformerWrapper

if "__main__" == __name__:
    embed = ViewEmbedding(2, 64)
    x = torch.FloatTensor([12, 68])
    y = embed(x)
    print(y.shape)