import torch.nn as nn
from linformer import Linformer, LinformerSelfAttention, LinformerLM
from models.vision_transformers import RelativePositionBias, FixedPositionalEmbedding
import torch


class LinformerSelfAttentionRelPos(LinformerSelfAttention):
    def __init__(self, dim, seq_len, k=256, rel_pos=False, sinusoidal_emb=False, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super(LinformerSelfAttentionRelPos, self).__init__(dim, seq_len, k=k, heads=heads, dim_head=dim_head,
                                                           one_kv_head=one_kv_head, share_kv=share_kv, dropout=dropout)
        if rel_pos:
            self.rel_pos = RelativePositionBias(heads=heads)

        if sinusoidal_emb:
            self.pia_pos_emb = FixedPositionalEmbedding(dim)


    def forward(self, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)

        if self.rel_pos:
            dots = self.rel_pos(dots)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Linformer2DEncoder(Linformer):
    def __init__(self, image_size, patch_size, dim, depth=1, emb_dropout=0., channel_size=3, k=128,
                 one_kv_head=True, share_kv = True):
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel_size * patch_size ** 2
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.patch_size = patch_size
        # self.rel_pos = RelativePositionBias(heads=1) if rel_pos_bias else None
        super(Linformer2DEncoder, self).__init__(dim=dim, seq_len=num_patches, depth=depth, k=k, one_kv_head=one_kv_head
                                                 , share_kv=share_kv, dropout=emb_dropout, num_tokens=num_patches)
        self.token_emb = None

    def forward(self, x):
        x = self.pos_emb(torch.arange(x.shape[1])) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out
