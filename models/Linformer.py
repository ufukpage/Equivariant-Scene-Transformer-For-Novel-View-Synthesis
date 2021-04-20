import torch.nn as nn
from linformer import Linformer

model = Linformer(
    dim = 512,
    seq_len = 4096,
    depth = 12,
    heads = 8,
    k = 256,
    one_kv_head = True,
    share_kv = True
)


class Linformer2DEncoder(Linformer):
    def __init__(self, image_size, patch_size, dim, depth=1, emb_dropout=0., channel_size=3, k=128,
                 one_kv_head=True, share_kv = True):
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel_size * patch_size ** 2
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.patch_size = patch_size
        super(Linformer2DEncoder, self).__init__(dim=dim, seq_len=num_patches, depth=depth, k=k, one_kv_head=one_kv_head
                                                 , share_kv=share_kv, dropout=emb_dropout)