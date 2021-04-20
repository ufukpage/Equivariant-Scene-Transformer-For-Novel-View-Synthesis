from linear_attention_transformer.images import ImageLinearAttention

attn =ImageLinearAttention(
  chan = 32,
  heads = 8,
  key_dim = 64       # can be decreased to 32 for more memory savings
)