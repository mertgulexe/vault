import torch

BATCH_SIZE = 4
D_MODEL = 512
VOCAB_SIZE = 8
MAX_LEN = 256
NUM_HEAD = 8

# from components import PositionalEncoding
# pp = PositionalEncoding(d_model=D_MODEL, context_size=MAX_LEN)
# x_enc = pp(torch.rand((2, 32, 512)))
# print(x_enc)
# print(x_enc.size())

# from components import Attention
# aa = Attention(d_model=D_MODEL)
# x_attn = aa(
#     torch.rand(size=(BATCH_SIZE, VOCAB_SIZE, D_MODEL))
# )
# print(x_attn)
# print(x_attn.size())