import torch
from torch import nn, Tensor
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_size: int,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_encoding = torch.zeros(size=(context_size, embed_dim))
        pos = torch.arange(0, context_size).unsqueeze(dim=1)
        dim = torch.arange(0, embed_dim, 2)

        sinusoid_term = pos / (10000 ** (dim / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(sinusoid_term)  # even
        pos_encoding[:, 1::2] = torch.cos(sinusoid_term)  # odd
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(dim=1)
        x = x + self.pos_encoding[0:seq_len, :]
        return self.dropout(x)


class Attention(nn.Module):
    """This class won't be used in the final implementation."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.Wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        div_term = self.embed_dim**0.5
        attn_matrix = torch.matmul(queries, keys.transpose(-1, -2)) / div_term
        attn_matrix = F.softmax(input=attn_matrix, dim=-1)
        print("Attention matrix shape --->", attn_matrix.shape)
        context_tensor = torch.matmul(attn_matrix, values)
        return context_tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Dimension of the model should be divisible by number of head."
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.constant_(tensor=self.q.bias, val=0)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.constant_(tensor=self.k.bias, val=0)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.constant_(tensor=self.v.bias, val=0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(tensor=self.out.bias, val=0)
    
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Tensor | None = None
    ) -> Tensor:
        previous_shape = query.size()  # [batch_size, seq_len, embed_dim]
        # reshape w.r.t. number of heads
        new_shape = (
            previous_shape[0],  # batch_size
            previous_shape[1],  # seq_len
            self.num_heads,
            self.head_dim
        )
        queries = self.q(query).view(*new_shape).transpose(1, 2)
        keys = self.k(key).view(*new_shape).transpose(1, 2)
        values = self.v(value).view(*new_shape).transpose(1, 2)
        # current shape: [B, NUM_HEAD, SEQ_LEN, HEAD_DIM]
        div_term = self.head_dim ** 0.5
        attn_logits = torch.matmul(
            input=queries,
            other=keys.transpose(-1, -2)
        ) / div_term
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float("-inf"))
        attn_matrix = F.softmax(input=attn_logits, dim=-1)
        context_tensor = torch.matmul(attn_matrix, values)
        # reshape back to [B, SEQ_LEN, NUM_HEAD, HEAD_DIM]
        context_tensor = context_tensor.transpose(1, 2).contiguous()
        # reshape back to [B, SEQ_LEN, D_MODEL] where D_MODEL = NUM_HEAD * HEAD_DIM
        context_tensor = context_tensor.view(previous_shape)
        return self.out(context_tensor)


class  PositionwiseFFNetwork(nn.Module):
    def __init__(self, embed_dim: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(tensor=self.linear1.bias, val=0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(tensor=self.linear2.bias, val=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x