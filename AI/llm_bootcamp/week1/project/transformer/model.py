import torch.nn as nn

from transformer.blocks import TransformerBlock
from components.rope import get_rotation_matrix


class Transformer(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            hidden_size, 
            num_heads, 
            window_size, 
            d_ff, 
            num_experts, 
            n_experts_per_token, 
            n_blocks,
            max_seq_len
        ):

        super().__init__()

        head_dim = hidden_size // num_heads
        period = 10000.0
        self.rotation_matrix = get_rotation_matrix(head_dim, max_seq_len, period)

        # TODO: instantiate the components
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=hidden_size
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                d_ff=d_ff,
                num_experts=num_experts,
                n_experts_per_token=n_experts_per_token,
                rotation_matrix=self.rotation_matrix
            ) for _ in range(n_blocks)
        ])
        self.out = nn.Linear(
            in_features=hidden_size,
            out_features=vocabulary_size
        )

    def forward(self, x):
        # TODO: implement for the forward method
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x
        