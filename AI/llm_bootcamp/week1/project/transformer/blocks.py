import torch.nn as nn

from components.norm_layers import RMSNorm
from components.attentions import EfficientSlidingWindowMultiheadAttention
from components.moe import MoeLayer


class TransformerBlock(nn.Module):
    def __init__(
          self, 
          hidden_size, 
          num_heads, 
          window_size, 
          d_ff, 
          num_experts, 
          n_experts_per_token,
          rotation_matrix
        ) -> None:
        super().__init__()

        # TODO: instantiate the different components
        self.rms_normalization_1 = RMSNorm(hidden_size=hidden_size)
        self.attention = EfficientSlidingWindowMultiheadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            rotation_matrix=rotation_matrix
        )
        self.rms_normalization_2 = RMSNorm(hidden_size=hidden_size)
        self.ff = MoeLayer(
            hidden_size=hidden_size,
            d_ff=d_ff,
            num_experts=num_experts,
            n_experts_per_token=n_experts_per_token
        )

    def forward(self, x):
        # TODO: implement for the forward logic
        x1 = self.rms_normalization_1(x)
        x1 = self.attention(x1)
        x = x + x1
        x2 = self.rms_normalization_2(x)
        x2 = self.ff(x2)
        return x2 + x
