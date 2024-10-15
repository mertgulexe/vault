import torch
import torch.nn as nn
import torch.nn.functional as F
from components.activations import SiGLU

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff):
        super().__init__()
        # TODO: instantiate 3 linear layers
        self.W1 = nn.Linear(hidden_size, d_ff)
        self.W2 = nn.Linear(hidden_size, d_ff)
        self.W3 = nn.Linear(d_ff, hidden_size)
        self.siglu = SiGLU(d_ff, d_ff)  # a SiLU or SiGLU activation function could be used as specified in the homework.

    def forward(self, x) -> torch.Tensor:
        # TODO: implement the expert logic
        x = self.W1(x) * self.W2(x)  # shape: [batch_size, context_length, d_ff]
        # x = F.silu(x)  # shape: [batch_size, context_length, d_ff]
        x = self.siglu(x)  # shape: [batch_size, context_length, d_ff]
        x = self.W3(x)  # shape: [batch_size, context_length, hidden_size]
        return x


class MoeLayer(nn.Module):
    def __init__(self, hidden_size, d_ff, num_experts, n_experts_per_token):
        super().__init__()

        self.num_experts = num_experts
        self.n_experts_per_token = n_experts_per_token

        # TODO: instantiate the experts and the gate
        self.experts = nn.ModuleList(
            [FeedForward(hidden_size, d_ff) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # TODO: pass the input x to the gate
        gate_output = self.gate(x)  # shape: [batch_size, seq_length, num_experts]
        # TODO: use torch.topk to get the topk values and indexes
        topk_vals, topk_indices = torch.topk(
            input=gate_output,
            k=self.n_experts_per_token,
            dim=-1
        )
        # TODO: pass the topk values to F.softmax to get the weights for each expert
        topk_weights = F.softmax(input=topk_vals, dim=-1)  # shape: [batch_size, seq_length, n_experts_per_token]
        # initiate the output tensor
        out = torch.zeros_like(input=x)
        for i, expert in enumerate(self.experts):
            # TODO: find the indexes of the hidden states that should be routed to the current expert
            batch_idx, token_idx, topk_idx = torch.where(
                condition=(topk_indices == i)
            )
            # TODO: update the out tensor
            out[batch_idx, token_idx] += torch.mul(
                input=topk_weights[batch_idx, token_idx, topk_idx, None],
                other=expert(x[batch_idx, token_idx])
            )
        return out


# My confusion about the line,
# other=expert(x[batch_idx, token_idx])
# is addressed by ChatGPT-4o:

# """
# Role of gate_output:
# The gate_output represents the raw logits produced by the gating network.
# It is used to decide which experts to activate and with what weight for each
# token in the sequence. It does not directly contain the feature data that
# needs to be processed by the experts.

# Purpose of Experts:
# The experts are specialized feed-forward neural networks that
# transform input data. For the MoE model to work correctly,
# the selected experts must process the original input feature data (x).
# The gating network only decides which experts to use and with
# what importance, but it does not modify the data itself.
# """