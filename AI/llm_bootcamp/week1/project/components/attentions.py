import torch
import torch.nn as nn
import torch.nn.functional as F
from components.rope import RoPE, get_rotation_matrix


class EfficientSlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, rotation_matrix):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        # create a position embedding attribute with RoPE
        self.rope = RoPE(rotation_matrix=rotation_matrix)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # giving input to the first linear layer of multi-head attention
        qkv = self.qkv_linear(x).reshape(  # reshaped to distribute the tensor to heads
            batch_size, seq_length, self.num_heads, self.head_dim * 3
        )

        # TODO: create the queries, keys and values
        # dividing the qkv variable into chunks to create query, key and value tensors
        queries, keys, values = qkv.transpose(1, 2).chunk(3, dim=-1)  # seq_length and num_heads dimensions are transposed
        
        # TODO: rotate the queries and keys using RoPE
        queries, keys = self.rope(queries=queries, keys=keys)        
        # TODO: pad the keys and values
        # padding keys by its "sequence length" dimension by a constant, zero
        keys_padded = F.pad(
            input=keys,
            pad=(0, 0, padding, padding),
            mode="constant",
            value=0
        )
        # padding values by its "sequence length" dimension by a constant, zero
        values_padded = F.pad(
            input=values,
            pad=(0, 0, padding, padding),  # (padding_left, padding_right, padding_top, padding_bottom)
            mode="constant",
            value=0
        )

        # TODO: Create sliding windows for keys and values
        # unfolding the keys to create sliding windows
        keys_windows = keys_padded.unfold(
            dimension=2, size=self.window_size, step=1
        )
        # unfolding the values to create sliding windows
        values_windows = values_padded.unfold(
            dimension=2, size=self.window_size, step=1
        )

        # TODO: Compute attention scores
        # calculate Einstein sum of queries and keys_windows
        scores = torch.einsum(
            "bhsd,bhsdw->bhsw",  # b: batch_size, h: num_heads, s: seq_length, d: head_dim, w: window_size
            queries, keys_windows
        )
        # normalise the logits and pass through the softmax to get the attention scores
        attentions = F.softmax(
            input=scores / (self.head_dim ** 0.5),
            dim=-1
        )

        # TODO: multiply attentions to values_windows
        # calculate the context tensor with shape: [batch_size, seq_length, num_heads, head_dim].
        context = torch.einsum("bhsw,bhsdw->bshd", attentions, values_windows)
        
        # TODO: Merge heads and combine the last two dimensions
        # TODO: perform the final linear transformation
        # concat the head outputs and give it to last linear layer
        output = self.out(
            context.reshape(shape=(batch_size, seq_length, self.hidden_size))
        )
        return output
   
    
class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        padding = self.window_size // 2

        # Compute Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reorder to (batch_size, num_heads, seq_length, 3 * head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Pad sequence for windowed attention
        keys = F.pad(keys, (0, 0, padding, padding), "constant", 0)
        values = F.pad(values, (0, 0, padding, padding), "constant", 0)

        # Initialize context tensors
        context = torch.zeros_like(queries)

        # Compute attention for each sliding window
        for i in range(seq_length):
            # Determine the start and end of the window
            start = i - padding
            end = i + padding + 1
            
            # Compute scores
            scores = torch.matmul(queries[:, :, i:i+1, :], keys[:, :, start:end, :].transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention to values and add to context
            context[:, :, i:i+1, :] += torch.matmul(attention, values[:, :, start:end, :])

        # Reshape context to (batch_size, seq_length, num_heads * head_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.hidden_size)

        # Final linear layer
        output = self.out(context)
        return output