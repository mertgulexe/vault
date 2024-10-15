import torch
import torch.nn as nn
from typing import Tuple


def get_rotation_matrix(dim: int, context_size: int, period: float) -> torch.Tensor:
    # compute a tensor of frequencies
    freqs = 1.0 / (
        period ** (
            torch.arange(
                start=0,
                end=dim,
                step=2,
                dtype=torch.float32
            )[0: (dim//2)] / dim  # dim//2 truncates to correct number of elements in case dim is odd.
        )
    )  # advised solution ---> freqs = 1.0 / (period ** (torch.arange(0, dim, 2) / dim))
    # compute a tensor of token indexes
    token_indexes = torch.arange(
        start=0,
        end=context_size,
        step=1,
        dtype=freqs.dtype
    )
    # compute the matrix thetas
    thetas = torch.outer(
        input=token_indexes,  # dimension: context_size
        vec2=freqs,  # dimension: dim//2
    )  # resulting dimension: [context_size, dim//2]
    # create the rotation matrix
    rotation_matrix = torch.polar(
        abs=torch.ones_like(
            input=thetas,
            dtype=thetas.dtype
        ),
        angle=thetas
    )
    return rotation_matrix


class RoPE(nn.Module):
    def __init__(self, rotation_matrix):
        super().__init__()
        self.rotation_matrix = rotation_matrix

    def forward(self, queries, keys):
        batch_size, num_heads, seq_length, head_dim = queries.size()

        # TODO: reshape to [batch_size, num_heads, seq_length, head_dim // 2 , 2]
        queries = queries.view(
            batch_size, num_heads, seq_length, head_dim // 2, 2
        )
        keys = keys.view(
            batch_size, num_heads, seq_length, head_dim // 2, 2
        )

        # TODO: transform into a complex tensor
        ASSERTION_WARNING = (
            "To view a tensor as complex numbers, "
            "its data type should be either float32 or float64."
        )
        assert queries.dtype in (torch.float32, torch.float64), ASSERTION_WARNING
        assert keys.dtype in (torch.float32, torch.float64), ASSERTION_WARNING
        queries_complex = torch.view_as_complex(input=queries)
        keys_complex = torch.view_as_complex(input=keys)

        # TODO: rotate the queries and keys
        truncated_rotation_matrix = self.rotation_matrix[0:seq_length, :]
        queries_rotated = truncated_rotation_matrix * queries_complex
        keys_rotated = truncated_rotation_matrix * keys_complex

        # TODO: conver to read and reshape back to [batch_size, num_heads, seq_length, head_dim]
        new_queries = torch.view_as_real(queries_rotated).reshape(batch_size, num_heads, seq_length, head_dim)
        new_keys = torch.view_as_real(keys_rotated).reshape(batch_size, num_heads, seq_length, head_dim)

        return new_queries, new_keys











def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)