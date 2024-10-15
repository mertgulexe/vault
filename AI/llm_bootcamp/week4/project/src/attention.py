from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math


class SparseAttention(LlamaAttention):

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, group_size_ratio=0.25):
        super().__init__(config, layer_idx)
        self.group_size_ratio = group_size_ratio

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        # use_cache: bool = False,
        # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # get the dimensions
        batch_size, seq_len, hidden_size = hidden_states.size()
        assert hidden_size % self.num_heads == 0, \
            "Hidden size must be divisible by number of heads!"

        # TODO: establish the group size
        group_size = math.floor(seq_len * self.group_size_ratio)
        # TODO: compute the number of groups
        num_group = math.floor(seq_len / group_size)

        # TODO: In Llama 3, the linear layers used to project into keys, queries, and values 
        # are called q_proj, k_proj, and v_proj, and they are attributes of the class. 
        # Compute the keys, queries and values in the forward function from the hidden_states.
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # TODO: Reshape the keys, queries, and values to have the shapes 
        # [batch_size, seq_len, num_heads, head_dim]. 
        # num_heads and head_dim are attributes of the class
        queries = queries.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        keys = keys.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        values = values.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # TODO: Then transpose the dimensions to get tensors of size 
        # [batch_size, num_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # TODO: rotate the keys and queries with RoPE.
        cos, sin = self.rotary_emb(x=values, seq_len=seq_len)
        queries, keys = apply_rotary_pos_emb(
            q=queries,
            k=keys,
            cos=cos,
            sin=sin,
            position_ids=position_ids
        )

        # TODO: Apply the function repeat_kv to potentially increase 
        # the number of heads for the keys and values.
        keys = repeat_kv(hidden_states=keys, n_rep=self.num_heads)
        values = repeat_kv(hidden_states=values, n_rep=self.num_heads)

        # TODO: apply the shift function to the queries, keys, and values
        queries, keys, values = [
            self.shift(
                qkv=qkv,
                batch_size=batch_size,
                seq_len=seq_len,
                group_size=group_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim
            ) for qkv in [queries, keys, values]
        ]
        # TODO: compute the interaction matrix between the queries and the keys
        attn_weights = torch.matmul(
            input=queries,
            other=keys.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # compute local mask
        local_mask = self.shift_attention_masks(
            simple=True, 
            attention_mask=attention_mask, 
            group_size=group_size, 
            batch_size=batch_size, 
            num_group=num_group, 
            num_heads=self.num_heads
        )

        # TODO: add the local attention masks to the interaction matrix and apply the softmax transformation
        attn_weights = nn.functional.softmax(
            input=attn_weights + local_mask,
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)
        
        # TODO: compute the new hidden states by computing the product between the values and the self-attentions.
        attn_output = torch.matmul(input=attn_weights, other=values)
        
        # TODO: At this point, the new hidden states should have a dimension 
        # [batch_size * num_group, num_heads, group_size, head_dim]. Reshape the hidden states 
        # to have dimension [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # TODO: Don't forget that half of the heads are shifted by -group_size // 2, 
        # so let's shift the new hidden states by group_size // 2.
        attn_output = attn_output.roll(shifts=group_size // 2, dim=2)

        # TODO: finally, let's reshape the hidden states back to [batch_size, seq_len, hidden_size], 
        # and project the resulting tensor with the output layer o_proj.

        attn_output = attn_output.reshape(
            batch_size, seq_len, hidden_size
        )
        new_hidden_states = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return new_hidden_states, attn_weights, past_key_value
    
    def shift(self, qkv, batch_size, seq_len, group_size, num_heads, head_dim):      
        # TODO: Implement the function. qkv can be any of the queries, keys, or values.
        # - First, use the roll function to shift half the heads by -group_size // 2.
        # - Then transpose the resulting tensor into the shape [batch_size, seq_len, num_heads, head_dim]
        # - Then reshape the resulting tensor into the shape 
        # [batch_size * (seq_len / group_size), group_size, num_heads, head_dim]. 
        # The idea is instead of having batch_size samples of input size seq_len, 
        # we are going to increase the number of samples to batch_size * (seq_len / group_size) 
        # or batch_size / group_size_ratio, but reduce the input size to group_size.
        # - Finally, transpose the resulting tensor to [batch_size * (seq_len / group_size), num_heads, group_size, head_dim].
        qkv = qkv.roll(shifts=-group_size // 2, dim=2)
        qkv = qkv.transpose(1, 2)
        qkv = qkv.reshape(
            batch_size * (seq_len / group_size),
            group_size,
            num_heads,
            head_dim
        )
        return qkv.transpose(1, 2)
    
    def shift_attention_masks(self, simple, attention_mask, group_size, batch_size, num_group, num_heads):
        
        if simple:
            # The way done by LongLoRA
            return attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)

        # shift the queries and the keys
        # [batch_size, 1, seq_len, seq_len]
        attention_mask_rolled = attention_mask.roll(-group_size // 2, dims=-1).roll(-group_size // 2, dims=-2)
        # reshape per group
        # [batch_size, 1, num_groups, group_size, num_groups, group_size]
        attention_mask = attention_mask.reshape(batch_size, 1, num_group, group_size, num_group, group_size)
        attention_mask_rolled = attention_mask_rolled.reshape(batch_size, 1, num_group, group_size, num_group, group_size)

        # repeat dim = 1 for half the number of heads
        attention_mask = attention_mask.expand(batch_size, num_heads // 2, num_group, group_size, num_group, group_size)
        attention_mask_rolled = attention_mask_rolled.expand(batch_size, num_heads // 2, num_group, group_size, num_group, group_size)
 
        # concatenate the non shifted attention masks with the shifted ones
        attention_mask = torch.cat([attention_mask, attention_mask_rolled], dim=1)

        # We want to keep only the masks for the groups on the diagonal
        # [num_group, 1, num_group, 1]
        diagonal_mask = torch.eye(num_group, dtype=torch.bool).unsqueeze(-2).unsqueeze(-1)
        # We repeat the pattern for all dimensions
        # [batch_size, num_heads, num_group, group_size, num_group, group_size]
        diagonal_mask = diagonal_mask.expand(batch_size, num_heads, num_group, group_size, num_group, group_size)

        # We keep only the masks for the groups on the diagonal
        local_attention_mask = attention_mask.masked_select(diagonal_mask)
        local_attention_mask = local_attention_mask.view(batch_size, num_heads, num_group, group_size, group_size)
        # We reshape to match the attention shape
        local_attention_mask = local_attention_mask.transpose(1, 2).reshape(batch_size * num_group, num_heads, group_size, group_size)

        return local_attention_mask