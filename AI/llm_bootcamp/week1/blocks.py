from torch import nn, Tensor
import torch.nn.functional as F
from components import (
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFFNetwork
)

class EncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            is_torch_import: bool = False
    ) -> None:
        super().__init__()
        if is_torch_import:
            MHAClass = nn.MultiheadAttention
        else:
            MHAClass = MultiHeadAttention
        self.self_attn = MHAClass(embed_dim=embed_dim,num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ff_network = PositionwiseFFNetwork(
            embed_dim=embed_dim, d_ff=d_ff, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.is_torch_import = is_torch_import
        # SELFNOTE: Tthere has to be 2 normalization layers because
        # each one of them has different learnable parameters individually.
        
    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        self_attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        if self.is_torch_import:
            self_attn_output = self_attn_output[0]
        x = self.layernorm1(input=self_attn_output + x)
        ff_network_output = self.ff_network(x)
        ff_network_output = self.dropout(ff_network_output)
        x = self.layernorm2(input=ff_network_output + x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            is_torch_import: bool = False
    ) -> None:
        super().__init__()
        if is_torch_import:
            MHAClass = nn.MultiheadAttention
        else:
            MHAClass = MultiHeadAttention
        self.self_attn = MHAClass(embed_dim, num_heads)
        self.cross_attn = MHAClass(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.ff_network = PositionwiseFFNetwork(embed_dim, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.is_torch_import = is_torch_import
        # SELFNOTE: There has to be 2 MultiHeadAttention layers as well
        # for the same reason!

    def forward(
            self,
            x: Tensor,
            encoder_output: Tensor,
            attn_mask: Tensor | None = None
        ) -> Tensor:
        self_attn_output = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )
        if self.is_torch_import:
            self_attn_output = self_attn_output[0]
        x = self.layernorm1(self_attn_output + x)
        cross_attn_output = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output
        )
        if self.is_torch_import:
            cross_attn_output = cross_attn_output[0]
        x = self.layernorm2(cross_attn_output + x)
        ff_network_output = self.ff_network(x)
        ff_network_output = self.dropout(ff_network_output)
        x = self.layernorm3(ff_network_output + x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            d_ff: int,
            dropout: float = 0.1,
            is_torch_import: bool = False
    ) -> None:
        super().__init__()
        enc_block = EncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            is_torch_import=is_torch_import
        )
        self.encoder_blocks = nn.ModuleList(
            modules=[enc_block for _ in range(num_blocks)]
        )
    
    def forward(
            self,
            x: Tensor,
            attn_mask: Tensor | None = None
    ) -> Tensor:
        for b in self.encoder_blocks:
            x = b(x, attn_mask=attn_mask)
            # SELFNOTE: x has to pass every encoder block.
            # self.encoder_blocks(x) is not a valid implementation!
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            d_ff: int,
            dropout: float = 0.1,
            is_torch_import: bool = False
    ) -> None:
        super().__init__()
        dec_block = DecoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            is_torch_import=is_torch_import
        )
        self.decoder_blocks = nn.ModuleList(
            [dec_block for _ in range(num_blocks)]
        )

    def forward(
            self,
            x: Tensor,
            encoder_output: Tensor,
            attn_mask: Tensor | None = None
    ) -> Tensor:
        for b in self.decoder_blocks:
            x = b(x=x, encoder_output=encoder_output, attn_mask=attn_mask)        
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            d_ff: int,
            context_size: int,
            vocab_size: int,
            dropout: float = 0.1,
            is_torch_import: bool = False
    ) -> None:
        super().__init__()
        self.encoder_emb_input = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.encoder_pos_enc_input = PositionalEncoding(
            embed_dim=embed_dim, context_size=context_size, dropout=dropout
        )
        self.encoder = Encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            d_ff=d_ff,
            dropout=dropout,
            is_torch_import=is_torch_import
        )
        self.decoder_emb_input = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.decoder_pos_enc_input = PositionalEncoding(
            embed_dim=embed_dim, context_size=context_size, dropout=dropout
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            d_ff=d_ff,
            dropout=dropout,
            is_torch_import=is_torch_import
        )
        self.output_layer = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size
        )
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
            self,
            encoder_input: Tensor,
            decoder_input: Tensor,
            attn_mask: Tensor | None = None
    ) -> Tensor:
        x_enc = self.encoder_emb_input(input=encoder_input)
        x_enc = self.encoder_pos_enc_input(x=x_enc)
        enc_output = self.encoder(x=x_enc)
        x_dec = self.decoder_emb_input(input=decoder_input)
        x_dec = self.decoder_pos_enc_input(x=x_dec)
        x = self.decoder(
            x=x_dec, encoder_output=enc_output, attn_mask=attn_mask
        )
        x = self.output_layer(input=x)
        x = self.dropout(input=x)
        return F.softmax(input=x, dim=-1)