# ===========================
# ./src/models/transformer.py
# ===========================

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding introduced by
    Vaswani et al. (2017).
    """

    def __init__(
            self,
            d_model: int,
            max_len: int = 1000,
    ):

        super().__init__()

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than zero."
            )

        if d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for this sinusoidal "
                "positional encoding implementation."
            )

        if max_len <= 0:
            raise ValueError(
                "max_len must be greater than zero."
            )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length "
                f"({x.size(1)}) exceeds "
                f"the positional encoding limit "
                f"({self.pe.size(1)})."
            )

        return (
            x
            + self.pe[
                :,
                :x.size(1),
                :
            ]
        )


class TranslationTransformer(nn.Module):
    """
    Sequence-to-sequence Transformer for neural machine
    translation.

    The model uses:
        - shared source/target token embeddings,
        - sinusoidal positional encoding,
        - PyTorch nn.Transformer,
        - tied input/output embedding weights.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_heads: int = 8,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.2,
            pad_id: int = 0,
            positional_encoding_max_len: int = 1000,
    ):

        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than zero."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than zero."
            )

        if n_heads <= 0:
            raise ValueError(
                "n_heads must be greater than zero."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        if positional_encoding_max_len <= 0:
            raise ValueError(
                "positional_encoding_max_len must be "
                "greater than zero."
            )

        self.d_model = d_model
        self.pad_id = pad_id

        self.positional_encoding_max_len = (
            positional_encoding_max_len
        )

        # -----------------------------------------------------
        # Shared source/target embedding
        # -----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        # -----------------------------------------------------
        # Sinusoidal positional encoding
        # -----------------------------------------------------

        self.pos_encoder = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=(
                    positional_encoding_max_len
                ),
            )
        )

        # -----------------------------------------------------
        # Transformer encoder-decoder
        # -----------------------------------------------------

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=(
                num_encoder_layers
            ),
            num_decoder_layers=(
                num_decoder_layers
            ),
            dim_feedforward=(
                dim_feedforward
            ),
            dropout=dropout,
            batch_first=True,
        )

        # -----------------------------------------------------
        # Vocabulary projection
        # -----------------------------------------------------

        self.output_layer = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # -----------------------------------------------------
        # Weight tying
        # -----------------------------------------------------

        self.output_layer.weight = (
            self.embedding.weight
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def make_src_mask(
            self,
            src: torch.Tensor,
    ) -> torch.Tensor:
        """
        Creates the source padding mask.

        True values indicate PAD positions that must be ignored.
        """

        return (
            src == self.pad_id
        )

    def make_tgt_mask(
            self,
            tgt: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Creates:
            1. target padding mask
            2. causal self-attention mask
        """

        tgt_pad_mask = (
            tgt == self.pad_id
        )

        seq_len = tgt.size(1)

        tgt_causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=tgt.device,
            ),
            diagonal=1,
        )

        return (
            tgt_pad_mask,
            tgt_causal_mask,
        )

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Masks
        # -----------------------------------------------------

        src_pad_mask = (
            self.make_src_mask(
                src
            )
        )

        (
            tgt_pad_mask,
            tgt_causal_mask,
        ) = self.make_tgt_mask(
            tgt
        )

        # -----------------------------------------------------
        # Source embeddings
        # -----------------------------------------------------

        src_emb = self.embedding(
            src
        )

        src_emb = (
            src_emb
            * math.sqrt(
                self.d_model
            )
        )

        src_emb = self.pos_encoder(
            src_emb
        )

        src_emb = self.dropout(
            src_emb
        )

        # -----------------------------------------------------
        # Target embeddings
        # -----------------------------------------------------

        tgt_emb = self.embedding(
            tgt
        )

        tgt_emb = (
            tgt_emb
            * math.sqrt(
                self.d_model
            )
        )

        tgt_emb = self.pos_encoder(
            tgt_emb
        )

        tgt_emb = self.dropout(
            tgt_emb
        )

        # -----------------------------------------------------
        # Transformer
        # -----------------------------------------------------

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=(
                tgt_causal_mask
            ),
            src_key_padding_mask=(
                src_pad_mask
            ),
            tgt_key_padding_mask=(
                tgt_pad_mask
            ),
            memory_key_padding_mask=(
                src_pad_mask
            ),
        )

        # -----------------------------------------------------
        # Vocabulary logits
        # -----------------------------------------------------

        logits = self.output_layer(
            out
        )

        return logits