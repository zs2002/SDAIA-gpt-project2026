"""Transformer block and GPT model components.

Implements the pre-norm TransformerBlock (GPT-2 style) and the full GPTModel
decoder-only transformer architecture.
"""

import torch.nn as nn
from torch import Tensor

from src.config import GPTConfig
from src.model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block.

    Architecture: LN → MHA → residual + dropout → LN → MLP → residual + dropout.
    The MLP uses GELU activation with inner dimension = 4 * d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize TransformerBlock.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate applied after attention and MLP outputs.
        """
        super().__init__()

        # Pre-norm layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward MLP: linear → GELU → linear
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # LN → MHA → residual + dropout
        x = x + self.dropout1(self.attn(self.ln1(x)))

        # LN → MLP → residual + dropout
        x = x + self.dropout2(self.mlp(self.ln2(x)))

        return x


class GPTModel(nn.Module):
    """Full GPT decoder-only transformer model.

    Assembles: token embedding + positional embedding + N TransformerBlocks
    + final LayerNorm + linear head (→ vocab_size).
    """

    def __init__(self, config: GPTConfig):
        """Initialize GPTModel from a GPTConfig.

        Args:
            config: GPTConfig with vocab_size, d_model, num_heads,
                    num_layers, max_seq_len, dropout_rate.
        """
        super().__init__()
        config.validate()
        self.config = config

        # Token and positional embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout after embeddings
        self.drop = nn.Dropout(config.dropout_rate)

        # N stacked transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.num_heads, config.dropout_rate)
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm and linear head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward pass: token IDs → logits.

        Args:
            token_ids: Input tensor of shape (batch, seq_len) with integer token IDs.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).

        Raises:
            ValueError: If seq_len exceeds config.max_seq_len.
        """
        _, seq_len = token_ids.size()

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum supported "
                f"length {self.config.max_seq_len}"
            )

        # Create position indices: (seq_len,)
        positions = token_ids.new_tensor(range(seq_len), dtype=token_ids.dtype)

        # Embeddings: token + positional
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm + linear head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
