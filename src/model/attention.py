"""Attention modules for the GPT model.

Implements ScaledDotProductAttention and MultiHeadAttention
following the GPT-2 style decoder-only transformer architecture.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with causal masking.

    Computes (Q @ K^T / sqrt(d_k)), applies a causal mask to prevent
    attending to future positions, applies softmax, then multiplies by V.
    """

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute scaled dot-product attention.

        Args:
            Q: Query tensor of shape (..., seq_len, d_k)
            K: Key tensor of shape (..., seq_len, d_k)
            V: Value tensor of shape (..., seq_len, d_k)
            mask: Optional mask tensor. If None, a causal mask is created.

        Returns:
            Output tensor of the same shape as V.
        """
        d_k = Q.size(-1)
        seq_len = Q.size(-2)

        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Create causal mask if not provided
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1,
            )

        # Apply causal mask: set future positions to -inf before softmax
        scores = scores.masked_fill(mask, float("-inf"))

        # Apply softmax and compute weighted sum
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Projects input to Q, K, V per head, applies ScaledDotProductAttention
    to each head, concatenates the results, and applies a final linear projection.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize MultiHeadAttention.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate applied after attention weights.

        Raises:
            ValueError: If d_model is not evenly divisible by num_heads.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention per head
        attn_output = self.attention(Q, K, V)

        # Apply dropout to attention output
        attn_output = self.dropout(attn_output)

        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.W_o(attn_output)

        return output
