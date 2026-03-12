"""Unit and property-based tests for attention modules."""

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.attention import ScaledDotProductAttention, MultiHeadAttention


# ===========================================================================
# Property-Based Tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 10: Attention output shape invariant
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=32),
    d_k=st.integers(min_value=1, max_value=64),
)
def test_property_attention_output_shape_invariant(batch_size, seq_len, d_k):
    """**Validates: Requirements 5.1, 5.4**

    For any batch size B, sequence length S, and dimension d_k, when the
    ScaledDotProductAttention module receives Q, K, V tensors of shape
    (B, S, d_k), the output tensor SHALL have shape (B, S, d_k).
    """
    attention = ScaledDotProductAttention()

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    with torch.no_grad():
        output = attention(Q, K, V)

    assert output.shape == (batch_size, seq_len, d_k), (
        f"Expected shape ({batch_size}, {seq_len}, {d_k}), got {output.shape}"
    )


# Feature: gpt-from-scratch, Property 11: Causal mask correctness
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=32),
    d_k=st.integers(min_value=1, max_value=64),
)
def test_property_causal_mask_correctness(batch_size, seq_len, d_k):
    """**Validates: Requirements 5.2, 5.3**

    For any input to the ScaledDotProductAttention module, the attention
    weights at position i for all positions j > i SHALL be zero (within
    floating-point tolerance). That is, the model never attends to future
    positions.
    """
    import math

    attention = ScaledDotProductAttention()

    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    with torch.no_grad():
        # Manually compute attention weights to inspect the causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply causal mask (same logic as the module)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)

        # Verify: for every position i, weights to future positions j > i must be zero
        for i in range(seq_len):
            future_weights = attn_weights[:, i, i + 1 :]
            assert torch.allclose(
                future_weights, torch.zeros_like(future_weights), atol=1e-6
            ), (
                f"Position {i} attends to future positions with weights: "
                f"{future_weights}"
            )

        # Also verify the actual module output is consistent with masked attention
        output = attention(Q, K, V)
        expected_output = torch.matmul(attn_weights, V)
        assert torch.allclose(output, expected_output, atol=1e-5), (
            "Module output does not match manually computed masked attention output"
        )


# Feature: gpt-from-scratch, Property 12: Multi-head attention shape invariant
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=32),
    num_heads=st.sampled_from([1, 2, 4, 8]),
    head_dim=st.integers(min_value=1, max_value=16),
)
def test_property_multi_head_attention_shape_invariant(batch_size, seq_len, num_heads, head_dim):
    """**Validates: Requirements 6.1, 6.4**

    For any batch size B, sequence length S, and model dimension d_model
    (where d_model is divisible by num_heads), the MultiHead_Attention module
    SHALL produce an output tensor of the same shape (B, S, d_model) as its input.
    """
    d_model = num_heads * head_dim

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = mha(x)

    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {output.shape}"
    )


# Feature: gpt-from-scratch, Property 13: Multi-head attention rejects non-divisible dimensions
@settings(max_examples=100)
@given(
    d_model=st.integers(min_value=1, max_value=256),
    num_heads=st.integers(min_value=1, max_value=64),
)
def test_property_multi_head_attention_rejects_non_divisible_dimensions(d_model, num_heads):
    """**Validates: Requirements 6.3**

    For any pair (d_model, num_heads) where d_model % num_heads != 0,
    constructing a MultiHead_Attention module SHALL raise a ValueError.
    """
    from hypothesis import assume

    assume(d_model % num_heads != 0)

    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
