"""Property-based tests for TransformerBlock and GPTModel."""

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.transformer import TransformerBlock


# ===========================================================================
# Property-Based Tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 14: Transformer block MLP inner dimension
@settings(max_examples=100)
@given(
    num_heads=st.sampled_from([1, 2, 4, 8]),
    head_dim=st.integers(min_value=1, max_value=32),
)
def test_property_transformer_block_mlp_inner_dimension(num_heads, head_dim):
    """**Validates: Requirements 7.2**

    For any TransformerBlock initialized with d_model, the feed-forward MLP's
    first linear layer SHALL have output features equal to 4 * d_model, and
    the second linear layer SHALL have input features equal to 4 * d_model.
    """
    d_model = num_heads * head_dim

    block = TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=0.0)

    # mlp is nn.Sequential(Linear(d_model, 4*d_model), GELU(), Linear(4*d_model, d_model))
    first_linear = block.mlp[0]
    second_linear = block.mlp[2]

    # First linear layer: output features == 4 * d_model
    assert first_linear.out_features == 4 * d_model, (
        f"First linear out_features: expected {4 * d_model}, got {first_linear.out_features}"
    )

    # Second linear layer: input features == 4 * d_model
    assert second_linear.in_features == 4 * d_model, (
        f"Second linear in_features: expected {4 * d_model}, got {second_linear.in_features}"
    )

# Feature: gpt-from-scratch, Property 15: Transformer block shape invariant
@settings(max_examples=100)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=1, max_value=32),
    num_heads=st.sampled_from([1, 2, 4, 8]),
    head_dim=st.integers(min_value=1, max_value=16),
)
def test_property_transformer_block_shape_invariant(batch_size, seq_len, num_heads, head_dim):
    """**Validates: Requirements 7.3**

    For any batch size B, sequence length S, and model dimension d_model,
    the TransformerBlock SHALL produce an output tensor of shape (B, S, d_model)
    given an input of the same shape.
    """
    d_model = num_heads * head_dim

    block = TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=0.0)
    block.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = block(x)

    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {output.shape}"
    )
