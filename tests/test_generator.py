"""Property-based tests for TextGenerator (generation module)."""

import sys
import os

# Ensure the project root is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.config import GPTConfig
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.generation.generator import TextGenerator


# ---------------------------------------------------------------------------
# Shared fixtures — tiny model + tokenizer for fast property tests
# ---------------------------------------------------------------------------

_SAMPLE_CORPUS = (
    "the cat sat on the mat "
    "the dog sat on the log "
    "a bird flew over the tree "
    "the sun shines bright today "
) * 20

_TOKENIZER = None
_MODEL = None
_GENERATOR = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = BPETokenizer(vocab_size=300)
        _TOKENIZER.train(_SAMPLE_CORPUS)
    return _TOKENIZER


def _get_model():
    global _MODEL
    if _MODEL is None:
        tok = _get_tokenizer()
        config = GPTConfig(
            vocab_size=len(tok.vocab),
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128,
            dropout_rate=0.0,
        )
        _MODEL = GPTModel(config)
        _MODEL.eval()
    return _MODEL


def _get_generator():
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = TextGenerator(_get_model(), _get_tokenizer())
    return _GENERATOR


# ===========================================================================
# Property-Based Tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 22: Greedy decoding determinism
@settings(max_examples=100)
@given(
    prompt=st.sampled_from(["the", "cat", "sat", "the cat", "a bird", "sun"]),
    max_new_tokens=st.integers(min_value=1, max_value=30),
)
def test_property_greedy_decoding_determinism(prompt, max_new_tokens):
    """**Validates: Requirements 11.3**

    For any prompt and trained model, calling the Text_Generator with
    temperature=0 twice SHALL produce identical output text and identical
    token ID lists.
    """
    gen = _get_generator()

    text1, ids1 = gen.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
    text2, ids2 = gen.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)

    assert text1 == text2, (
        f"Greedy text mismatch for prompt={prompt!r}: {text1!r} != {text2!r}"
    )
    assert ids1 == ids2, (
        f"Greedy token IDs mismatch for prompt={prompt!r}: {ids1} != {ids2}"
    )


# Feature: gpt-from-scratch, Property 23: Generation length bounded by max_new_tokens
@settings(max_examples=100)
@given(
    prompt=st.sampled_from(["the", "cat", "sat", "the cat", "a bird", "sun"]),
    max_new_tokens=st.integers(min_value=1, max_value=50),
)
def test_property_generation_length_bounded_by_max_new_tokens(prompt, max_new_tokens):
    """**Validates: Requirements 11.4**

    For any generation call with max_new_tokens=N, the number of newly
    generated tokens SHALL be at most N.
    """
    gen = _get_generator()

    _text, ids = gen.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=0
    )

    assert len(ids) <= max_new_tokens, (
        f"Generated {len(ids)} tokens but max_new_tokens={max_new_tokens} "
        f"for prompt={prompt!r}"
    )


# Feature: gpt-from-scratch, Property 24: Generation output consistency
@settings(max_examples=100)
@given(
    prompt=st.sampled_from(["the", "cat", "sat", "the cat", "a bird", "sun"]),
    max_new_tokens=st.integers(min_value=1, max_value=30),
)
def test_property_generation_output_consistency(prompt, max_new_tokens):
    """**Validates: Requirements 11.1, 11.5**

    For any generation call, the Text_Generator SHALL return a tuple
    (text, token_ids) where decoding token_ids with the tokenizer
    produces text.
    """
    gen = _get_generator()
    tok = _get_tokenizer()

    text, ids = gen.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=0
    )

    decoded = tok.decode(ids)
    assert text == decoded, (
        f"Output consistency failed for prompt={prompt!r}: "
        f"text={text!r} != decode(ids)={decoded!r}, ids={ids}"
    )
