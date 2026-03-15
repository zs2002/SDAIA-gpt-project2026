"""Property-based test for seed-based reproducibility (Property 27)."""

import sys
import os

# Ensure the project root is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import GPTConfig, TrainConfig
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.pretrain import set_seed, PretrainPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A small fixed corpus used for all test runs.  It must be large enough to
# produce meaningful train/val splits and batches for the tiny model config.
_SAMPLE_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be that is the question. "
    "All that glitters is not gold. "
    "Knowledge is power and power is knowledge. "
) * 20  # repeat to ensure enough tokens


def _make_tokenizer(corpus: str, vocab_size: int = 300) -> BPETokenizer:
    """Train a tiny BPE tokenizer on *corpus*."""
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(corpus)
    return tok


def _run_training(seed: int, corpus: str, tokenizer: BPETokenizer) -> dict:
    """Create a fresh model + pipeline and run a short training, returning the
    final model state dict (moved to CPU with cloned tensors)."""
    gpt_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        dropout_rate=0.0,  # deterministic — no dropout randomness
    )

    train_config = TrainConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        batch_size=2,
        max_steps=3,
        warmup_steps=1,
        log_interval=1,
        eval_interval=100,   # skip eval during short run
        save_interval=100,   # skip checkpoint saving
        patience=3,
        seed=seed,
        train_split=0.9,
    )

    set_seed(seed)
    model = GPTModel(gpt_config)
    pipeline = PretrainPipeline(model, train_config, tokenizer)
    pipeline.train(corpus=corpus)

    # Return a CPU copy of every parameter tensor
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ===========================================================================
# Property-Based Tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 27: Reproducibility via seed
@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_reproducibility_via_seed(seed):
    """**Validates: Requirements 13.2**

    For any random seed value, two training runs initialized with the same
    seed and identical configuration SHALL produce identical model parameter
    values after the same number of steps (on the same hardware).
    """
    # Train a shared tokenizer once (deterministic given the same corpus)
    tokenizer = _make_tokenizer(_SAMPLE_CORPUS)

    # Run 1
    state1 = _run_training(seed, _SAMPLE_CORPUS, tokenizer)

    # Run 2 — same seed, same config, same corpus
    state2 = _run_training(seed, _SAMPLE_CORPUS, tokenizer)

    # Every parameter must be identical
    assert state1.keys() == state2.keys(), (
        f"State dict keys differ: {state1.keys()} vs {state2.keys()}"
    )
    for key in state1:
        assert torch.equal(state1[key], state2[key]), (
            f"Parameter '{key}' differs between runs with seed={seed}.\n"
            f"  max abs diff = {(state1[key] - state2[key]).abs().max().item()}"
        )
