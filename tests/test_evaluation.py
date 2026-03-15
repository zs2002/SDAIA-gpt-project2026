"""Property-based and unit tests for evaluation modules.

Properties 25 and 26 from the design document.
"""

import math

import pytest
from hypothesis import given, settings, strategies as st

from src.evaluation.metrics import Evaluator
from src.evaluation.error_analysis import ErrorAnalyzer


# ---------------------------------------------------------------------------
# Feature: gpt-from-scratch, Property 25: Perplexity equals exp(loss)
# ---------------------------------------------------------------------------
# **Validates: Requirements 12.1**


@settings(max_examples=100)
@given(
    avg_loss=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_perplexity_equals_exp_loss(avg_loss: float) -> None:
    """For any non-negative average cross-entropy loss L, perplexity == exp(L)."""
    evaluator = Evaluator()
    perplexity = evaluator.perplexity_from_loss(avg_loss)
    expected = math.exp(avg_loss)
    assert math.isclose(perplexity, expected, rel_tol=1e-9), (
        f"perplexity_from_loss({avg_loss}) = {perplexity}, expected exp({avg_loss}) = {expected}"
    )


# ---------------------------------------------------------------------------
# Feature: gpt-from-scratch, Property 26: Error analysis detects repetition
# ---------------------------------------------------------------------------
# **Validates: Requirements 12.5**


# Strategy: generate a word, repeat an n-gram of that word more than threshold times
@st.composite
def repetitive_text_strategy(draw):
    """Generate text containing a repeated n-gram (n >= 3) exceeding threshold."""
    # Pick n-gram size (3 to 6)
    n = draw(st.integers(min_value=3, max_value=6))
    # Pick threshold (1 to 5)
    threshold = draw(st.integers(min_value=1, max_value=5))
    # Number of repetitions: strictly more than threshold
    reps = threshold + draw(st.integers(min_value=1, max_value=5))

    # Generate n distinct words for the n-gram
    words = draw(
        st.lists(
            st.from_regex(r"[a-z]{2,8}", fullmatch=True),
            min_size=n,
            max_size=n,
        )
    )
    ngram_phrase = " ".join(words)

    # Build text: some prefix + repeated n-gram + some suffix
    prefix_words = draw(
        st.lists(
            st.from_regex(r"[a-z]{2,6}", fullmatch=True),
            min_size=0,
            max_size=3,
        )
    )
    suffix_words = draw(
        st.lists(
            st.from_regex(r"[a-z]{2,6}", fullmatch=True),
            min_size=0,
            max_size=3,
        )
    )

    parts = []
    if prefix_words:
        parts.append(" ".join(prefix_words))
    # Repeat the n-gram phrase `reps` times
    parts.extend([ngram_phrase] * reps)
    if suffix_words:
        parts.append(" ".join(suffix_words))

    text = " ".join(parts)
    return text, n, threshold


@settings(max_examples=100)
@given(data=repetitive_text_strategy())
def test_error_analysis_detects_repetition(data) -> None:
    """For any text with a repeated n-gram (n>=3) above threshold, ErrorAnalyzer flags repetition."""
    text, ngram_size, threshold = data
    analyzer = ErrorAnalyzer(ngram_size=ngram_size, repetition_threshold=threshold)
    report = analyzer.analyze([text])
    assert report["repetition"]["count"] >= 1, (
        f"ErrorAnalyzer failed to detect repetition in text with "
        f"n={ngram_size}, threshold={threshold}:\n{text[:300]}"
    )
