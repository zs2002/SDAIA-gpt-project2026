"""Property-based tests for training pipelines (pretrain + finetune helpers)."""

import sys
import os

# Ensure the project root is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.training.pretrain import get_lr, detect_overfitting
from src.training.finetune import create_loss_mask, format_sft_example


# ===========================================================================
# Property-Based Tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 18: Learning rate schedule correctness
@settings(max_examples=100)
@given(
    warmup_steps=st.integers(min_value=1, max_value=500),
    extra_steps=st.integers(min_value=1, max_value=500),
    peak_lr=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_learning_rate_schedule_correctness(warmup_steps, extra_steps, peak_lr):
    """**Validates: Requirements 9.3**

    For any warmup_steps W and max_steps M (where W < M), the learning rate
    schedule SHALL satisfy: (a) LR at step 0 is near zero, (b) LR increases
    monotonically during steps [0, W], (c) LR at step W equals the peak
    learning rate, and (d) LR decreases monotonically during steps [W, M]
    following cosine decay.
    """
    max_steps = warmup_steps + extra_steps  # guarantees W < M

    # (a) LR at step 0 is near zero
    lr_0 = get_lr(0, warmup_steps, max_steps, peak_lr)
    assert lr_0 < peak_lr * 0.01 + 1e-12, (
        f"LR at step 0 should be near zero, got {lr_0} (peak={peak_lr})"
    )

    # (b) LR increases monotonically during warmup [0, W]
    prev_lr = lr_0
    for step in range(1, warmup_steps + 1):
        lr = get_lr(step, warmup_steps, max_steps, peak_lr)
        assert lr >= prev_lr - 1e-12, (
            f"LR should increase during warmup: step {step - 1}→{step}, "
            f"{prev_lr} → {lr}"
        )
        prev_lr = lr

    # (c) LR at step W equals the peak learning rate
    lr_at_warmup = get_lr(warmup_steps, warmup_steps, max_steps, peak_lr)
    assert abs(lr_at_warmup - peak_lr) < 1e-9, (
        f"LR at warmup_steps should equal peak_lr: got {lr_at_warmup}, expected {peak_lr}"
    )

    # (d) LR decreases monotonically during cosine decay [W, M]
    prev_lr = lr_at_warmup
    for step in range(warmup_steps + 1, max_steps + 1):
        lr = get_lr(step, warmup_steps, max_steps, peak_lr)
        assert lr <= prev_lr + 1e-12, (
            f"LR should decrease during decay: step {step - 1}→{step}, "
            f"{prev_lr} → {lr}"
        )
        prev_lr = lr


# Feature: gpt-from-scratch, Property 19: Overfitting detection
@settings(max_examples=100)
@given(
    patience=st.integers(min_value=1, max_value=10),
    base_loss=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    prefix_len=st.integers(min_value=0, max_value=5),
    increments=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
def test_property_overfitting_detection(patience, base_loss, prefix_len, increments):
    """**Validates: Requirements 9.7**

    For any sequence of validation losses where the last patience values are
    strictly increasing, the overfitting detection function SHALL flag an
    overfitting warning.
    """
    # Build a sequence where the last (patience) values are strictly increasing.
    # We need patience+1 values at the tail that are strictly increasing
    # (detect_overfitting checks patience consecutive *increases*, needing patience+1 values).
    tail_len = patience + 1
    # Use enough increments — cycle if needed
    used_increments = []
    for i in range(tail_len - 1):
        used_increments.append(increments[i % len(increments)])

    # Build the strictly increasing tail
    tail = [base_loss]
    for inc in used_increments:
        tail.append(tail[-1] + inc)

    # Optionally prepend some arbitrary prefix values
    prefix = [base_loss + 5.0 - i * 0.5 for i in range(prefix_len)]

    val_losses = prefix + tail

    result = detect_overfitting(val_losses, patience)
    assert result is True, (
        f"Expected overfitting detected with patience={patience}, "
        f"val_losses tail={tail}, full={val_losses}"
    )


# Feature: gpt-from-scratch, Property 20: SFT loss mask covers output tokens only
@settings(max_examples=100)
@given(
    instruction_len=st.integers(min_value=0, max_value=100),
    output_len=st.integers(min_value=0, max_value=100),
)
def test_property_sft_loss_mask_covers_output_tokens_only(instruction_len, output_len):
    """**Validates: Requirements 10.1**

    For any SFT training example with instruction length L_i and output length
    L_o, the loss mask SHALL be zero for the first L_i token positions and one
    for the remaining L_o token positions.
    """
    total_len = instruction_len + output_len
    mask = create_loss_mask(instruction_len, total_len)

    # Correct length
    assert len(mask) == total_len, (
        f"Mask length should be {total_len}, got {len(mask)}"
    )

    # First instruction_len positions are 0
    for i in range(instruction_len):
        assert mask[i] == 0, (
            f"Mask at instruction position {i} should be 0, got {mask[i]}"
        )

    # Remaining output_len positions are 1
    for i in range(instruction_len, total_len):
        assert mask[i] == 1, (
            f"Mask at output position {i} should be 1, got {mask[i]}"
        )


# Feature: gpt-from-scratch, Property 21: SFT example formatting
@settings(max_examples=100)
@given(
    instruction=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    output=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    has_input=st.booleans(),
    input_text=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    separator=st.text(min_size=1, max_size=10, alphabet=st.characters(blacklist_categories=("Cs",))),
)
def test_property_sft_example_formatting(instruction, output, has_input, input_text, separator):
    """**Validates: Requirements 10.2**

    For any valid SFT record with instruction I, optional input X, and output O,
    the formatted training string SHALL contain I, the separator token, and O in
    that order. If X is present, it SHALL appear between I and the separator.
    """
    # Filter out degenerate cases where fields are substrings of each other,
    # making positional assertions on str.index() ambiguous.
    all_parts = [instruction, separator, output]
    if has_input:
        all_parts.append(input_text)
    assume(len(set(all_parts)) == len(all_parts))

    record = {"instruction": instruction, "output": output}
    if has_input:
        record["input"] = input_text

    formatted = format_sft_example(record, separator)

    # The formatted string must contain instruction, separator, and output
    assert instruction in formatted, (
        f"Formatted string should contain instruction: {instruction!r}"
    )
    assert separator in formatted, (
        f"Formatted string should contain separator: {separator!r}"
    )
    assert output in formatted, (
        f"Formatted string should contain output: {output!r}"
    )

    # Instruction appears before separator, separator appears before output
    instr_idx = formatted.index(instruction)
    sep_idx = formatted.index(separator, instr_idx + len(instruction))
    out_idx = formatted.index(output, sep_idx + len(separator))

    assert instr_idx < sep_idx < out_idx, (
        f"Order violation: instruction@{instr_idx}, separator@{sep_idx}, output@{out_idx}"
    )

    # If input is present, it should appear between instruction and separator
    if has_input:
        assert input_text in formatted, (
            f"Formatted string should contain input: {input_text!r}"
        )
        input_idx = formatted.index(input_text, instr_idx + len(instruction))
        assert input_idx < sep_idx, (
            f"Input should appear between instruction and separator: "
            f"instruction@{instr_idx}, input@{input_idx}, separator@{sep_idx}"
        )
