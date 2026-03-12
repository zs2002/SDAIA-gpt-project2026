"""Unit tests for DataPreprocessor class."""

import json
import os
import tempfile

import pytest

from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


@pytest.fixture
def preprocessor_no_arabic():
    return DataPreprocessor(normalize_arabic=False)


def _write_temp_file(content: str | bytes, suffix: str = ".txt") -> str:
    """Write content to a temp file and return its path."""
    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, mode, encoding=encoding) as f:  # type: ignore[call-overload]
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_basic_utf8_file(self, preprocessor):
        path = _write_temp_file("Hello world\nLine two\n")
        result = preprocessor.clean_text(path)
        # Normalize line endings for cross-platform compatibility
        assert result.replace("\r\n", "\n") == "Hello world\nLine two\n"
        os.unlink(path)

    def test_removes_html_tags(self, preprocessor):
        path = _write_temp_file("<p>Hello</p> <b>world</b>")
        result = preprocessor.clean_text(path)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result
        os.unlink(path)

    def test_removes_control_characters(self, preprocessor):
        path = _write_temp_file("Hello\x00\x01\x02world")
        result = preprocessor.clean_text(path)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result
        assert "world" in result
        os.unlink(path)

    def test_preserves_tabs_and_newlines(self, preprocessor):
        path = _write_temp_file("Hello\tworld\nNew line\r\n")
        result = preprocessor.clean_text(path)
        assert "\t" in result
        assert "\n" in result
        os.unlink(path)

    def test_arabic_alef_normalization(self, preprocessor):
        # آ أ إ should all become ا
        path = _write_temp_file("أحمد إبراهيم آمن")
        result = preprocessor.clean_text(path)
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result
        assert "ا" in result
        os.unlink(path)

    def test_arabic_normalization_disabled(self, preprocessor_no_arabic):
        path = _write_temp_file("أحمد إبراهيم")
        result = preprocessor_no_arabic.clean_text(path)
        assert "أ" in result
        assert "إ" in result
        os.unlink(path)

    def test_invalid_utf8_raises_with_offset(self, preprocessor):
        # Valid ASCII followed by invalid byte sequence
        invalid_bytes = b"Hello \xff world"
        path = _write_temp_file(invalid_bytes)
        with pytest.raises(UnicodeDecodeError) as exc_info:
            preprocessor.clean_text(path)
        assert exc_info.value.start == 6  # byte offset of \xff
        os.unlink(path)

    def test_file_not_found(self, preprocessor):
        with pytest.raises(FileNotFoundError):
            preprocessor.clean_text("/nonexistent/path/file.txt")

    def test_empty_file(self, preprocessor):
        path = _write_temp_file("")
        result = preprocessor.clean_text(path)
        assert result == ""
        os.unlink(path)

    def test_mixed_arabic_english(self, preprocessor):
        path = _write_temp_file("Hello أهلا World عالم")
        result = preprocessor.clean_text(path)
        assert "Hello" in result
        assert "World" in result
        assert "اهلا" in result  # أ normalized to ا
        os.unlink(path)


# ---------------------------------------------------------------------------
# get_corpus_stats tests
# ---------------------------------------------------------------------------


class TestGetCorpusStats:
    def test_basic_stats(self, preprocessor):
        text = "Line one\nLine two\nLine three\n"
        stats = preprocessor.get_corpus_stats(text)
        assert stats["line_count"] == 3
        assert stats["char_count"] == len(text)
        assert stats["byte_size"] == len(text.encode("utf-8"))

    def test_empty_string(self, preprocessor):
        stats = preprocessor.get_corpus_stats("")
        assert stats["line_count"] == 0
        assert stats["char_count"] == 0
        assert stats["byte_size"] == 0

    def test_single_line_no_trailing_newline(self, preprocessor):
        stats = preprocessor.get_corpus_stats("Hello")
        assert stats["line_count"] == 1

    def test_arabic_text_byte_size(self, preprocessor):
        text = "مرحبا"
        stats = preprocessor.get_corpus_stats(text)
        assert stats["char_count"] == 5
        # Arabic chars are multi-byte in UTF-8
        assert stats["byte_size"] > stats["char_count"]


# ---------------------------------------------------------------------------
# validate_corpus tests
# ---------------------------------------------------------------------------


class TestValidateCorpus:
    def test_meets_line_threshold(self, preprocessor):
        text = "\n".join(f"Line {i}" for i in range(1000))
        assert preprocessor.validate_corpus(text, min_lines=1000, min_bytes=999_999_999) is True

    def test_meets_byte_threshold(self, preprocessor):
        text = "x" * 2_000_000
        assert preprocessor.validate_corpus(text, min_lines=999_999, min_bytes=2_000_000) is True

    def test_meets_both(self, preprocessor):
        text = "\n".join("x" * 2000 for _ in range(1000))
        assert preprocessor.validate_corpus(text, min_lines=1000, min_bytes=2_000_000) is True

    def test_meets_neither(self, preprocessor):
        text = "short"
        assert preprocessor.validate_corpus(text, min_lines=1000, min_bytes=2_000_000) is False

    def test_or_semantics_lines_only(self, preprocessor):
        # Enough lines but not enough bytes
        text = "\n".join("a" for _ in range(1001))
        assert preprocessor.validate_corpus(text, min_lines=1000, min_bytes=999_999_999) is True

    def test_or_semantics_bytes_only(self, preprocessor):
        # Enough bytes but not enough lines
        text = "a" * 3_000_000
        assert preprocessor.validate_corpus(text, min_lines=999_999, min_bytes=2_000_000) is True


# ---------------------------------------------------------------------------
# parse_sft_data tests
# ---------------------------------------------------------------------------


class TestParseSftData:
    def test_valid_jsonl(self, preprocessor):
        records = [
            {"instruction": "Translate", "output": "ترجم"},
            {"instruction": "Summarize", "output": "لخص", "input": "some text"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 2
        assert result[0]["instruction"] == "Translate"
        assert result[1]["input"] == "some text"
        os.unlink(path)

    def test_valid_json_list(self, preprocessor):
        records = [
            {"instruction": "Translate", "output": "ترجم"},
            {"instruction": "Summarize", "output": "لخص"},
        ]
        path = _write_temp_file(json.dumps(records), suffix=".json")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 2
        os.unlink(path)

    def test_skips_missing_instruction(self, preprocessor):
        records = [
            {"output": "ترجم"},  # missing instruction
            {"instruction": "Summarize", "output": "لخص"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 1
        assert result[0]["instruction"] == "Summarize"
        os.unlink(path)

    def test_skips_missing_output(self, preprocessor):
        records = [
            {"instruction": "Translate"},  # missing output
            {"instruction": "Summarize", "output": "لخص"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 1
        os.unlink(path)

    def test_skips_empty_instruction(self, preprocessor):
        records = [
            {"instruction": "", "output": "ترجم"},
            {"instruction": "Summarize", "output": "لخص"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 1
        os.unlink(path)

    def test_skips_empty_output(self, preprocessor):
        records = [
            {"instruction": "Translate", "output": ""},
            {"instruction": "Summarize", "output": "لخص"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 1
        os.unlink(path)

    def test_skips_whitespace_only_instruction(self, preprocessor):
        records = [
            {"instruction": "   ", "output": "ترجم"},
            {"instruction": "Summarize", "output": "لخص"},
        ]
        content = "\n".join(json.dumps(r) for r in records)
        path = _write_temp_file(content, suffix=".jsonl")
        result = preprocessor.parse_sft_data(path)
        assert len(result) == 1
        os.unlink(path)

    def test_file_not_found(self, preprocessor):
        with pytest.raises(FileNotFoundError):
            preprocessor.parse_sft_data("/nonexistent/file.jsonl")

    def test_invalid_json_raises(self, preprocessor):
        path = _write_temp_file("not valid json\n", suffix=".jsonl")
        with pytest.raises(json.JSONDecodeError):
            preprocessor.parse_sft_data(path)
        os.unlink(path)

    def test_preserves_optional_input_field(self, preprocessor):
        records = [{"instruction": "Q", "output": "A", "input": "context"}]
        path = _write_temp_file(json.dumps(records), suffix=".json")
        result = preprocessor.parse_sft_data(path)
        assert "input" in result[0]
        assert result[0]["input"] == "context"
        os.unlink(path)

    def test_no_input_field_when_absent(self, preprocessor):
        records = [{"instruction": "Q", "output": "A"}]
        path = _write_temp_file(json.dumps(records), suffix=".json")
        result = preprocessor.parse_sft_data(path)
        assert "input" not in result[0]
        os.unlink(path)


# ---------------------------------------------------------------------------
# validate_sft_dataset tests
# ---------------------------------------------------------------------------


class TestValidateSftDataset:
    def test_meets_threshold(self, preprocessor):
        records = [
            {"instruction": f"Q{i}", "output": f"A{i}"} for i in range(200)
        ]
        assert preprocessor.validate_sft_dataset(records, min_pairs=200) is True

    def test_below_threshold(self, preprocessor):
        records = [
            {"instruction": f"Q{i}", "output": f"A{i}"} for i in range(199)
        ]
        assert preprocessor.validate_sft_dataset(records, min_pairs=200) is False

    def test_duplicates_not_counted(self, preprocessor):
        records = [{"instruction": "Q", "output": "A"}] * 300
        assert preprocessor.validate_sft_dataset(records, min_pairs=2) is False

    def test_unique_pairs_counted(self, preprocessor):
        records = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q1", "output": "A2"},  # same instruction, different output
            {"instruction": "Q2", "output": "A1"},  # different instruction, same output
        ]
        assert preprocessor.validate_sft_dataset(records, min_pairs=3) is True

    def test_empty_records(self, preprocessor):
        assert preprocessor.validate_sft_dataset([], min_pairs=1) is False


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================

import re
import string

from hypothesis import given, settings
from hypothesis import strategies as st


# Feature: gpt-from-scratch, Property 1: Preprocessor produces valid UTF-8 output
@settings(max_examples=100)
@given(
    raw_text=st.text(
        alphabet=st.characters(
            categories=("L", "M", "N", "P", "S", "Z"),
        ),
        min_size=0,
        max_size=500,
    ).flatmap(
        lambda base: st.tuples(st.just(base), st.booleans(), st.booleans()).map(
            lambda t: (
                t[0]
                + ("<b>tag</b>" if t[1] else "")
                + ("\x00\x01\x02" if t[2] else "")
            )
        )
    ),
)
def test_property_preprocessor_valid_utf8(raw_text):
    """**Validates: Requirements 1.1, 1.2**

    For any raw text input (including HTML tags, control characters, mixed
    encodings), the cleaned output SHALL contain only valid UTF-8 characters
    with no HTML tags or control characters remaining, while all meaningful
    alphabetic and numeric content is preserved.
    """
    preprocessor = DataPreprocessor(normalize_arabic=False)
    path = _write_temp_file(raw_text)
    try:
        result = preprocessor.clean_text(path)

        # Output must be valid UTF-8
        result.encode("utf-8")

        # No HTML tags remaining
        assert not re.search(r"<[^>]+>", result), "HTML tags found in output"

        # No control characters remaining (except \t, \n, \r)
        control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        assert not control_re.search(result), "Control characters found in output"

        # All alphanumeric characters from the original (outside HTML tags and
        # control chars) should be preserved in the output
        stripped_of_html = re.sub(r"<[^>]+>", "", raw_text)
        stripped_of_ctrl = control_re.sub("", stripped_of_html)
        for ch in stripped_of_ctrl:
            if ch.isalnum():
                assert ch in result, f"Alphanumeric char '{ch}' lost during cleaning"
    finally:
        os.unlink(path)


# Feature: gpt-from-scratch, Property 2: Arabic normalization is idempotent
@settings(max_examples=100)
@given(
    arabic_text=st.text(
        alphabet=st.sampled_from(
            list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإؤئ") + list("ًٌٍَُِّْ")
        ),
        min_size=1,
        max_size=200,
    ),
)
def test_property_arabic_normalization_idempotent(arabic_text):
    """**Validates: Requirements 1.3**

    For any Arabic text string, applying normalization once produces the same
    result as applying it twice: normalize(normalize(text)) == normalize(text).
    """
    preprocessor = DataPreprocessor(normalize_arabic=True, alef_normalization=True)

    # Write text, clean once
    path1 = _write_temp_file(arabic_text)
    try:
        once = preprocessor.clean_text(path1)
    finally:
        os.unlink(path1)

    # Write cleaned text, clean again
    path2 = _write_temp_file(once)
    try:
        twice = preprocessor.clean_text(path2)
    finally:
        os.unlink(path2)

    assert once == twice, (
        f"Normalization not idempotent:\n  once={once!r}\n  twice={twice!r}"
    )


# Feature: gpt-from-scratch, Property 3: Corpus validation thresholds
@settings(max_examples=100)
@given(
    num_lines=st.integers(min_value=0, max_value=2000),
    line_length=st.integers(min_value=1, max_value=500),
    min_lines=st.integers(min_value=1, max_value=1500),
    min_bytes=st.integers(min_value=1, max_value=500_000),
)
def test_property_corpus_validation_thresholds(num_lines, line_length, min_lines, min_bytes):
    """**Validates: Requirements 1.4**

    For any text corpus, validate_corpus returns True if and only if the corpus
    has >= min_lines lines OR >= min_bytes bytes.
    """
    preprocessor = DataPreprocessor()

    # Build a corpus with the given number of lines
    if num_lines == 0:
        corpus = ""
    else:
        corpus = "\n".join("a" * line_length for _ in range(num_lines))

    stats = preprocessor.get_corpus_stats(corpus)
    expected = stats["line_count"] >= min_lines or stats["byte_size"] >= min_bytes
    actual = preprocessor.validate_corpus(corpus, min_lines=min_lines, min_bytes=min_bytes)

    assert actual == expected, (
        f"validate_corpus mismatch: lines={stats['line_count']}, bytes={stats['byte_size']}, "
        f"min_lines={min_lines}, min_bytes={min_bytes}, expected={expected}, got={actual}"
    )


# Feature: gpt-from-scratch, Property 4: SFT record validation
@settings(max_examples=100)
@given(
    records=st.lists(
        st.fixed_dictionaries(
            {},
            optional={
                "instruction": st.one_of(st.none(), st.text(min_size=0, max_size=50)),
                "output": st.one_of(st.none(), st.text(min_size=0, max_size=50)),
                "input": st.text(min_size=0, max_size=50),
            },
        ),
        min_size=0,
        max_size=20,
    ),
)
def test_property_sft_record_validation(records):
    """**Validates: Requirements 2.1, 2.2**

    For any list of JSON records, parse_sft_data returns only records with
    non-empty "instruction" and "output" fields. Records missing either field
    or having empty values are excluded.
    """
    preprocessor = DataPreprocessor()

    # Write records as JSON list to a temp file
    content = json.dumps(records)
    path = _write_temp_file(content, suffix=".json")
    try:
        result = preprocessor.parse_sft_data(path)
    finally:
        os.unlink(path)

    # Every returned record must have non-empty instruction and output
    for r in result:
        assert "instruction" in r and isinstance(r["instruction"], str)
        assert r["instruction"].strip(), "Returned record has empty instruction"
        assert "output" in r and isinstance(r["output"], str)
        assert r["output"].strip(), "Returned record has empty output"

    # Count how many input records should be valid
    expected_count = 0
    for rec in records:
        instr = rec.get("instruction")
        out = rec.get("output")
        if (
            isinstance(instr, str)
            and instr.strip()
            and isinstance(out, str)
            and out.strip()
        ):
            expected_count += 1

    assert len(result) == expected_count, (
        f"Expected {expected_count} valid records, got {len(result)}"
    )


# Feature: gpt-from-scratch, Property 5: SFT dataset size validation
@settings(max_examples=100)
@given(
    num_unique=st.integers(min_value=0, max_value=50),
    num_duplicates=st.integers(min_value=0, max_value=10),
    min_pairs=st.integers(min_value=1, max_value=60),
)
def test_property_sft_dataset_size_validation(num_unique, num_duplicates, min_pairs):
    """**Validates: Requirements 2.3**

    For any list of SFT records, validate_sft_dataset returns True if and only
    if the number of unique (instruction, output) pairs >= min_pairs.
    """
    preprocessor = DataPreprocessor()

    # Build records with exactly num_unique unique pairs
    records = [
        {"instruction": f"instr_{i}", "output": f"out_{i}"}
        for i in range(num_unique)
    ]
    # Add duplicates of the first record (if any exist)
    if num_unique > 0:
        records.extend(
            [{"instruction": "instr_0", "output": "out_0"}] * num_duplicates
        )

    expected = num_unique >= min_pairs
    actual = preprocessor.validate_sft_dataset(records, min_pairs=min_pairs)

    assert actual == expected, (
        f"validate_sft_dataset mismatch: unique={num_unique}, "
        f"min_pairs={min_pairs}, expected={expected}, got={actual}"
    )
