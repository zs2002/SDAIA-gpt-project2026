"""Byte-level BPE tokenizer with Arabic UTF-8 support."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from src.tokenizer.vocab import Vocabulary


class BPETokenizer:
    """Byte-Pair Encoding tokenizer operating on UTF-8 bytes.

    Special tokens ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>`` occupy IDs 0-3.
    The tokenizer learns merge rules from a text corpus and uses them to
    encode/decode arbitrary strings.  Because it operates at the byte level,
    Arabic (and any other multi-byte UTF-8) text is handled without splitting
    within a single code point.
    """

    SPECIAL_TOKENS: dict[str, int] = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    def __init__(self, vocab_size: int = 8000) -> None:
        """Initialise with a target vocabulary size.

        Args:
            vocab_size: Desired vocabulary size (including special tokens and
                the 256 base byte tokens).
        """
        self.vocab_size = vocab_size
        self.vocab = Vocabulary()
        self.merges: list[tuple[str, str]] = []

        # Seed special tokens first (IDs 0-3)
        for token, _expected_id in self.SPECIAL_TOKENS.items():
            self.vocab.add_token(token)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str) -> None:
        """Learn BPE merge rules from *text*.

        Populates ``self.merges`` and ``self.vocab`` so that the total
        vocabulary size equals ``self.vocab_size``.
        """
        # Reset state for a fresh training run
        self.vocab = Vocabulary()
        self.merges = []
        for token in self.SPECIAL_TOKENS:
            self.vocab.add_token(token)

        # Add the 256 base byte tokens
        for byte_val in range(256):
            self.vocab.add_token(self._byte_token(byte_val))

        # Convert corpus to list of "words" (each word is a list of byte tokens)
        words = self._corpus_to_words(text)

        # Iteratively merge the most frequent pair
        num_merges = self.vocab_size - len(self.vocab)
        for _ in range(num_merges):
            pair_counts = self._count_pairs(words)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add_token(merged_token)
            self.merges.append(best_pair)
            words = self._apply_merge(words, best_pair, merged_token)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token IDs using learned merges."""
        if not text:
            return []

        tokens = [self._byte_token(b) for b in text.encode("utf-8")]

        # Apply merges in learned order
        for left, right in self.merges:
            tokens = self._apply_merge_to_sequence(tokens, left, right)

        return [self.vocab.get_id(t) for t in tokens]

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to a text string.

        Unknown IDs are replaced with ``<unk>``.
        """
        tokens = [self.vocab.get_token(tid) for tid in token_ids]
        raw_bytes = bytearray()
        for token in tokens:
            if token in self.SPECIAL_TOKENS:
                # Special tokens are not byte-representable; emit <unk> bytes
                raw_bytes.extend("<unk>".encode("utf-8"))
                continue
            raw_bytes.extend(self._token_to_bytes(token))
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocabulary and merges to a JSON file at *path*."""
        data: dict[str, Any] = {
            "token_to_id": self.vocab.token_to_id,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load vocabulary and merges from a JSON file at *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = Vocabulary()
        for token, tid in sorted(data["token_to_id"].items(), key=lambda x: x[1]):
            added_id = self.vocab.add_token(token)
            # Ensure IDs match the saved order
            assert added_id == tid, f"ID mismatch for {token!r}: {added_id} != {tid}"

        self.merges = [tuple(pair) for pair in data["merges"]]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _byte_token(byte_val: int) -> str:
        """Return the canonical string representation of a single byte.

        Uses ``<0xHH>`` notation so byte tokens never collide with real text
        or special tokens.
        """
        return f"<0x{byte_val:02X}>"

    @staticmethod
    def _token_to_bytes(token: str) -> bytes:
        """Convert a (possibly merged) token back to raw bytes."""
        result = bytearray()
        i = 0
        while i < len(token):
            if token[i:i + 1] == "<" and i + 5 <= len(token) and token[i:i + 3] == "<0x" and token[i + 5] == ">":
                hex_str = token[i + 3:i + 5]
                result.append(int(hex_str, 16))
                i += 6
            else:
                # Shouldn't happen for well-formed tokens, but be safe
                result.append(ord(token[i]))
                i += 1
        return bytes(result)

    @staticmethod
    def _corpus_to_words(text: str) -> list[list[str]]:
        """Split corpus into whitespace-delimited words, each as a list of byte tokens."""
        words: list[list[str]] = []
        for word in text.split():
            byte_tokens = [BPETokenizer._byte_token(b) for b in word.encode("utf-8")]
            if byte_tokens:
                words.append(byte_tokens)
        return words

    @staticmethod
    def _count_pairs(words: list[list[str]]) -> Counter:
        """Count adjacent token pairs across all words."""
        counts: Counter = Counter()
        for word in words:
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += 1
        return counts

    @staticmethod
    def _apply_merge(
        words: list[list[str]],
        pair: tuple[str, str],
        merged: str,
    ) -> list[list[str]]:
        """Replace all occurrences of *pair* in *words* with *merged*."""
        new_words: list[list[str]] = []
        for word in words:
            new_words.append(BPETokenizer._apply_merge_to_sequence(word, pair[0], pair[1], merged))
        return new_words

    @staticmethod
    def _apply_merge_to_sequence(
        tokens: list[str],
        left: str,
        right: str,
        merged: str | None = None,
    ) -> list[str]:
        """Merge adjacent *left*+*right* tokens into *merged* in a token list."""
        if merged is None:
            merged = left + right
        result: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                result.append(merged)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result
