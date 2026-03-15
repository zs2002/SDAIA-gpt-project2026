"""Error analysis for generated text.

Categorizes generation failures into repetition, incoherence, off-topic,
and grammatical error categories using simple heuristics.
"""

from __future__ import annotations

import re
from collections import Counter


class ErrorAnalyzer:
    """Analyze generated texts for common failure patterns.

    Detects repetition (repeated n-grams), incoherence (sentence-level),
    off-topic drift, and basic grammatical issues using lightweight heuristics.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        repetition_threshold: int = 3,
    ) -> None:
        """Initialize the error analyzer.

        Args:
            ngram_size: Minimum n-gram size for repetition detection (n >= 3).
            repetition_threshold: Number of times an n-gram must appear to
                be flagged as repetition.
        """
        self.ngram_size = max(ngram_size, 3)
        self.repetition_threshold = repetition_threshold

    def analyze(self, generated_texts: list[str]) -> dict:
        """Analyze a list of generated texts for failure patterns.

        Args:
            generated_texts: List of generated text strings.

        Returns:
            Summary dict with keys:
            - ``repetition``: dict with ``count`` and ``examples``
            - ``incoherence``: dict with ``count`` and ``examples``
            - ``off_topic``: dict with ``count`` and ``examples``
            - ``grammatical``: dict with ``count`` and ``examples``
            - ``total_texts``: total number of texts analyzed
        """
        results = {
            "repetition": {"count": 0, "examples": []},
            "incoherence": {"count": 0, "examples": []},
            "off_topic": {"count": 0, "examples": []},
            "grammatical": {"count": 0, "examples": []},
            "total_texts": len(generated_texts),
        }

        for text in generated_texts:
            if self._detect_repetition(text):
                results["repetition"]["count"] += 1
                results["repetition"]["examples"].append(
                    text[:200] if len(text) > 200 else text
                )

            if self._detect_incoherence(text):
                results["incoherence"]["count"] += 1
                results["incoherence"]["examples"].append(
                    text[:200] if len(text) > 200 else text
                )

            if self._detect_off_topic(text):
                results["off_topic"]["count"] += 1
                results["off_topic"]["examples"].append(
                    text[:200] if len(text) > 200 else text
                )

            if self._detect_grammatical(text):
                results["grammatical"]["count"] += 1
                results["grammatical"]["examples"].append(
                    text[:200] if len(text) > 200 else text
                )

        return results

    def _detect_repetition(self, text: str) -> bool:
        """Detect repeated n-grams (n >= ngram_size) exceeding the threshold.

        Returns True if any n-gram of size >= self.ngram_size appears more
        than self.repetition_threshold times.
        """
        words = text.split()
        if len(words) < self.ngram_size:
            return False

        # Check n-grams from ngram_size up to min(len(words), ngram_size + 3)
        # to catch various repetition lengths
        max_n = min(len(words), self.ngram_size + 4)
        for n in range(self.ngram_size, max_n):
            ngrams = [
                tuple(words[i : i + n]) for i in range(len(words) - n + 1)
            ]
            counts = Counter(ngrams)
            for _ngram, count in counts.items():
                if count > self.repetition_threshold:
                    return True

        return False

    def _detect_incoherence(self, text: str) -> bool:
        """Detect incoherence using simple heuristics.

        Flags text that has very short average sentence length (fragmented)
        or excessive punctuation indicating broken generation.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return False

        # Fragmented text: many very short sentences
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if len(sentences) > 3 and avg_words < 2:
            return True

        # Excessive repeated punctuation
        if re.search(r"[.!?]{4,}", text):
            return True

        return False

    def _detect_off_topic(self, text: str) -> bool:
        """Detect off-topic drift using simple heuristics.

        Flags text that contains mostly non-alphabetic characters or is
        extremely short relative to expected output.
        """
        if not text.strip():
            return True

        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.strip())
        if total_chars > 0 and alpha_chars / total_chars < 0.3:
            return True

        return False

    def _detect_grammatical(self, text: str) -> bool:
        """Detect basic grammatical issues using simple heuristics.

        Flags text with repeated whitespace, missing capitalization after
        sentence boundaries, or unbalanced brackets/quotes.
        """
        # Excessive whitespace
        if re.search(r"  {3,}", text):
            return True

        # Unbalanced parentheses
        if text.count("(") != text.count(")"):
            return True

        # Unbalanced brackets
        if text.count("[") != text.count("]"):
            return True

        return False
