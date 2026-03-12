"""Data preprocessing for pretraining corpora and SFT instruction datasets."""

import json
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Arabic alef variants to normalize → bare alef (U+0627)
_ALEF_VARIANTS = re.compile("[\u0622\u0623\u0625]")  # آ أ إ → ا

# HTML tag pattern
_HTML_TAG = re.compile(r"<[^>]+>")

# Control characters (C0/C1) except common whitespace (\t \n \r)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class DataPreprocessor:
    """Clean and validate raw text corpora and SFT instruction datasets."""

    def __init__(
        self, normalize_arabic: bool = True, alef_normalization: bool = True
    ):
        """Configure preprocessing strategies.

        Args:
            normalize_arabic: Whether to apply Arabic text normalization.
            alef_normalization: Whether to normalize alef variants to bare alef.
        """
        self.normalize_arabic = normalize_arabic
        self.alef_normalization = alef_normalization

    def clean_text(self, filepath: str) -> str:
        """Read a UTF-8 file, remove HTML/control chars, normalize Arabic.

        Args:
            filepath: Path to the input text file.

        Returns:
            Cleaned text string.

        Raises:
            UnicodeDecodeError: If the file contains invalid UTF-8, with the
                byte offset of the first invalid sequence.
            FileNotFoundError: If the file does not exist.
        """
        raw_bytes = _read_file_bytes(filepath)
        text = _decode_utf8_strict(raw_bytes)

        # Remove HTML tags
        text = _HTML_TAG.sub("", text)

        # Remove control characters (keep \t, \n, \r)
        text = _CONTROL_CHARS.sub("", text)

        # Arabic normalization
        if self.normalize_arabic and self.alef_normalization:
            text = _ALEF_VARIANTS.sub("\u0627", text)

        return text

    def get_corpus_stats(self, text: str) -> dict:
        """Return corpus statistics.

        Args:
            text: The corpus text.

        Returns:
            Dict with 'line_count', 'byte_size', and 'char_count'.
        """
        return {
            "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            "byte_size": len(text.encode("utf-8")),
            "char_count": len(text),
        }

    def validate_corpus(
        self, text: str, min_lines: int = 1000, min_bytes: int = 2_000_000
    ) -> bool:
        """Check corpus meets minimum size requirements.

        Returns True if the corpus has >= min_lines lines OR >= min_bytes bytes.

        Args:
            text: The corpus text.
            min_lines: Minimum number of lines.
            min_bytes: Minimum byte size.

        Returns:
            True if the corpus meets at least one threshold.
        """
        stats = self.get_corpus_stats(text)
        meets = stats["line_count"] >= min_lines or stats["byte_size"] >= min_bytes
        if not meets:
            logger.warning(
                "Corpus below minimum size: %d lines (need %d), %d bytes (need %d)",
                stats["line_count"],
                min_lines,
                stats["byte_size"],
                min_bytes,
            )
        return meets

    def parse_sft_data(self, filepath: str) -> list[dict]:
        """Parse JSONL or JSON list file and validate SFT records.

        Each valid record must have a non-empty 'instruction' and non-empty
        'output' field. An optional 'input' field is preserved if present.
        Invalid records are skipped with a logged warning.

        Args:
            filepath: Path to a JSONL or JSON file.

        Returns:
            List of valid SFT record dicts.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        raw_text = _read_file_text(filepath)
        records = _parse_json_or_jsonl(raw_text)

        valid: list[dict] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                logger.warning("Record %d: not a dict, skipping", idx)
                continue

            instruction = record.get("instruction")
            output = record.get("output")

            if not instruction or not isinstance(instruction, str) or not instruction.strip():
                logger.warning("Record %d: missing or empty 'instruction' field, skipping", idx)
                continue

            if not output or not isinstance(output, str) or not output.strip():
                logger.warning("Record %d: missing or empty 'output' field, skipping", idx)
                continue

            entry: dict = {
                "instruction": instruction,
                "output": output,
            }
            if "input" in record:
                entry["input"] = record["input"]

            valid.append(entry)

        return valid

    def validate_sft_dataset(
        self, records: list[dict], min_pairs: int = 200
    ) -> bool:
        """Check SFT dataset has enough unique instruction-response pairs.

        Args:
            records: List of validated SFT record dicts.
            min_pairs: Minimum number of unique (instruction, output) pairs.

        Returns:
            True if the dataset meets the threshold.
        """
        unique_pairs = {
            (r["instruction"], r["output"]) for r in records
        }
        meets = len(unique_pairs) >= min_pairs
        if not meets:
            logger.warning(
                "SFT dataset has %d unique pairs (need %d)",
                len(unique_pairs),
                min_pairs,
            )
        return meets


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_file_bytes(filepath: str) -> bytes:
    """Read file as raw bytes."""
    with open(filepath, "rb") as f:
        return f.read()


def _read_file_text(filepath: str) -> str:
    """Read file as UTF-8 text."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _decode_utf8_strict(data: bytes) -> str:
    """Decode bytes as UTF-8, raising UnicodeDecodeError with byte offset."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end, e.reason
        ) from None


def _parse_json_or_jsonl(text: str) -> list:
    """Parse text as either a JSON list or line-delimited JSONL.

    Tries JSON list first; if that fails or doesn't yield a list,
    falls back to JSONL (one JSON object per line).
    """
    stripped = text.strip()

    # Try JSON list first
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed

    # Fall back to JSONL
    records: list = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON at line {line_num}: {e.msg}",
                e.doc,
                e.pos,
            ) from None
    return records
