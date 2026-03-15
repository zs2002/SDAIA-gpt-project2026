"""Text generation with configurable sampling strategies.

Implements autoregressive text generation from a GPT model with support for
greedy decoding, top-k sampling, top-p (nucleus) sampling, and temperature
scaling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer


class TextGenerator:
    """Autoregressive text generator for GPT models.

    Supports greedy decoding (temperature=0), temperature-scaled sampling,
    top-k filtering, and top-p (nucleus) filtering. Generation stops at
    ``<eos>`` token or ``max_new_tokens`` limit.
    """

    EOS_TOKEN_ID = BPETokenizer.SPECIAL_TOKENS["<eos>"]

    def __init__(self, model: GPTModel, tokenizer: BPETokenizer) -> None:
        """Initialize with a GPT model and BPE tokenizer.

        Args:
            model: A GPTModel instance (should have loaded weights).
            tokenizer: A trained BPETokenizer for encoding/decoding text.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> tuple[str, list[int]]:
        """Generate text from a prompt string.

        Args:
            prompt: Input text to condition generation on. Must be non-empty.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. 0 means greedy (argmax).
            top_k: If set, keep only the top-k highest-probability tokens.
            top_p: If set, keep the smallest set of tokens whose cumulative
                probability exceeds p (nucleus sampling).

        Returns:
            A tuple of (generated_text, generated_token_ids) where
            generated_token_ids contains only the newly generated tokens
            (not the prompt tokens).

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        self.model.eval()

        # Encode prompt to token IDs
        prompt_ids = self.tokenizer.encode(prompt)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated_ids: list[int] = []
        max_seq_len = self.model.config.max_seq_len

        for _ in range(max_new_tokens):
            # Truncate input to max_seq_len (keep most recent tokens)
            if input_ids.size(1) > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]

            # Forward pass — logits shape: (1, seq_len, vocab_size)
            logits = self.model(input_ids)

            # Take logits for the last position
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Select next token
            next_token_id = self._sample_token(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            generated_ids.append(next_token_id)

            # Stop on <eos>
            if next_token_id == self.EOS_TOKEN_ID:
                break

            # Append to input sequence for next iteration
            next_tensor = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=device
            )
            input_ids = torch.cat([input_ids, next_tensor], dim=1)

        # Decode only the generated tokens
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text, generated_ids

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> int:
        """Sample a single token from logits with the given strategy.

        Args:
            logits: Raw logits of shape (1, vocab_size).
            temperature: Sampling temperature (0 = greedy).
            top_k: Optional top-k filter.
            top_p: Optional nucleus (top-p) filter.

        Returns:
            The selected token ID as a Python int.
        """
        logits = logits.squeeze(0)  # (vocab_size,)

        # Greedy decoding
        if temperature == 0:
            return logits.argmax(dim=-1).item()

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = logits.topk(top_k).values[-1]
            logits = logits.where(logits >= kth_val, torch.tensor(float("-inf"), device=logits.device))

        # Top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # Mask tokens beyond the cumulative probability threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back to original ordering
            logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
