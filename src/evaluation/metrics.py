"""Evaluation metrics: perplexity, sample generation, and loss curve plotting.

Provides the Evaluator class for computing perplexity on a test set,
generating sample texts from prompts, and plotting training/validation
loss curves.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch import Tensor

from src.model.transformer import GPTModel


class Evaluator:
    """Evaluation utilities for GPT models.

    Computes perplexity, generates sample texts, and plots loss curves.
    """

    @torch.no_grad()
    def compute_perplexity(self, model: GPTModel, test_data: Tensor) -> float:
        """Compute perplexity as exp(average cross-entropy loss) on *test_data*.

        Args:
            model: A GPTModel instance.
            test_data: 1-D tensor of token IDs representing the test set.

        Returns:
            Perplexity value (float). Returns inf if the test data is too
            short to form at least one input-target pair.
        """
        model.eval()
        max_seq_len = model.config.max_seq_len

        if test_data.dim() != 1 or test_data.size(0) < 2:
            return float("inf")

        device = next(model.parameters()).device
        test_data = test_data.to(device)

        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_chunks = 0

        # Process test data in chunks of max_seq_len + 1 (input + target)
        for start in range(0, test_data.size(0) - 1, max_seq_len):
            end = min(start + max_seq_len + 1, test_data.size(0))
            chunk = test_data[start:end]
            if chunk.size(0) < 2:
                break

            input_ids = chunk[:-1].unsqueeze(0)  # (1, seq_len)
            targets = chunk[1:]  # (seq_len,)

            logits = model(input_ids)  # (1, seq_len, vocab_size)
            logits = logits.squeeze(0)  # (seq_len, vocab_size)

            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            num_chunks += 1

        if num_chunks == 0:
            return float("inf")

        avg_loss = total_loss / num_chunks
        return math.exp(avg_loss)

    def generate_samples(
        self,
        generator,
        prompts: list[str],
        output_dir: str,
    ) -> list[str]:
        """Generate sample texts and save them to *output_dir*.

        Args:
            generator: A TextGenerator instance.
            prompts: List of prompt strings.
            output_dir: Directory to save generated samples.

        Returns:
            List of generated text strings.
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_texts: list[str] = []

        for i, prompt in enumerate(prompts):
            text, _ids = generator.generate(prompt)
            generated_texts.append(text)

            filepath = os.path.join(output_dir, f"sample_{i:03d}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated:\n{text}\n")

        return generated_texts

    def plot_loss_curves(
        self,
        train_losses: list,
        val_losses: list,
        output_dir: str,
    ) -> None:
        """Plot and save training/validation loss curves.

        Args:
            train_losses: List of training loss values.
            val_losses: List of validation loss values.
            output_dir: Directory to save the plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_losses:
            ax.plot(range(len(train_losses)), train_losses, label="Train Loss")
        if val_losses:
            ax.plot(range(len(val_losses)), val_losses, label="Val Loss")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(output_dir, "loss_curves.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def perplexity_from_loss(avg_loss: float) -> float:
        """Compute perplexity from an average cross-entropy loss value.

        Args:
            avg_loss: Non-negative average cross-entropy loss.

        Returns:
            exp(avg_loss).
        """
        return math.exp(avg_loss)
