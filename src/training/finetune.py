"""Supervised Fine-Tuning (SFT) pipeline for GPT model.

Implements the SFT training loop that fine-tunes a pretrained GPT model on
instruction-response pairs. Loss is computed only on output tokens (instruction
tokens are masked). Checkpoints are saved to checkpoints/finetuned/.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.config import SFTConfig
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.pretrain import set_seed

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Standalone helpers (testable independently)
# ------------------------------------------------------------------


def format_sft_example(record: dict, separator: str) -> str:
    """Format a single SFT record into a training string.

    Concatenates instruction + optional input + separator + output.

    Args:
        record: Dict with keys ``instruction``, ``output``, and optionally ``input``.
        separator: Separator string placed between instruction/input and output.

    Returns:
        Formatted training string.
    """
    parts = [record["instruction"]]
    if record.get("input"):
        parts.append(record["input"])
    parts_str = "\n".join(parts)
    return parts_str + separator + record["output"]


def create_loss_mask(instruction_len: int, total_len: int) -> list[int]:
    """Create a loss mask that is 0 for instruction tokens and 1 for output tokens.

    Args:
        instruction_len: Number of tokens belonging to the instruction portion.
        total_len: Total number of tokens in the sequence.

    Returns:
        List of ints (0 or 1) of length *total_len*.
    """
    return [0] * instruction_len + [1] * (total_len - instruction_len)


# ------------------------------------------------------------------
# SFTPipeline
# ------------------------------------------------------------------


class SFTPipeline:
    """Supervised fine-tuning pipeline for GPT model.

    Loads a pretrained checkpoint, sets up an optimizer with a lower learning
    rate, and fine-tunes on instruction-response pairs with loss computed
    only on output tokens.
    """

    def __init__(
        self,
        model: GPTModel,
        config: SFTConfig,
        tokenizer: BPETokenizer,
    ) -> None:
        """Initialise the SFT pipeline.

        Args:
            model: GPTModel instance (weights may already be loaded from a
                   pretrained checkpoint by the caller).
            config: SFTConfig with fine-tuning hyperparameters.
            tokenizer: Trained BPETokenizer used to encode examples.
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # AdamW with lower LR than pretraining
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        # Loss function — reduction='none' so we can apply the mask
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_examples(
        self, sft_data: list[dict]
    ) -> list[tuple[list[int], list[int]]]:
        """Tokenise and prepare (token_ids, loss_mask) pairs for each SFT record.

        Returns:
            List of (token_ids, loss_mask) tuples.
        """
        examples: list[tuple[list[int], list[int]]] = []
        separator = self.config.separator_tokens
        max_seq_len = self.model.config.max_seq_len

        for record in sft_data:
            # Build instruction portion (everything before the output)
            instruction_text = record["instruction"]
            if record.get("input"):
                instruction_text = instruction_text + "\n" + record["input"]
            instruction_text += separator

            instruction_ids = self.tokenizer.encode(instruction_text)
            output_ids = self.tokenizer.encode(record["output"])

            full_ids = instruction_ids + output_ids

            # Truncate to max_seq_len if needed
            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]
                # Adjust instruction_len if truncation cuts into output
                instr_len = min(len(instruction_ids), max_seq_len)
            else:
                instr_len = len(instruction_ids)

            mask = create_loss_mask(instr_len, len(full_ids))
            examples.append((full_ids, mask))

        return examples

    def _get_batch(
        self,
        examples: list[tuple[list[int], list[int]]],
        indices: list[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build a padded batch from the given example indices.

        Returns:
            (input_ids, target_ids, loss_mask) — all of shape (batch, seq_len).
        """
        batch_tokens: list[list[int]] = []
        batch_masks: list[list[int]] = []

        for idx in indices:
            tokens, mask = examples[idx]
            batch_tokens.append(tokens)
            batch_masks.append(mask)

        # Pad to the longest sequence in this batch
        max_len = max(len(t) for t in batch_tokens)
        pad_id = self.tokenizer.SPECIAL_TOKENS.get("<pad>", 0)

        padded_tokens: list[list[int]] = []
        padded_masks: list[list[int]] = []
        for tokens, mask in zip(batch_tokens, batch_masks):
            pad_len = max_len - len(tokens)
            padded_tokens.append(tokens + [pad_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)  # mask=0 for padding

        token_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(padded_masks, dtype=torch.float, device=self.device)

        # Input is tokens[:-1], target is tokens[1:]
        input_ids = token_tensor[:, :-1]
        target_ids = token_tensor[:, 1:]
        # Shift mask to align with targets (mask[1:])
        loss_mask = mask_tensor[:, 1:]

        return input_ids, target_ids, loss_mask

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_val_loss(
        self,
        val_examples: list[tuple[list[int], list[int]]],
    ) -> float:
        """Compute average masked validation loss over all val examples.

        Args:
            val_examples: List of (token_ids, loss_mask) tuples.

        Returns:
            Average cross-entropy loss on output tokens only.
        """
        if not val_examples:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Process in batches
        bs = self.config.batch_size
        for start in range(0, len(val_examples), bs):
            end = min(start + bs, len(val_examples))
            indices = list(range(start, end))
            input_ids, target_ids, loss_mask = self._get_batch(val_examples, indices)

            logits = self.model(input_ids)
            # logits: (batch, seq_len, vocab_size)
            loss_per_token = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )
            loss_per_token = loss_per_token.view_as(target_ids)

            masked_loss = (loss_per_token * loss_mask).sum()
            num_tokens = loss_mask.sum()

            total_loss += masked_loss.item()
            total_tokens += num_tokens.item()

        self.model.train()
        if total_tokens == 0:
            return 0.0
        return total_loss / total_tokens

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        step: int,
        train_losses: list[float],
        val_losses: list[float],
        checkpoint_dir: str,
    ) -> None:
        """Save a fine-tuning checkpoint.

        Args:
            step: Current training step.
            train_losses: Logged training losses so far.
            val_losses: Logged validation losses so far.
            checkpoint_dir: Directory to save the checkpoint file.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "step": step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "seed": self.config.seed,
        }
        path = os.path.join(checkpoint_dir, f"sft_checkpoint_step_{step}.pt")
        torch.save(checkpoint, path)
        logger.info("Saved SFT checkpoint at step %d to %s", step, path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, sft_data: list[dict]) -> dict[str, list[float]]:
        """Run the SFT training loop.

        Fine-tunes on instruction-response pairs with loss computed only on
        output tokens. Saves checkpoints to ``checkpoints/finetuned/``.

        Args:
            sft_data: List of SFT records, each a dict with ``instruction``,
                      ``output``, and optionally ``input`` keys.

        Returns:
            ``{'train_losses': [...], 'val_losses': [...]}``
        """
        set_seed(self.config.seed)

        # Prepare all examples
        all_examples = self._prepare_examples(sft_data)

        # Split into train/val (90/10)
        split_idx = max(1, int(len(all_examples) * 0.9))
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]

        checkpoint_dir = os.path.join("checkpoints", "finetuned")
        train_losses: list[float] = []
        val_losses: list[float] = []

        self.model.train()
        bs = self.config.batch_size

        for step in range(1, self.config.max_steps + 1):
            # Sample a random batch from training examples
            num_examples = len(train_examples)
            indices = torch.randint(0, num_examples, (min(bs, num_examples),)).tolist()

            input_ids, target_ids, loss_mask = self._get_batch(train_examples, indices)

            # Forward pass
            logits = self.model(input_ids)

            # Compute per-token loss
            loss_per_token = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )
            loss_per_token = loss_per_token.view_as(target_ids)

            # Apply mask: loss only on output tokens
            masked_loss = (loss_per_token * loss_mask).sum()
            num_output_tokens = loss_mask.sum()

            if num_output_tokens > 0:
                loss = masked_loss / num_output_tokens
            else:
                loss = masked_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training loss
            if step % self.config.log_interval == 0:
                train_losses.append(loss.item())
                logger.info(
                    "SFT Step %d/%d | Train Loss: %.4f",
                    step,
                    self.config.max_steps,
                    loss.item(),
                )

            # Evaluate on validation set
            if step % self.config.eval_interval == 0:
                val_loss = self._compute_val_loss(val_examples)
                val_losses.append(val_loss)
                logger.info(
                    "SFT Step %d/%d | Val Loss: %.4f",
                    step,
                    self.config.max_steps,
                    val_loss,
                )

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self._save_checkpoint(step, train_losses, val_losses, checkpoint_dir)

        # Final checkpoint
        self._save_checkpoint(
            self.config.max_steps, train_losses, val_losses, checkpoint_dir
        )

        return {"train_losses": train_losses, "val_losses": val_losses}
