"""Pretraining pipeline for GPT model.

Implements the full pretraining loop with AdamW optimizer, linear warmup +
cosine decay LR schedule, next-token prediction via cross-entropy loss,
train/val splitting, checkpoint saving, and overfitting detection.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.config import GPTConfig, TrainConfig
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Standalone helpers (testable independently)
# ------------------------------------------------------------------


def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float) -> float:
    """Compute learning rate for a given training step.

    Linear warmup from 0 to *peak_lr* over *warmup_steps*, then cosine
    decay from *peak_lr* to 0 over the remaining steps.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        peak_lr: Peak learning rate reached at the end of warmup.

    Returns:
        Learning rate for the given step.
    """
    if max_steps <= 0:
        return 0.0
    if warmup_steps <= 0:
        # No warmup — pure cosine decay from peak_lr
        progress = min(step / max(max_steps, 1), 1.0)
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * step / warmup_steps
    if step >= max_steps:
        return 0.0
    # Cosine decay
    decay_steps = max_steps - warmup_steps
    decay_progress = (step - warmup_steps) / decay_steps
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def detect_overfitting(val_losses: list[float], patience: int) -> bool:
    """Check whether validation loss has been increasing for *patience* consecutive evals.

    Args:
        val_losses: List of validation loss values in chronological order.
        patience: Number of consecutive increases required to flag overfitting.

    Returns:
        ``True`` if the last *patience* validation losses are strictly
        increasing, ``False`` otherwise.
    """
    if patience <= 0 or len(val_losses) < patience + 1:
        return False
    recent = val_losses[-(patience + 1):]
    return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))


# ------------------------------------------------------------------
# Seed utility
# ------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# PretrainPipeline
# ------------------------------------------------------------------


class PretrainPipeline:
    """Pretraining pipeline for GPT model using next-token prediction.

    Sets up AdamW optimizer, LR schedule (warmup + cosine decay), data
    splits, and runs the training loop with checkpoint saving and
    overfitting detection.
    """

    def __init__(
        self,
        model: GPTModel,
        config: TrainConfig,
        tokenizer: BPETokenizer,
    ) -> None:
        """Initialise the pretraining pipeline.

        Args:
            model: GPTModel instance to train.
            config: TrainConfig with all training hyperparameters.
            tokenizer: Trained BPETokenizer used to encode the corpus.
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, corpus: str) -> tuple[Tensor, Tensor]:
        """Tokenise *corpus* and split into train/val tensors.

        Returns:
            (train_ids, val_ids) — 1-D tensors of token IDs.
        """
        token_ids = self.tokenizer.encode(corpus)
        data = torch.tensor(token_ids, dtype=torch.long)
        split_idx = int(len(data) * self.config.train_split)
        return data[:split_idx], data[split_idx:]

    def _get_batch(self, data: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Sample a random batch of (input, target) pairs from *data*.

        Each sample is a contiguous chunk of *seq_len* tokens; the target
        is the same chunk shifted by one position.

        Returns:
            (x, y) each of shape (batch_size, seq_len).
        """
        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            # Data too short — use what we have
            x = data[:-1].unsqueeze(0)
            y = data[1:].unsqueeze(0)
            return x.to(self.device), y.to(self.device)

        indices = torch.randint(0, max_start, (self.config.batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in indices])
        y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in indices])
        return x.to(self.device), y.to(self.device)


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_val_loss(self, val_data: Tensor, seq_len: int, num_batches: int = 5) -> float:
        """Compute average validation loss over *num_batches* random batches.

        Args:
            val_data: 1-D tensor of validation token IDs.
            seq_len: Sequence length for each sample.
            num_batches: Number of batches to average over.

        Returns:
            Average cross-entropy loss on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        for _ in range(num_batches):
            x, y = self._get_batch(val_data, seq_len)
            logits = self.model(x)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item()
        self.model.train()
        return total_loss / num_batches

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
        """Save a training checkpoint to *checkpoint_dir*.

        The checkpoint includes model state, optimizer state, configs,
        step number, losses, and seed for full reproducibility.

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
        path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint at step %d to %s", step, path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, corpus: str | None = None, token_ids: list[int] | None = None) -> dict[str, list[float]]:
        """Run the pretraining loop.

        Either *corpus* (raw text) or *token_ids* (pre-tokenised) must be
        provided.  Returns a dict with ``train_losses`` and ``val_losses``.

        Args:
            corpus: Raw text corpus to tokenise and train on.
            token_ids: Pre-tokenised corpus as a list of integer IDs.

        Returns:
            ``{'train_losses': [...], 'val_losses': [...]}``
        """
        # Seed for reproducibility
        set_seed(self.config.seed)

        # Prepare data
        if token_ids is not None:
            data = torch.tensor(token_ids, dtype=torch.long)
            split_idx = int(len(data) * self.config.train_split)
            train_data, val_data = data[:split_idx], data[split_idx:]
        elif corpus is not None:
            train_data, val_data = self._prepare_data(corpus)
        else:
            raise ValueError("Either corpus or token_ids must be provided")

        seq_len = self.model.config.max_seq_len
        checkpoint_dir = os.path.join("checkpoints", "pretrained")

        train_losses: list[float] = []
        val_losses: list[float] = []

        self.model.train()

        for step in range(1, self.config.max_steps + 1):
            # Update learning rate
            lr = get_lr(
                step,
                self.config.warmup_steps,
                self.config.max_steps,
                self.config.learning_rate,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            x, y = self._get_batch(train_data, seq_len)
            logits = self.model(x)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training loss
            if step % self.config.log_interval == 0:
                train_losses.append(loss.item())
                logger.info(
                    "Step %d/%d | LR: %.6f | Train Loss: %.4f",
                    step,
                    self.config.max_steps,
                    lr,
                    loss.item(),
                )

            # Evaluate on validation set
            if step % self.config.eval_interval == 0:
                val_loss = self._compute_val_loss(val_data, seq_len)
                val_losses.append(val_loss)
                logger.info(
                    "Step %d/%d | Val Loss: %.4f",
                    step,
                    self.config.max_steps,
                    val_loss,
                )

                # Overfitting detection
                if detect_overfitting(val_losses, self.config.patience):
                    logger.warning(
                        "Overfitting detected: validation loss increased for "
                        "%d consecutive evaluations",
                        self.config.patience,
                    )

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self._save_checkpoint(step, train_losses, val_losses, checkpoint_dir)

        # Final checkpoint
        self._save_checkpoint(
            self.config.max_steps, train_losses, val_losses, checkpoint_dir
        )

        return {"train_losses": train_losses, "val_losses": val_losses}
