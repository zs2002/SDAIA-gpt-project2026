"""Configuration dataclasses for GPT model, pretraining, and supervised fine-tuning."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for the GPT model architecture."""

    vocab_size: int = 8000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If d_model is not divisible by num_heads,
                        if any numeric value is non-positive,
                        or if dropout_rate is outside [0, 1].
        """
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {self.dropout_rate}"
            )
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )


@dataclass
class TrainConfig:
    """Configuration for the pretraining pipeline."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 32
    max_steps: int = 10000
    warmup_steps: int = 500
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    patience: int = 3
    seed: int = 42
    train_split: float = 0.9


@dataclass
class SFTConfig:
    """Configuration for the supervised fine-tuning pipeline."""

    learning_rate: float = 1e-5
    batch_size: int = 16
    max_steps: int = 2000
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 500
    separator_tokens: str = "\n### "
    seed: int = 42
