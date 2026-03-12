"""Unit tests for GPTConfig, TrainConfig, and SFTConfig dataclasses."""

import pytest

from src.config import GPTConfig, SFTConfig, TrainConfig


class TestGPTConfig:
    """Tests for GPTConfig dataclass and its validate() method."""

    def test_default_values(self):
        config = GPTConfig()
        assert config.vocab_size == 8000
        assert config.d_model == 256
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.max_seq_len == 512
        assert config.dropout_rate == 0.1

    def test_validate_passes_with_defaults(self):
        config = GPTConfig()
        config.validate()  # should not raise

    def test_validate_passes_custom_valid(self):
        config = GPTConfig(vocab_size=1000, d_model=64, num_heads=4, num_layers=2, max_seq_len=128, dropout_rate=0.0)
        config.validate()

    def test_validate_d_model_not_divisible_by_num_heads(self):
        config = GPTConfig(d_model=100, num_heads=3)
        with pytest.raises(ValueError, match="divisible"):
            config.validate()

    def test_validate_vocab_size_non_positive(self):
        config = GPTConfig(vocab_size=0)
        with pytest.raises(ValueError, match="vocab_size"):
            config.validate()

    def test_validate_vocab_size_negative(self):
        config = GPTConfig(vocab_size=-1)
        with pytest.raises(ValueError, match="vocab_size"):
            config.validate()

    def test_validate_d_model_non_positive(self):
        config = GPTConfig(d_model=0)
        with pytest.raises(ValueError, match="d_model"):
            config.validate()

    def test_validate_num_heads_non_positive(self):
        config = GPTConfig(num_heads=0)
        with pytest.raises(ValueError, match="num_heads"):
            config.validate()

    def test_validate_num_layers_non_positive(self):
        config = GPTConfig(num_layers=0)
        with pytest.raises(ValueError, match="num_layers"):
            config.validate()

    def test_validate_max_seq_len_non_positive(self):
        config = GPTConfig(max_seq_len=0)
        with pytest.raises(ValueError, match="max_seq_len"):
            config.validate()

    def test_validate_dropout_rate_negative(self):
        config = GPTConfig(dropout_rate=-0.1)
        with pytest.raises(ValueError, match="dropout_rate"):
            config.validate()

    def test_validate_dropout_rate_above_one(self):
        config = GPTConfig(dropout_rate=1.1)
        with pytest.raises(ValueError, match="dropout_rate"):
            config.validate()

    def test_validate_dropout_rate_zero(self):
        config = GPTConfig(dropout_rate=0.0)
        config.validate()  # boundary: should pass

    def test_validate_dropout_rate_one(self):
        config = GPTConfig(dropout_rate=1.0)
        config.validate()  # boundary: should pass


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        config = TrainConfig()
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.batch_size == 32
        assert config.max_steps == 10000
        assert config.warmup_steps == 500
        assert config.log_interval == 100
        assert config.eval_interval == 500
        assert config.save_interval == 1000
        assert config.patience == 3
        assert config.seed == 42
        assert config.train_split == 0.9

    def test_custom_values(self):
        config = TrainConfig(learning_rate=1e-3, batch_size=64, seed=123)
        assert config.learning_rate == 1e-3
        assert config.batch_size == 64
        assert config.seed == 123


class TestSFTConfig:
    """Tests for SFTConfig dataclass."""

    def test_default_values(self):
        config = SFTConfig()
        assert config.learning_rate == 1e-5
        assert config.batch_size == 16
        assert config.max_steps == 2000
        assert config.log_interval == 50
        assert config.eval_interval == 200
        assert config.save_interval == 500
        assert config.separator_tokens == "\n### "
        assert config.seed == 42

    def test_custom_values(self):
        config = SFTConfig(learning_rate=5e-6, separator_tokens="<sep>", seed=99)
        assert config.learning_rate == 5e-6
        assert config.separator_tokens == "<sep>"
        assert config.seed == 99
