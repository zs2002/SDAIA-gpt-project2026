"""Main pretraining script.

Chains: data loading → preprocessing → tokenizer training → model creation
→ pretraining → checkpoint saving.

Usage:
    python run_pretrain.py [--corpus data/pretrain/data.txt]
                           [--vocab-size 8000]
                           [--max-steps 10000]
                           [--batch-size 32]
                           [--seed 42]
"""

import argparse
import os
import sys

from src.config import GPTConfig, TrainConfig
from src.data.preprocessor import DataPreprocessor
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.pretrain import PretrainPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GPT model")
    parser.add_argument("--corpus", default="data/pretrain/data.txt",
                        help="Path to pretraining corpus text file")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Preprocess corpus
    print("=== Step 1: Preprocessing corpus ===")
    dp = DataPreprocessor(normalize_arabic=True)
    corpus = dp.clean_text(args.corpus)
    stats = dp.get_corpus_stats(corpus)
    print(f"Corpus: {stats['line_count']} lines, "
          f"{stats['byte_size']:,} bytes, {stats['char_count']:,} chars")

    # 2. Train tokenizer
    print("\n=== Step 2: Training BPE tokenizer ===")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    tok_path = os.path.join("checkpoints", "pretrained", "tokenizer.json")
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tokenizer.save(tok_path)
    print(f"Tokenizer saved to {tok_path}")

    # 3. Create model
    print("\n=== Step 3: Creating GPT model ===")
    gpt_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )
    gpt_config.validate()
    model = GPTModel(gpt_config)
    print(f"Parameters: {model.count_parameters():,}")

    # 4. Pretrain
    print("\n=== Step 4: Pretraining ===")
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    pipeline = PretrainPipeline(model, train_config, tokenizer)
    results = pipeline.train(corpus=corpus)

    print(f"\nPretraining complete.")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    if results['val_losses']:
        print(f"Final val loss:   {results['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
