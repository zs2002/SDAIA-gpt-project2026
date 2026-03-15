"""Main fine-tuning script.

Chains: SFT data loading → preprocessing → checkpoint loading → SFT training
→ checkpoint saving.

Usage:
    python run_finetune.py [--sft-data data/finetune/arabic_sft.jsonl]
                           [--checkpoint checkpoints/pretrained/checkpoint_latest.pt]
                           [--tokenizer checkpoints/pretrained/tokenizer.json]
                           [--max-steps 2000]
"""

import argparse
import os

import torch

from src.config import GPTConfig, SFTConfig
from src.data.preprocessor import DataPreprocessor
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.finetune import SFTPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GPT model with SFT")
    parser.add_argument("--sft-data", default="data/finetune/arabic_sft.jsonl",
                        help="Path to SFT JSONL file")
    parser.add_argument("--checkpoint",
                        default="checkpoints/pretrained/checkpoint_latest.pt",
                        help="Path to pretrained checkpoint")
    parser.add_argument("--tokenizer",
                        default="checkpoints/pretrained/tokenizer.json",
                        help="Path to saved tokenizer JSON")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load tokenizer
    print("=== Step 1: Loading tokenizer ===")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # 2. Load pretrained checkpoint
    print("\n=== Step 2: Loading pretrained checkpoint ===")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gpt_config = ckpt["config"]
    model = GPTModel(gpt_config)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    print(f"Parameters: {model.count_parameters():,}")

    # 3. Parse SFT data
    print("\n=== Step 3: Parsing SFT data ===")
    dp = DataPreprocessor(normalize_arabic=True)
    sft_records = dp.parse_sft_data(args.sft_data)
    print(f"Valid SFT records: {len(sft_records)}")

    # 4. Fine-tune
    print("\n=== Step 4: Fine-tuning ===")
    sft_config = SFTConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    pipeline = SFTPipeline(model, sft_config, tokenizer)
    results = pipeline.train(sft_records)

    print(f"\nFine-tuning complete.")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    if results['val_losses']:
        print(f"Final val loss:   {results['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
