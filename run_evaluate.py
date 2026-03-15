"""Main evaluation script.

Chains: checkpoint loading → perplexity computation → sample generation
→ loss plotting → error analysis.

Usage:
    python run_evaluate.py [--checkpoint checkpoints/finetuned/checkpoint_latest.pt]
                           [--tokenizer checkpoints/pretrained/tokenizer.json]
"""

import argparse
import json
import os

import torch

from src.config import GPTConfig
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.metrics import Evaluator
from src.generation.generator import TextGenerator
from src.model.transformer import GPTModel
from src.tokenizer.bpe_tokenizer import BPETokenizer


SAMPLE_PROMPTS = [
    "في يوم من الأيام",
    "The quick brown fox",
    "كان هناك ملك عادل",
    "Once upon a time",
    "مرحبا بالعالم",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT model")
    parser.add_argument("--checkpoint",
                        default="checkpoints/finetuned/checkpoint_latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer",
                        default="checkpoints/pretrained/tokenizer.json",
                        help="Path to saved tokenizer JSON")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for evaluation outputs")
    args = parser.parse_args()

    # 1. Load tokenizer and model
    print("=== Step 1: Loading model ===")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gpt_config = ckpt["config"]
    model = GPTModel(gpt_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    evaluator = Evaluator()
    generator = TextGenerator(model, tokenizer)

    # 2. Compute perplexity (if train losses available in checkpoint)
    print("\n=== Step 2: Perplexity ===")
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    if val_losses:
        final_ppl = evaluator.perplexity_from_loss(val_losses[-1])
        print(f"Perplexity (from val loss): {final_ppl:.2f}")
    else:
        print("No validation losses in checkpoint, skipping perplexity.")

    # 3. Generate samples
    print("\n=== Step 3: Sample generation ===")
    samples_dir = os.path.join(args.output_dir, "sample_generations")
    samples = evaluator.generate_samples(generator, SAMPLE_PROMPTS, samples_dir)
    for prompt, text in zip(SAMPLE_PROMPTS, samples):
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:120]}...")

    # 4. Plot loss curves
    print("\n=== Step 4: Loss curves ===")
    if train_losses or val_losses:
        plots_dir = os.path.join(args.output_dir, "plots")
        evaluator.plot_loss_curves(train_losses, val_losses, plots_dir)
        print(f"Loss curves saved to {plots_dir}/")
    else:
        print("No loss data in checkpoint, skipping plots.")

    # 5. Error analysis
    print("\n=== Step 5: Error analysis ===")
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(samples)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
